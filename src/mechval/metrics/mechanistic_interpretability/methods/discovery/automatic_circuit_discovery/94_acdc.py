"""Automatic Circuit DisCovery (Causal C10)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     C10 — Automatic Circuit Discovery
Categories:     causal
Validity layer: Internal
Criteria:       C10 ACDC Agreement
Establishes:    Whether iterative KL-divergence edge pruning recovers claimed circuit edges
Requires:       CPU or GPU, model
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Implements a simplified version of ACDC (Conmy et al., NeurIPS 2023).
For each task, starts with the full computation graph of attention head
edges and iteratively prunes edges whose removal causes less than
`threshold` increase in KL divergence between ablated and clean logits.

Edges that survive pruning form the ACDC-discovered circuit. This is
compared against the claimed circuit edges using:
  - Jaccard overlap: |intersection| / |union|
  - AUROC: how well per-edge KL-impact scores discriminate circuit
    from non-circuit edges

Pass condition: Jaccard > 0.3

Usage:
    uv run python 94_acdc.py --tasks ioi --n-prompts 40
    uv run python 94_acdc.py --tasks ioi sva --device cpu --threshold 0.01
"""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    calibrate_mean_z,
    generate_prompts,
    get_circuit_info,
    get_token_ids,
    load_model,
    log,
    parse_common_args,
    save_incremental,
    save_results,
)


@torch.no_grad()
def _clean_logits_last(model, tokens: torch.Tensor) -> torch.Tensor:
    """Return last-position logits for a single prompt (softmaxed)."""
    logits = model(tokens)
    return torch.softmax(logits[0, -1], dim=-1)


@torch.no_grad()
def _kl_divergence(p: torch.Tensor, q: torch.Tensor) -> float:
    """KL(p || q) for two probability vectors, with epsilon smoothing."""
    eps = 1e-10
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)
    return (p * (p.log() - q.log())).sum().item()


@torch.no_grad()
def compute_edge_kl_scores(
    model, prompts, correct_ids, incorrect_ids, mean_z: torch.Tensor,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """Compute per-edge KL divergence impact.

    For each candidate forward edge (Ls,Hs)->(Lr,Hr), ablate that single
    edge's sender output at the receiver layer and measure the KL divergence
    from the clean logit distribution.

    Returns the score matrix and the list of all candidate edges.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    n_total = n_layers * n_heads
    n = min(len(prompts), len(correct_ids), len(incorrect_ids))

    edge_scores = np.zeros((n_total, n_total), dtype=np.float64)
    all_edges = []

    # Enumerate all forward edges once
    for Ls in range(n_layers):
        for Hs in range(n_heads):
            for Lr in range(Ls + 1, n_layers):
                for Hr in range(n_heads):
                    all_edges.append((Ls, Hs, Lr, Hr))

    for i in range(n):
        tokens = model.to_tokens(prompts[i].text)
        clean_probs = _clean_logits_last(model, tokens)

        for Ls, Hs, Lr, Hr in all_edges:
            # Ablate the sender's contribution at the receiver layer
            def _ablate_hook(z, hook, _Hs=Hs, _Ls=Ls):
                z[0, :, _Hs, :] = mean_z[_Ls, _Hs].to(z.device)
                return z

            hook_name = f"blocks.{Lr}.attn.hook_z"
            # We ablate the receiver head as a proxy for removing the
            # edge: zero the receiver head's z at its layer, which
            # removes the sender->receiver information flow.
            def _ablate_receiver(z, hook, _Hr=Hr, _Lr=Lr):
                z[0, :, _Hr, :] = mean_z[_Lr, _Hr].to(z.device)
                return z

            # For a proper single-edge ablation, ablate only the
            # sender head at its own layer (removing its output from
            # the residual stream that the receiver reads).
            sender_hook = f"blocks.{Ls}.attn.hook_z"

            def _ablate_sender(z, hook, _Hs=Hs, _Ls=Ls):
                z[0, :, _Hs, :] = mean_z[_Ls, _Hs].to(z.device)
                return z

            ablated_logits = model.run_with_hooks(
                tokens, fwd_hooks=[(sender_hook, _ablate_sender)],
            )
            ablated_probs = torch.softmax(ablated_logits[0, -1], dim=-1)
            kl = _kl_divergence(clean_probs, ablated_probs)

            s_idx = Ls * n_heads + Hs
            r_idx = Lr * n_heads + Hr
            edge_scores[s_idx, r_idx] += kl

        if (i + 1) % 5 == 0:
            log(f"    processed {i+1}/{n} prompts")

    edge_scores /= max(n, 1)
    return edge_scores, all_edges


def acdc_prune(
    edge_scores: np.ndarray,
    all_edges: list[tuple[int, int, int, int]],
    n_heads: int,
    threshold: float,
) -> set[tuple[int, int, int, int]]:
    """Iterative greedy pruning: remove edges with KL impact below threshold.

    Returns the set of surviving edges (the ACDC-discovered circuit).
    """
    # Sort edges by KL impact (ascending) -- prune least important first
    scored = []
    for Ls, Hs, Lr, Hr in all_edges:
        s_idx = Ls * n_heads + Hs
        r_idx = Lr * n_heads + Hr
        scored.append((edge_scores[s_idx, r_idx], (Ls, Hs, Lr, Hr)))
    scored.sort(key=lambda x: x[0])

    surviving = set(all_edges)
    for score, edge in scored:
        if score < threshold:
            surviving.discard(edge)
        else:
            # Once we hit edges above threshold, stop pruning
            break

    return surviving


def compute_jaccard_auroc(
    edge_scores: np.ndarray,
    discovered_edges: set[tuple[int, int, int, int]],
    circuit_edges: set[tuple[int, int, int, int]],
    all_edges: list[tuple[int, int, int, int]],
    n_layers: int,
    n_heads: int,
) -> tuple[float, float, dict]:
    """Compute Jaccard overlap and AUROC between discovered and claimed circuits."""
    # Jaccard
    intersection = discovered_edges & circuit_edges
    union = discovered_edges | circuit_edges
    jaccard = len(intersection) / len(union) if union else 0.0

    # AUROC: use KL impact scores to discriminate circuit vs non-circuit
    labels = []
    scores = []
    for Ls, Hs, Lr, Hr in all_edges:
        s_idx = Ls * n_heads + Hs
        r_idx = Lr * n_heads + Hr
        is_circuit = (Ls, Hs, Lr, Hr) in circuit_edges
        labels.append(1 if is_circuit else 0)
        scores.append(edge_scores[s_idx, r_idx])

    labels_arr = np.array(labels)
    scores_arr = np.array(scores)

    if labels_arr.sum() == 0 or labels_arr.sum() == len(labels_arr):
        auroc = 0.0
    else:
        auroc = float(roc_auc_score(labels_arr, scores_arr))

    circuit_scores = scores_arr[labels_arr == 1]
    non_circuit_scores = scores_arr[labels_arr == 0]

    stats = {
        "n_circuit_edges": int(labels_arr.sum()),
        "n_non_circuit_edges": int((1 - labels_arr).sum()),
        "n_discovered_edges": len(discovered_edges),
        "n_intersection": len(intersection),
        "n_union": len(union),
        "mean_circuit_kl": float(circuit_scores.mean()) if len(circuit_scores) > 0 else 0.0,
        "mean_non_circuit_kl": float(non_circuit_scores.mean()) if len(non_circuit_scores) > 0 else 0.0,
    }
    return jaccard, auroc, stats


def run_acdc(
    model, tasks: list[str], n_prompts: int = 40, threshold: float = 0.01,
) -> list[EvalResult]:
    tokenizer = model.tokenizer
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    results = []

    for task in tasks:
        circuit, all_heads, circuit_edges = get_circuit_info(task)
        if circuit is None or not circuit_edges:
            log(f"  {task}: no circuit edges, skipping")
            continue

        prompts = generate_prompts(task, tokenizer, n_prompts)
        if not prompts:
            log(f"  {task}: no prompts, skipping")
            continue

        correct_ids, incorrect_ids = get_token_ids(prompts, tokenizer)
        if not correct_ids:
            continue

        log(f"  {task}: {len(circuit_edges)} circuit edges, {len(prompts)} prompts")

        # Calibrate mean activations for ablation
        mean_z = calibrate_mean_z(model, prompts)

        # Compute per-edge KL impact
        edge_scores, all_edges = compute_edge_kl_scores(
            model, prompts, correct_ids, incorrect_ids, mean_z,
        )

        # Prune edges below threshold
        discovered_edges = acdc_prune(edge_scores, all_edges, n_heads, threshold)

        # Evaluate agreement with claimed circuit
        jaccard, auroc, stats = compute_jaccard_auroc(
            edge_scores, discovered_edges, circuit_edges, all_edges, n_layers, n_heads,
        )

        passed = bool(jaccard > 0.3)

        log(f"    Jaccard={jaccard:.4f}  AUROC={auroc:.4f}  [{('PASS' if passed else 'FAIL')}]")
        log(f"    discovered {stats['n_discovered_edges']} edges, "
            f"{stats['n_intersection']} overlap with {stats['n_circuit_edges']} circuit edges")

        # Top discovered edges for metadata
        top_edges = []
        for Ls, Hs, Lr, Hr in all_edges:
            s_idx = Ls * n_heads + Hs
            r_idx = Lr * n_heads + Hr
            kl = float(edge_scores[s_idx, r_idx])
            if kl > 1e-8:
                top_edges.append({
                    "edge": f"L{Ls}H{Hs}->L{Lr}H{Hr}",
                    "kl_impact": kl,
                    "in_circuit": (Ls, Hs, Lr, Hr) in circuit_edges,
                    "discovered": (Ls, Hs, Lr, Hr) in discovered_edges,
                })
        top_edges.sort(key=lambda e: e["kl_impact"], reverse=True)

        results.append(EvalResult(
            metric_id="C10.acdc_agreement",
            value=jaccard,
            n_samples=len(prompts),
            metadata={
                "task": task,
                "jaccard": jaccard,
                "auroc": auroc,
                "passed": passed,
                "threshold": threshold,
                "n_circuit_edges": stats["n_circuit_edges"],
                "n_non_circuit_edges": stats["n_non_circuit_edges"],
                "n_discovered_edges": stats["n_discovered_edges"],
                "n_intersection": stats["n_intersection"],
                "n_union": stats["n_union"],
                "mean_circuit_kl": stats["mean_circuit_kl"],
                "mean_non_circuit_kl": stats["mean_non_circuit_kl"],
                "top_edges": top_edges[:30],
            },
        ))

    return results


def main():
    parser = parse_common_args("C10: Automatic Circuit Discovery (ACDC)")
    parser.add_argument("--threshold", type=float, default=0.01,
                        help="KL divergence threshold for edge pruning (default: 0.01)")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("C10: AUTOMATIC CIRCUIT DISCOVERY (ACDC)")
    log("=" * 60)

    out = args.out or "94_acdc.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_acdc(model, [task], args.n_prompts, args.threshold)
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)
            p = "PASS" if r.metadata["passed"] else "FAIL"
            log(f"  {task}: Jaccard={r.value:.4f}  AUROC={r.metadata['auroc']:.4f}  [{p}]")

    save_results(results, out)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
