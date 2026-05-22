"""Position-Aware Edge Attribution Patching (Causal C14)
=====================================================
Instrument:     C14 — Position-Aware EAP
Categories:     causal
Validity layer: Internal
Criteria:       C14 Position-Aware Edge Attribution Discrimination
Establishes:    Whether position-aware gradient-based edge attribution scores
                discriminate circuit edges from non-circuit edges
Requires:       CPU or GPU, model
=====================================================

Extends standard EAP (Syed et al. 2023) following Haklay & Belinkov
(ACL 2025 Oral) by treating each (head, position) pair as a distinct
node.  For each prompt, computes hook_z outputs and gradients at every
position, yielding edge scores between (Ls, Hs, pos_s) and
(Lr, Hr, pos_r) via the dot product of z_s[pos_s] and grad_r[pos_r].

For comparison against head-level circuit edges the metric aggregates
over positions in two ways: position-averaged and max-over-positions.
Both are evaluated via AUROC.

Pass condition: AUROC > 0.70 (on the max-over-positions aggregation)

Usage:
    uv run python 98_position_aware_eap.py --tasks ioi --n-prompts 40
    uv run python 98_position_aware_eap.py --tasks ioi sva --device cpu
"""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    generate_prompts,
    get_circuit_info,
    get_token_ids,
    load_model,
    log,
    parse_common_args,
    save_incremental,
    save_results,
)


def compute_position_aware_eap_scores(
    model, prompts, correct_ids, incorrect_ids,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute position-aware EAP edge scores.

    Returns two (n_total, n_total) arrays — one averaged over position
    pairs, one taking the max absolute score over position pairs — where
    n_total = n_layers * n_heads.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    n = min(len(prompts), len(correct_ids), len(incorrect_ids))
    n_total = n_layers * n_heads

    avg_scores = np.zeros((n_total, n_total), dtype=np.float64)
    max_scores = np.zeros((n_total, n_total), dtype=np.float64)

    for i in range(n):
        tokens = model.to_tokens(prompts[i].text)
        seq_len = tokens.shape[1]
        model.zero_grad()

        cache_dict: dict[int, torch.Tensor] = {}

        def make_cache_hook(layer):
            def hook_fn(z, hook):
                cache_dict[layer] = z
                z.retain_grad()
                return z
            return hook_fn

        fwd_hooks = [
            (f"blocks.{L}.attn.hook_z", make_cache_hook(L))
            for L in range(n_layers)
        ]

        with torch.enable_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)
            logit_diff = logits[0, -1, correct_ids[i]] - logits[0, -1, incorrect_ids[i]]
            logit_diff.backward()

        # For each sender-receiver head pair, compute the full
        # position-cross-position dot product and aggregate.
        for Ls in range(n_layers):
            z_s = cache_dict[Ls][0].detach()  # (seq_len, n_heads, d_head)
            for Lr in range(Ls + 1, n_layers):
                z_r = cache_dict[Lr]
                if z_r.grad is None:
                    continue
                g_r = z_r.grad[0].detach()  # (seq_len, n_heads, d_head)

                # Dot products across all position pairs:
                # cross[pos_s, Hs, pos_r, Hr] = sum_d z_s[pos_s,Hs,d] * g_r[pos_r,Hr,d]
                # Shape: (seq_len, n_heads, seq_len, n_heads)
                cross = torch.einsum("shd,rhd->shrd", z_s, g_r)
                # Reduce over position pairs per head pair
                # cross_abs shape: (n_heads, n_heads) for max, (n_heads, n_heads) for mean
                cross_np = cross.cpu().numpy()

                for Hs in range(n_heads):
                    s_idx = Ls * n_heads + Hs
                    for Hr in range(n_heads):
                        r_idx = Lr * n_heads + Hr
                        pair_scores = cross_np[:, Hs, :, Hr]  # (seq_len, seq_len)
                        avg_scores[s_idx, r_idx] += pair_scores.mean()
                        # Max absolute score over all position pairs
                        abs_vals = np.abs(pair_scores)
                        max_scores[s_idx, r_idx] += abs_vals.max()

        model.zero_grad()

        if (i + 1) % 10 == 0:
            log(f"    processed {i+1}/{n} prompts")

    avg_scores /= max(n, 1)
    max_scores /= max(n, 1)
    return avg_scores, max_scores


def compute_auroc(
    edge_scores: np.ndarray,
    circuit_edges: set[tuple[int, int, int, int]],
    n_layers: int,
    n_heads: int,
) -> tuple[float, dict]:
    labels = []
    scores = []
    for Ls in range(n_layers):
        for Hs in range(n_heads):
            s_idx = Ls * n_heads + Hs
            for Lr in range(Ls + 1, n_layers):
                for Hr in range(n_heads):
                    r_idx = Lr * n_heads + Hr
                    is_circuit = (Ls, Hs, Lr, Hr) in circuit_edges
                    labels.append(1 if is_circuit else 0)
                    scores.append(abs(edge_scores[s_idx, r_idx]))

    labels_arr = np.array(labels)
    scores_arr = np.array(scores)

    if labels_arr.sum() == 0 or labels_arr.sum() == len(labels_arr):
        return 0.0, {"n_circuit": int(labels_arr.sum()), "n_total": len(labels_arr)}

    auroc = float(roc_auc_score(labels_arr, scores_arr))

    circuit_scores = scores_arr[labels_arr == 1]
    non_circuit_scores = scores_arr[labels_arr == 0]

    return auroc, {
        "n_circuit_edges": int(labels_arr.sum()),
        "n_non_circuit_edges": int((1 - labels_arr).sum()),
        "mean_circuit_score": float(circuit_scores.mean()),
        "mean_non_circuit_score": float(non_circuit_scores.mean()),
        "median_circuit_score": float(np.median(circuit_scores)),
        "median_non_circuit_score": float(np.median(non_circuit_scores)),
    }


def run_position_aware_eap(
    model, tasks: list[str], n_prompts: int = 40,
) -> list[EvalResult]:
    tokenizer = model.tokenizer
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    results = []

    for task in tasks:
        circuit, all_heads, all_edges = get_circuit_info(task)
        if circuit is None or not all_edges:
            log(f"  {task}: no circuit edges, skipping")
            continue

        prompts = generate_prompts(task, tokenizer, n_prompts)
        if not prompts:
            log(f"  {task}: no prompts, skipping")
            continue

        correct_ids, incorrect_ids = get_token_ids(prompts, tokenizer)
        if not correct_ids:
            continue

        log(f"  {task}: {len(all_edges)} circuit edges, {len(prompts)} prompts")

        avg_scores, max_scores = compute_position_aware_eap_scores(
            model, prompts, correct_ids, incorrect_ids,
        )

        auroc_avg, stats_avg = compute_auroc(avg_scores, all_edges, n_layers, n_heads)
        auroc_max, stats_max = compute_auroc(max_scores, all_edges, n_layers, n_heads)

        # Primary metric is the max-over-positions aggregation
        passed = bool(auroc_max > 0.70)

        log(f"    AUROC(avg)={auroc_avg:.4f}  AUROC(max)={auroc_max:.4f}  "
            f"[{'PASS' if passed else 'FAIL'}]")
        log(f"    circuit edges: {stats_max.get('n_circuit_edges', 0)}, "
            f"mean|score|={stats_max.get('mean_circuit_score', 0):.4f}")
        log(f"    non-circuit:   {stats_max.get('n_non_circuit_edges', 0)}, "
            f"mean|score|={stats_max.get('mean_non_circuit_score', 0):.4f}")

        # Build top-edges list from max-over-positions scores
        top_edges = []
        for Ls in range(n_layers):
            for Hs in range(n_heads):
                s_idx = Ls * n_heads + Hs
                for Lr in range(Ls + 1, n_layers):
                    for Hr in range(n_heads):
                        r_idx = Lr * n_heads + Hr
                        score_max = float(max_scores[s_idx, r_idx])
                        score_avg = float(avg_scores[s_idx, r_idx])
                        if abs(score_max) > 1e-6:
                            top_edges.append({
                                "edge": f"L{Ls}H{Hs}->L{Lr}H{Hr}",
                                "score_max": score_max,
                                "score_avg": score_avg,
                                "in_circuit": (Ls, Hs, Lr, Hr) in all_edges,
                            })
        top_edges.sort(key=lambda e: abs(e["score_max"]), reverse=True)

        results.append(EvalResult(
            metric_id="C14.position_aware_eap",
            value=auroc_max,
            n_samples=len(prompts),
            metadata={
                "task": task,
                "auroc_max": auroc_max,
                "auroc_avg": auroc_avg,
                "passed": passed,
                "threshold": 0.70,
                "n_circuit_edges": stats_max.get("n_circuit_edges", 0),
                "n_non_circuit_edges": stats_max.get("n_non_circuit_edges", 0),
                "mean_circuit_score_max": stats_max.get("mean_circuit_score", 0),
                "mean_non_circuit_score_max": stats_max.get("mean_non_circuit_score", 0),
                "median_circuit_score_max": stats_max.get("median_circuit_score", 0),
                "median_non_circuit_score_max": stats_max.get("median_non_circuit_score", 0),
                "mean_circuit_score_avg": stats_avg.get("mean_circuit_score", 0),
                "mean_non_circuit_score_avg": stats_avg.get("mean_non_circuit_score", 0),
                "top_edges": top_edges[:30],
            },
        ))

    return results


def main():
    parser = parse_common_args("C14: Position-Aware Edge Attribution Patching")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("C14: POSITION-AWARE EDGE ATTRIBUTION PATCHING")
    log("=" * 60)

    out = args.out or "98_position_aware_eap.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_position_aware_eap(model, [task], args.n_prompts)
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)
            p = "PASS" if r.metadata["passed"] else "FAIL"
            log(f"  {task}: AUROC(max)={r.value:.4f}  [{p}]")

    save_results(results, out)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
