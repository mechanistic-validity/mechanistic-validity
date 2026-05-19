"""Edge Attribution Patching (Causal C7)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     C07 — Edge Attribution Patching
Categories:     causal
Validity layer: Internal
Criteria:       C7 Edge Attribution Discrimination
Establishes:    Whether gradient-based edge attribution scores discriminate circuit edges from non-circuit edges
Requires:       CPU or GPU, model
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For each task, computes edge-level attribution scores between all pairs
of attention heads using the EAP method (Syed et al. 2023). For each
edge (sender -> receiver), the score is the mean over prompts of the
dot product between the sender's hook_z output and the receiver's
hook_z gradient (both at the last token position).

Circuit edges are treated as positives, all other forward edges as
negatives. AUROC measures how well attribution scores discriminate them.

Pass condition: AUROC > 0.70

Usage:
    uv run python 91_eap.py --tasks ioi --n-prompts 40
    uv run python 91_eap.py --tasks ioi sva --device cpu
"""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from mechanistic_validity.metrics.common import (
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


def compute_eap_scores(model, prompts, correct_ids, incorrect_ids) -> np.ndarray:
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    n = min(len(prompts), len(correct_ids), len(incorrect_ids))
    n_total = n_layers * n_heads
    edge_scores = np.zeros((n_total, n_total), dtype=np.float64)

    for i in range(n):
        tokens = model.to_tokens(prompts[i].text)
        model.zero_grad()

        cache_dict = {}

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

        for Ls in range(n_layers):
            z_s = cache_dict[Ls][0, -1].detach()
            for Lr in range(Ls + 1, n_layers):
                z_r = cache_dict[Lr]
                if z_r.grad is None:
                    continue
                g_r = z_r.grad[0, -1].detach()
                for Hs in range(n_heads):
                    s_idx = Ls * n_heads + Hs
                    s_vec = z_s[Hs]
                    for Hr in range(n_heads):
                        r_idx = Lr * n_heads + Hr
                        edge_scores[s_idx, r_idx] += torch.dot(s_vec, g_r[Hr]).item()

        model.zero_grad()

        if (i + 1) % 10 == 0:
            log(f"    processed {i+1}/{n} prompts")

    edge_scores /= max(n, 1)
    return edge_scores


def compute_auroc(edge_scores: np.ndarray, circuit_edges: set[tuple[int, int, int, int]],
                  n_layers: int, n_heads: int) -> tuple[float, dict]:
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

    labels = np.array(labels)
    scores = np.array(scores)

    if labels.sum() == 0 or labels.sum() == len(labels):
        return 0.0, {"n_circuit": int(labels.sum()), "n_total": len(labels)}

    auroc = float(roc_auc_score(labels, scores))

    circuit_scores = scores[labels == 1]
    non_circuit_scores = scores[labels == 0]

    return auroc, {
        "n_circuit_edges": int(labels.sum()),
        "n_non_circuit_edges": int((1 - labels).sum()),
        "mean_circuit_score": float(circuit_scores.mean()),
        "mean_non_circuit_score": float(non_circuit_scores.mean()),
        "median_circuit_score": float(np.median(circuit_scores)),
        "median_non_circuit_score": float(np.median(non_circuit_scores)),
    }


def run_eap(model, tasks: list[str], n_prompts: int = 40) -> list[EvalResult]:
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

        edge_scores = compute_eap_scores(model, prompts, correct_ids, incorrect_ids)
        auroc, stats = compute_auroc(edge_scores, all_edges, n_layers, n_heads)
        passed = bool(auroc > 0.70)

        log(f"    AUROC={auroc:.4f}  [{('PASS' if passed else 'FAIL')}]")
        log(f"    circuit edges: {stats.get('n_circuit_edges', 0)}, "
            f"mean|score|={stats.get('mean_circuit_score', 0):.4f}")
        log(f"    non-circuit:   {stats.get('n_non_circuit_edges', 0)}, "
            f"mean|score|={stats.get('mean_non_circuit_score', 0):.4f}")

        top_edges = []
        for Ls in range(n_layers):
            for Hs in range(n_heads):
                s_idx = Ls * n_heads + Hs
                for Lr in range(Ls + 1, n_layers):
                    for Hr in range(n_heads):
                        r_idx = Lr * n_heads + Hr
                        score = float(edge_scores[s_idx, r_idx])
                        if abs(score) > 1e-6:
                            top_edges.append({
                                "edge": f"L{Ls}H{Hs}->L{Lr}H{Hr}",
                                "score": score,
                                "in_circuit": (Ls, Hs, Lr, Hr) in all_edges,
                            })
        top_edges.sort(key=lambda e: abs(e["score"]), reverse=True)

        results.append(EvalResult(
            metric_id="C7.eap_auroc",
            value=auroc,
            n_samples=len(prompts),
            metadata={
                "task": task,
                "auroc": auroc,
                "passed": passed,
                "threshold": 0.70,
                "n_circuit_edges": stats.get("n_circuit_edges", 0),
                "n_non_circuit_edges": stats.get("n_non_circuit_edges", 0),
                "mean_circuit_score": stats.get("mean_circuit_score", 0),
                "mean_non_circuit_score": stats.get("mean_non_circuit_score", 0),
                "median_circuit_score": stats.get("median_circuit_score", 0),
                "median_non_circuit_score": stats.get("median_non_circuit_score", 0),
                "top_edges": top_edges[:30],
            },
        ))

    return results


def main():
    parser = parse_common_args("C7: Edge Attribution Patching")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("C7: EDGE ATTRIBUTION PATCHING (EAP)")
    log("=" * 60)

    out = args.out or "91_eap.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_eap(model, [task], args.n_prompts)
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)
            p = "PASS" if r.metadata["passed"] else "FAIL"
            log(f"  {task}: AUROC={r.value:.4f}  [{p}]")

    save_results(results, out)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
