"""K-Composition Matrix
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Metric:         B10 — K-Composition
Categories:     structural
Validity layer: Construct
Criteria:       C2 Structural plausibility, I3 Specificity
Establishes:    Circuit heads communicate via direct weight composition
Requires:       CPU, model
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Measures pairwise K-composition between circuit heads:
    K_comp(sender, receiver) = ‖W_O(sender) · W_K(receiver)‖_F

High K-composition means the sender's output subspace aligns with the
receiver's key subspace — a direct weight-space communication channel.

Per-circuit outputs:
  - mean/max/std of K-comp across all circuit head pairs
  - number of "strong" edges (K-comp z-score > 1.0 vs background)
  - hierarchy depth (longest chain of strong edges)
  - background statistics (mean/std over all 10296 head pairs)

Usage:
    uv run python B10_k_composition.py --tasks ioi sva greater_than
    uv run python B10_k_composition.py --device cpu
"""

import numpy as np
import torch

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    get_circuit_heads,
    load_completed_tasks,
    load_model,
    log,
    parse_common_args,
    save_incremental,
    save_results,
)


@torch.no_grad()
def compute_k_composition_matrix(model):
    """Compute pairwise K-composition for all head pairs.

    Returns (n_total_heads, n_total_heads) matrix where entry [i, j] is
    ‖W_O[head_i] · W_K[head_j]‖_F.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    n_total = n_layers * n_heads

    W_O_all = []
    W_K_all = []
    for L in range(n_layers):
        W_O = model.blocks[L].attn.W_O.float()  # (n_heads, d_head, d_model)
        W_K = model.blocks[L].attn.W_K.float()  # (n_heads, d_model, d_head)
        for H in range(n_heads):
            W_O_all.append(W_O[H])  # (d_head, d_model)
            W_K_all.append(W_K[H])  # (d_model, d_head)

    kcomp = np.zeros((n_total, n_total))
    for i in range(n_total):
        for j in range(n_total):
            if i == j:
                continue
            # W_O[i] is (d_head, d_model), W_K[j] is (d_model, d_head)
            product = W_O_all[i] @ W_K_all[j]  # (d_head, d_head)
            kcomp[i, j] = float(product.norm().item())

    return kcomp


def head_idx(L, H, n_heads):
    return L * n_heads + H


def longest_strong_chain(circuit_heads, kcomp, n_heads, threshold):
    """Find longest chain of strong K-comp edges (sender layer < receiver layer)."""
    sorted_heads = sorted(circuit_heads)
    head_to_idx = {h: head_idx(h[0], h[1], n_heads) for h in sorted_heads}

    adj = {}
    for src in sorted_heads:
        adj[src] = []
        for dst in sorted_heads:
            if dst[0] > src[0]:
                i, j = head_to_idx[src], head_to_idx[dst]
                if kcomp[i, j] > threshold:
                    adj[src].append(dst)

    memo = {}

    def dfs(node):
        if node in memo:
            return memo[node]
        best = 1
        for nxt in adj.get(node, []):
            best = max(best, 1 + dfs(nxt))
        memo[node] = best
        return best

    return max((dfs(h) for h in sorted_heads), default=1)


JSONL_FILE = "B10_k_composition.jsonl"


@torch.no_grad()
def run(model=None, tasks=None, device="cpu", model_name="gpt2", save=True,
        output_dir=None, resume=True):
    if model is None:
        model = load_model(model_name, device)
    tasks = tasks or CIRCUIT_TASKS
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    if output_dir is not None:
        from mechval.metrics.common import set_data_dir
        set_data_dir(output_dir)

    done_tasks, prior_results = load_completed_tasks(JSONL_FILE) if resume else (set(), [])

    log("=" * 60)
    log("B10: K-COMPOSITION MATRIX")
    log("=" * 60)

    log("Computing pairwise K-composition for all head pairs...")
    kcomp = compute_k_composition_matrix(model)

    mask = np.ones_like(kcomp, dtype=bool)
    np.fill_diagonal(mask, False)
    all_vals = kcomp[mask]
    bg_mean = float(np.mean(all_vals))
    bg_std = float(np.std(all_vals))
    log(f"  Background: mean={bg_mean:.2f}, std={bg_std:.2f}, "
        f"P95={float(np.percentile(all_vals, 95)):.2f}")

    results = []
    for task in tasks:
        if task in done_tasks:
            log(f"  {task}: already done, skipping")
            continue

        circuit_heads = get_circuit_heads(task)
        if not circuit_heads:
            log(f"  {task}: no circuit, skipping")
            continue

        sorted_heads = sorted(circuit_heads)
        log(f"  {task}: {len(sorted_heads)} circuit heads")

        circuit_vals = []
        per_edge = {}
        for src in sorted_heads:
            for dst in sorted_heads:
                if src == dst:
                    continue
                i = head_idx(src[0], src[1], n_heads)
                j = head_idx(dst[0], dst[1], n_heads)
                val = kcomp[i, j]
                circuit_vals.append(val)
                per_edge[f"L{src[0]}H{src[1]}->L{dst[0]}H{dst[1]}"] = round(val, 2)

        circuit_vals = np.array(circuit_vals)
        strong_threshold = bg_mean + bg_std
        n_strong = int((circuit_vals > strong_threshold).sum())
        depth = longest_strong_chain(circuit_heads, kcomp, n_heads, strong_threshold)

        mean_kcomp = float(np.mean(circuit_vals))
        max_kcomp = float(np.max(circuit_vals))
        std_kcomp = float(np.std(circuit_vals))

        log(f"    mean={mean_kcomp:.2f}  max={max_kcomp:.2f}  "
            f"strong_edges={n_strong}  depth={depth}")

        top_edges = dict(sorted(per_edge.items(), key=lambda x: -x[1])[:10])

        result = EvalResult(
            metric_id="B10.k_composition_mean",
            value=mean_kcomp,
            baseline_random=bg_mean,
            n_samples=len(circuit_vals),
            metadata={
                "task": task,
                "metric": "k_composition",
                "circuit_mean": mean_kcomp,
                "circuit_max": max_kcomp,
                "circuit_std": std_kcomp,
                "background_mean": bg_mean,
                "background_std": bg_std,
                "n_strong_edges": n_strong,
                "strong_edge_threshold": round(strong_threshold, 2),
                "hierarchy_depth": depth,
                "ratio_to_background": round(mean_kcomp / (bg_mean + 1e-10), 3),
                "top_10_edges": top_edges,
            },
        )
        results.append(result)
        if save:
            save_incremental(result, JSONL_FILE)

    if save and results:
        save_results(results, "B10_k_composition.json")
    log(f"\nDone. {len(results)} new results across {len(tasks)} tasks.")
    return results


@torch.no_grad()
def main():
    parser = parse_common_args("B10: K-Composition Matrix")
    args = parser.parse_args()
    return run(
        model=None, tasks=args.tasks, device=args.device, model_name=args.model,
        save=True, output_dir=getattr(args, "data_dir", None),
    )


if __name__ == "__main__":
    main()
