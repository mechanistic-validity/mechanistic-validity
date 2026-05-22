"""Copying Score
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Metric:         B11 — Copying Score
Categories:     structural
Validity layer: Construct
Criteria:       C2 Structural plausibility
Establishes:    Whether circuit heads act as token copiers in weight space
Requires:       CPU, model
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Measures the "copying score" of each head: the largest eigenvalue of
W_E^T @ W_OV @ W_U. A large positive eigenvalue means the head's OV
circuit maps input tokens to the same output tokens (copying behavior).

Per-circuit outputs:
  - mean/max/std of copying scores across circuit heads
  - ratio vs non-circuit heads
  - per-head scores

Usage:
    uv run python B11_copying_score.py --tasks ioi sva greater_than
    uv run python B11_copying_score.py --device cpu
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

JSONL_FILE = "B11_copying_score.jsonl"


@torch.no_grad()
def compute_copying_scores(model, top_k_eigenvalues: int = 5):
    """Compute copying score for every head.

    Returns dict mapping (L, H) -> {score, top_eigenvalues}.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    W_E = model.W_E.float()  # (d_vocab, d_model)
    W_U = model.W_U.float()  # (d_model, d_vocab)

    scores = {}
    for L in range(n_layers):
        W_V = model.blocks[L].attn.W_V.float()  # (n_heads, d_model, d_head)
        W_O = model.blocks[L].attn.W_O.float()  # (n_heads, d_head, d_model)
        for H in range(n_heads):
            W_OV = W_V[H] @ W_O[H]  # (d_model, d_model)
            # W_E @ W_OV @ W_U is (d_vocab, d_vocab) — too large.
            # Same nonzero eigenvalues as (W_U @ W_E) @ W_OV which is (d_model, d_model).
            M = (W_U @ W_E) @ W_OV  # (d_model, d_model)
            eigvals = torch.linalg.eigvalsh(
                (M + M.T) / 2  # symmetrize for real eigenvalues
            ).cpu().numpy()
            eigvals = np.sort(eigvals)[::-1]

            score = float(eigvals[0])
            scores[(L, H)] = {
                "score": score,
                "top_eigenvalues": eigvals[:top_k_eigenvalues].tolist(),
                "trace": float(np.sum(eigvals)),
            }

    return scores


@torch.no_grad()
def run(model=None, tasks=None, device="cpu", model_name="gpt2", save=True,
        output_dir=None, resume=True):
    if model is None:
        model = load_model(model_name, device)
    tasks = tasks or CIRCUIT_TASKS

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    all_heads = {(L, H) for L in range(n_layers) for H in range(n_heads)}

    if output_dir is not None:
        from mechval.metrics.common import set_data_dir
        set_data_dir(output_dir)

    done_tasks, _ = load_completed_tasks(JSONL_FILE) if resume else (set(), [])

    log("=" * 60)
    log("B11: COPYING SCORE")
    log("=" * 60)

    log("Computing copying scores for all heads...")
    copy_scores = compute_copying_scores(model)

    results = []
    for task in tasks:
        if task in done_tasks:
            log(f"  {task}: already done, skipping")
            continue

        circuit_heads = get_circuit_heads(task)
        if not circuit_heads:
            log(f"  {task}: no circuit, skipping")
            continue

        non_circuit = all_heads - circuit_heads
        log(f"  {task}: {len(circuit_heads)} circuit heads")

        circuit_vals = [copy_scores[(L, H)]["score"] for L, H in circuit_heads]
        non_circuit_vals = [copy_scores[(L, H)]["score"] for L, H in non_circuit]

        mean_circuit = float(np.mean(circuit_vals))
        mean_non_circuit = float(np.mean(non_circuit_vals))
        max_circuit = float(np.max(circuit_vals))
        ratio = mean_circuit / (abs(mean_non_circuit) + 1e-10)

        log(f"    circuit_mean={mean_circuit:.2f}  non_circuit={mean_non_circuit:.2f}  "
            f"max={max_circuit:.2f}  ratio={ratio:.3f}")

        per_head = {f"L{L}H{H}": round(copy_scores[(L, H)]["score"], 2)
                    for L, H in sorted(circuit_heads)}

        result = EvalResult(
            metric_id="B11.copying_score",
            value=mean_circuit,
            baseline_random=mean_non_circuit,
            n_samples=len(circuit_heads),
            metadata={
                "task": task,
                "metric": "copying_score",
                "circuit_mean": mean_circuit,
                "circuit_max": max_circuit,
                "circuit_std": float(np.std(circuit_vals)),
                "non_circuit_mean": mean_non_circuit,
                "ratio": ratio,
                "per_head": per_head,
            },
        )
        results.append(result)
        if save:
            save_incremental(result, JSONL_FILE)

    if save and results:
        save_results(results, "B11_copying_score.json")
    log(f"\nDone. {len(results)} new results across {len(tasks)} tasks.")
    return results


@torch.no_grad()
def main():
    parser = parse_common_args("B11: Copying Score")
    args = parser.parse_args()
    return run(
        model=None, tasks=args.tasks, device=args.device, model_name=args.model,
        save=True, output_dir=getattr(args, "data_dir", None),
    )


if __name__ == "__main__":
    main()
