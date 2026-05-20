"""QK Frobenius Norm & Singular Value Gap
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Metric:         B12 — QK Norms
Categories:     structural
Validity layer: Construct
Criteria:       C2 Structural plausibility, C3 Task specificity
Establishes:    Attention focus patterns from weight geometry
Requires:       CPU, model
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Two complementary measurements on W_Q @ W_K^T per head:

1. QK Frobenius norm — ‖W_Q W_K^T‖_F.
   High norm = head has strong attention preferences in weight space.
   PTH heads (e.g. L4H11 in RTI = 58.16) are 3-4x typical heads.

2. QK singular value gap — ratio of top-1 to top-2 singular values.
   High gap = attention dominated by single direction (focused, PTH-like).
   Low gap = distributed attention across many directions (copier-like).

Per-circuit outputs:
  - mean/max/min/std of both metrics across circuit heads
  - ratio vs non-circuit heads
  - per-head values

Usage:
    uv run python B12_qk_norms.py --tasks ioi sva greater_than
    uv run python B12_qk_norms.py --device cpu
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

JSONL_FILE = "B12_qk_norms.jsonl"


@torch.no_grad()
def compute_qk_metrics(model):
    """Compute QK Frobenius norm and SV gap for every head.

    Returns dict mapping (L, H) -> {qk_frob_norm, sv_gap, top_sv}.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    metrics = {}
    for L in range(n_layers):
        W_Q = model.blocks[L].attn.W_Q.float()  # (n_heads, d_model, d_head)
        W_K = model.blocks[L].attn.W_K.float()  # (n_heads, d_model, d_head)
        for H in range(n_heads):
            # W_QK = W_Q^T @ W_K -> (d_head, d_head) but we want the
            # full QK circuit: (d_model, d_model) = W_Q @ W_K^T
            W_QK = W_Q[H] @ W_K[H].T  # (d_model, d_model)

            frob_norm = float(W_QK.norm().item())
            sv = torch.linalg.svdvals(W_QK).cpu().numpy()

            sv_gap = float(sv[0] / (sv[1] + 1e-10)) if len(sv) > 1 else float("inf")

            metrics[(L, H)] = {
                "qk_frob_norm": frob_norm,
                "sv_gap": sv_gap,
                "top_sv": float(sv[0]),
            }

    return metrics


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
    log("B12: QK FROBENIUS NORM & SINGULAR VALUE GAP")
    log("=" * 60)

    log("Computing QK metrics for all heads...")
    qk_metrics = compute_qk_metrics(model)

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

        for metric_name in ["qk_frob_norm", "sv_gap"]:
            circuit_vals = [qk_metrics[(L, H)][metric_name] for L, H in circuit_heads]
            non_circuit_vals = [qk_metrics[(L, H)][metric_name] for L, H in non_circuit]

            mean_circuit = float(np.mean(circuit_vals))
            mean_non_circuit = float(np.mean(non_circuit_vals))
            max_circuit = float(np.max(circuit_vals))
            min_circuit = float(np.min(circuit_vals))
            ratio = mean_circuit / (mean_non_circuit + 1e-10)

            label = "QK Frobenius norm" if metric_name == "qk_frob_norm" else "QK SV gap"
            log(f"    {label}: circuit={mean_circuit:.2f}  "
                f"non_circuit={mean_non_circuit:.2f}  "
                f"max={max_circuit:.2f}  ratio={ratio:.3f}")

            per_head = {f"L{L}H{H}": round(qk_metrics[(L, H)][metric_name], 2)
                        for L, H in sorted(circuit_heads)}

            result = EvalResult(
                metric_id=f"B12.{metric_name}",
                value=mean_circuit,
                baseline_random=mean_non_circuit,
                n_samples=len(circuit_heads),
                metadata={
                    "task": task,
                    "metric": metric_name,
                    "circuit_mean": mean_circuit,
                    "circuit_max": max_circuit,
                    "circuit_min": min_circuit,
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
        save_results(results, "B12_qk_norms.json")
    log(f"\nDone. {len(results)} new results across {len(tasks)} tasks.")
    return results


@torch.no_grad()
def main():
    parser = parse_common_args("B12: QK Norms & SV Gap")
    args = parser.parse_args()
    return run(
        model=None, tasks=args.tasks, device=args.device, model_name=args.model,
        save=True, output_dir=getattr(args, "data_dir", None),
    )


if __name__ == "__main__":
    main()
