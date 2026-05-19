"""Polysemanticity Index
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     B07 — Polysemanticity
Categories:     structural
Validity layer: Construct
Criteria:       C3 Task specificity
Establishes:    Circuit heads are less polysemantic than non-circuit heads
Requires:       GPU, model
Doc:            /instruments_v2/structural/b07-polysemanticity
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Measures how "polysemantic" each circuit head's OV matrix is via three
complementary metrics:

1. Effective rank of W_OV — exp(entropy of normalized singular values).
   High effective rank = many active directions = polysemantic.

2. Participation ratio of singular values — (sum(s))^2 / sum(s^2).
   Ranges from 1 (single dominant SV) to n (all SVs equal).

3. Unembedding fan-out — number of W_U columns (token directions) that
   the OV matrix projects onto with |cosine| > 0.1.
   High fan-out = head writes to many output tokens = polysemantic.

Compares circuit heads vs non-circuit heads on all three measures.

Usage:
    uv run python 52_polysemanticity.py --tasks ioi sva greater_than
    uv run python 52_polysemanticity.py --device cpu
"""

import numpy as np
import torch

from mechanistic_validity.instruments.common import (
    CIRCUIT_TASKS,
    EvalResult,
    get_circuit_heads,
    load_model,
    log,
    parse_common_args,
    save_results,
)


def effective_rank(sv: np.ndarray) -> float:
    """exp(entropy of normalized singular values)."""
    sv = sv[sv > 1e-10]
    if len(sv) == 0:
        return 0.0
    p = sv / sv.sum()
    entropy = -(p * np.log(p)).sum()
    return float(np.exp(entropy))


def participation_ratio(sv: np.ndarray) -> float:
    """(sum s_i)^2 / sum(s_i^2). Ranges from 1 to n."""
    sv = sv[sv > 1e-10]
    if len(sv) == 0:
        return 0.0
    return float((sv.sum() ** 2) / (sv ** 2).sum())


@torch.no_grad()
def compute_polysemanticity(model, cosine_threshold: float = 0.1):
    """Compute polysemanticity metrics for every head.

    Returns dict mapping (L, H) -> {eff_rank, participation, fan_out}.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    W_U = model.W_U.float()  # (d_model, d_vocab)

    # Normalize unembedding columns for cosine computation
    W_U_normed = W_U / W_U.norm(dim=0, keepdim=True).clamp(min=1e-10)

    metrics = {}
    for L in range(n_layers):
        W_V = model.blocks[L].attn.W_V  # (n_heads, d_model, d_head)
        W_O = model.blocks[L].attn.W_O  # (n_heads, d_head, d_model)
        for H in range(n_heads):
            wov = (W_V[H] @ W_O[H]).float()  # (d_model, d_model)
            sv = torch.linalg.svdvals(wov).cpu().numpy()

            eff_r = effective_rank(sv)
            part_r = participation_ratio(sv)

            # Fan-out: number of unembedding directions with |cosine| > threshold
            # Use top singular vector as the head's "output direction"
            _, _, Vt = torch.linalg.svd(wov, full_matrices=False)
            top_output_dir = Vt[0]  # (d_model,) — dominant output direction
            top_output_normed = top_output_dir / top_output_dir.norm().clamp(min=1e-10)

            # Cosine with each unembedding column
            cosines = (top_output_normed @ W_U_normed).abs()  # (d_vocab,)
            fan_out = int((cosines > cosine_threshold).sum().item())

            metrics[(L, H)] = {
                "eff_rank": eff_r,
                "participation": part_r,
                "fan_out": fan_out,
            }

    return metrics


@torch.no_grad()
def main():
    parser = parse_common_args("B52: Polysemanticity Index")
    parser.add_argument("--cosine-threshold", type=float, default=0.1,
                        help="Cosine threshold for fan-out count")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("B52: POLYSEMANTICITY INDEX")
    log("=" * 60)

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    all_heads = {(L, H) for L in range(n_layers) for H in range(n_heads)}

    log("Computing polysemanticity metrics for all heads...")
    poly_metrics = compute_polysemanticity(model, args.cosine_threshold)

    results = []
    for task in tasks:
        circuit_heads = get_circuit_heads(task)
        if not circuit_heads:
            log(f"  {task}: no circuit, skipping")
            continue

        non_circuit = all_heads - circuit_heads
        log(f"  {task}: {len(circuit_heads)} circuit heads")

        # Aggregate metrics
        for metric_name in ["eff_rank", "participation", "fan_out"]:
            circuit_vals = [poly_metrics[(L, H)][metric_name] for L, H in circuit_heads]
            non_circuit_vals = [poly_metrics[(L, H)][metric_name] for L, H in non_circuit]

            mean_circuit = float(np.mean(circuit_vals))
            mean_non_circuit = float(np.mean(non_circuit_vals))
            ratio = mean_circuit / (mean_non_circuit + 1e-10)

            log(f"    {metric_name}: circuit={mean_circuit:.2f}  "
                f"non_circuit={mean_non_circuit:.2f}  ratio={ratio:.3f}")

            per_head = {f"L{L}H{H}": poly_metrics[(L, H)][metric_name]
                        for L, H in sorted(circuit_heads)}

            results.append(EvalResult(
                metric_id=f"B52.polysemanticity_{metric_name}",
                value=mean_circuit,
                baseline_random=mean_non_circuit,
                n_samples=len(circuit_heads),
                metadata={
                    "task": task,
                    "metric": metric_name,
                    "circuit_mean": mean_circuit,
                    "non_circuit_mean": mean_non_circuit,
                    "ratio": ratio,
                    "circuit_std": float(np.std(circuit_vals)),
                    "per_head": per_head,
                    "interpretation": "ratio<1 = circuit heads more monosemantic",
                },
            ))

    out = args.out or "52_polysemanticity.json"
    save_results(results, out)
    log(f"\nDone. {len(results)} results across {len(tasks)} tasks.")


if __name__ == "__main__":
    main()
