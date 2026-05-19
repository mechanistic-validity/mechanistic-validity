"""Norm Trajectory Through Layers
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     B05 — Norm Trajectory
Categories:     structural
Validity layer: Construct
Criteria:       C2 Structural plausibility
Establishes:    Circuit heads have distinctive norm magnitude profiles across layers
Requires:       CPU, model weights only
Doc:            /instruments_v2/structural/b05-norm-trajectory
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Tracks how the Frobenius norm of OV matrices for circuit heads vs
non-circuit heads evolves across layers. This reveals the "contribution
magnitude profile" of the circuit — whether it peaks early, late, or is
distributed.

Reports per task:
- Peak layer (layer with highest mean circuit OV norm)
- Decay rate (linear slope of norm across layers)
- Whether circuit heads have systematically higher norms (ratio)
- Full per-layer norm trajectories

Usage:
    uv run python 51_norm_trajectory.py --tasks ioi sva greater_than
    uv run python 51_norm_trajectory.py --device cpu
"""

import numpy as np
import torch

from mechanistic_validity.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    get_circuit_heads,
    load_model,
    log,
    parse_common_args,
    save_results,
)


@torch.no_grad()
def compute_ov_norms(model) -> np.ndarray:
    """Compute Frobenius norm of W_OV for each (layer, head).

    Returns array of shape (n_layers, n_heads).
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    norms = np.zeros((n_layers, n_heads))

    for L in range(n_layers):
        W_V = model.blocks[L].attn.W_V  # (n_heads, d_model, d_head)
        W_O = model.blocks[L].attn.W_O  # (n_heads, d_head, d_model)
        for H in range(n_heads):
            wov = (W_V[H] @ W_O[H]).float()
            norms[L, H] = wov.norm().item()

    return norms


def linear_slope(values: np.ndarray) -> float:
    """Compute linear regression slope over layer indices."""
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=np.float64)
    x_centered = x - x.mean()
    y_centered = values - values.mean()
    denom = (x_centered ** 2).sum()
    if denom < 1e-10:
        return 0.0
    return float((x_centered * y_centered).sum() / denom)


@torch.no_grad()
def main():
    parser = parse_common_args("B51: Norm Trajectory Through Layers")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("B51: NORM TRAJECTORY THROUGH LAYERS")
    log("=" * 60)

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    all_heads = {(L, H) for L in range(n_layers) for H in range(n_heads)}

    log("Computing OV Frobenius norms...")
    ov_norms = compute_ov_norms(model)

    results = []
    for task in tasks:
        circuit_heads = get_circuit_heads(task)
        if not circuit_heads:
            log(f"  {task}: no circuit, skipping")
            continue

        non_circuit = all_heads - circuit_heads
        log(f"  {task}: {len(circuit_heads)} circuit heads")

        # Per-layer mean norms for circuit vs non-circuit
        circuit_trajectory = np.zeros(n_layers)
        non_circuit_trajectory = np.zeros(n_layers)
        circuit_counts = np.zeros(n_layers)
        non_circuit_counts = np.zeros(n_layers)

        for L, H in circuit_heads:
            circuit_trajectory[L] += ov_norms[L, H]
            circuit_counts[L] += 1

        for L, H in non_circuit:
            non_circuit_trajectory[L] += ov_norms[L, H]
            non_circuit_counts[L] += 1

        # Average per layer (avoid division by zero)
        circuit_trajectory = np.divide(
            circuit_trajectory, circuit_counts,
            out=np.zeros_like(circuit_trajectory),
            where=circuit_counts > 0,
        )
        non_circuit_trajectory = np.divide(
            non_circuit_trajectory, non_circuit_counts,
            out=np.zeros_like(non_circuit_trajectory),
            where=non_circuit_counts > 0,
        )

        # Metrics
        active_layers = circuit_counts > 0
        if not active_layers.any():
            continue

        peak_layer = int(np.argmax(circuit_trajectory))
        circuit_slope = linear_slope(circuit_trajectory[active_layers])
        non_circuit_slope = linear_slope(non_circuit_trajectory)

        # Overall norm ratio
        mean_circuit_norm = float(np.mean([ov_norms[L, H] for L, H in circuit_heads]))
        mean_non_circuit_norm = float(np.mean([ov_norms[L, H] for L, H in non_circuit]))
        norm_ratio = mean_circuit_norm / (mean_non_circuit_norm + 1e-10)

        log(f"    Peak layer: {peak_layer}  Slope: {circuit_slope:.4f}  "
            f"Norm ratio: {norm_ratio:.3f}")

        results.append(EvalResult(
            metric_id="B51.norm_trajectory",
            value=norm_ratio,
            baseline_random=1.0,
            n_samples=len(circuit_heads),
            metadata={
                "task": task,
                "peak_layer": peak_layer,
                "circuit_slope": circuit_slope,
                "non_circuit_slope": non_circuit_slope,
                "mean_circuit_norm": mean_circuit_norm,
                "mean_non_circuit_norm": mean_non_circuit_norm,
                "norm_ratio": norm_ratio,
                "circuit_trajectory": circuit_trajectory.tolist(),
                "non_circuit_trajectory": non_circuit_trajectory.tolist(),
                "circuit_heads_per_layer": circuit_counts.tolist(),
            },
        ))

    out = args.out or "51_norm_trajectory.json"
    save_results(results, out)
    log(f"\nDone. {len(results)} results across {len(tasks)} tasks.")


if __name__ == "__main__":
    main()
