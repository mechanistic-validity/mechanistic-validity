"""Weight Decomposition (NMF on OV Matrices)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     B08 — ICA/NMF
Categories:     structural
Validity layer: Construct
Criteria:       C2 Structural plausibility
Establishes:    Circuit heads share more low-rank weight structure than random subsets
Requires:       CPU, model weights only
Doc:            /instruments_v2/structural/b08-ica-nmf
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Applies Non-negative Matrix Factorization to the absolute values of OV
matrices from circuit heads to find shared weight-space structure. Stacks
|W_OV| matrices and decomposes to measure how much structure circuit heads
share compared to random head subsets.

Reports:
- Number of components needed to explain 90% of variance
- Reconstruction error for circuit heads vs random head subsets
- Whether circuit heads share more low-rank weight structure

Falls back to SVD-based truncated decomposition if sklearn NMF is
unavailable.

Usage:
    uv run python 50_weight_decomposition.py --tasks ioi greater_than
    uv run python 50_weight_decomposition.py --device cpu
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "causal"))
from _common import (
    CIRCUIT_TASKS,
    EvalResult,
    get_circuit_heads,
    load_model,
    log,
    parse_common_args,
    save_results,
)

try:
    from sklearn.decomposition import NMF
    HAS_NMF = True
except ImportError:
    HAS_NMF = False


def decompose_and_measure(stacked: np.ndarray, n_components: int) -> float:
    """Decompose stacked matrix and return relative reconstruction error."""
    if HAS_NMF:
        nmf = NMF(n_components=n_components, max_iter=300, random_state=42)
        W = nmf.fit_transform(stacked)
        H = nmf.components_
        reconstruction = W @ H
    else:
        # SVD fallback on absolute values (already non-negative)
        U, S, Vt = np.linalg.svd(stacked, full_matrices=False)
        reconstruction = (U[:, :n_components] * S[:n_components]) @ Vt[:n_components]
        reconstruction = np.abs(reconstruction)

    error = np.linalg.norm(stacked - reconstruction) / (np.linalg.norm(stacked) + 1e-10)
    return float(error)


def components_for_variance(stacked: np.ndarray, threshold: float = 0.90) -> int:
    """Number of SVD components needed to explain `threshold` fraction of variance."""
    U, S, Vt = np.linalg.svd(stacked, full_matrices=False)
    cumvar = np.cumsum(S ** 2) / (np.sum(S ** 2) + 1e-10)
    idx = np.searchsorted(cumvar, threshold)
    return int(idx + 1)


@torch.no_grad()
def get_ov_matrices(model) -> dict[tuple[int, int], np.ndarray]:
    """Extract |W_OV| for every head, flattened to 1D."""
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    ov_flat = {}
    for L in range(n_layers):
        W_V = model.blocks[L].attn.W_V  # (n_heads, d_model, d_head)
        W_O = model.blocks[L].attn.W_O  # (n_heads, d_head, d_model)
        for H in range(n_heads):
            wov = (W_V[H] @ W_O[H]).float().cpu().numpy()
            ov_flat[(L, H)] = np.abs(wov).flatten()
    return ov_flat


@torch.no_grad()
def main():
    parser = parse_common_args("B50: Weight Decomposition (NMF on OV)")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("B50: WEIGHT DECOMPOSITION (NMF ON OV MATRICES)")
    log(f"  Backend: {'sklearn NMF' if HAS_NMF else 'SVD fallback'}")
    log("=" * 60)

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    all_heads = [(L, H) for L in range(n_layers) for H in range(n_heads)]

    log("Extracting OV matrices...")
    ov_flat = get_ov_matrices(model)

    rng = np.random.default_rng(42)
    results = []

    for task in tasks:
        circuit_heads = get_circuit_heads(task)
        if len(circuit_heads) < 3:
            log(f"  {task}: too few circuit heads ({len(circuit_heads)}), skipping")
            continue

        log(f"  {task}: {len(circuit_heads)} circuit heads")

        # Stack circuit heads' OV matrices
        circuit_list = sorted(circuit_heads)
        stacked_circuit = np.stack([ov_flat[h] for h in circuit_list])

        # Components for 90% variance
        n_comp_90 = components_for_variance(stacked_circuit)
        log(f"    Components for 90% variance: {n_comp_90}/{len(circuit_list)}")

        # Reconstruction error with half the components
        n_comp = max(2, len(circuit_list) // 2)
        circuit_error = decompose_and_measure(stacked_circuit, n_comp)

        # Random baselines: sample same number of heads, compute error
        n_baselines = min(args.n_random_baselines, 50)
        random_errors = []
        for _ in range(n_baselines):
            random_heads = rng.choice(len(all_heads), size=len(circuit_list), replace=False)
            stacked_random = np.stack([ov_flat[all_heads[i]] for i in random_heads])
            random_errors.append(decompose_and_measure(stacked_random, n_comp))

        mean_random_error = float(np.mean(random_errors))
        sharing_ratio = circuit_error / (mean_random_error + 1e-10)

        log(f"    Reconstruction error: circuit={circuit_error:.4f}  "
            f"random={mean_random_error:.4f}  ratio={sharing_ratio:.3f}")

        results.append(EvalResult(
            metric_id="B50.weight_decomposition",
            value=circuit_error,
            baseline_random=mean_random_error,
            n_samples=len(circuit_list),
            metadata={
                "task": task,
                "n_components_90pct": n_comp_90,
                "n_components_used": n_comp,
                "circuit_recon_error": circuit_error,
                "random_recon_error_mean": mean_random_error,
                "random_recon_error_std": float(np.std(random_errors)),
                "sharing_ratio": sharing_ratio,
                "interpretation": "ratio<1 means circuit heads share more structure",
                "backend": "nmf" if HAS_NMF else "svd",
            },
        ))

    out = args.out or "50_weight_decomposition.json"
    save_results(results, out)
    log(f"\nDone. {len(results)} results across {len(tasks)} tasks.")


if __name__ == "__main__":
    main()
