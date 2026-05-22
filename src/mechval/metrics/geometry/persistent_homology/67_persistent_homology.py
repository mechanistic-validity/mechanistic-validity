"""Persistent Homology
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     E09 — Persistent Homology
Categories:     representational
Validity layer: Construct
Criteria:       C2 Structural plausibility
Establishes:    Circuit layers have distinct topological structure in activation space
Requires:       GPU, model
Doc:            /instruments_v2/representational/e09-persistent-homology
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Computes a simplified topological descriptor of the residual stream activation
manifold at each layer. Builds a cosine distance matrix, then tracks how the
number of connected components (0-th Betti number) decreases as the distance
threshold epsilon grows -- a Vietoris-Rips-inspired H0 persistence summary.
Reports the diameter (epsilon at full connectivity) and merge rate. Compares
H0 persistence between circuit-active layers and non-circuit layers.

Uses scipy.spatial.distance and scipy.cluster.hierarchy for efficient
single-linkage clustering (equivalent to tracking H0 persistence).

Usage:
    uv run python 67_persistent_homology.py --tasks ioi induction
    uv run python 67_persistent_homology.py --device cpu --n-prompts 50
"""

import numpy as np
import torch
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    generate_prompts,
    get_circuit_heads,
    load_model,
    log,
    parse_common_args,
    save_results,
)


def compute_h0_persistence(distance_matrix_condensed: np.ndarray, n_points: int
                           ) -> dict[str, float]:
    """Compute H0 persistence summary from condensed distance matrix.

    Uses single-linkage clustering: each merge corresponds to a component death.
    Returns diameter, mean merge distance, and normalized AUC of the Betti curve.
    """
    # Single-linkage gives H0 persistence diagram directly
    Z = linkage(distance_matrix_condensed, method="single")
    # Z[:,2] = merge distances (sorted ascending)
    merge_distances = Z[:, 2]

    diameter = float(merge_distances[-1]) if len(merge_distances) > 0 else 0.0
    mean_merge = float(merge_distances.mean()) if len(merge_distances) > 0 else 0.0

    # Betti curve AUC: integral of b0(eps) over [0, diameter]
    # b0 starts at n_points and decreases by 1 at each merge
    if diameter < 1e-12:
        return {"diameter": 0.0, "mean_merge": 0.0, "betti_auc_normalized": 0.0}

    # Compute normalized AUC (area under Betti curve / (n_points * diameter))
    betti_auc = 0.0
    prev_eps = 0.0
    current_b0 = n_points
    for d in merge_distances:
        betti_auc += current_b0 * (d - prev_eps)
        current_b0 -= 1
        prev_eps = d
    # Last component persists to diameter (already included as last merge)
    betti_auc_normalized = betti_auc / (n_points * diameter)

    return {
        "diameter": diameter,
        "mean_merge": mean_merge,
        "betti_auc_normalized": betti_auc_normalized,
    }


@torch.no_grad()
def collect_layer_residuals(model, prompts, device: str) -> dict[int, np.ndarray]:
    """Collect last-position residual stream at each layer."""
    n_layers = model.cfg.n_layers
    residuals: dict[int, list[np.ndarray]] = {L: [] for L in range(n_layers)}

    for p in prompts:
        tokens = model.to_tokens(p.text)
        _, cache = model.run_with_cache(
            tokens, names_filter=lambda n: "hook_resid_post" in n
        )
        for L in range(n_layers):
            h = cache[f"blocks.{L}.hook_resid_post"][0, -1].cpu().numpy()
            residuals[L].append(h)

    return {L: np.stack(v) for L, v in residuals.items()}


@torch.no_grad()
def main():
    parser = parse_common_args("E06: Persistent Homology (H0 Betti Curve)")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)
    results = []

    log("=" * 60)
    log("E06: PERSISTENT HOMOLOGY (H0)")
    log("=" * 60)

    for task in tasks:
        log(f"\n--- Task: {task} ---")
        circuit_heads = get_circuit_heads(task)
        if not circuit_heads:
            log(f"  No circuit heads for {task}, skipping")
            continue

        prompts = generate_prompts(task, model.tokenizer, args.n_prompts)
        if not prompts:
            log(f"  No prompts for {task}, skipping")
            continue

        log(f"  Collecting residuals ({len(prompts)} prompts)...")
        residuals = collect_layer_residuals(model, prompts, args.device)

        circuit_layers = sorted(set(L for L, _ in circuit_heads))
        non_circuit_layers = [L for L in range(model.cfg.n_layers) if L not in circuit_layers]

        persistence_per_layer = {}
        for L in range(model.cfg.n_layers):
            X = residuals[L]
            if X.shape[0] < 3:
                continue
            # Cosine distance
            dists = pdist(X, metric="cosine")
            # Replace NaN (from zero vectors) with max distance
            dists = np.nan_to_num(dists, nan=2.0)
            persistence_per_layer[L] = compute_h0_persistence(dists, X.shape[0])

        # Aggregate by circuit vs non-circuit layers
        circuit_diameters = [persistence_per_layer[L]["diameter"]
                            for L in circuit_layers if L in persistence_per_layer]
        non_circuit_diameters = [persistence_per_layer[L]["diameter"]
                                 for L in non_circuit_layers if L in persistence_per_layer]
        circuit_auc = [persistence_per_layer[L]["betti_auc_normalized"]
                       for L in circuit_layers if L in persistence_per_layer]
        non_circuit_auc = [persistence_per_layer[L]["betti_auc_normalized"]
                           for L in non_circuit_layers if L in persistence_per_layer]

        mean_circuit_diam = float(np.mean(circuit_diameters)) if circuit_diameters else 0.0
        mean_non_circuit_diam = float(np.mean(non_circuit_diameters)) if non_circuit_diameters else 0.0
        mean_circuit_auc = float(np.mean(circuit_auc)) if circuit_auc else 0.0
        mean_non_circuit_auc = float(np.mean(non_circuit_auc)) if non_circuit_auc else 0.0

        log(f"  Circuit layers diameter: {mean_circuit_diam:.4f} (n={len(circuit_diameters)})")
        log(f"  Non-circuit layers diameter: {mean_non_circuit_diam:.4f} (n={len(non_circuit_diameters)})")
        log(f"  Circuit Betti AUC: {mean_circuit_auc:.4f}")
        log(f"  Non-circuit Betti AUC: {mean_non_circuit_auc:.4f}")

        results.append(EvalResult(
            metric_id="E06.persistent_homology_h0",
            value=mean_circuit_diam,
            baseline_random=mean_non_circuit_diam,
            n_samples=len(prompts),
            metadata={
                "task": task,
                "persistence_per_layer": persistence_per_layer,
                "mean_circuit_diameter": mean_circuit_diam,
                "mean_non_circuit_diameter": mean_non_circuit_diam,
                "mean_circuit_betti_auc": mean_circuit_auc,
                "mean_non_circuit_betti_auc": mean_non_circuit_auc,
                "circuit_layers": circuit_layers,
                "n_prompts": len(prompts),
            },
        ))

    out = args.out or "67_persistent_homology.json"
    save_results(results, out)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
