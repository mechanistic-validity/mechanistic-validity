"""Geodesic Distance
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     E07b — Geodesic Distance
Categories:     representational
Validity layer: Construct
Criteria:       C2 Structural plausibility
Establishes:    Circuit layers have higher manifold curvature indicating nonlinear computation
Requires:       GPU, model
Doc:            /instruments_v2/representational/e07b-geodesic-distance
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Measures the curvature of the activation manifold by comparing geodesic
distances (shortest paths on a k-NN graph) to Euclidean distances. A ratio
geodesic/euclidean > 1 indicates curvature -- the manifold is not flat and
straight-line interpolation does not reflect the data geometry. Higher
distortion at circuit layers suggests nonlinear computation is occurring.

Builds a k-NN graph on residual stream activations at each layer, computes
shortest-path (geodesic) distances via Dijkstra, and reports the mean
distortion ratio.

Usage:
    uv run python 68_geodesic_distance.py --tasks ioi induction
    uv run python 68_geodesic_distance.py --device cpu --n-prompts 50
"""

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.spatial.distance import cdist, pdist, squareform

from mechanistic_validity.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    generate_prompts,
    get_circuit_heads,
    load_model,
    log,
    parse_common_args,
    save_results,
)

K_NEIGHBORS = 10


def build_knn_graph(X: np.ndarray, k: int) -> csr_matrix:
    """Build a symmetric k-NN graph with Euclidean edge weights."""
    n = X.shape[0]
    dist_matrix = squareform(pdist(X, metric="euclidean"))

    # For each point, keep only k nearest neighbors
    rows, cols, data = [], [], []
    for i in range(n):
        dists = dist_matrix[i]
        # Exclude self (distance 0)
        neighbor_idx = np.argsort(dists)[1:k + 1]
        for j in neighbor_idx:
            rows.append(i)
            cols.append(j)
            data.append(dists[j])

    # Make symmetric
    graph = csr_matrix((data, (rows, cols)), shape=(n, n))
    graph = graph.maximum(graph.T)
    return graph


def compute_distortion_ratio(X: np.ndarray, k: int) -> dict[str, float]:
    """Compute mean geodesic/euclidean distortion ratio.

    Returns mean and max distortion over all reachable pairs.
    """
    n = X.shape[0]
    if n < k + 1:
        return {"mean_distortion": 1.0, "max_distortion": 1.0, "unreachable_frac": 1.0}

    graph = build_knn_graph(X, k)
    geodesic = shortest_path(graph, directed=False)
    euclidean = squareform(pdist(X, metric="euclidean"))

    # Only consider pairs that are reachable (geodesic < inf) and non-self
    mask = np.isfinite(geodesic) & (euclidean > 1e-10)
    np.fill_diagonal(mask, False)

    if mask.sum() == 0:
        return {"mean_distortion": 1.0, "max_distortion": 1.0, "unreachable_frac": 1.0}

    ratios = geodesic[mask] / euclidean[mask]
    unreachable_frac = 1.0 - mask.sum() / (n * (n - 1))

    return {
        "mean_distortion": float(ratios.mean()),
        "max_distortion": float(ratios.max()),
        "unreachable_frac": float(unreachable_frac),
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
    parser = parse_common_args("E08: Geodesic vs Euclidean Distance Ratio")
    parser.add_argument("--k", type=int, default=K_NEIGHBORS,
                        help="Number of nearest neighbors for graph (default: 10)")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)
    k = args.k
    results = []

    log("=" * 60)
    log("E08: GEODESIC DISTANCE (MANIFOLD CURVATURE)")
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

        log(f"  Collecting residuals ({len(prompts)} prompts, k={k})...")
        residuals = collect_layer_residuals(model, prompts, args.device)

        circuit_layers = sorted(set(L for L, _ in circuit_heads))
        non_circuit_layers = [L for L in range(model.cfg.n_layers) if L not in circuit_layers]

        distortion_per_layer = {}
        for L in range(model.cfg.n_layers):
            X = residuals[L]
            if X.shape[0] < k + 1:
                log(f"    Layer {L}: too few points ({X.shape[0]}), skipping")
                continue
            distortion_per_layer[L] = compute_distortion_ratio(X, k)

        # Aggregate
        circuit_distortions = [distortion_per_layer[L]["mean_distortion"]
                              for L in circuit_layers if L in distortion_per_layer]
        non_circuit_distortions = [distortion_per_layer[L]["mean_distortion"]
                                   for L in non_circuit_layers if L in distortion_per_layer]

        mean_circuit = float(np.mean(circuit_distortions)) if circuit_distortions else 1.0
        mean_non_circuit = float(np.mean(non_circuit_distortions)) if non_circuit_distortions else 1.0

        log(f"  Circuit layers mean distortion: {mean_circuit:.4f}")
        log(f"  Non-circuit layers mean distortion: {mean_non_circuit:.4f}")
        log(f"  Distortion per layer:")
        for L in sorted(distortion_per_layer):
            marker = "*" if L in circuit_layers else " "
            log(f"    {marker} Layer {L:2d}: {distortion_per_layer[L]['mean_distortion']:.4f}")

        results.append(EvalResult(
            metric_id="E08.geodesic_distance",
            value=mean_circuit,
            baseline_random=mean_non_circuit,
            n_samples=len(prompts),
            metadata={
                "task": task,
                "distortion_per_layer": distortion_per_layer,
                "mean_circuit_distortion": mean_circuit,
                "mean_non_circuit_distortion": mean_non_circuit,
                "circuit_layers": circuit_layers,
                "k_neighbors": k,
                "n_prompts": len(prompts),
            },
        ))

    out = args.out or "68_geodesic_distance.json"
    save_results(results, out)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
