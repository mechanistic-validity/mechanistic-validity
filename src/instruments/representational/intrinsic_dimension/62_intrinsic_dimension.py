"""Intrinsic Dimension via Two-NN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     E07 — Intrinsic Dimension
Categories:     representational, structural
Validity layer: Construct
Criteria:       C2 Structural plausibility
Establishes:    Circuit layers operate on lower-dimensional manifolds than non-circuit layers
Requires:       GPU, model
Doc:            /instruments_v2/representational/e07-intrinsic-dimension
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Estimates the intrinsic dimension (ID) of residual stream activations
at each layer using the Two-NN estimator (Facco et al. 2017).

Method: For each data point, compute mu = r2/r1 (ratio of second to
first nearest neighbor distance). The MLE of intrinsic dimension is:
    ID = n / sum(log(mu_i))

Lower ID at circuit-critical layers suggests the computation is
constrained to a low-dimensional manifold, indicating structured
rather than distributed processing.

Usage:
    uv run python 62_intrinsic_dimension.py --tasks ioi sva --device cpu
    uv run python 62_intrinsic_dimension.py --device cuda --n-prompts 60
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "causal"))
from _common import (
    CIRCUIT_TASKS,
    EvalResult,
    generate_prompts,
    get_circuit_heads,
    load_model,
    log,
    parse_common_args,
    save_results,
)


def two_nn_intrinsic_dimension(X: np.ndarray) -> float:
    """Estimate intrinsic dimension using the Two-NN method (Facco et al. 2017).

    X: (n_samples, n_features). Requires n_samples >= 3.
    Returns estimated intrinsic dimension.
    """
    n = X.shape[0]
    if n < 3:
        return 0.0

    # Compute pairwise distances
    dists = np.zeros((n, n))
    for i in range(n):
        diff = X - X[i]
        dists[i] = np.sqrt(np.sum(diff ** 2, axis=1))

    # For each point, find r1 (nearest) and r2 (second nearest)
    log_mu_sum = 0.0
    valid_count = 0

    for i in range(n):
        d = dists[i].copy()
        d[i] = np.inf  # exclude self
        sorted_d = np.sort(d)
        r1 = sorted_d[0]
        r2 = sorted_d[1]
        if r1 > 1e-10:
            mu = r2 / r1
            log_mu_sum += np.log(mu)
            valid_count += 1

    if valid_count == 0 or log_mu_sum < 1e-10:
        return 0.0

    return float(valid_count / log_mu_sum)


@torch.no_grad()
def collect_residual_activations(model, prompts, layer: int) -> np.ndarray:
    """Collect last-token residual stream at a given layer. Returns (n_prompts, d_model)."""
    activations = []
    hook_name = f"blocks.{layer}.hook_resid_post"
    for p in prompts:
        tokens = model.to_tokens(p.text)
        _, cache = model.run_with_cache(tokens, names_filter=lambda n: n == hook_name)
        act = cache[hook_name][0, -1].cpu().float().numpy()
        activations.append(act)
    return np.stack(activations, axis=0)


@torch.no_grad()
def main():
    parser = parse_common_args("E62: Intrinsic Dimension (Two-NN)")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)
    tokenizer = model.tokenizer
    n_layers = model.cfg.n_layers

    log("=" * 60)
    log("E62: INTRINSIC DIMENSION (Two-NN estimator)")
    log("=" * 60)

    results = []

    for task in tasks:
        circuit_heads = get_circuit_heads(task)
        if not circuit_heads:
            log(f"  {task}: no circuit, skipping")
            continue

        circuit_layers = {L for L, _ in circuit_heads}
        non_circuit_layers = {L for L in range(n_layers)} - circuit_layers

        prompts = generate_prompts(task, tokenizer, args.n_prompts)
        if not prompts or len(prompts) < 5:
            log(f"  {task}: insufficient prompts")
            continue

        log(f"  {task}: {len(prompts)} prompts, circuit layers={sorted(circuit_layers)}")

        per_layer_id = []
        circuit_ids = []
        non_circuit_ids = []

        for layer in range(n_layers):
            acts = collect_residual_activations(model, prompts, layer)
            intrinsic_dim = two_nn_intrinsic_dimension(acts)
            per_layer_id.append(intrinsic_dim)

            if layer in circuit_layers:
                circuit_ids.append(intrinsic_dim)
            else:
                non_circuit_ids.append(intrinsic_dim)

            if layer % 3 == 0:
                log(f"    layer {layer}: ID={intrinsic_dim:.2f}")

        mean_circuit_id = float(np.mean(circuit_ids)) if circuit_ids else 0.0
        mean_non_circuit_id = float(np.mean(non_circuit_ids)) if non_circuit_ids else 0.0

        # Ratio <1 means circuit layers have lower ID (more constrained)
        id_ratio = mean_circuit_id / mean_non_circuit_id if mean_non_circuit_id > 0 else 1.0

        log(f"    circuit ID={mean_circuit_id:.2f}, non-circuit ID={mean_non_circuit_id:.2f}, "
            f"ratio={id_ratio:.3f}")

        results.append(EvalResult(
            metric_id="E62.intrinsic_dim_ratio",
            value=id_ratio,
            n_samples=len(prompts),
            metadata={
                "task": task,
                "mean_circuit_id": mean_circuit_id,
                "mean_non_circuit_id": mean_non_circuit_id,
                "per_layer_id": per_layer_id,
                "circuit_layers": sorted(circuit_layers),
            },
        ))
        results.append(EvalResult(
            metric_id="E62.mean_intrinsic_dim",
            value=float(np.mean(per_layer_id)),
            n_samples=len(prompts),
            metadata={
                "task": task,
                "per_layer_id": per_layer_id,
                "min_id": float(np.min(per_layer_id)),
                "max_id": float(np.max(per_layer_id)),
                "min_layer": int(np.argmin(per_layer_id)),
                "max_layer": int(np.argmax(per_layer_id)),
            },
        ))

    out = args.out or "62_intrinsic_dimension.json"
    save_results(results, out)
    log(f"\nDone. {len(results)} metrics.")


if __name__ == "__main__":
    main()
