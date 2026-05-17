"""PCA Dimensionality
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     E06 — PCA Dimensionality
Categories:     representational
Validity layer: Construct
Criteria:       C2 Structural plausibility
Establishes:    Circuit layers use fewer effective dimensions than non-circuit layers
Requires:       CPU/GPU, model
Doc:            /instruments_v2/representational/e06-pca-dimensionality
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Collects residual stream activations at each layer across prompts and
measures the effective dimensionality of the representation space.

Metrics:
  - Effective dimensionality: number of PCA components needed for 90%
    variance explained.
  - Participation ratio: PR = (sum lambda)^2 / sum(lambda^2), measuring
    how many dimensions actively contribute.
  - Comparison: circuit layers vs non-circuit layers.

Lower effective dimensionality at circuit layers suggests more
structured, constrained representations during task computation.

Usage:
    uv run python 60_pca_dimensionality.py --tasks ioi sva --device cpu
    uv run python 60_pca_dimensionality.py --device cuda --n-prompts 60
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


def effective_dimensionality(eigenvalues: np.ndarray, threshold: float = 0.9) -> int:
    """Number of components needed to explain threshold fraction of variance."""
    total = eigenvalues.sum()
    if total < 1e-12:
        return 0
    cumulative = np.cumsum(eigenvalues) / total
    return int(np.searchsorted(cumulative, threshold) + 1)


def participation_ratio(eigenvalues: np.ndarray) -> float:
    """PR = (sum lambda)^2 / sum(lambda^2). Higher = more distributed."""
    s1 = eigenvalues.sum()
    s2 = (eigenvalues ** 2).sum()
    if s2 < 1e-12:
        return 0.0
    return float(s1 ** 2 / s2)


@torch.no_grad()
def collect_residual_activations(model, prompts, layer: int) -> np.ndarray:
    """Collect last-token residual stream at a given layer. Returns (n_prompts, d_model)."""
    activations = []
    hook_name = f"blocks.{layer}.hook_resid_post"
    for p in prompts:
        tokens = model.to_tokens(p.text)
        _, cache = model.run_with_cache(tokens, names_filter=lambda n: n == hook_name)
        act = cache[hook_name][0, -1].cpu().float().numpy()  # (d_model,)
        activations.append(act)
    return np.stack(activations, axis=0)


@torch.no_grad()
def main():
    parser = parse_common_args("E60: PCA Dimensionality")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)
    tokenizer = model.tokenizer
    n_layers = model.cfg.n_layers

    log("=" * 60)
    log("E60: PCA DIMENSIONALITY")
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
        if not prompts:
            continue

        log(f"  {task}: {len(prompts)} prompts, circuit layers={sorted(circuit_layers)}")

        per_layer_eff_dim = []
        per_layer_pr = []
        circuit_eff_dims = []
        circuit_prs = []
        non_circuit_eff_dims = []
        non_circuit_prs = []

        for layer in range(n_layers):
            acts = collect_residual_activations(model, prompts, layer)
            # Center the data
            acts_centered = acts - acts.mean(axis=0, keepdims=True)
            # Compute covariance eigenvalues via SVD (more stable)
            _, s, _ = np.linalg.svd(acts_centered, full_matrices=False)
            eigenvalues = s ** 2 / (len(acts) - 1)

            eff_dim = effective_dimensionality(eigenvalues)
            pr = participation_ratio(eigenvalues)

            per_layer_eff_dim.append(eff_dim)
            per_layer_pr.append(pr)

            if layer in circuit_layers:
                circuit_eff_dims.append(eff_dim)
                circuit_prs.append(pr)
            else:
                non_circuit_eff_dims.append(eff_dim)
                non_circuit_prs.append(pr)

        mean_circuit_dim = float(np.mean(circuit_eff_dims)) if circuit_eff_dims else 0.0
        mean_non_circuit_dim = float(np.mean(non_circuit_eff_dims)) if non_circuit_eff_dims else 0.0
        mean_circuit_pr = float(np.mean(circuit_prs)) if circuit_prs else 0.0
        mean_non_circuit_pr = float(np.mean(non_circuit_prs)) if non_circuit_prs else 0.0

        # Ratio: circuit dim / non-circuit dim (<1 means circuit is lower-dimensional)
        dim_ratio = mean_circuit_dim / mean_non_circuit_dim if mean_non_circuit_dim > 0 else 1.0

        log(f"    circuit_dim={mean_circuit_dim:.1f}, non_circuit_dim={mean_non_circuit_dim:.1f}, "
            f"ratio={dim_ratio:.3f}")
        log(f"    circuit_PR={mean_circuit_pr:.1f}, non_circuit_PR={mean_non_circuit_pr:.1f}")

        results.append(EvalResult(
            metric_id="E60.effective_dimensionality_ratio",
            value=dim_ratio,
            n_samples=len(prompts),
            metadata={
                "task": task,
                "mean_circuit_dim": mean_circuit_dim,
                "mean_non_circuit_dim": mean_non_circuit_dim,
                "per_layer_eff_dim": per_layer_eff_dim,
                "circuit_layers": sorted(circuit_layers),
            },
        ))
        results.append(EvalResult(
            metric_id="E60.participation_ratio_circuit",
            value=mean_circuit_pr,
            n_samples=len(prompts),
            metadata={
                "task": task,
                "mean_circuit_pr": mean_circuit_pr,
                "mean_non_circuit_pr": mean_non_circuit_pr,
                "per_layer_pr": per_layer_pr,
            },
        ))

    out = args.out or "60_pca_dimensionality.json"
    save_results(results, out)
    log(f"\nDone. {len(results)} metrics.")


if __name__ == "__main__":
    main()
