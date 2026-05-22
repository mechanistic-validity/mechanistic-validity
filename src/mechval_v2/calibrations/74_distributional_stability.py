"""Distributional Stability Across Subsets
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     F01 — Bootstrap Stability
Categories:     measurement
Validity layer: Measurement
Criteria:       S3 Distributional Stability (proposed)
Establishes:    Whether activation statistics are stable across data subsets
Requires:       CPU, model
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Splits prompts into 5 random subsets and computes per-head logit
attribution statistics (mean, variance, skewness) on each. Reports the
coefficient of variation (CV) of each statistic across subsets.
Pass condition: CV < 0.20 for mean and variance.

Usage:
    uv run python 74_distributional_stability.py --tasks ioi sva
    uv run python 74_distributional_stability.py --device cpu --n-prompts 100
"""

import numpy as np
import torch

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    generate_prompts,
    get_circuit_heads,
    get_token_ids,
    load_model,
    log,
    parse_common_args,
    save_results,
)

N_SUBSETS = 5
CV_THRESHOLD = 0.20


def _skewness(x: np.ndarray) -> float:
    """Compute sample skewness (Fisher definition)."""
    n = len(x)
    if n < 3:
        return 0.0
    mu = x.mean()
    s = x.std(ddof=1)
    if s < 1e-12:
        return 0.0
    return float((n / ((n - 1) * (n - 2))) * np.sum(((x - mu) / s) ** 3))


def _cv(values: np.ndarray) -> float:
    """Coefficient of variation: std/|mean|."""
    mu = np.abs(values.mean())
    if mu < 1e-12:
        return float("inf")
    return float(values.std(ddof=1) / mu)


@torch.no_grad()
def compute_head_attributions(model, prompts, correct_ids) -> dict[tuple[int, int], np.ndarray]:
    """Compute logit attribution per head per prompt.

    Attribution = z[:, -1, head, :] @ W_O[layer, head] @ W_U[:, correct_token].
    Returns dict mapping (layer, head) -> array of shape (n_prompts,).
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    W_U = model.W_U

    attributions: dict[tuple[int, int], list[float]] = {
        (L, H): [] for L in range(n_layers) for H in range(n_heads)
    }

    for i, p in enumerate(prompts):
        if i >= len(correct_ids):
            break
        tokens = model.to_tokens(p.text)
        _, cache = model.run_with_cache(tokens, names_filter=lambda n: "hook_z" in n)
        target_col = W_U[:, correct_ids[i]]
        for L in range(n_layers):
            z = cache[f"blocks.{L}.attn.hook_z"]
            W_O_layer = model.W_O[L]
            z_last = z[0, -1]
            for H in range(n_heads):
                contrib = z_last[H] @ W_O_layer[H] @ target_col
                attributions[(L, H)].append(contrib.item())

    return {k: np.array(v) for k, v in attributions.items()}


def compute_subset_stats(attributions: dict[tuple[int, int], np.ndarray],
                         circuit_heads: set[tuple[int, int]],
                         n_subsets: int = N_SUBSETS,
                         seed: int = 42) -> dict:
    """Split attributions into subsets, compute stats per subset, report CV."""
    rng = np.random.RandomState(seed)
    n_prompts = len(next(iter(attributions.values())))
    indices = np.arange(n_prompts)
    rng.shuffle(indices)
    subsets = np.array_split(indices, n_subsets)

    per_head_results = {}
    for h in sorted(circuit_heads):
        vals = attributions[h]
        subset_means = []
        subset_vars = []
        subset_skews = []

        for sub_idx in subsets:
            sub_vals = vals[sub_idx]
            if len(sub_vals) < 3:
                continue
            subset_means.append(float(sub_vals.mean()))
            subset_vars.append(float(sub_vals.var(ddof=1)))
            subset_skews.append(_skewness(sub_vals))

        subset_means = np.array(subset_means)
        subset_vars = np.array(subset_vars)
        subset_skews = np.array(subset_skews)

        cv_mean = _cv(subset_means) if len(subset_means) > 1 else float("inf")
        cv_var = _cv(subset_vars) if len(subset_vars) > 1 else float("inf")
        cv_skew = _cv(subset_skews) if len(subset_skews) > 1 else float("inf")

        per_head_results[f"L{h[0]}H{h[1]}"] = {
            "cv_mean": cv_mean,
            "cv_variance": cv_var,
            "cv_skewness": cv_skew,
            "subset_means": subset_means.tolist(),
            "subset_variances": subset_vars.tolist(),
            "subset_skewness": subset_skews.tolist(),
            "pass_mean": cv_mean < CV_THRESHOLD,
            "pass_variance": cv_var < CV_THRESHOLD,
        }

    return per_head_results


def run_distributional_stability(model, tasks: list[str],
                                 n_prompts: int = 40) -> list[EvalResult]:
    tokenizer = model.tokenizer
    results = []

    for task in tasks:
        circuit_heads = get_circuit_heads(task)
        if not circuit_heads:
            log(f"  {task}: no circuit, skipping")
            continue

        prompts = generate_prompts(task, tokenizer, n_prompts)
        if not prompts:
            continue

        correct_ids, incorrect_ids = get_token_ids(prompts, tokenizer)
        if not correct_ids:
            continue

        log(f"  {task} ({len(circuit_heads)} heads, {len(correct_ids)} prompts)...")

        attributions = compute_head_attributions(model, prompts, correct_ids)
        per_head = compute_subset_stats(attributions, circuit_heads)

        # Aggregate: fraction of circuit heads that pass both mean and variance CV
        n_pass_mean = sum(1 for v in per_head.values() if v["pass_mean"])
        n_pass_var = sum(1 for v in per_head.values() if v["pass_variance"])
        n_pass_both = sum(1 for v in per_head.values()
                         if v["pass_mean"] and v["pass_variance"])
        n_total = len(per_head)

        frac_pass = n_pass_both / n_total if n_total > 0 else 0.0

        # Mean CV across circuit heads
        mean_cv_mean = float(np.mean([v["cv_mean"] for v in per_head.values()
                                       if np.isfinite(v["cv_mean"])])) if per_head else float("inf")
        mean_cv_var = float(np.mean([v["cv_variance"] for v in per_head.values()
                                      if np.isfinite(v["cv_variance"])])) if per_head else float("inf")

        log(f"    pass(mean CV<{CV_THRESHOLD}): {n_pass_mean}/{n_total}, "
            f"pass(var CV<{CV_THRESHOLD}): {n_pass_var}/{n_total}, "
            f"pass(both): {n_pass_both}/{n_total}")
        log(f"    avg CV(mean)={mean_cv_mean:.3f}, avg CV(var)={mean_cv_var:.3f}")

        results.append(EvalResult(
            metric_id="S3.distributional_stability",
            value=frac_pass,
            n_samples=len(correct_ids),
            metadata={
                "task": task,
                "n_circuit_heads": n_total,
                "n_subsets": N_SUBSETS,
                "cv_threshold": CV_THRESHOLD,
                "n_pass_mean": n_pass_mean,
                "n_pass_variance": n_pass_var,
                "n_pass_both": n_pass_both,
                "mean_cv_mean": mean_cv_mean,
                "mean_cv_variance": mean_cv_var,
                "per_head": per_head,
            },
        ))

    return results


def main():
    parser = parse_common_args("S3: Distributional Stability")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("S3: DISTRIBUTIONAL STABILITY ACROSS SUBSETS")
    log("=" * 60)

    results = run_distributional_stability(model, tasks, args.n_prompts)

    out = args.out or "74_distributional_stability.json"
    save_results(results, out)

    log(f"\nDone. {len(results)} tasks evaluated.")
    for r in results:
        t = r.metadata["task"]
        log(f"  {t}: frac_pass={r.value:.3f} "
            f"(CV_mean={r.metadata['mean_cv_mean']:.3f}, "
            f"CV_var={r.metadata['mean_cv_variance']:.3f})")


if __name__ == "__main__":
    main()
