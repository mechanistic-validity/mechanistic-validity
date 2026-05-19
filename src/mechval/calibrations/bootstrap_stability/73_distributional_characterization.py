"""Distributional Characterization (Activation Statistics)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     F01 — Bootstrap Stability
Categories:     measurement
Validity layer: Measurement
Criteria:       S1 Distributional Characterization (proposed)
Establishes:    Full distributional profile of circuit component activations
Requires:       CPU, model
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For each circuit head, computes full distributional statistics of logit
attributions: mean (with bootstrap 95% CI), variance, skewness, kurtosis,
sparsity, and effective rank. Compares circuit heads vs non-circuit heads.

Usage:
    uv run python 73_distributional_characterization.py --tasks ioi sva
    uv run python 73_distributional_characterization.py --device cpu --n-prompts 60
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

SPARSITY_THRESHOLD = 0.01
N_BOOTSTRAP_CI = 1000


def _bootstrap_ci(values: np.ndarray, n_bootstrap: int = N_BOOTSTRAP_CI,
                  alpha: float = 0.05) -> tuple[float, float]:
    """Compute bootstrap confidence interval for the mean."""
    n = len(values)
    if n < 2:
        return float(values[0]) if n == 1 else 0.0, 0.0
    rng = np.random.RandomState(42)
    boot_means = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        boot_means[b] = values[idx].mean()
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return lo, hi


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


def _kurtosis(x: np.ndarray) -> float:
    """Compute excess kurtosis (Fisher definition)."""
    n = len(x)
    if n < 4:
        return 0.0
    mu = x.mean()
    s = x.std(ddof=1)
    if s < 1e-12:
        return 0.0
    m4 = np.mean((x - mu) ** 4)
    return float(m4 / (s ** 4) - 3.0)


def _effective_rank(matrix: np.ndarray) -> float:
    """Effective rank via (sum sigma)^2 / sum sigma^2."""
    if matrix.shape[0] < 2 or matrix.shape[1] < 1:
        return 1.0
    s = np.linalg.svd(matrix, compute_uv=False)
    s = s[s > 1e-12]
    if len(s) == 0:
        return 0.0
    sum_s = s.sum()
    sum_s2 = (s ** 2).sum()
    if sum_s2 < 1e-20:
        return 0.0
    return float(sum_s ** 2 / sum_s2)


@torch.no_grad()
def compute_head_attributions(model, prompts, correct_ids) -> dict[tuple[int, int], np.ndarray]:
    """Compute logit attribution per head per prompt.

    Attribution = z[:, -1, head, :] @ W_O[layer, head] @ W_U[:, correct_token].
    Returns dict mapping (layer, head) -> array of shape (n_prompts,).
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    W_U = model.W_U  # (d_model, d_vocab)

    attributions: dict[tuple[int, int], list[float]] = {
        (L, H): [] for L in range(n_layers) for H in range(n_heads)
    }

    for i, p in enumerate(prompts):
        if i >= len(correct_ids):
            break
        tokens = model.to_tokens(p.text)
        _, cache = model.run_with_cache(tokens, names_filter=lambda n: "hook_z" in n)

        target_col = W_U[:, correct_ids[i]]  # (d_model,)

        for L in range(n_layers):
            z = cache[f"blocks.{L}.attn.hook_z"]  # (1, seq, n_heads, d_head)
            W_O_layer = model.W_O[L]  # (n_heads, d_head, d_model)
            z_last = z[0, -1]  # (n_heads, d_head)
            for H in range(n_heads):
                # z_last[H] @ W_O[H] @ target_col
                contrib = z_last[H] @ W_O_layer[H] @ target_col
                attributions[(L, H)].append(contrib.item())

    return {k: np.array(v) for k, v in attributions.items()}


def compute_head_stats(values: np.ndarray) -> dict:
    """Compute full distributional profile for a single head."""
    mean = float(values.mean())
    ci_lo, ci_hi = _bootstrap_ci(values)
    var = float(values.var(ddof=1)) if len(values) > 1 else 0.0
    std = float(np.sqrt(var))
    skew = _skewness(values)
    kurt = _kurtosis(values)
    sparsity = float((np.abs(values) < SPARSITY_THRESHOLD).mean())
    return {
        "mean": mean,
        "ci_low": ci_lo,
        "ci_high": ci_hi,
        "variance": var,
        "std": std,
        "skewness": skew,
        "kurtosis": kurt,
        "sparsity": sparsity,
    }


@torch.no_grad()
def compute_effective_ranks(model, prompts, correct_ids,
                            heads: set[tuple[int, int]]) -> dict[tuple[int, int], float]:
    """Compute effective rank of activation vectors across prompts for given heads."""
    n_prompts = min(len(prompts), len(correct_ids))
    W_U = model.W_U

    head_vectors: dict[tuple[int, int], list[np.ndarray]] = {h: [] for h in heads}

    for i in range(n_prompts):
        tokens = model.to_tokens(prompts[i].text)
        _, cache = model.run_with_cache(tokens, names_filter=lambda n: "hook_z" in n)
        for L, H in heads:
            z = cache[f"blocks.{L}.attn.hook_z"]
            z_last = z[0, -1, H].cpu().numpy()  # (d_head,)
            head_vectors[(L, H)].append(z_last)

    ranks = {}
    for h, vecs in head_vectors.items():
        if len(vecs) < 2:
            ranks[h] = 1.0
            continue
        matrix = np.stack(vecs, axis=0)  # (n_prompts, d_head)
        ranks[h] = _effective_rank(matrix)
    return ranks


def run_distributional_characterization(model, tasks: list[str],
                                        n_prompts: int = 40) -> list[EvalResult]:
    tokenizer = model.tokenizer
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    all_heads = {(L, H) for L in range(n_layers) for H in range(n_heads)}
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

        non_circuit_heads = all_heads - circuit_heads

        log(f"  {task} ({len(circuit_heads)} circuit heads, {len(non_circuit_heads)} non-circuit)...")

        # Compute logit attributions for all heads
        attributions = compute_head_attributions(model, prompts, correct_ids)

        # Compute stats per head
        circuit_stats = {}
        for h in sorted(circuit_heads):
            vals = attributions[h]
            circuit_stats[f"L{h[0]}H{h[1]}"] = compute_head_stats(vals)

        non_circuit_stats = {}
        for h in sorted(non_circuit_heads):
            vals = attributions[h]
            non_circuit_stats[f"L{h[0]}H{h[1]}"] = compute_head_stats(vals)

        # Aggregate circuit vs non-circuit means for comparison
        circuit_means = np.array([attributions[h].mean() for h in circuit_heads])
        non_circuit_means = np.array([attributions[h].mean() for h in non_circuit_heads])

        circuit_vars = np.array([attributions[h].var(ddof=1) for h in circuit_heads
                                 if len(attributions[h]) > 1])
        non_circuit_vars = np.array([attributions[h].var(ddof=1) for h in non_circuit_heads
                                     if len(attributions[h]) > 1])

        # Effective rank for circuit heads
        ranks = compute_effective_ranks(model, prompts, correct_ids, circuit_heads)

        # Summary statistic: ratio of mean |attribution| circuit vs non-circuit
        circuit_mag = float(np.abs(circuit_means).mean()) if len(circuit_means) > 0 else 0.0
        non_circuit_mag = float(np.abs(non_circuit_means).mean()) if len(non_circuit_means) > 0 else 0.0
        attribution_ratio = circuit_mag / non_circuit_mag if non_circuit_mag > 1e-8 else float("inf")

        log(f"    circuit mean |attr|={circuit_mag:.4f}, non-circuit={non_circuit_mag:.4f}, "
            f"ratio={attribution_ratio:.2f}")
        log(f"    circuit mean var={circuit_vars.mean():.6f}, "
            f"non-circuit mean var={non_circuit_vars.mean():.6f}")

        rank_values = {f"L{h[0]}H{h[1]}": r for h, r in ranks.items()}
        mean_rank = float(np.mean(list(ranks.values()))) if ranks else 0.0
        log(f"    mean effective rank (circuit heads)={mean_rank:.2f}")

        results.append(EvalResult(
            metric_id="S1.distributional_characterization",
            value=attribution_ratio,
            n_samples=len(correct_ids),
            metadata={
                "task": task,
                "n_circuit_heads": len(circuit_heads),
                "n_non_circuit_heads": len(non_circuit_heads),
                "circuit_head_stats": circuit_stats,
                "circuit_mean_magnitude": circuit_mag,
                "non_circuit_mean_magnitude": non_circuit_mag,
                "circuit_mean_variance": float(circuit_vars.mean()) if len(circuit_vars) > 0 else 0.0,
                "non_circuit_mean_variance": float(non_circuit_vars.mean()) if len(non_circuit_vars) > 0 else 0.0,
                "effective_ranks": rank_values,
                "mean_effective_rank": mean_rank,
            },
        ))

    return results


def main():
    parser = parse_common_args("S1: Distributional Characterization")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("S1: DISTRIBUTIONAL CHARACTERIZATION")
    log("=" * 60)

    results = run_distributional_characterization(model, tasks, args.n_prompts)

    out = args.out or "73_distributional_characterization.json"
    save_results(results, out)

    log(f"\nDone. {len(results)} tasks evaluated.")
    for r in results:
        t = r.metadata["task"]
        log(f"  {t}: attribution_ratio={r.value:.3f}")


if __name__ == "__main__":
    main()
