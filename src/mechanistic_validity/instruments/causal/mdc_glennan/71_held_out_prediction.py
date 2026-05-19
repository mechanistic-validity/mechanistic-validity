"""Held-Out Function Prediction
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     A05 — MDC/Glennan Mechanisms
Categories:     causal
Validity layer: Internal
Criteria:       F2 Held-Out Prediction (proposed)
Establishes:    Whether claimed component function predicts behavior on novel inputs
Requires:       GPU or CPU, model
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For each circuit head:
1. Split prompts into train (first half) and test (second half).
2. On train set: compute the principal output direction of the head's hook_z
   (first right singular vector of the z activations matrix).
3. On test set: predict activation magnitude as projection onto the train-derived
   principal direction. Measure Pearson correlation between predicted and actual
   activation norms.
4. Compare circuit heads vs random non-circuit heads — circuit heads should
   have more predictable (consistent) function across inputs.

High prediction correlation means the head behaves consistently: its function
on novel inputs is predictable from its function on training inputs.

Usage:
    uv run python 71_held_out_prediction.py --tasks ioi sva
    uv run python 71_held_out_prediction.py --device cpu --n-prompts 60
"""

import numpy as np
import torch

from mechanistic_validity.instruments.common import (
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


@torch.no_grad()
def collect_head_z(model, prompts, layer: int, head: int) -> torch.Tensor:
    """Collect hook_z at last position for a head across all prompts.

    Returns: (n_prompts, d_head)
    """
    zs = []
    for p in prompts:
        tokens = model.to_tokens(p.text)
        _, cache = model.run_with_cache(
            tokens,
            names_filter=lambda n, _L=layer: n == f"blocks.{_L}.attn.hook_z",
        )
        z = cache[f"blocks.{layer}.attn.hook_z"][0, -1, head]  # (d_head,)
        zs.append(z.cpu())
    return torch.stack(zs)


def pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation between two 1-D arrays."""
    if len(x) < 3:
        return 0.0
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denom = np.sqrt((x_centered ** 2).sum() * (y_centered ** 2).sum())
    if denom < 1e-12:
        return 0.0
    return float((x_centered * y_centered).sum() / denom)


def held_out_prediction_correlation(z_all: torch.Tensor) -> float:
    """Split activations into train/test, predict test from train-derived direction.

    z_all: (n_prompts, d_head)
    Returns Pearson r between predicted and actual activation magnitudes on test set.
    """
    n = z_all.shape[0]
    if n < 6:
        return 0.0

    n_train = n // 2
    z_train = z_all[:n_train].float()
    z_test = z_all[n_train:].float()

    # Compute principal direction from train set via SVD
    z_train_centered = z_train - z_train.mean(dim=0, keepdim=True)
    U, S, Vt = torch.linalg.svd(z_train_centered, full_matrices=False)
    if S[0].item() < 1e-10:
        return 0.0
    principal_dir = Vt[0]  # (d_head,)

    # Train-set mean activation (for the mean component)
    train_mean = z_train.mean(dim=0)  # (d_head,)

    # Predict test: project each test activation onto the principal direction
    # The "prediction" is how much each test sample aligns with the train pattern
    test_centered = z_test - train_mean.unsqueeze(0)
    predicted_magnitude = (test_centered @ principal_dir).numpy()

    # Actual: activation norm of each test sample (centered)
    actual_magnitude = test_centered.norm(dim=-1).numpy()

    # Use absolute predicted magnitude (direction doesn't matter for magnitude prediction)
    return pearson_correlation(np.abs(predicted_magnitude), actual_magnitude)


def run_held_out_prediction(model, tasks: list[str],
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
            log(f"  {task}: no prompts, skipping")
            continue

        log(f"  {task} ({len(circuit_heads)} heads, {len(prompts)} prompts)...")

        non_circuit = all_heads - circuit_heads

        # Circuit heads: held-out prediction correlation
        circuit_corrs = {}
        for L, H in sorted(circuit_heads):
            z_all = collect_head_z(model, prompts, L, H)
            corr = held_out_prediction_correlation(z_all)
            circuit_corrs[f"L{L}H{H}"] = corr
            log(f"    L{L}H{H}: prediction_r={corr:.4f}")

        # Baseline: sample of non-circuit heads
        rng = np.random.RandomState(42)
        n_baseline = min(len(non_circuit), max(len(circuit_heads), 10))
        baseline_heads = sorted(non_circuit)
        rng.shuffle(baseline_heads)
        baseline_heads = baseline_heads[:n_baseline]

        baseline_corrs = []
        for L, H in baseline_heads:
            z_all = collect_head_z(model, prompts, L, H)
            corr = held_out_prediction_correlation(z_all)
            baseline_corrs.append(corr)

        mean_circuit = float(np.mean(list(circuit_corrs.values())))
        mean_baseline = float(np.mean(baseline_corrs)) if baseline_corrs else 0.0

        # Fraction of circuit heads that beat the baseline median
        baseline_median = float(np.median(baseline_corrs)) if baseline_corrs else 0.0
        n_above = sum(1 for v in circuit_corrs.values() if v > baseline_median)
        frac_above = n_above / len(circuit_corrs) if circuit_corrs else 0.0

        log(f"    circuit mean r={mean_circuit:.4f}  baseline mean r={mean_baseline:.4f}")
        log(f"    {n_above}/{len(circuit_corrs)} circuit heads above baseline median "
            f"({baseline_median:.4f})")

        results.append(EvalResult(
            metric_id="F2.held_out_prediction",
            value=mean_circuit,
            baseline_random=mean_baseline,
            n_samples=len(circuit_heads),
            metadata={
                "task": task,
                "per_head_correlation": circuit_corrs,
                "baseline_mean_correlation": mean_baseline,
                "baseline_median_correlation": baseline_median,
                "frac_above_baseline_median": frac_above,
                "n_circuit_heads": len(circuit_heads),
                "n_baseline_heads": n_baseline,
                "n_prompts_train": len(prompts) // 2,
                "n_prompts_test": len(prompts) - len(prompts) // 2,
                "interpretation": "Pearson r of train-predicted vs actual test activation magnitude (higher = more predictable)",
            },
        ))

    return results


def main():
    parser = parse_common_args("F2: Held-Out Function Prediction")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("F2: HELD-OUT FUNCTION PREDICTION")
    log("=" * 60)

    results = run_held_out_prediction(model, tasks, args.n_prompts)

    out = args.out or "71_held_out_prediction.json"
    save_results(results, out)

    log(f"\nDone. {len(results)} tasks evaluated.")
    for r in results:
        t = r.metadata["task"]
        log(f"  {t}: mean_r={r.value:.4f} (baseline={r.baseline_random:.4f})")


if __name__ == "__main__":
    main()
