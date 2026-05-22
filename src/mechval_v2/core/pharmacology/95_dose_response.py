"""Ablation Dose-Response (Behavioral)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     B95 — Ablation Dose-Response
Categories:     behavioral
Validity layer: Internal
Criteria:       Monotonic degradation under graded ablation
Establishes:    Whether circuit ablation produces smooth, monotonic performance loss
                rather than sudden collapse or plateau
Requires:       CPU or GPU, model
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Sweeps ablation strength from 0% to 100% across circuit heads and
measures logit-diff at each level. For each ablation fraction, each
circuit head's output is interpolated between clean and mean-ablated:
    output = (1 - frac) * clean + frac * mean

Metrics:
  - Monotonicity: fraction of adjacent ablation-level pairs where
    performance decreases. Should be 1.0 for perfect monotonic
    degradation.
  - Selectivity: ratio of dose-response slope for circuit heads vs
    random heads. Circuit heads should degrade faster.

Pass condition: monotonicity >= 0.8 AND selectivity > 2.0

Usage:
    uv run python 95_dose_response.py --tasks ioi --n-prompts 40
    uv run python 95_dose_response.py --tasks ioi sva --device cpu
"""

import numpy as np
import torch

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    calibrate_mean_z,
    generate_prompts,
    get_circuit_heads,
    get_token_ids,
    load_model,
    log,
    logit_diff_from_logits,
    parse_common_args,
    save_incremental,
    save_results,
)

ABLATION_FRACTIONS = [0.0, 0.25, 0.5, 0.75, 1.0]


def make_dose_hooks(
    heads: set[tuple[int, int]],
    mean_z: torch.Tensor,
    frac: float,
) -> list[tuple[str, callable]]:
    """Interpolate circuit heads between clean and mean-ablated at strength `frac`.

    output = (1 - frac) * clean + frac * mean
    """
    by_layer: dict[int, list[int]] = {}
    for L, H in heads:
        by_layer.setdefault(L, []).append(H)

    hooks = []
    for layer, head_list in by_layer.items():
        def _hook(z, hook, _layer=layer, _heads=head_list, _frac=frac):
            for H in _heads:
                mean_val = mean_z[_layer, H].to(z.device)
                z[0, :, H, :] = (1 - _frac) * z[0, :, H, :] + _frac * mean_val
            return z
        hooks.append((f"blocks.{layer}.attn.hook_z", _hook))
    return hooks


@torch.no_grad()
def sweep_dose_response(
    model, prompts, correct_ids, incorrect_ids,
    heads: set[tuple[int, int]], mean_z: torch.Tensor,
    fractions: list[float],
) -> list[float]:
    """Return mean logit-diff at each ablation fraction."""
    per_frac_lds: list[list[float]] = [[] for _ in fractions]

    for i, p in enumerate(prompts):
        if i >= len(correct_ids):
            break
        tokens = model.to_tokens(p.text)
        for fi, frac in enumerate(fractions):
            if frac == 0.0:
                logits = model(tokens)
            else:
                hooks = make_dose_hooks(heads, mean_z, frac)
                logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
            ld = logit_diff_from_logits(logits, correct_ids[i], incorrect_ids[i])
            per_frac_lds[fi].append(ld)

    return [float(np.mean(lds)) if lds else 0.0 for lds in per_frac_lds]


def compute_monotonicity(values: list[float]) -> float:
    """Fraction of adjacent pairs where value strictly decreases."""
    if len(values) < 2:
        return 1.0
    n_decreasing = sum(1 for a, b in zip(values[:-1], values[1:]) if b < a)
    return n_decreasing / (len(values) - 1)


def compute_slope(values: list[float], fractions: list[float]) -> float:
    """Total drop per unit ablation fraction (linear slope)."""
    if len(values) < 2 or abs(fractions[-1] - fractions[0]) < 1e-8:
        return 0.0
    return (values[0] - values[-1]) / (fractions[-1] - fractions[0])


@torch.no_grad()
def run_dose_response(
    model, tasks: list[str], n_prompts: int = 40,
    n_random_baselines: int = 20,
) -> list[EvalResult]:
    tokenizer = model.tokenizer
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    all_heads = [(L, H) for L in range(n_layers) for H in range(n_heads)]
    rng = np.random.RandomState(42)
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

        correct_ids, incorrect_ids = get_token_ids(prompts, tokenizer)
        if not correct_ids:
            continue

        log(f"  {task}: {len(circuit_heads)} heads, {len(prompts)} prompts")
        mean_z = calibrate_mean_z(model, prompts, n_calibration=min(100, len(prompts)))

        # Circuit dose-response curve
        circuit_curve = sweep_dose_response(
            model, prompts, correct_ids, incorrect_ids,
            circuit_heads, mean_z, ABLATION_FRACTIONS,
        )
        monotonicity = compute_monotonicity(circuit_curve)
        circuit_slope = compute_slope(circuit_curve, ABLATION_FRACTIONS)

        log(f"    circuit curve: {[f'{v:.3f}' for v in circuit_curve]}")
        log(f"    monotonicity={monotonicity:.2f}  slope={circuit_slope:.3f}")

        # Random baseline: same-size random head sets
        k = len(circuit_heads)
        random_slopes = []
        for _ in range(n_random_baselines):
            rand_idx = rng.choice(len(all_heads), size=k, replace=False)
            rand_heads = {all_heads[j] for j in rand_idx}
            rand_curve = sweep_dose_response(
                model, prompts, correct_ids, incorrect_ids,
                rand_heads, mean_z, ABLATION_FRACTIONS,
            )
            random_slopes.append(compute_slope(rand_curve, ABLATION_FRACTIONS))

        mean_random_slope = float(np.mean(random_slopes))
        selectivity = (circuit_slope / mean_random_slope
                       if abs(mean_random_slope) > 1e-8 else float("inf"))

        passed = bool(monotonicity >= 0.8 and selectivity > 2.0)

        log(f"    random_slope={mean_random_slope:.3f}  selectivity={selectivity:.2f}"
            f"  [{'PASS' if passed else 'FAIL'}]")

        results.append(EvalResult(
            metric_id="B95.dose_response",
            value=monotonicity,
            baseline_random=mean_random_slope,
            n_samples=len(prompts),
            metadata={
                "task": task,
                "fractions": ABLATION_FRACTIONS,
                "circuit_curve": circuit_curve,
                "monotonicity": monotonicity,
                "circuit_slope": circuit_slope,
                "mean_random_slope": mean_random_slope,
                "selectivity": selectivity,
                "n_circuit_heads": k,
                "circuit_heads": sorted(circuit_heads),
                "n_random_baselines": n_random_baselines,
                "passed": passed,
                "threshold_monotonicity": 0.8,
                "threshold_selectivity": 2.0,
            },
        ))

    return results


def main():
    parser = parse_common_args("B95: Ablation Dose-Response")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("B95: ABLATION DOSE-RESPONSE")
    log("=" * 60)

    out = args.out or "95_dose_response.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_dose_response(model, [task], args.n_prompts)
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)
            p = "PASS" if r.metadata["passed"] else "FAIL"
            log(f"  {task}: monotonicity={r.value:.2f}  "
                f"selectivity={r.metadata['selectivity']:.2f}  [{p}]")

    save_results(results, out)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
