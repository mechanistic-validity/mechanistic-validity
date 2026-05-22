"""Steering-Bench Reliability (Behavioral B21)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     B21 — Steering-Bench Reliability
Categories:     behavioral
Validity layer: External
Criteria:       Propensity-corrected steerability with dose-response
Establishes:    Whether artifact directions reliably steer model behavior
                after correcting for baseline propensity (ceiling effects)
Requires:       CPU or GPU, model, artifact adapter
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Implements the Steering-Bench decomposition from Tan et al. (NeurIPS 2024,
UCL/FAR AI). The key insight is that raw steerability conflates baseline
model propensity with genuine causal effect. This metric decomposes
steering evaluation into three components:

  1. **Propensity**: P(correct) without steering — the baseline behavior.
  2. **Raw steerability**: change in P(correct) when adding the artifact
     direction at a hook point with various coefficients.
  3. **Propensity-corrected steerability**: steerability / (1 - propensity),
     which corrects for ceiling effects (a model already at P=0.9 can only
     improve by 0.1, so raw steerability underestimates the causal effect).

Additionally measures dose-response linearity (R^2 of coefficient vs effect)
to verify the steering signal is graded rather than binary.

Pass condition: corrected_steerability > 0.15

Usage:
    # Programmatic (from metric_registry dispatch):
    run_steering_reliability(model, artifact=artifact,
        hook_name="blocks.5.hook_resid_pre", tasks=["ioi"])

    # CLI:
    uv run python 102_steering_reliability.py --model gpt2 \\
        --hook blocks.5.hook_resid_pre --tasks ioi

Reference:
    Tan et al., "Steering-Bench: Evaluating Representation Engineering
    in Language Models", NeurIPS 2024.
    https://github.com/dtch1997/steering-bench
"""

import numpy as np
import torch

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    generate_prompts,
    get_token_ids,
    load_model,
    log,
    parse_common_args,
    save_incremental,
    save_results,
)


# ---------------------------------------------------------------------------
# Core: propensity measurement
# ---------------------------------------------------------------------------

@torch.no_grad()
def _measure_propensity(
    model, prompts, correct_ids: list[int], incorrect_ids: list[int],
) -> float:
    """Measure P(correct) without any steering intervention.

    For each prompt, checks whether the model assigns higher probability
    to the correct token than the incorrect token at the last position.
    Returns the fraction of prompts where the model is correct.
    """
    n_correct = 0
    n_valid = min(len(prompts), len(correct_ids))
    if n_valid == 0:
        return 0.0

    for i in range(n_valid):
        tokens = model.to_tokens(prompts[i].text)
        logits = model(tokens)
        last_logits = logits[0, -1]
        if last_logits[correct_ids[i]] > last_logits[incorrect_ids[i]]:
            n_correct += 1

    return n_correct / n_valid


# ---------------------------------------------------------------------------
# Core: steered measurement
# ---------------------------------------------------------------------------

@torch.no_grad()
def _measure_steered_accuracy(
    model, prompts, correct_ids: list[int], incorrect_ids: list[int],
    direction: torch.Tensor, hook_name: str, coeff: float,
) -> float:
    """Measure P(correct) when adding coeff * direction at hook_name.

    Returns the fraction of prompts where the steered model picks the
    correct token over the incorrect token.
    """
    n_correct = 0
    n_valid = min(len(prompts), len(correct_ids))
    if n_valid == 0:
        return 0.0

    for i in range(n_valid):
        tokens = model.to_tokens(prompts[i].text)

        def steering_hook(act, hook, _d=direction, _c=coeff):
            act[:, :, :] = act + _c * _d.to(act.device)
            return act

        logits = model.run_with_hooks(
            tokens, fwd_hooks=[(hook_name, steering_hook)],
        )
        last_logits = logits[0, -1]
        if last_logits[correct_ids[i]] > last_logits[incorrect_ids[i]]:
            n_correct += 1

    return n_correct / n_valid


# ---------------------------------------------------------------------------
# Dose-response linearity
# ---------------------------------------------------------------------------

def _dose_response_r2(
    coefficients: list[float], effects: list[float],
) -> float:
    """Compute R^2 of linear fit between steering coefficients and effects.

    Returns 0.0 if there are fewer than 2 data points or zero variance.
    """
    if len(coefficients) < 2:
        return 0.0

    x = np.array(coefficients, dtype=np.float64)
    y = np.array(effects, dtype=np.float64)

    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if ss_tot < 1e-12:
        return 0.0

    # Linear regression: y = a*x + b
    n = len(x)
    sx = np.sum(x)
    sy = np.sum(y)
    sxy = np.sum(x * y)
    sxx = np.sum(x * x)
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:
        return 0.0

    a = (n * sxy - sx * sy) / denom
    b = (sy - a * sx) / n
    y_pred = a * x + b

    ss_res = np.sum((y - y_pred) ** 2)
    return float(1.0 - ss_res / ss_tot)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_steering_reliability(
    model,
    tasks: list[str] | None = None,
    artifact=None,
    hook_name: str | None = None,
    n_prompts: int = 30,
    steering_coefficients: list[float] | None = None,
) -> list[EvalResult]:
    """Run Steering-Bench reliability evaluation.

    For each task, measures propensity (baseline P(correct)), raw
    steerability at each coefficient, propensity-corrected steerability,
    and dose-response linearity.

    Args:
        model: HookedTransformer instance.
        tasks: List of task names to evaluate.
        artifact: ArtifactAdapter with directions() method providing
            steering directions as a tensor.
        hook_name: Hook point for steering (e.g. "blocks.5.hook_resid_pre").
        n_prompts: Number of prompts per task.
        steering_coefficients: Coefficients to sweep for dose-response.

    Returns:
        List of EvalResult, one per task.
    """
    if tasks is None:
        tasks = CIRCUIT_TASKS
    if steering_coefficients is None:
        steering_coefficients = [0.5, 1.0, 2.0, 5.0]

    if artifact is None:
        log("  WARNING: no artifact adapter provided, skipping steering reliability")
        return []

    effective_hook = hook_name or getattr(
        getattr(artifact, "manifest", None), "hook_point", None,
    )
    if not effective_hook:
        effective_hook = "blocks.5.hook_resid_pre"

    # Get directions from artifact — shape (n_features, d_model) or (layers, n_features, d_model)
    dirs = artifact.directions()
    if dirs.ndim == 3:
        dirs = dirs.mean(dim=0)
    # Use the mean direction across all features as the steering vector
    # (or first direction if artifact provides a single canonical direction)
    if dirs.shape[0] == 1:
        steering_direction = dirs[0]
    else:
        steering_direction = dirs.mean(dim=0)
    steering_direction = steering_direction / (steering_direction.norm() + 1e-12)

    tokenizer = model.tokenizer

    log(f"  Steering-Bench: {len(tasks)} tasks, hook={effective_hook}, "
        f"coeffs={steering_coefficients}")

    results = []

    for task in tasks:
        prompts = generate_prompts(task, tokenizer, n_prompts)
        if not prompts:
            log(f"  {task}: no prompts, skipping")
            continue

        correct_ids, incorrect_ids = get_token_ids(prompts, tokenizer)
        if not correct_ids:
            log(f"  {task}: no valid token pairs, skipping")
            continue

        n_valid = min(len(prompts), len(correct_ids))
        log(f"  {task}: {n_valid} valid prompts")

        # 1. Propensity: baseline P(correct) without steering
        propensity = _measure_propensity(model, prompts, correct_ids, incorrect_ids)
        log(f"    propensity (baseline P(correct)) = {propensity:.3f}")

        # 2. Steerability at each coefficient
        per_coeff_accuracy = {}
        per_coeff_raw_steerability = {}
        for coeff in steering_coefficients:
            steered_acc = _measure_steered_accuracy(
                model, prompts, correct_ids, incorrect_ids,
                steering_direction, effective_hook, coeff,
            )
            per_coeff_accuracy[coeff] = steered_acc
            per_coeff_raw_steerability[coeff] = steered_acc - propensity

        # 3. Find optimal coefficient (largest positive raw steerability)
        best_coeff = max(steering_coefficients, key=lambda c: per_coeff_raw_steerability[c])
        raw_steerability = per_coeff_raw_steerability[best_coeff]

        # 4. Propensity-corrected steerability
        if propensity < 1.0 - 1e-8:
            corrected_steerability = raw_steerability / (1.0 - propensity)
        else:
            # Model is already at ceiling; corrected steerability is undefined
            corrected_steerability = 0.0

        # 5. Dose-response linearity
        effects = [per_coeff_raw_steerability[c] for c in steering_coefficients]
        r2 = _dose_response_r2(steering_coefficients, effects)

        # Pass/fail
        passed = bool(corrected_steerability > 0.15)

        log(f"    raw_steerability = {raw_steerability:.3f} (best coeff={best_coeff})")
        log(f"    corrected_steerability = {corrected_steerability:.3f}")
        log(f"    dose_response R^2 = {r2:.3f}")
        log(f"    [{'PASS' if passed else 'FAIL'}]")

        results.append(EvalResult(
            metric_id="B21.steering_reliability",
            value=corrected_steerability,
            n_samples=n_valid,
            metadata={
                "task": task,
                "propensity": propensity,
                "raw_steerability": raw_steerability,
                "corrected_steerability": corrected_steerability,
                "dose_response_r2": r2,
                "optimal_coefficient": best_coeff,
                "steering_coefficients": steering_coefficients,
                "per_coeff_accuracy": {str(k): v for k, v in per_coeff_accuracy.items()},
                "per_coeff_raw_steerability": {str(k): v for k, v in per_coeff_raw_steerability.items()},
                "hook_name": effective_hook,
                "passed": passed,
                "threshold_corrected_steerability": 0.15,
            },
        ))

    # Log aggregate summary
    if results:
        mean_propensity = np.mean([r.metadata["propensity"] for r in results])
        mean_corrected = np.mean([r.metadata["corrected_steerability"] for r in results])
        mean_r2 = np.mean([r.metadata["dose_response_r2"] for r in results])
        n_passed = sum(1 for r in results if r.metadata["passed"])
        log(f"  SUMMARY: mean_propensity={mean_propensity:.3f}, "
            f"mean_corrected_steerability={mean_corrected:.3f}, "
            f"mean_dose_response_r2={mean_r2:.3f}")
        log(f"  passed: {n_passed}/{len(results)}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = parse_common_args("B21: Steering-Bench Reliability")
    parser.add_argument("--hook", default=None,
                        help="Hook point (e.g. blocks.5.hook_resid_pre)")
    parser.add_argument("--artifact-path", default=None,
                        help="SAE release ID for artifact")
    parser.add_argument("--sae-id", default=None,
                        help="SAE ID within release")
    parser.add_argument("--coefficients", type=float, nargs="+",
                        default=None,
                        help="Steering coefficients to sweep (default: 0.5 1.0 2.0 5.0)")
    args = parser.parse_args()

    model = load_model(args.model, args.device)

    artifact = None
    if args.artifact_path:
        from mechval.lib.artifacts import SAEAdapter
        artifact = SAEAdapter.from_pretrained(
            release=args.artifact_path,
            sae_id=args.sae_id or "",
            hook_point=args.hook or "",
        )

    tasks = args.tasks or CIRCUIT_TASKS

    log("=" * 60)
    log("B21: STEERING-BENCH RELIABILITY")
    log("=" * 60)

    out = args.out or "102_steering_reliability.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_steering_reliability(
            model, [task],
            artifact=artifact,
            hook_name=args.hook,
            n_prompts=args.n_prompts,
            steering_coefficients=args.coefficients,
        )
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)
            p = "PASS" if r.metadata["passed"] else "FAIL"
            log(f"  {task}: corrected_steerability={r.value:.3f}  "
                f"propensity={r.metadata['propensity']:.3f}  [{p}]")

    save_results(results, out)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
