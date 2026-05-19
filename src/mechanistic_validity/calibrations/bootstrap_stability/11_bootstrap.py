"""Bootstrap Stability
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     F01 — Bootstrap Stability
Categories:     measurement
Validity layer: Measurement
Criteria:       M1 Reliability
Establishes:    Circuit metrics have tight confidence intervals under resampling
Requires:       GPU, model
Doc:            /instruments_v2/measurement/f01-bootstrap-stability
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Wraps any inner metric (default: faithfulness via activation patching)
and computes bootstrap confidence intervals. Reports F +/- 1.96*sigma
as a 95% CI.

No new dependencies beyond scipy (already installed).

Usage:
    uv run python 11_bootstrap.py --tasks ioi greater_than --n-prompts 40
    uv run python 11_bootstrap.py --inner completeness --n-bootstrap 500
"""

import numpy as np
import torch

from mechanistic_validity.metrics.common import (
    ALL_TASKS,
    CIRCUIT_TASKS,
    DATA_DIR,
    EvalResult,
    calibrate_mean_z,
    compute_completeness,
    compute_faithfulness,
    generate_prompts,
    get_circuit_heads,
    get_token_ids,
    load_model,
    log,
    parse_common_args,
    save_results,
)


def bootstrap_metric(model, prompts, correct_ids, incorrect_ids,
                     circuit_heads, mean_z, metric_fn, n_bootstrap=1000,
                     seed=42) -> tuple[float, float, float, float]:
    """Bootstrap a metric over prompt subsamples.

    Returns (point_estimate, ci_low, ci_high, sigma).
    """
    rng = np.random.RandomState(seed)
    n = len(prompts)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0

    point = metric_fn(model, prompts, correct_ids, incorrect_ids,
                      circuit_heads, mean_z)

    boot_values = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot_prompts = [prompts[i] for i in idx]
        boot_correct = [correct_ids[i] for i in idx]
        boot_incorrect = [incorrect_ids[i] for i in idx]
        val = metric_fn(model, boot_prompts, boot_correct, boot_incorrect,
                        circuit_heads, mean_z)
        boot_values.append(val)

    boot_arr = np.array(boot_values)
    sigma = float(np.std(boot_arr))
    ci_low = float(np.percentile(boot_arr, 2.5))
    ci_high = float(np.percentile(boot_arr, 97.5))

    return point, ci_low, ci_high, sigma


def run_bootstrap(model, tasks: list[str], n_prompts: int = 40,
                  n_bootstrap: int = 1000, inner: str = "faithfulness") -> list[EvalResult]:
    tokenizer = model.tokenizer
    results = []

    metric_fn = compute_faithfulness if inner == "faithfulness" else compute_completeness

    for task in tasks:
        heads = get_circuit_heads(task)
        if not heads:
            log(f"  {task}: no circuit, skipping")
            continue

        prompts = generate_prompts(task, tokenizer, n_prompts)
        if not prompts:
            log(f"  {task}: no prompts, skipping")
            continue

        correct_ids, incorrect_ids = get_token_ids(prompts, tokenizer)
        if not correct_ids:
            continue

        log(f"  {task} ({len(heads)} heads, {len(prompts)} prompts, B={n_bootstrap})...")
        mean_z = calibrate_mean_z(model, prompts, n_calibration=min(100, len(prompts)))

        point, ci_low, ci_high, sigma = bootstrap_metric(
            model, prompts, correct_ids, incorrect_ids,
            heads, mean_z, metric_fn, n_bootstrap=n_bootstrap,
        )

        log(f"    {inner}={point:.3f} [{ci_low:.3f}, {ci_high:.3f}] sigma={sigma:.3f}")

        results.append(EvalResult(
            metric_id=f"C11.bootstrap_{inner}",
            value=point,
            ci_low=ci_low,
            ci_high=ci_high,
            n_samples=len(prompts),
            metadata={
                "task": task,
                "inner_metric": inner,
                "n_bootstrap": n_bootstrap,
                "sigma": sigma,
                "n_circuit_heads": len(heads),
                "circuit_heads": sorted(heads),
            },
        ))

    return results


def main():
    parser = parse_common_args("C11: Bootstrap Stability")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--inner", default="faithfulness",
                        choices=["faithfulness", "completeness"])
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("C11: BOOTSTRAP STABILITY")
    log("=" * 60)

    results = run_bootstrap(model, tasks, args.n_prompts, args.n_bootstrap, args.inner)

    out = args.out or f"11_bootstrap_{args.inner}.json"
    save_results(results, out)

    log(f"\nDone. {len(results)} tasks evaluated.")
    for r in results:
        t = r.metadata["task"]
        log(f"  {t}: {r.value:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}]")


if __name__ == "__main__":
    main()
