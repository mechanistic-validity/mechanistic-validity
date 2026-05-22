"""Multiple Comparisons Correction
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     M93 — Multiple Comparisons Correction
Categories:     measurement
Validity layer: Measurement
Criteria:       M5 Statistical Rigor
Establishes:    Significant results survive family-wise error correction across instruments
Requires:       CPU, data-only
Doc:            /instruments_v2/measurement/m93-multiple-comparisons
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Meta-instrument that reads existing instrument result JSON files,
collects all testable claims (value vs baseline_random pairs and
pass/fail verdicts), then applies Bonferroni and Benjamini-Hochberg
corrections to estimate how many results survive multiple-testing
adjustment.

Pass condition: >50% of nominally significant results survive BH
correction at alpha=0.05.

Usage:
    uv run python 93_multiple_comparisons.py
    uv run python 93_multiple_comparisons.py --tasks ioi sva
"""
import math

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    DATA_DIR,
    EvalResult,
    load_results,
    log,
    parse_common_args,
    save_incremental,
    save_results,
)

ALPHA = 0.05


def _approximate_p_value(value: float, baseline: float, n_samples: int,
                         ci_low: float | None, ci_high: float | None) -> float | None:
    """Derive an approximate two-sided p-value from value, baseline, and CI or sample size.

    If CI bounds are available, estimate SE from them. Otherwise fall back to a
    conservative approximation using sqrt(n). Returns None when there is not
    enough information to compute anything meaningful.
    """
    if n_samples < 2:
        return None
    diff = abs(value - baseline)
    if diff < 1e-15:
        return 1.0

    se = None
    if ci_low is not None and ci_high is not None:
        ci_width = ci_high - ci_low
        if ci_width > 0:
            se = ci_width / (2 * 1.96)

    if se is None:
        scale = max(abs(value), abs(baseline), 1e-8)
        se = scale / math.sqrt(n_samples)

    if se < 1e-15:
        return 0.0

    z = diff / se
    p = 2.0 * (1.0 - _normal_cdf(z))
    return p


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def benjamini_hochberg(p_values: list[float], alpha: float = ALPHA) -> list[bool]:
    """Return a boolean mask: True if the test survives BH correction."""
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    survives = [False] * m
    max_k = -1
    for k_minus_1, (orig_idx, p) in enumerate(indexed):
        k = k_minus_1 + 1
        threshold = (k / m) * alpha
        if p <= threshold:
            max_k = k_minus_1
    if max_k >= 0:
        for j in range(max_k + 1):
            survives[indexed[j][0]] = True
    return survives


def bonferroni(p_values: list[float], alpha: float = ALPHA) -> list[bool]:
    """Return a boolean mask: True if the test survives Bonferroni correction."""
    m = len(p_values)
    if m == 0:
        return []
    threshold = alpha / m
    return [p <= threshold for p in p_values]


def _collect_testable_results(task: str) -> list[dict]:
    """Scan all numbered JSON files in DATA_DIR for results matching task."""
    testable = []
    for path in sorted(DATA_DIR.glob("[0-9][0-9]_*.json")):
        data = load_results(path.name)
        if data is None:
            continue
        if not isinstance(data, list):
            continue
        for item in data:
            md = item.get("metadata", {})
            item_task = md.get("task")
            if item_task != task:
                continue

            value = item.get("value")
            baseline = item.get("baseline_random")
            n_samples = item.get("n_samples", 0)
            ci_low = item.get("ci_low")
            ci_high = item.get("ci_high")
            passed = md.get("passed")
            metric_id = item.get("metric_id", path.stem)

            p_val = None
            if value is not None and baseline is not None and n_samples >= 2:
                p_val = _approximate_p_value(value, baseline, n_samples, ci_low, ci_high)

            if p_val is None and passed is not None:
                p_val = 0.01 if passed else 0.50

            if p_val is None:
                continue

            testable.append({
                "source_file": path.name,
                "metric_id": metric_id,
                "task": task,
                "value": value,
                "baseline_random": baseline,
                "p_value": p_val,
                "nominally_significant": p_val < ALPHA,
                "passed": passed,
                "n_samples": n_samples,
            })
    return testable


def _analyze_task(task: str) -> EvalResult | None:
    """Run multiple comparisons analysis for one task."""
    testable = _collect_testable_results(task)
    if not testable:
        log(f"  {task}: no testable results found")
        return None

    p_values = [t["p_value"] for t in testable]
    m = len(p_values)
    n_nominal = sum(1 for p in p_values if p < ALPHA)

    bh_mask = benjamini_hochberg(p_values, ALPHA)
    bonf_mask = bonferroni(p_values, ALPHA)

    n_bh = sum(bh_mask)
    n_bonf = sum(bonf_mask)
    fwer_estimate = 1.0 - (1.0 - ALPHA) ** m

    bh_survival_rate = n_bh / n_nominal if n_nominal > 0 else 0.0
    passed = bh_survival_rate > 0.50

    per_test = []
    for i, t in enumerate(testable):
        per_test.append({
            "source_file": t["source_file"],
            "metric_id": t["metric_id"],
            "p_value": t["p_value"],
            "nominally_significant": t["nominally_significant"],
            "survives_bh": bh_mask[i],
            "survives_bonferroni": bonf_mask[i],
        })

    log(f"  {task}: {m} tests, {n_nominal} nominal, {n_bh} survive BH, "
        f"{n_bonf} survive Bonferroni, FWER={fwer_estimate:.4f}, "
        f"BH survival rate={bh_survival_rate:.3f} -> {'PASS' if passed else 'FAIL'}")

    return EvalResult(
        metric_id="M93.multiple_comparisons",
        value=bh_survival_rate,
        n_samples=m,
        metadata={
            "task": task,
            "alpha": ALPHA,
            "n_tests": m,
            "n_nominally_significant": n_nominal,
            "n_survive_bh": n_bh,
            "n_survive_bonferroni": n_bonf,
            "bh_survival_rate": bh_survival_rate,
            "fwer_estimate": fwer_estimate,
            "bonferroni_threshold": ALPHA / m if m > 0 else None,
            "passed": passed,
            "per_test": per_test,
        },
    )


def main():
    parser = parse_common_args("M93: Multiple Comparisons Correction (data-only)")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS

    log("=" * 60)
    log("M93: MULTIPLE COMPARISONS CORRECTION")
    log("=" * 60)
    log(f"Tasks: {tasks}")
    log(f"DATA_DIR: {DATA_DIR}")
    log(f"Alpha: {ALPHA}")

    all_results: list[EvalResult] = []
    out = args.out or "93_multiple_comparisons.json"
    incremental_out = out.replace(".json", ".jsonl")

    for task in tasks:
        log(f"\n--- {task} ---")
        result = _analyze_task(task)
        if result is not None:
            all_results.append(result)
            save_incremental(result, incremental_out)

    if all_results:
        all_p = []
        for r in all_results:
            per_test = r.metadata.get("per_test", [])
            all_p.extend(t["p_value"] for t in per_test)

        if all_p:
            global_bh = benjamini_hochberg(all_p, ALPHA)
            global_bonf = bonferroni(all_p, ALPHA)
            n_global_nominal = sum(1 for p in all_p if p < ALPHA)
            n_global_bh = sum(global_bh)
            n_global_bonf = sum(global_bonf)
            global_fwer = 1.0 - (1.0 - ALPHA) ** len(all_p)
            global_bh_rate = n_global_bh / n_global_nominal if n_global_nominal > 0 else 0.0
            global_passed = global_bh_rate > 0.50

            log(f"\n--- GLOBAL (all tasks pooled) ---")
            log(f"  {len(all_p)} total tests, {n_global_nominal} nominal, "
                f"{n_global_bh} survive BH, {n_global_bonf} survive Bonferroni")
            log(f"  FWER={global_fwer:.4f}, BH survival rate={global_bh_rate:.3f} "
                f"-> {'PASS' if global_passed else 'FAIL'}")

            all_results.append(EvalResult(
                metric_id="M93.multiple_comparisons_global",
                value=global_bh_rate,
                n_samples=len(all_p),
                metadata={
                    "task": "ALL",
                    "alpha": ALPHA,
                    "n_tests": len(all_p),
                    "n_nominally_significant": n_global_nominal,
                    "n_survive_bh": n_global_bh,
                    "n_survive_bonferroni": n_global_bonf,
                    "bh_survival_rate": global_bh_rate,
                    "fwer_estimate": global_fwer,
                    "bonferroni_threshold": ALPHA / len(all_p) if all_p else None,
                    "passed": global_passed,
                },
            ))

    save_results(all_results, out)

    log(f"\nDone. {len(all_results)} results saved.")


if __name__ == "__main__":
    main()
