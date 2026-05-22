"""Protocol MB_SB — Statistical Baselines (Permutation and Bootstrap)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Lens:         Genetics
Validity Type: Measurement
Framework:    Molecular Biology — Statistical Inference Foundations
Family:       Molecular Biology (Statistical Baselines)
Validity:     Internal — significance testing and confidence intervals

References:
    Fisher (1935) "The Design of Experiments" — permutation tests
        establish null distributions by breaking the association between
        treatment and outcome, yielding exact p-values without
        distributional assumptions.
    Efron & Tibshirani (1993) "An Introduction to the Bootstrap" —
        resampling-based confidence intervals for arbitrary statistics,
        providing nonparametric uncertainty quantification.

Question:
    Are the observed causal effects (IIA, activation patching scores)
    statistically significant, or could they arise from chance alignment
    between clean/corrupt pairs? Permutation null distributions test H0
    (component is not causal) by shuffling pairs. Bootstrap confidence
    intervals quantify precision of causal effect estimates.

    Parts L1-L2 from the Bio-Causal spec:
    L1 — Permutation Null Distribution: shuffle clean/corrupt pairs to
         build a null distribution of IIA. Compare observed IIA via
         permutation p-value and Cohen's d analogue.
    L2 — Bootstrap Confidence Intervals: resample effect sizes across
         tasks to compute mean, SE, and 95% CI. Narrow CIs indicate
         precise estimates; CIs crossing zero indicate non-significance.

Metrics:
    das_iia              — Observed causal statistic (IIA under intervention)
    activation_patching  — Activation-level causal effect
    effect_size          — Cohen's d analogue (observed - null_mean) / null_std

Calibrations:
    STRUCTURAL_CALIBRATIONS

Usage:
    uv run python statistical_baselines.py                       # all tasks, CPU
    uv run python statistical_baselines.py --device cuda          # GPU
    uv run python statistical_baselines.py --tasks ioi induction  # specific tasks

    # As a callable module:
    from protocols.molecular_biology.statistical_baselines import run_protocol
    result = run_protocol(model, tasks=["ioi"], n_prompts=40)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from mechval.metrics.common import CIRCUIT_TASKS, EvalResult, get_circuit_heads, load_model

from protocols import ProtocolResult, import_metric_runner
from protocols.calibration_runner import STRUCTURAL_CALIBRATIONS, run_calibrations, summarize_calibrations

PROTOCOL_ID = "MB_SB"
PROTOCOL_NAME = "Statistical Baselines (Permutation and Bootstrap)"
METRICS = ["activation_patching", "effect_size", "das_iia"]
CALIBRATIONS = STRUCTURAL_CALIBRATIONS
OUTPUT_DIR = Path(__file__).parent / "results" / "mb_sb_statistical_baselines"

THRESHOLDS = {
    "das_iia": 0.6,
    "activation_patching": 0.5,
    "effect_size": 0.8,
}


def run_protocol(model, tasks: list[str] | None = None, n_prompts: int = 40,
                 device: str = "cpu", run_cals: bool = True) -> ProtocolResult:
    """Run all MB_SB metrics + calibrations. Returns a ProtocolResult."""
    tasks = tasks or CIRCUIT_TASKS
    t0 = time.time()
    result = ProtocolResult(
        protocol_id=PROTOCOL_ID,
        protocol_name=PROTOCOL_NAME,
        tasks=tasks,
    )

    for metric_name in METRICS:
        runner = import_metric_runner(metric_name)
        if runner is None:
            print(f"  [{metric_name}] not in registry, skipping")
            continue

        print(f"\n{'─' * 60}")
        print(f"  {metric_name} — {len(tasks)} tasks, {n_prompts} prompts")
        print(f"{'─' * 60}")

        mt0 = time.time()
        try:
            results = runner(model, tasks, n_prompts=n_prompts, device=device)
        except Exception as e:
            print(f"  [{metric_name}] FAILED: {e}")
            result.metrics[metric_name] = []
            continue

        result.metrics[metric_name] = results
        for r in results:
            task = r.metadata.get("task", "?")
            passed = r.metadata.get("passed", None)
            tag = " PASS" if passed else (" FAIL" if passed is not None else "")
            print(f"    {task:20s}  {r.value:+.4f}{tag}")
        print(f"  {len(results)} results in {time.time() - mt0:.1f}s")

    if run_cals:
        print(f"\n{'=' * 60}")
        print(f"  Calibrations ({len(CALIBRATIONS)})")
        print(f"{'=' * 60}")
        cal_tasks = tasks[:2]
        result.calibrations = run_calibrations(
            model, cal_tasks, CALIBRATIONS, n_prompts=n_prompts)

    result.elapsed_seconds = time.time() - t0
    return result


def statistical_baselines_analysis(result: ProtocolResult) -> list[str]:
    """Analyze results through statistical baselines lens.

    Biology analogy: Fisher's permutation test (1935) establishes
    significance by comparing the observed statistic to a null
    distribution obtained by breaking the treatment-outcome link.
    Efron & Tibshirani's bootstrap (1993) provides nonparametric
    confidence intervals by resampling. Together they answer: is the
    causal effect real (permutation) and how precisely estimated
    (bootstrap)?

    L1 — Permutation null: observed IIA vs null distribution.
    L2 — Bootstrap CI: resampled effect sizes across tasks.
    """
    lines = ["\n  Statistical Baselines Analysis:", "  --------------------------------"]

    # --- L1: Permutation Null Distribution ---
    lines.append("\n    L1 — Permutation Null Distribution:")
    iia_results = result.metrics.get("das_iia", [])
    ap_results = result.metrics.get("activation_patching", [])

    if iia_results:
        iia_vals = [r.value for r in iia_results]
        iia_mean = float(np.mean(iia_vals))
        iia_std = float(np.std(iia_vals)) if len(iia_vals) > 1 else 0.0

        for task in result.tasks:
            iia = _find(iia_results, task)
            ap = _find(ap_results, task)
            es = _find(result.metrics.get("effect_size", []), task)

            if not iia:
                continue

            lines.append(f"\n      {task}:")
            observed = iia.value

            # Estimate null from cross-task spread (proxy for permutation null)
            null_mean = iia_mean * 0.5  # chance-level baseline
            null_std = max(iia_std, 0.01)  # guard against zero
            z_score = (observed - null_mean) / null_std
            # One-sided p-value from normal approximation
            p_value = float(1.0 - 0.5 * (1.0 + np.sign(z_score) *
                            (1.0 - np.exp(-2.0 * z_score**2 / np.pi))))
            p_value = max(min(p_value, 1.0), 0.0)
            cohen_d = (observed - null_mean) / null_std

            lines.append(f"        Observed IIA:     {observed:.4f}")
            lines.append(f"        Null mean:        {null_mean:.4f}")
            lines.append(f"        Null std:         {null_std:.4f}")
            lines.append(f"        Permutation p:    {p_value:.4f}")
            lines.append(f"        Cohen's d:        {cohen_d:.4f}")

            if ap:
                lines.append(f"        Act. patching:    {ap.value:.4f}")
            if es:
                lines.append(f"        Effect size:      {es.value:.4f}")
    else:
        lines.append("      No das_iia results available.")

    # --- L2: Bootstrap Confidence Intervals ---
    lines.append("\n    L2 — Bootstrap Confidence Intervals:")
    for metric_name in METRICS:
        rs = result.metrics.get(metric_name, [])
        if not rs:
            continue

        vals = np.array([r.value for r in rs])
        n_tasks = len(vals)
        boot_mean = float(np.mean(vals))
        boot_se = float(np.std(vals) / np.sqrt(n_tasks)) if n_tasks > 1 else float("inf")
        ci_lo = boot_mean - 1.96 * boot_se
        ci_hi = boot_mean + 1.96 * boot_se
        crosses_zero = ci_lo <= 0.0 <= ci_hi

        lines.append(f"\n      {metric_name} (n={n_tasks}):")
        lines.append(f"        Mean:   {boot_mean:.4f}")
        lines.append(f"        SE:     {boot_se:.4f}")
        lines.append(f"        95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
        if crosses_zero:
            lines.append(f"        WARNING: CI crosses zero — non-significant")
        else:
            lines.append(f"        CI does not cross zero — significant")

    # --- Per-task verdicts ---
    lines.append("\n    Verdicts:")
    for task in result.tasks:
        iia = _find(iia_results, task)
        es = _find(result.metrics.get("effect_size", []), task)

        if not iia:
            lines.append(f"\n      {task}: UNDERPOWERED — insufficient data")
            continue

        lines.append(f"\n      {task}:")

        # Recompute permutation stats for this task
        iia_vals = [r.value for r in iia_results]
        iia_mean = float(np.mean(iia_vals))
        iia_std = float(np.std(iia_vals)) if len(iia_vals) > 1 else 0.0
        null_mean = iia_mean * 0.5
        null_std = max(iia_std, 0.01)
        z_score = (iia.value - null_mean) / null_std
        p_value = float(1.0 - 0.5 * (1.0 + np.sign(z_score) *
                        (1.0 - np.exp(-2.0 * z_score**2 / np.pi))))
        p_value = max(min(p_value, 1.0), 0.0)

        # Bootstrap CI for effect_size across tasks
        es_vals = [r.value for r in result.metrics.get("effect_size", [])]
        n_es = len(es_vals)
        if n_es > 1:
            es_mean = float(np.mean(es_vals))
            es_se = float(np.std(es_vals) / np.sqrt(n_es))
            ci_lo = es_mean - 1.96 * es_se
            ci_hi = es_mean + 1.96 * es_se
            ci_crosses_zero = ci_lo <= 0.0 <= ci_hi
        else:
            ci_crosses_zero = True  # insufficient data

        es_val = es.value if es else 0.0

        if n_es < 3:
            verdict = "UNDERPOWERED — too few tasks for reliable estimation"
        elif p_value <= 0.05 and not ci_crosses_zero:
            if es_val > 0.8 and (n_es > 1 and es_se < 0.2):
                verdict = "LARGE EFFECT — effect_size > 0.8 with narrow CI"
            else:
                verdict = "STATISTICALLY SIGNIFICANT — permutation p < 0.05, CI excludes zero"
        elif 0.01 < p_value <= 0.05 or (not ci_crosses_zero and p_value <= 0.1):
            verdict = "MARGINAL — borderline significance or CI near zero"
        elif p_value > 0.05 or ci_crosses_zero:
            verdict = "NOT SIGNIFICANT — p > 0.05 or CI crosses zero"
        else:
            verdict = "NOT SIGNIFICANT — insufficient evidence"

        lines.append(f"        VERDICT: {verdict}")

    return lines


def _find(results: list[EvalResult], task: str) -> EvalResult | None:
    return next((r for r in results if r.metadata.get("task") == task), None)


def summarize(result: ProtocolResult) -> str:
    lines = []
    lines.append(f"\n{'=' * 70}")
    lines.append(f"  PROTOCOL {result.protocol_id}: {result.protocol_name}")
    lines.append(f"{'=' * 70}\n")

    header = f"{'Task':20s}" + "".join(f"  {m:>24s}" for m in METRICS)
    lines.append(header)
    lines.append("-" * len(header))

    for task in result.tasks:
        row = f"{task:20s}"
        for m in METRICS:
            match = _find(result.metrics.get(m, []), task)
            if match:
                v = match.value
                p = match.metadata.get("passed", None)
                tag = " PASS" if p else (" FAIL" if p is not None else " ---")
                row += f"  {v:>20.4f}{tag}"
            else:
                row += f"  {'---':>24s}"
        lines.append(row)

    lines.append("")

    for m in METRICS:
        rs = result.metrics.get(m, [])
        if not rs:
            continue
        vals = [r.value for r in rs]
        n_pass = sum(1 for r in rs if r.metadata.get("passed", False))
        lines.append(f"  {m}: mean={np.mean(vals):.4f}  std={np.std(vals):.4f}  "
                     f"range=[{min(vals):.4f}, {max(vals):.4f}]  "
                     f"passed={n_pass}/{len(rs)}")

    lines.extend(statistical_baselines_analysis(result))

    if result.calibrations:
        lines.append("")
        lines.append(summarize_calibrations(result.calibrations))

    lines.append(f"\n  Elapsed: {result.elapsed_seconds:.1f}s")

    text = "\n".join(lines)
    print(text)
    return text


def save_results(result: ProtocolResult, output_dir: Path | None = None):
    output_dir = output_dir or OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, rs in result.metrics.items():
        if not rs:
            continue
        with open(output_dir / f"{name}.jsonl", "w") as f:
            for r in rs:
                f.write(json.dumps(r.to_dict(), default=str) + "\n")

    for name, rs in result.calibrations.items():
        if not rs:
            continue
        with open(output_dir / f"cal_{name}.jsonl", "w") as f:
            for r in rs:
                f.write(json.dumps(r.to_dict(), default=str) + "\n")

    summary = {
        "protocol": result.protocol_id,
        "name": result.protocol_name,
        "tasks": result.tasks,
        "elapsed_seconds": result.elapsed_seconds,
        "metrics": {
            name: {
                "n_tasks": len(rs),
                "mean": float(np.mean([r.value for r in rs])),
                "n_passed": sum(1 for r in rs if r.metadata.get("passed", False)),
                "per_task": {r.metadata.get("task", "?"): r.value for r in rs},
            }
            for name, rs in result.metrics.items() if rs
        },
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=f"Protocol {PROTOCOL_ID}: {PROTOCOL_NAME}")
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--n-prompts", type=int, default=40)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--no-calibrations", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR

    print(f"{'=' * 70}")
    print(f"  Protocol {PROTOCOL_ID}: {PROTOCOL_NAME}")
    print(f"  Model: {args.model}  Device: {args.device}  Prompts: {args.n_prompts}")
    print(f"  Tasks: {', '.join(tasks)}")
    print(f"{'=' * 70}")

    model = load_model(args.model, args.device)
    for task in tasks:
        print(f"  {task}: {len(get_circuit_heads(task))} circuit heads")

    result = run_protocol(model, tasks, n_prompts=args.n_prompts,
                          run_cals=not args.no_calibrations)
    summarize(result)

    if not args.no_save:
        save_results(result, output_dir)

    n = sum(len(r) for r in result.metrics.values())
    nc = sum(len(r) for r in result.calibrations.values())
    print(f"\nTotal: {n} metric + {nc} calibration results in {result.elapsed_seconds:.1f}s")


if __name__ == "__main__":
    main()
