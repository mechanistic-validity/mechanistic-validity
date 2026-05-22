"""Protocol EX02 — Psychometric Analysis (DIF)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Lens:         Measurement Theory
Validity Type: Measurement
Framework:    https://mechanistic-validity.github.io/mechanistic-validity/framework/metrics/cognitive_science/psychometrics/
Family:       Cognitive Science (Psychometrics)
Validity:     External — measurement fairness across circuit populations

References:
    Holland & Wainer (1993) "Differential Item Functioning"
    Lord (1980) "Applications of Item Response Theory to Practical Testing Problems"
    Embretson & Reise (2000) "Item Response Theory for Psychologists"

Question:
    Do metrics behave differently across circuits in ways unrelated to the
    circuits' actual properties? DIF (Differential Item Functioning) tests
    whether measurement items are fair — whether a metric score means the
    same thing for different circuits. A metric with large DIF is biased:
    it conflates circuit identity with circuit quality.

Metrics:
    dif — Differential Item Functioning effect size across circuit groups

Usage:
    uv run python psychometrics.py
    uv run python psychometrics.py --tasks ioi induction --device cuda

    from protocols.cognitive_science.psychometrics import run_protocol
    result = run_protocol(model, tasks=["ioi"])
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from mechval.metrics.common import CIRCUIT_TASKS, EvalResult, get_circuit_heads, load_model

from protocols import ProtocolResult, import_metric_runner
from protocols.calibration_runner import STRUCTURAL_CALIBRATIONS, run_calibrations, summarize_calibrations

PROTOCOL_ID = "EX02"
PROTOCOL_NAME = "Psychometric Analysis (DIF)"
METRICS = ["dif"]
CALIBRATIONS = STRUCTURAL_CALIBRATIONS
OUTPUT_DIR = Path(__file__).parent / "results" / "ex02_psychometrics"

THRESHOLDS = {
    "dif": 0.25,
}




def run_protocol(model, tasks: list[str] | None = None, n_prompts: int = 40,
                 run_cals: bool = True) -> ProtocolResult:
    """Run all EX02 metrics + calibrations. Returns a ProtocolResult."""
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
            results = runner(model, tasks, n_prompts=n_prompts)
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


def dif_analysis(result: ProtocolResult) -> list[str]:
    """Analyze DIF results to flag measurement unfairness."""
    lines = ["\n  Differential Item Functioning Analysis:", "  ----------------------------------------"]

    dif_results = result.metrics.get("dif", [])
    if not dif_results:
        lines.append("    No DIF results available.")
        return lines

    negligible = []
    moderate = []
    large = []

    for r in dif_results:
        task = r.metadata.get("task", "?")
        v = abs(r.value)
        if v < 0.1:
            negligible.append((task, r.value))
        elif v < THRESHOLDS["dif"]:
            moderate.append((task, r.value))
        else:
            large.append((task, r.value))

    if large:
        lines.append("\n    LARGE DIF (measurement bias likely):")
        for task, v in large:
            lines.append(f"      {task:20s}  |DIF| = {abs(v):.4f}")
            lines.append(f"        Metric scores for this task mean different things for "
                         f"different circuits — results are NOT directly comparable")

    if moderate:
        lines.append("\n    MODERATE DIF (possible measurement bias):")
        for task, v in moderate:
            lines.append(f"      {task:20s}  |DIF| = {abs(v):.4f}")
            lines.append(f"        Some measurement non-invariance detected — compare "
                         f"with caution across circuit groups")

    if negligible:
        lines.append("\n    NEGLIGIBLE DIF (measurement is fair):")
        for task, v in negligible:
            lines.append(f"      {task:20s}  |DIF| = {abs(v):.4f}")

    total = len(dif_results)
    n_fair = len(negligible)
    n_biased = len(large)

    if n_biased == 0:
        verdict = (f"All {total} tasks show fair measurement (no large DIF) — "
                   f"metric scores are comparable across circuits")
    elif n_biased < total / 2:
        verdict = (f"{n_biased}/{total} tasks show large DIF — "
                   f"metric is partially biased; interpret flagged tasks with caution")
    else:
        verdict = (f"{n_biased}/{total} tasks show large DIF — "
                   f"metric is systematically biased and should not be used for "
                   f"cross-circuit comparison without correction")

    lines.append(f"\n    VERDICT: {verdict}")
    lines.append(f"    Fair: {n_fair}  Moderate: {len(moderate)}  "
                 f"Biased: {n_biased}  (total: {total})")

    return lines


def _find(results: list[EvalResult], task: str) -> EvalResult | None:
    return next((r for r in results if r.metadata.get("task") == task), None)


def summarize(result: ProtocolResult) -> str:
    lines = []
    lines.append(f"\n{'=' * 70}")
    lines.append(f"  PROTOCOL {result.protocol_id}: {result.protocol_name}")
    lines.append(f"{'=' * 70}\n")

    header = f"{'Task':20s}" + "".join(f"  {m:>16s}" for m in METRICS)
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
                row += f"  {v:>12.4f}{tag}"
            else:
                row += f"  {'---':>16s}"
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

    lines.extend(dif_analysis(result))

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
