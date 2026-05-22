"""Protocol A05 — MDC / Glennan Mechanisms
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Lens:         Neuroscience
Validity Type: Internal
Framework:    https://mechanistic-validity.github.io/mechanistic-validity/framework/metrics/causal/a05-mdc-glennan/
Family:       Causal (Mechanistic — component-level explanation)
Validity:     Internal — I2 Sufficiency, I5 Mechanistic Explanation

References:
    Machamer, Darden & Craver (2000) "Thinking about Mechanisms" — MDC framework
    Glennan (2017) "The New Mechanical Philosophy" — updated mechanistic account
    Bechtel & Abrahamsen (2005) "Explanation: A Mechanist Alternative" — mechanistic explanation

Question:
    Does the circuit constitute a genuine MECHANISM in the philosophical sense?
    A mechanism requires:

    (1) OPERATION SPECIFICATION: Can we describe what each component does
        (not just that it matters, but HOW it contributes)?
    (2) PREDICTION: Does understanding the mechanism let us predict held-out
        behavior (generalization beyond training distribution)?
    (3) REPLACEMENT: Can we replace a component with a synthetic one that
        performs the same operation and preserve function?
    (4) PROCEDURE: Can the mechanism's operation be specified as a procedure
        (algorithm) that a different system could execute?
    (5) COMPOSITION: Do the components combine hierarchically — is the
        mechanism decomposable into sub-mechanisms?
    (6) LOGIC GATES: Can component interactions be described as logical
        operations (AND, OR, inhibition)?

Metrics:
    operation_specification — Can each component's operation be described?
    held_out_prediction     — Does the mechanism predict held-out behavior?
    replacement_test        — Can components be replaced with synthetic equivalents?
    procedure_specification — Can the mechanism be specified as an algorithm?
    composition_test        — Is the mechanism hierarchically decomposable?
    logic_gates             — Can interactions be described as logic gates?

Calibrations:
    bootstrap           — Are metric values stable across resampled prompts?
    seed_variance       — Are results reproducible across random seeds?
    ablation_invariance — Do results change under different ablation methods?
    method_invariance   — Zero vs mean vs resample ablation agreement
    convergent_validity — Do different causal metrics agree?

Usage:
    uv run python a05_mdc_glennan.py                       # all tasks, CPU
    uv run python a05_mdc_glennan.py --device cuda          # GPU
    uv run python a05_mdc_glennan.py --tasks ioi induction  # specific tasks

    # As a callable module:
    from protocols.causal.a05_mdc_glennan import run_protocol
    result = run_protocol(model, tasks=["ioi"], n_prompts=40)
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
from protocols.calibration_runner import CAUSAL_CALIBRATIONS, run_calibrations, summarize_calibrations

PROTOCOL_ID = "A05"
PROTOCOL_NAME = "MDC / Glennan Mechanisms"
METRICS = ["operation_specification", "held_out_prediction", "replacement_test",
           "procedure_specification", "composition_test", "logic_gates"]
CALIBRATIONS = CAUSAL_CALIBRATIONS
OUTPUT_DIR = Path(__file__).parent / "results" / "a05_mdc_glennan"

THRESHOLDS = {
    "operation_specification": 0.5,
    "held_out_prediction": 0.5,
    "replacement_test": 0.7,
    "procedure_specification": 0.5,
    "composition_test": 0.5,
    "logic_gates": 0.5,
}




def run_protocol(model, tasks: list[str] | None = None, n_prompts: int = 40,
                 run_cals: bool = True) -> ProtocolResult:
    """Run all A05 metrics + calibrations. Returns a ProtocolResult."""
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
            tag = " ✓" if passed else (" ✗" if passed is not None else "")
            print(f"    {task:20s}  {r.value:+.4f}{tag}")
        print(f"  {len(results)} results in {time.time() - mt0:.1f}s")

    if run_cals:
        print(f"\n{'═' * 60}")
        print(f"  Calibrations ({len(CALIBRATIONS)})")
        print(f"{'═' * 60}")
        cal_tasks = tasks[:2]
        result.calibrations = run_calibrations(
            model, cal_tasks, CALIBRATIONS, n_prompts=n_prompts)

    result.elapsed_seconds = time.time() - t0
    return result


def mechanism_analysis(result: ProtocolResult) -> list[str]:
    """Analyze results through MDC/Glennan mechanistic criteria."""
    lines = ["\n  Mechanistic Explanation Analysis:", "  ─────────────────────────────────"]

    criterion_names = {
        "operation_specification": "Operation Specification",
        "held_out_prediction": "Predictive Power",
        "replacement_test": "Component Replaceability",
        "procedure_specification": "Procedural Specification",
        "composition_test": "Hierarchical Composition",
        "logic_gates": "Logical Structure",
    }

    for task in result.tasks:
        lines.append(f"\n    {task}:")

        criteria_met = []

        for metric_name, criterion_label in criterion_names.items():
            r = _find(result.metrics.get(metric_name, []), task)
            if r is None:
                continue

            threshold = THRESHOLDS.get(metric_name, 0.5)
            passed = r.value > threshold
            status = "PASS" if passed else "FAIL"
            lines.append(f"      {criterion_label}: {r.value:.4f} [{status}]")

            if passed:
                criteria_met.append(criterion_label)

        n = len(criteria_met)
        total = len(criterion_names)
        if n == total:
            verdict = "Full mechanistic explanation — all criteria satisfied"
        elif n >= total // 2:
            verdict = f"Partial mechanism ({n}/{total} criteria: {', '.join(criteria_met)})"
        elif n >= 1:
            verdict = f"Weak mechanism ({n}/{total} criteria: {', '.join(criteria_met)})"
        else:
            verdict = "Not a mechanism — no criteria satisfied"
        lines.append(f"      VERDICT: {verdict}")

    return lines


def _find(results: list[EvalResult], task: str) -> EvalResult | None:
    return next((r for r in results if r.metadata.get("task") == task), None)


def _top_heads(r: EvalResult, k: int) -> str:
    ph = r.metadata.get("per_head", {})
    if not ph:
        return ""
    items = sorted(ph.items(),
                   key=lambda kv: abs(kv[1].get("effect", kv[1]) if isinstance(kv[1], dict) else kv[1]),
                   reverse=True)[:k]
    return ", ".join(f"{h}" for h, _ in items)


def summarize(result: ProtocolResult) -> str:
    lines = []
    lines.append(f"\n{'═' * 70}")
    lines.append(f"  PROTOCOL {result.protocol_id}: {result.protocol_name}")
    lines.append(f"{'═' * 70}\n")

    # Task x Metric matrix
    header = f"{'Task':20s}" + "".join(f"  {m:>20s}" for m in METRICS)
    lines.append(header)
    lines.append("─" * len(header))

    for task in result.tasks:
        row = f"{task:20s}"
        for m in METRICS:
            match = _find(result.metrics.get(m, []), task)
            if match:
                v = match.value
                p = match.metadata.get("passed", None)
                tag = " ✓" if p else (" ✗" if p is not None else " —")
                row += f"  {v:>18.4f}{tag}"
            else:
                row += f"  {'—':>20s}"
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

    lines.extend(mechanism_analysis(result))

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

    print(f"{'═' * 70}")
    print(f"  Protocol {PROTOCOL_ID}: {PROTOCOL_NAME}")
    print(f"  Model: {args.model}  Device: {args.device}  Prompts: {args.n_prompts}")
    print(f"  Tasks: {', '.join(tasks)}")
    print(f"{'═' * 70}")

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
