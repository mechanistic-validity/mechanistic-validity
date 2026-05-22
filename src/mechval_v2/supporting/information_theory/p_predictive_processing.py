"""Protocol SM_PP — Predictive Processing
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Lens:         Information Theory
Validity Type: Construct
Framework:    https://mechanistic-validity.github.io/mechanistic-validity/framework/metrics/cognitive_science/predictive-processing/
Family:       Cognitive Science (Predictive Processing)
Validity:     External — free energy minimization and prediction error dynamics

References:
    Friston (2010) "The free-energy principle: a unified brain theory?"
    Clark (2013) "Whatever next? Predictive brains, situated agents, and the future of cognitive science"
    Rao & Ballard (1999) "Predictive coding in the visual cortex"

Question:
    Does the circuit minimize prediction error? Is there a free energy
    gradient across layers? Does entropy cascade through the circuit
    hierarchy? A negative free energy gradient indicates the circuit
    progressively reduces surprise. Low prediction error means the circuit
    generates accurate top-down predictions. Decreasing entropy across
    layers indicates progressive refinement of representations.

Metrics:
    free_energy_gradient — Layer-wise free energy gradient (negative = reducing surprise)
    prediction_error     — Prediction error magnitude between circuit predictions and targets
    entropy_cascade      — Entropy change across circuit hierarchy (decreasing = refinement)

Usage:
    uv run python predictive_processing.py
    uv run python predictive_processing.py --tasks ioi induction --device cuda

    from protocols.cognitive_science.predictive_processing import run_protocol
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

PROTOCOL_ID = "SM_PP"
PROTOCOL_NAME = "Predictive Processing"
METRICS = ["free_energy_gradient", "prediction_error", "entropy_cascade"]
CALIBRATIONS = STRUCTURAL_CALIBRATIONS
OUTPUT_DIR = Path(__file__).parent / "results" / "sm_pp_predictive_processing"

THRESHOLDS = {
    "free_energy_gradient": 0.0,
    "prediction_error": 0.5,
    "entropy_cascade": 0.0,
}




def run_protocol(model, tasks: list[str] | None = None, n_prompts: int = 40,
                 run_cals: bool = True) -> ProtocolResult:
    """Run all SM_PP metrics + calibrations. Returns a ProtocolResult."""
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


def predictive_analysis(result: ProtocolResult) -> list[str]:
    """Analyze results through the predictive processing lens."""
    lines = ["\n  Predictive Processing Analysis:", "  --------------------------------"]

    for task in result.tasks:
        lines.append(f"\n    {task}:")

        fe = _find(result.metrics.get("free_energy_gradient", []), task)
        pe = _find(result.metrics.get("prediction_error", []), task)
        ec = _find(result.metrics.get("entropy_cascade", []), task)

        evidence_for_pp = 0
        total_measures = 0

        if fe:
            total_measures += 1
            g = fe.value
            lines.append(f"      Free energy gradient = {g:+.4f}")
            if g < -0.1:
                lines.append(f"        Strong negative gradient — circuit actively minimizes surprise")
                evidence_for_pp += 1
            elif g < 0.0:
                lines.append(f"        Weak negative gradient — mild surprise reduction")
                evidence_for_pp += 1
            elif g < 0.1:
                lines.append(f"        Near-zero gradient — no clear free energy minimization")
            else:
                lines.append(f"        Positive gradient — circuit INCREASES surprise across layers")

        if pe:
            total_measures += 1
            e = pe.value
            lines.append(f"      Prediction error = {e:.4f}")
            if e < 0.2:
                lines.append(f"        Very low prediction error — accurate top-down predictions")
                evidence_for_pp += 1
            elif e < THRESHOLDS["prediction_error"]:
                lines.append(f"        Moderate prediction error — partial predictive capacity")
                evidence_for_pp += 1
            else:
                lines.append(f"        High prediction error — poor top-down predictions")

        if ec:
            total_measures += 1
            c = ec.value
            lines.append(f"      Entropy cascade = {c:+.4f}")
            if c < -0.1:
                lines.append(f"        Strong entropy decrease — progressive representation refinement")
                evidence_for_pp += 1
            elif c < 0.0:
                lines.append(f"        Mild entropy decrease — some hierarchical refinement")
                evidence_for_pp += 1
            else:
                lines.append(f"        Non-decreasing entropy — no hierarchical refinement")

        if total_measures > 0:
            if evidence_for_pp == total_measures:
                verdict = "Consistent with predictive processing (all criteria met)"
            elif evidence_for_pp > 0:
                verdict = (f"Partial predictive processing ({evidence_for_pp}/{total_measures} "
                           f"criteria) — may reflect feedforward rather than predictive computation")
            else:
                verdict = "No evidence of predictive processing dynamics"
            lines.append(f"      VERDICT: {verdict}")

    return lines


def _find(results: list[EvalResult], task: str) -> EvalResult | None:
    return next((r for r in results if r.metadata.get("task") == task), None)


def summarize(result: ProtocolResult) -> str:
    lines = []
    lines.append(f"\n{'=' * 70}")
    lines.append(f"  PROTOCOL {result.protocol_id}: {result.protocol_name}")
    lines.append(f"{'=' * 70}\n")

    header = f"{'Task':20s}" + "".join(f"  {m:>20s}" for m in METRICS)
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
                row += f"  {v:>16.4f}{tag}"
            else:
                row += f"  {'---':>20s}"
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

    lines.extend(predictive_analysis(result))

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
