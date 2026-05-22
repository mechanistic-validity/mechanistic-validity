"""Protocol MB_MR — Mendelian Randomization
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Lens:         Genetics
Validity Type: Internal
Framework:    Molecular Biology — Instrumental Variable Causal Inference
Family:       Molecular Biology (Mendelian Randomization)
Validity:     External — causal effect estimation via instruments

References:
    Davey Smith & Hemani (2014) "Mendelian randomization: genetic anchors
        for causal inference in epidemiological studies"
    Peters, Buehlmann, Meinshausen (2016) "Causal inference by using
        invariant prediction: identification and confidence intervals"

Question:
    Can we use upstream variables as instrumental variables for causal
    effect estimation? Path patching uses token position as an instrument.
    Cross-task transfer = two-sample MR (estimate in one population,
    validate in another). Cross-model invariance = in vivo generalization
    (does the causal relationship hold in a different organism?).

Metrics:
    path_patching          — Instrument validity: position-specific causal paths
    cross_task_transfer    — Two-sample MR: train on one task, test on another
    cross_model_invariance — In vivo generalization: same circuit across models

Calibrations:
    STRUCTURAL_CALIBRATIONS

Usage:
    uv run python mendelian_randomization.py                       # all tasks, CPU
    uv run python mendelian_randomization.py --device cuda          # GPU
    uv run python mendelian_randomization.py --tasks ioi induction  # specific tasks

    # As a callable module:
    from protocols.molecular_biology.mendelian_randomization import run_protocol
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

PROTOCOL_ID = "MB_MR"
PROTOCOL_NAME = "Mendelian Randomization"
METRICS = ["path_patching", "cross_task_transfer", "cross_model_invariance"]
CALIBRATIONS = STRUCTURAL_CALIBRATIONS
OUTPUT_DIR = Path(__file__).parent / "results" / "mb_mr_mendelian_randomization"

THRESHOLDS = {
    "path_patching": 0.5,
    "cross_task_transfer": 0.4,
    "cross_model_invariance": 0.6,
}


def run_protocol(model, tasks: list[str] | None = None, n_prompts: int = 40,
                 device: str = "cpu", run_cals: bool = True) -> ProtocolResult:
    """Run all MB_MR metrics + calibrations. Returns a ProtocolResult."""
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


def mr_analysis(result: ProtocolResult) -> list[str]:
    """Analyze results through Mendelian Randomization lens.

    Biology analogy: MR uses genetic variants as instruments for causal
    effect estimation. Three validity conditions: (1) the instrument
    (path) affects the exposure (circuit component), (2) the instrument
    is independent of confounders, (3) the instrument affects outcome
    only through the exposure. Two-sample MR validates in a second
    population (task). In vivo checks generalization to another organism
    (model).
    """
    lines = ["\n  Mendelian Randomization Analysis:", "  ---------------------------------"]

    for task in result.tasks:
        pp = _find(result.metrics.get("path_patching", []), task)
        ct = _find(result.metrics.get("cross_task_transfer", []), task)
        cm = _find(result.metrics.get("cross_model_invariance", []), task)

        lines.append(f"\n    {task}:")

        evidence = []

        if pp:
            label = "valid instrument" if pp.value > 0.5 else ("weak instrument" if pp.value > 0.2 else "invalid")
            lines.append(f"      Instrument validity (path_patching):   {pp.value:.4f} — {label}")
            if pp.value > 0.5:
                evidence.append("instrument")

        if ct:
            label = "consistent" if ct.value > 0.4 else ("partial" if ct.value > 0.2 else "inconsistent")
            lines.append(f"      Two-sample MR (cross_task_transfer):   {ct.value:.4f} — {label}")
            if ct.value > 0.4:
                evidence.append("two-sample")

        if cm:
            label = "generalizes" if cm.value > 0.6 else ("partial" if cm.value > 0.3 else "species-specific")
            lines.append(f"      In vivo (cross_model_invariance):      {cm.value:.4f} — {label}")
            if cm.value > 0.6:
                evidence.append("in-vivo")

        n = len(evidence)
        if n == 3:
            verdict = "ROBUST CAUSAL CLAIM — all MR criteria satisfied"
        elif n == 2:
            verdict = f"MODERATE — {', '.join(evidence)} supported"
        elif n == 1:
            verdict = f"WEAK — only {evidence[0]} supported; possible pleiotropy"
        else:
            verdict = "NO SUPPORT — instrument conditions not met"
        lines.append(f"      VERDICT: {verdict}")

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

    lines.extend(mr_analysis(result))

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
