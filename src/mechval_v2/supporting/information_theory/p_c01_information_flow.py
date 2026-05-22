"""Protocol C01 — Information Flow
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Lens:         Information Theory
Validity Type: Construct
Framework:    https://mechanistic-validity.github.io/mechanistic-validity/framework/metrics/structural/c01-mutual-information/
Family:       Information-theoretic (information flow and directed dependencies)
Validity:     Internal — I1 Necessity; Construct — C1 Convergent

References:
    Cover & Thomas (2006) "Elements of Information Theory"
    Schreiber (2000) "Measuring Information Transfer" — transfer entropy
    Granger (1969) "Investigating Causal Relations by Econometric Models"
    Geiger et al. (2021) "Causal Abstractions of Neural Networks"

Question:
    How does information flow through the circuit? Do circuit components
    share mutual information? Is there directed information transfer (transfer
    entropy) between layers? Does Granger causality confirm the hypothesized
    information flow direction?

Metrics:
    mutual_information  — MI between circuit component activations
    conditional_mi      — Conditional MI controlling for confounders
    transfer_entropy    — Directed information transfer between components
    granger_causality   — Granger-causal relationships between components

Calibrations:
    measurement_invariance — Are information metrics stable across prompts?
    convergent_validity    — Do information metrics agree with causal metrics?
    discriminant_validity  — Do information metrics distinguish circuits?

Usage:
    uv run python c01_information_flow.py                       # all tasks, CPU
    uv run python c01_information_flow.py --device cuda          # GPU
    uv run python c01_information_flow.py --tasks ioi induction  # specific tasks

    # As a callable module:
    from protocols.information.c01_information_flow import run_protocol
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
from protocols.calibration_runner import STRUCTURAL_CALIBRATIONS, run_calibrations, summarize_calibrations

PROTOCOL_ID = "C01"
PROTOCOL_NAME = "Information Flow"
METRICS = ["mutual_information", "conditional_mi", "transfer_entropy", "granger_causality"]
CALIBRATIONS = STRUCTURAL_CALIBRATIONS
OUTPUT_DIR = Path(__file__).parent / "results" / "c01_information_flow"

THRESHOLDS = {
    "mutual_information": 0.1,
    "conditional_mi": 0.05,
    "transfer_entropy": 0.05,
    "granger_causality": 0.05,
}




def run_protocol(model, tasks: list[str] | None = None, n_prompts: int = 40,
                 run_cals: bool = True) -> ProtocolResult:
    """Run all C01 metrics + calibrations. Returns a ProtocolResult."""
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


def information_flow_analysis(result: ProtocolResult) -> list[str]:
    """Analyze results through information flow lens."""
    lines = ["\n  Information Flow Analysis:", "  ──────────────────────────"]

    for task in result.tasks:
        lines.append(f"\n    {task}:")

        mi = _find(result.metrics.get("mutual_information", []), task)
        cmi = _find(result.metrics.get("conditional_mi", []), task)
        te = _find(result.metrics.get("transfer_entropy", []), task)
        gc = _find(result.metrics.get("granger_causality", []), task)

        channels = []

        if mi:
            lines.append(f"      Mutual information: {mi.value:.4f} bits")
            if mi.value > THRESHOLDS["mutual_information"]:
                channels.append("shared information")
                lines.append(f"        → Components share significant mutual information")

        if cmi:
            lines.append(f"      Conditional MI: {cmi.value:.4f} bits")
            if cmi.value > THRESHOLDS["conditional_mi"]:
                channels.append("direct channel")
                lines.append(f"        → Direct information channel (controlling confounders)")
            else:
                lines.append(f"        → Shared info may be mediated by confounders")

        if te:
            lines.append(f"      Transfer entropy: {te.value:.4f} bits")
            if te.value > THRESHOLDS["transfer_entropy"]:
                channels.append("directed flow")
                lines.append(f"        → Directed information transfer confirmed")
            direction = te.metadata.get("direction", None)
            if direction:
                lines.append(f"        Direction: {direction}")

        if gc:
            lines.append(f"      Granger causality: {gc.value:.4f}")
            if gc.value > THRESHOLDS["granger_causality"]:
                channels.append("Granger-causal")
                lines.append(f"        → Granger-causal relationship between components")

        if channels:
            verdict = "Active information flow: " + ", ".join(channels)
        else:
            verdict = "Weak information flow — components may be independent"
        lines.append(f"      VERDICT: {verdict}")

    return lines


def _find(results: list[EvalResult], task: str) -> EvalResult | None:
    return next((r for r in results if r.metadata.get("task") == task), None)


def summarize(result: ProtocolResult) -> str:
    lines = []
    lines.append(f"\n{'═' * 70}")
    lines.append(f"  PROTOCOL {result.protocol_id}: {result.protocol_name}")
    lines.append(f"{'═' * 70}\n")

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

    lines.extend(information_flow_analysis(result))

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
