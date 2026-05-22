"""Protocol IT — Information Theory (Full)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Lens:         Information Theory
Validity Type: Internal
Framework:    https://mechanistic-validity.github.io/mechanistic-validity/framework/metrics/cross_discipline/information-theory/
Family:       Cross-discipline (Information Theory)
Validity:     External — full information-theoretic characterization

References:
    Shannon (1948) "A Mathematical Theory of Communication"
    Kolmogorov (1965) "Three approaches to the quantitative definition of information"
    Cover & Thomas (2006) "Elements of Information Theory"

Question:
    This umbrella protocol combines all information-theoretic metrics:
    channel capacity (throughput), rate-distortion (compression efficiency),
    and Kolmogorov complexity (fundamental algorithmic complexity). Together
    they characterize the circuit's information processing from complementary
    angles.

Metrics:
    channel_capacity      — Information-theoretic capacity of circuit channels
    rate_distortion       — Compression-fidelity tradeoff characterization
    kolmogorov_complexity — Estimated algorithmic complexity of circuit computation

Usage:
    uv run python information_theory.py
    uv run python information_theory.py --tasks ioi induction --device cuda

    from protocols.cross_discipline.information_theory import run_protocol
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

PROTOCOL_ID = "IT"
PROTOCOL_NAME = "Information Theory (Full)"
METRICS = ["channel_capacity", "rate_distortion", "kolmogorov_complexity"]
CALIBRATIONS = STRUCTURAL_CALIBRATIONS
OUTPUT_DIR = Path(__file__).parent / "results" / "it_information_theory"

THRESHOLDS = {
    "channel_capacity": 0.5,
    "rate_distortion": 0.5,
    "kolmogorov_complexity": 0.5,
}




def run_protocol(model, tasks: list[str] | None = None, n_prompts: int = 40,
                 run_cals: bool = True) -> ProtocolResult:
    """Run all IT metrics + calibrations. Returns a ProtocolResult."""
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


def information_theory_analysis(result: ProtocolResult) -> list[str]:
    """Protocol-specific interpretive analysis."""
    lines = ["\n  Information Theory Analysis:", "  -----------------------------"]

    for task in result.tasks:
        lines.append(f"\n    {task}:")

        cc = _find(result.metrics.get("channel_capacity", []), task)
        rd = _find(result.metrics.get("rate_distortion", []), task)
        kc = _find(result.metrics.get("kolmogorov_complexity", []), task)

        if cc:
            lines.append(f"      Channel capacity = {cc.value:.4f}")
        if rd:
            lines.append(f"      Rate-distortion = {rd.value:.4f}")
        if kc:
            lines.append(f"      Kolmogorov complexity = {kc.value:.4f}")

        properties = []

        # Capacity: throughput characterization
        if cc and cc.value > THRESHOLDS["channel_capacity"]:
            properties.append("high-throughput")

        # Rate-distortion: compression efficiency
        if rd and rd.value > THRESHOLDS["rate_distortion"]:
            properties.append("efficient-compression")

        # Kolmogorov: algorithmic simplicity
        if kc:
            if kc.value < 0.3:
                properties.append("simple-algorithm")
            elif kc.value > 0.7:
                properties.append("complex-algorithm")

        # Joint interpretation
        if cc and rd and kc:
            if (cc.value > THRESHOLDS["channel_capacity"]
                    and rd.value > THRESHOLDS["rate_distortion"]
                    and kc.value < 0.5):
                lines.append(f"        High capacity + efficient coding + low complexity")
                lines.append(f"        Circuit is an efficient, simple information processor")
            elif (cc.value > THRESHOLDS["channel_capacity"]
                  and kc.value > 0.7):
                lines.append(f"        High capacity but high complexity")
                lines.append(f"        Circuit may be over-parameterized for the task")
            elif (cc.value < THRESHOLDS["channel_capacity"]
                  and rd.value < THRESHOLDS["rate_distortion"]):
                lines.append(f"        Low capacity and lossy compression")
                lines.append(f"        Circuit is information-bottlenecked")

        if properties:
            verdict = f"Information-theoretic profile: {', '.join(properties)}"
        else:
            verdict = "Insufficient data for information-theoretic characterization"
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

    lines.extend(information_theory_analysis(result))

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
