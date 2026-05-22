"""Protocol D02 — Generalization & Transfer
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Lens:         Pharmacology
Validity Type: External
Framework:    https://mechanistic-validity.github.io/mechanistic-validity/framework/metrics/structural/d09-generalization-gap/
Family:       Behavioral (generalization and transfer analysis)
Validity:     External — E1 Generalization, E2 Ecological validity

References:
    Geiger et al. (2021) "Causal Abstractions of Neural Networks" — IIA transfer
    Olsson et al. (2022) "In-context Learning and Induction Heads" — cross-task
    Rissanen (1978) "Modeling by Shortest Data Description" — MDL principle
    Nanda et al. (2023) "Progress Measures for Grokking" — generalization gap

Question:
    Does the circuit generalize beyond the specific prompts it was validated on?
    Does the explanation transfer to related tasks? Is the circuit hypothesis
    compressible (low MDL), or does it overfit to the training distribution?

Metrics:
    ce_delta              — Cross-entropy change under circuit ablation
    per_token_nll         — Per-token negative log-likelihood impact
    calibration           — Calibration of circuit-ablated model predictions
    generalization_gap    — Gap between in-distribution and OOD performance
    mdl_compression       — Minimum description length of the circuit hypothesis
    normative_account     — Does the circuit match a normative/algorithmic account?
    error_boundary        — Where does the circuit explanation break down?

Calibrations:
    bootstrap                — Are metric values stable across resampled prompts?
    distributional_stability — Are distributions stable across samples?
    internal_consistency     — Do generalization sub-metrics agree?
    convergent_validity      — Do generalization metrics agree with causal metrics?

Usage:
    uv run python d02_generalization.py                       # all tasks, CPU
    uv run python d02_generalization.py --device cuda          # GPU
    uv run python d02_generalization.py --tasks ioi induction  # specific tasks

    # As a callable module:
    from protocols.behavioral.d02_generalization import run_protocol
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
from protocols.calibration_runner import BEHAVIORAL_CALIBRATIONS, run_calibrations, summarize_calibrations

PROTOCOL_ID = "D02"
PROTOCOL_NAME = "Generalization & Transfer"
METRICS = [
    "ce_delta", "per_token_nll", "calibration", "generalization_gap",
    "mdl_compression", "normative_account", "error_boundary",
]
CALIBRATIONS = BEHAVIORAL_CALIBRATIONS
OUTPUT_DIR = Path(__file__).parent / "results" / "d02_generalization"

THRESHOLDS = {
    "ce_delta": 0.5,
    "generalization_gap": 0.2,
    "mdl_compression": 0.5,
    "calibration": 0.1,
}




def run_protocol(model, tasks: list[str] | None = None, n_prompts: int = 40,
                 run_cals: bool = True) -> ProtocolResult:
    """Run all D02 metrics + calibrations. Returns a ProtocolResult."""
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


def generalization_analysis(result: ProtocolResult) -> list[str]:
    """Analyze results through generalization lens."""
    lines = ["\n  Generalization Analysis:", "  ────────────────────────"]

    for task in result.tasks:
        lines.append(f"\n    {task}:")

        ce = _find(result.metrics.get("ce_delta", []), task)
        nll = _find(result.metrics.get("per_token_nll", []), task)
        cal = _find(result.metrics.get("calibration", []), task)
        gap = _find(result.metrics.get("generalization_gap", []), task)
        mdl = _find(result.metrics.get("mdl_compression", []), task)
        norm = _find(result.metrics.get("normative_account", []), task)
        err = _find(result.metrics.get("error_boundary", []), task)

        properties = []

        if ce:
            lines.append(f"      CE delta: {ce.value:.4f}")
            if ce.value > THRESHOLDS["ce_delta"]:
                properties.append("large CE impact")

        if nll:
            lines.append(f"      Per-token NLL: {nll.value:.4f}")

        if cal:
            lines.append(f"      Calibration error: {cal.value:.4f}")
            if cal.value < THRESHOLDS["calibration"]:
                properties.append("well-calibrated")

        if gap:
            lines.append(f"      Generalization gap: {gap.value:.4f}")
            if gap.value < THRESHOLDS["generalization_gap"]:
                properties.append("generalizes")
                lines.append(f"        → Small gap: circuit generalizes to OOD inputs")
            else:
                lines.append(f"        → Large gap: circuit may overfit to prompt distribution")

        if mdl:
            lines.append(f"      MDL compression: {mdl.value:.4f}")
            if mdl.value > THRESHOLDS["mdl_compression"]:
                properties.append("compressible")
                lines.append(f"        → High compression: parsimonious explanation")

        if norm:
            lines.append(f"      Normative account: {norm.value:.4f}")
            if norm.value > 0.5:
                properties.append("normatively grounded")

        if err:
            lines.append(f"      Error boundary: {err.value:.4f}")

        if properties:
            verdict = "Generalizable circuit: " + ", ".join(properties)
        else:
            verdict = "Limited generalization — interpret with caution"
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

    lines.extend(generalization_analysis(result))

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
