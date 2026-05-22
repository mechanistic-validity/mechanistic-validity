"""Protocol B01 — Spectral / SVD Analysis
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Lens:         Measurement Theory
Validity Type: Measurement
Framework:    https://mechanistic-validity.github.io/mechanistic-validity/framework/metrics/structural/b01-svd-spectral/
Family:       Structural (weight-space analysis, no forward pass required)
Validity:     Construct — C1 Convergent, C2 Discriminant

References:
    Elhage et al. (2021) "A Mathematical Framework for Transformer Circuits"
    Millidge & Black (2022) "The Singular Value Decompositions of Transformer
        Weight Matrices are Highly Interpretable"
    Henighan et al. (2023) "Superposition, Memorization, and Double Descent"

Question:
    Does the circuit's weight structure reveal interpretable spectral modes?
    Are the singular values concentrated (low effective rank) or diffuse?
    Do the top singular vectors align with known task-relevant directions?

Metrics:
    spectral_svd          — SVD decomposition and singular value analysis
    weight_extended       — Extended weight-space statistics (norms, spectra)
    capacity_utilization  — Effective rank / capacity usage of weight matrices
    k_alignment           — Alignment of top-k singular vectors with task subspace

Calibrations:
    measurement_invariance — Are results stable across different task prompts?
    convergent_validity    — Do spectral metrics agree with other structural metrics?
    discriminant_validity  — Do spectral metrics distinguish circuits from baselines?

Usage:
    uv run python b01_spectral_svd.py                       # all tasks, CPU
    uv run python b01_spectral_svd.py --device cuda          # GPU
    uv run python b01_spectral_svd.py --tasks ioi induction  # specific tasks

    # As a callable module:
    from protocols.structural.b01_spectral_svd import run_protocol
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

PROTOCOL_ID = "B01"
PROTOCOL_NAME = "Spectral / SVD Analysis"
METRICS = ["spectral_svd", "weight_extended", "capacity_utilization", "k_alignment"]
CALIBRATIONS = STRUCTURAL_CALIBRATIONS
OUTPUT_DIR = Path(__file__).parent / "results" / "b01_spectral_svd"

THRESHOLDS = {
    "capacity_utilization": 0.3,
    "k_alignment": 0.5,
}




def run_protocol(model, tasks: list[str] | None = None, n_prompts: int = 40,
                 run_cals: bool = True) -> ProtocolResult:
    """Run all B01 metrics + calibrations. Returns a ProtocolResult."""
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


def spectral_analysis(result: ProtocolResult) -> list[str]:
    """Analyze results through spectral / SVD lens."""
    lines = ["\n  Spectral Analysis:", "  ──────────────────"]

    for task in result.tasks:
        lines.append(f"\n    {task}:")

        cap = _find(result.metrics.get("capacity_utilization", []), task)
        kalign = _find(result.metrics.get("k_alignment", []), task)
        svd = _find(result.metrics.get("spectral_svd", []), task)
        wext = _find(result.metrics.get("weight_extended", []), task)

        if svd:
            lines.append(f"      SVD analysis:  score = {svd.value:+.4f}")

        if wext:
            lines.append(f"      Weight stats:  score = {wext.value:+.4f}")

        if cap:
            lines.append(f"      Capacity utilization: {cap.value:.4f}")
            if cap.value < THRESHOLDS["capacity_utilization"]:
                lines.append(f"        → Low rank: circuit uses a small subspace")
            else:
                lines.append(f"        → High rank: circuit uses a diffuse subspace")

        if kalign:
            lines.append(f"      K-alignment: {kalign.value:.4f}")
            if kalign.value > THRESHOLDS["k_alignment"]:
                lines.append(f"        → Top singular vectors align with task directions")
            else:
                lines.append(f"        → Weak alignment with task-relevant subspace")

        if cap and kalign:
            if cap.value < THRESHOLDS["capacity_utilization"] and kalign.value > THRESHOLDS["k_alignment"]:
                verdict = "Clean circuit: low-rank with aligned singular vectors"
            elif cap.value < THRESHOLDS["capacity_utilization"]:
                verdict = "Low-rank but poorly aligned — compression without structure"
            elif kalign.value > THRESHOLDS["k_alignment"]:
                verdict = "High-rank but aligned — distributed but task-relevant"
            else:
                verdict = "Diffuse and unaligned — weak spectral signature"
            lines.append(f"      VERDICT: {verdict}")

    return lines


def _find(results: list[EvalResult], task: str) -> EvalResult | None:
    return next((r for r in results if r.metadata.get("task") == task), None)


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

    lines.extend(spectral_analysis(result))

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
