"""Protocol WC_M2 — Angular Steering (Rotation Geometry)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Lens:         Geometry
Validity Type: Construct
Framework:    Wildcard — Rotation Geometry
Family:       Wildcard (Angular Steering)
Validity:     External — rotational causal intervention

References:
    NeurIPS 2025 Spotlight "Angular Steering"
    Rodrigues rotation formula

Question:
    Does angular steering (rotation in the h-d plane preserving norm)
    reveal qualitatively different circuit properties than additive
    steering (h' = h + alpha*d, which changes norm)?  The parameter
    space is bounded [-90, +90] degrees instead of unbounded alpha,
    so angular sensitivity characterizes the circuit's directional
    threshold.

Metrics:
    dose_response  — Angular sensitivity: slope of behavioral change
                     vs rotation angle (steep = toggle, shallow = gradual)
    effect_size    — Maximum behavioral range achievable through rotation
    das_iia        — Alignment of circuit causal structure with rotational
                     intervention (DAS interchange intervention accuracy)

Calibrations:
    STRUCTURAL_CALIBRATIONS

Usage:
    uv run python angular_steering.py                       # all tasks, CPU
    uv run python angular_steering.py --device cuda          # GPU
    uv run python angular_steering.py --tasks ioi induction  # specific tasks

    # As a callable module:
    from protocols.wildcard.angular_steering import run_protocol
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

PROTOCOL_ID = "WC_M2"
PROTOCOL_NAME = "Angular Steering (Rotation Geometry)"
METRICS = ["dose_response", "effect_size", "das_iia"]
CALIBRATIONS = STRUCTURAL_CALIBRATIONS
OUTPUT_DIR = Path(__file__).parent / "results" / "wc_m2_angular_steering"

THRESHOLDS = {
    "dose_response": 0.5,
    "effect_size": 0.8,
    "das_iia": 0.6,
}


def run_protocol(model, tasks: list[str] | None = None, n_prompts: int = 40,
                 device: str = "cpu", run_cals: bool = True) -> ProtocolResult:
    """Run all WC_M2 metrics + calibrations. Returns a ProtocolResult."""
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


def angular_steering_analysis(result: ProtocolResult) -> list[str]:
    """Analyze results through Angular Steering (Rotation Geometry) lens.

    Instead of additive steering h' = h + alpha*d (which changes norm),
    angular steering rotates h toward d in the h-d plane, preserving norm
    via the Rodrigues rotation formula.  The parameter space is bounded
    [-90, +90] degrees instead of unbounded alpha.

    - dose_response characterizes the circuit's angular sensitivity:
      steep -> narrow angular threshold (small rotation triggers large
      behavioral change, like a toggle); shallow -> requires large angular
      displacement (gradual behavior).
    - effect_size assesses the maximum behavioral range achievable through
      rotation alone (norm-preserving).
    - das_iia assesses whether the circuit's causal structure aligns with
      rotational intervention (interchange intervention accuracy).
    """
    lines = ["\n  Angular Steering Analysis:", "  --------------------------"]

    for task in result.tasks:
        dr = _find(result.metrics.get("dose_response", []), task)
        es = _find(result.metrics.get("effect_size", []), task)
        di = _find(result.metrics.get("das_iia", []), task)

        lines.append(f"\n    {task}:")

        if dr:
            if dr.value > 0.7:
                label = "steep sigmoid (angular toggle)"
            elif dr.value > 0.3:
                label = "smooth angular response"
            else:
                label = "rotationally insensitive"
            lines.append(f"      Angular sensitivity (dose_response): {dr.value:.4f} — {label}")

        if es:
            label = "large range" if es.value > 0.8 else ("moderate range" if es.value > 0.4 else "small range")
            lines.append(f"      Rotational range (effect_size):      {es.value:.4f} — {label}")

        if di:
            label = "aligned" if di.value > 0.6 else ("partial" if di.value > 0.3 else "misaligned")
            lines.append(f"      Causal alignment (das_iia):          {di.value:.4f} — {label}")

        # Determine verdict
        dr_val = dr.value if dr else 0.0
        es_val = es.value if es else 0.0
        di_val = di.value if di else 0.0

        if es_val > 0.8 and di_val < 0.3:
            verdict = "NORM-DOMINATED — additive better than rotational"
        elif dr_val > 0.7:
            verdict = "ANGULAR TOGGLE — small rotation triggers transition"
        elif dr_val > 0.3:
            verdict = "GRADUAL ROTATOR — smooth angular response"
        else:
            verdict = "ROTATIONALLY INSENSITIVE — direction doesn't matter, magnitude does"

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

    lines.extend(angular_steering_analysis(result))

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
