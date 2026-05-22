"""Protocol WC_M1 --- PID Activation Steering (Control Theory)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Lens:         Control Theory
Validity Type: Internal
Framework:    Wildcard --- Control Theory
Family:       Wildcard (PID Steering)
Validity:     External --- activation steering as feedback control

References:
    ICLR 2026 "Activation Steering with a Feedback Controller"
    Ziegler-Nichols tuning method for gain selection

Question:
    Can we model activation steering as a PID controller? Standard
    steering is P-only (constant alpha). PID adds:
      - I (Integral): accumulated error across layers eliminates
        steady-state drift
      - D (Derivative): rate-of-change damping prevents overshoot

    The dose-response curve of a circuit tells us which control regime
    applies: steep responses need D-term damping, shallow responses
    need I-term accumulation, and moderate responses may work with
    P-only steering.

Metrics:
    dose_response      --- Circuit response curve steepness
    sigma_ablation     --- Noise tolerance (integral accumulation budget)
    effect_size        --- Overall control authority

Calibrations:
    STRUCTURAL_CALIBRATIONS

Usage:
    uv run python pid_steering.py                       # all tasks, CPU
    uv run python pid_steering.py --device cuda          # GPU
    uv run python pid_steering.py --tasks ioi induction  # specific tasks

    # As a callable module:
    from protocols.wildcard.pid_steering import run_protocol
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

PROTOCOL_ID = "WC_M1"
PROTOCOL_NAME = "PID Activation Steering (Control Theory)"
METRICS = ["dose_response", "sigma_ablation", "effect_size"]
CALIBRATIONS = STRUCTURAL_CALIBRATIONS
OUTPUT_DIR = Path(__file__).parent / "results" / "wc_m1_pid_steering"

THRESHOLDS = {
    "dose_response": 0.5,
    "sigma_ablation": 0.5,
    "effect_size": 0.8,
}


def run_protocol(model, tasks: list[str] | None = None, n_prompts: int = 40,
                 device: str = "cpu", run_cals: bool = True) -> ProtocolResult:
    """Run all WC_M1 metrics + calibrations. Returns a ProtocolResult."""
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


def pid_steering_analysis(result: ProtocolResult) -> list[str]:
    """Analyze results through PID control theory lens.

    Control theory analogy: standard activation steering applies a
    constant gain (P-only controller). The dose-response curve reveals
    the system's transfer function:
      - Steep (> 0.7): high open-loop gain, prone to overshoot.
        Needs D-term (derivative) damping per Ziegler-Nichols tuning.
      - Shallow (< 0.3): low gain, steady-state error persists.
        Needs I-term (integral) to accumulate correction across layers.
      - Moderate (0.3-0.7): P-only may suffice if effect_size confirms
        adequate control authority.

    sigma_ablation estimates noise tolerance --- how much integral
    accumulation the system can absorb before breaking (analogous to
    integral windup limit). effect_size measures overall control
    authority (loop gain).

    References:
        ICLR 2026 "Activation Steering with a Feedback Controller"
        Ziegler-Nichols tuning method for gain selection
    """
    lines = ["\n  PID Activation Steering Analysis:", "  ---------------------------------"]

    for task in result.tasks:
        dr = _find(result.metrics.get("dose_response", []), task)
        sa = _find(result.metrics.get("sigma_ablation", []), task)
        es = _find(result.metrics.get("effect_size", []), task)

        lines.append(f"\n    {task}:")

        if dr:
            if dr.value > 0.7:
                label = "steep — high gain, overshoot risk"
            elif dr.value < 0.3:
                label = "shallow — steady-state error"
            else:
                label = "moderate — P-only may suffice"
            lines.append(f"      Dose-response (transfer function):     {dr.value:.4f} — {label}")

        if sa:
            if sa.value > 0.5:
                label = "high noise tolerance — large integral budget"
            else:
                label = "low noise tolerance — integral windup risk"
            lines.append(f"      Sigma ablation (windup limit):         {sa.value:.4f} — {label}")

        if es:
            if es.value > 0.8:
                label = "strong control authority"
            elif es.value > 0.5:
                label = "moderate control authority"
            elif es.value > 0.2:
                label = "weak control authority"
            else:
                label = "no control authority"
            lines.append(f"      Effect size (loop gain):               {es.value:.4f} — {label}")

        # Determine verdict
        dr_val = dr.value if dr else None
        es_val = es.value if es else None

        if es_val is not None and es_val < 0.2:
            verdict = "UNCONTROLLABLE"
        elif dr_val is not None and dr_val > 0.7:
            verdict = "RESPONSIVE — NEEDS DAMPING"
        elif dr_val is not None and dr_val < 0.3:
            verdict = "SLUGGISH — NEEDS INTEGRAL"
        elif dr_val is not None and es_val is not None and 0.3 < dr_val < 0.7 and es_val > 0.5:
            verdict = "WELL-CONTROLLED"
        else:
            verdict = "INDETERMINATE — insufficient data"
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

    lines.extend(pid_steering_analysis(result))

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
