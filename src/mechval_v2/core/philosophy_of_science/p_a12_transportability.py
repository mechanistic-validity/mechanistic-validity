"""Protocol A12 — Causal Transportability
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Lens:         Philosophy of Science
Validity Type: External
Framework:    https://mechanistic-validity.github.io/mechanistic-validity/framework/metrics/causal/a12-transportability/
Family:       Causal (Cross-Model Generalization)
Validity:     External — E1 Generalization, E2 Cross-Model Invariance

References:
    Pearl & Bareinboim (2011) "Transportability of Causal and Statistical
        Relations: A Formal Approach", AAAI

Question:
    Does the circuit structure transport across models? If the same
    functional circuit (e.g., induction heads at similar relative
    positions) appears in independently trained models, that is strong
    evidence the circuit reflects a genuine computational structure
    rather than an artifact of a particular training run.

    High cross-model invariance means the circuit is a robust feature
    of the architecture + task, not an idiosyncrasy of one set of
    learned weights.

Metrics:
    cross_model_invariance — Correlation of circuit-level metrics (ablation
                             effects, attention patterns) across independently
                             trained model instances

Calibrations:
    bootstrap           — Are metric values stable across resampled prompts?
    seed_variance       — Are results reproducible across random seeds?
    ablation_invariance — Do results change under different ablation methods?
    method_invariance   — Zero vs mean vs resample ablation agreement
    convergent_validity — Do different causal metrics agree?

Usage:
    uv run python a12_transportability.py                       # all tasks, CPU
    uv run python a12_transportability.py --device cuda          # GPU
    uv run python a12_transportability.py --tasks ioi induction  # specific tasks

    # As a callable module:
    from protocols.causal.a12_transportability import run_protocol
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

PROTOCOL_ID = "A12"
PROTOCOL_NAME = "Causal Transportability"
METRICS = ["cross_model_invariance"]
CALIBRATIONS = CAUSAL_CALIBRATIONS
OUTPUT_DIR = Path(__file__).parent / "results" / "a12_transportability"

THRESHOLDS = {
    "cross_model_invariance": 0.5,
}




def run_protocol(model, tasks: list[str] | None = None, n_prompts: int = 40,
                 run_cals: bool = True) -> ProtocolResult:
    """Run all A12 metrics + calibrations. Returns a ProtocolResult."""
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


def transportability_analysis(result: ProtocolResult) -> list[str]:
    """Analyze cross-model transportability of circuit structure."""
    lines = ["\n  Transportability Analysis:", "  ──────────────────────────"]

    cmi_results = result.metrics.get("cross_model_invariance", [])

    for task in result.tasks:
        lines.append(f"\n    {task}:")

        match = _find(cmi_results, task)
        if not match:
            lines.append(f"      No cross-model invariance data available")
            continue

        lines.append(f"      Cross-model invariance: {match.value:+.4f}")

        model_pairs = match.metadata.get("model_pairs", {})
        if model_pairs:
            for pair, corr in model_pairs.items():
                val = corr if isinstance(corr, (int, float)) else corr.get("correlation", corr)
                lines.append(f"        {pair}: {val:.4f}")

        if match.value >= 0.7:
            lines.append(f"      -> Strong transportability: circuit is architecture-intrinsic")
            verdict = "Circuit transports across models"
        elif match.value >= THRESHOLDS["cross_model_invariance"]:
            lines.append(f"      -> Moderate transportability: core structure preserved")
            verdict = "Partial transportability"
        else:
            lines.append(f"      -> Low transportability: circuit may be training-run specific")
            verdict = "Circuit does not transport"
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

    lines.extend(transportability_analysis(result))

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
