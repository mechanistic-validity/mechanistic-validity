"""Protocol D01 — Faithfulness & Recovery
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Lens:         Pharmacology
Validity Type: Construct
Framework:    https://mechanistic-validity.github.io/mechanistic-validity/framework/metrics/structural/d01-faithfulness/
Family:       Behavioral (activation-based, requires forward pass)
Validity:     Internal — I1 Necessity, I2 Sufficiency; External — E1 Generalization

References:
    Wang et al. (2022) "Interpretability in the Wild" — logit diff recovery
    Conmy et al. (2023) "Towards Automated Circuit Discovery" — faithfulness
    Chan et al. (2022) "Causal Scrubbing" — corrupt-and-restore paradigm
    Cohen & Saphra (2024) "Evaluating the Faithfulness of Circuit Hypotheses"

Question:
    Is the circuit a faithful explanation of the model's behavior? When the
    circuit is ablated, does the model fail? When only the circuit is kept,
    does the model succeed? How large is the effect, and does it follow a
    dose-response curve?

Metrics:
    effect_size                — Cohen's d for circuit ablation effect
    dose_response              — Does partial ablation produce graded effects?
    corrupt_restore_behavioral — Corrupt model, restore circuit: recovery rate
    output_variants            — Logit diff recovery under circuit ablation
    output_variants_kl         — KL divergence between full and ablated model
    output_variants_topk       — Top-k accuracy preservation under ablation
    mean_centered_logit        — Mean-centered logit diff (bias-corrected)

Calibrations:
    bootstrap                  — Are metric values stable across resampled prompts?
    distributional_stability   — Are distributions stable across samples?
    internal_consistency       — Do sub-metrics agree with each other?
    convergent_validity        — Do faithfulness metrics agree with causal metrics?

Usage:
    uv run python d01_faithfulness.py                       # all tasks, CPU
    uv run python d01_faithfulness.py --device cuda          # GPU
    uv run python d01_faithfulness.py --tasks ioi induction  # specific tasks

    # As a callable module:
    from protocols.behavioral.d01_faithfulness import run_protocol
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

PROTOCOL_ID = "D01"
PROTOCOL_NAME = "Faithfulness & Recovery"
METRICS = [
    "effect_size", "dose_response", "corrupt_restore_behavioral",
    "output_variants", "output_variants_kl", "output_variants_topk",
    "mean_centered_logit",
]
CALIBRATIONS = BEHAVIORAL_CALIBRATIONS
OUTPUT_DIR = Path(__file__).parent / "results" / "d01_faithfulness"

THRESHOLDS = {
    "effect_size": 0.8,
    "corrupt_restore_behavioral": 0.5,
    "output_variants": 0.7,
    "output_variants_kl": 1.0,
    "output_variants_topk": 0.7,
}




def run_protocol(model, tasks: list[str] | None = None, n_prompts: int = 40,
                 run_cals: bool = True) -> ProtocolResult:
    """Run all D01 metrics + calibrations. Returns a ProtocolResult."""
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


def faithfulness_analysis(result: ProtocolResult) -> list[str]:
    """Analyze results through faithfulness lens."""
    lines = ["\n  Faithfulness Analysis:", "  ──────────────────────"]

    for task in result.tasks:
        lines.append(f"\n    {task}:")

        es = _find(result.metrics.get("effect_size", []), task)
        dr = _find(result.metrics.get("dose_response", []), task)
        cr = _find(result.metrics.get("corrupt_restore_behavioral", []), task)
        ov = _find(result.metrics.get("output_variants", []), task)
        kl = _find(result.metrics.get("output_variants_kl", []), task)
        topk = _find(result.metrics.get("output_variants_topk", []), task)
        mcl = _find(result.metrics.get("mean_centered_logit", []), task)

        evidence = []

        if es:
            lines.append(f"      Effect size (Cohen's d): {es.value:.4f}")
            if es.value > THRESHOLDS["effect_size"]:
                evidence.append("large effect")
                lines.append(f"        → Large effect: ablation strongly disrupts output")

        if dr:
            lines.append(f"      Dose-response: {dr.value:.4f}")
            if dr.value > 0.5:
                evidence.append("dose-response")
                lines.append(f"        → Graded response to partial ablation")

        if cr:
            lines.append(f"      Corrupt-restore recovery: {cr.value:.4f}")
            if cr.value > THRESHOLDS["corrupt_restore_behavioral"]:
                evidence.append("recoverable")
                lines.append(f"        → Circuit alone restores behavior in corrupted model")

        if ov:
            lines.append(f"      Logit diff recovery: {ov.value:.4f}")
        if kl:
            lines.append(f"      KL divergence: {kl.value:.4f}")
        if topk:
            lines.append(f"      Top-k preservation: {topk.value:.4f}")
        if mcl:
            lines.append(f"      Mean-centered logit: {mcl.value:+.4f}")

        n = len(evidence)
        if n >= 3:
            verdict = "Highly faithful: strong effect, dose-response, and recovery"
        elif n >= 1:
            verdict = f"Partially faithful ({', '.join(evidence)})"
        else:
            verdict = "Low faithfulness: circuit may not explain model behavior"
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

    lines.extend(faithfulness_analysis(result))

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
