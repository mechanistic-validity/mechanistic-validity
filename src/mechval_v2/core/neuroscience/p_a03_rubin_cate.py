"""Protocol A03 — Rubin CATE (Average Treatment Effects)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Lens:         Neuroscience
Validity Type: Internal
Framework:    https://mechanistic-validity.github.io/mechanistic-validity/framework/metrics/causal/a03-rubin-cate/
Family:       Causal (Rung 2 — Intervention)
Validity:     Internal — I1 Necessity, I4 Specificity

References:
    Rubin (1974) "Estimating Causal Effects of Treatments in Randomized
        and Nonrandomized Studies" — potential outcomes framework
    Holland (1986) "Statistics and Causal Inference" — fundamental problem of causal inference
    Imbens & Rubin (2015) "Causal Inference for Statistics, Social, and
        Biomedical Sciences" — modern treatment effects theory

Question:
    What is the average treatment effect (ATE) of ablating the circuit?
    Is the effect specific to the circuit (localized) or does it spill
    over to non-circuit components?

    High ATE with high specificity means the circuit is a precise causal
    unit — removing it has a large, targeted effect. High ATE with low
    specificity means the intervention is too broad; the "treatment"
    affects components outside the circuit hypothesis.

Metrics:
    cate                     — Conditional Average Treatment Effect
    intervention_specificity — Is the effect localized to the circuit?

Calibrations:
    bootstrap           — Are metric values stable across resampled prompts?
    seed_variance       — Are results reproducible across random seeds?
    ablation_invariance — Do results change under different ablation methods?
    method_invariance   — Zero vs mean vs resample ablation agreement
    convergent_validity — Do different causal metrics agree?

Usage:
    uv run python a03_rubin_cate.py                       # all tasks, CPU
    uv run python a03_rubin_cate.py --device cuda          # GPU
    uv run python a03_rubin_cate.py --tasks ioi induction  # specific tasks

    # As a callable module:
    from protocols.causal.a03_rubin_cate import run_protocol
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

PROTOCOL_ID = "A03"
PROTOCOL_NAME = "Rubin CATE (Average Treatment Effects)"
METRICS = ["cate", "intervention_specificity"]
CALIBRATIONS = CAUSAL_CALIBRATIONS
OUTPUT_DIR = Path(__file__).parent / "results" / "a03_rubin_cate"

THRESHOLDS = {
    "cate": 0.5,
    "intervention_specificity": 0.7,
}




def run_protocol(model, tasks: list[str] | None = None, n_prompts: int = 40,
                 run_cals: bool = True) -> ProtocolResult:
    """Run all A03 metrics + calibrations. Returns a ProtocolResult."""
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


def ate_analysis(result: ProtocolResult) -> list[str]:
    """Analyze results through Rubin's potential outcomes lens."""
    lines = ["\n  Treatment Effect Analysis:", "  ──────────────────────────"]

    for task in result.tasks:
        lines.append(f"\n    {task}:")

        cate = _find(result.metrics.get("cate", []), task)
        spec = _find(result.metrics.get("intervention_specificity", []), task)

        if cate:
            ate = cate.value
            lines.append(f"      ATE = {ate:+.4f}")
            if abs(ate) > THRESHOLDS["cate"]:
                lines.append(f"        Large treatment effect — circuit ablation substantially changes output")
            else:
                lines.append(f"        Small treatment effect — circuit may be redundant or non-essential")

        if spec:
            s = spec.value
            lines.append(f"      Specificity = {s:.4f}")
            if s > THRESHOLDS["intervention_specificity"]:
                lines.append(f"        High specificity — effect is localized to the circuit")
            else:
                lines.append(f"        Low specificity — intervention has off-target effects")

        # Combined verdict
        if cate and spec:
            if abs(cate.value) > THRESHOLDS["cate"] and spec.value > THRESHOLDS["intervention_specificity"]:
                verdict = "Strong targeted effect — circuit is a precise causal unit"
            elif abs(cate.value) > THRESHOLDS["cate"] and spec.value <= THRESHOLDS["intervention_specificity"]:
                verdict = "Large but diffuse effect — intervention too broad, refine circuit boundary"
            elif abs(cate.value) <= THRESHOLDS["cate"] and spec.value > THRESHOLDS["intervention_specificity"]:
                verdict = "Small but specific effect — circuit is precise but not essential"
            else:
                verdict = "Small diffuse effect — circuit hypothesis needs revision"
            lines.append(f"      VERDICT: {verdict}")

    return lines


def _find(results: list[EvalResult], task: str) -> EvalResult | None:
    return next((r for r in results if r.metadata.get("task") == task), None)


def _top_heads(r: EvalResult, k: int) -> str:
    ph = r.metadata.get("per_head", {})
    if not ph:
        return ""
    items = sorted(ph.items(),
                   key=lambda kv: abs(kv[1].get("effect", kv[1]) if isinstance(kv[1], dict) else kv[1]),
                   reverse=True)[:k]
    return ", ".join(f"{h}" for h, _ in items)


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

    lines.extend(ate_analysis(result))

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
