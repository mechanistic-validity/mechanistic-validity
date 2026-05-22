"""Protocol A02 — Counterfactual DAS (Causal Abstraction)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Lens:         Neuroscience
Validity Type: Internal
Framework:    https://mechanistic-validity.github.io/mechanistic-validity/framework/metrics/causal/a02-counterfactual-das/
Family:       Causal (Rung 3 — Counterfactual)
Validity:     Internal — I2 Sufficiency, I3 Causal Abstraction

References:
    Geiger et al. (2021) "Causal Abstractions of Neural Networks" — causal abstraction framework
    Geiger et al. (2023) "Finding Alignments Between Interpretable Causal
        Variables and Distributed Neural Representations" — DAS
    Wu et al. (2023) "Interpretability at Scale" — interchange intervention training

Question:
    Does a high-level causal model (the circuit hypothesis) form a valid
    causal abstraction of the neural network? Can we align interpretable
    causal variables to distributed representations via DAS and achieve
    high interchange intervention accuracy (IIA)?

    High IIA means the circuit's causal structure faithfully mirrors the
    model's internal computation. Low IIA means the proposed abstraction
    is a poor match — the model's causal structure differs from the hypothesis.

Metrics:
    das_iia                     — DAS interchange intervention accuracy
    iia_variants                — IIA under alternative intervention strategies
    corrupt_restore             — Corrupt-and-restore patching
    multi_axis_iia              — Multi-variable simultaneous IIA
    counterfactual_consistency  — Do counterfactual predictions match?
    path_patching               — Edge-level path patching
    intermediate_state_prediction — Predict intermediate representations

Calibrations:
    bootstrap           — Are metric values stable across resampled prompts?
    seed_variance       — Are results reproducible across random seeds?
    ablation_invariance — Do results change under different ablation methods?
    method_invariance   — Zero vs mean vs resample ablation agreement
    convergent_validity — Do different causal metrics agree?

Usage:
    uv run python a02_counterfactual_das.py                       # all tasks, CPU
    uv run python a02_counterfactual_das.py --device cuda          # GPU
    uv run python a02_counterfactual_das.py --tasks ioi induction  # specific tasks

    # As a callable module:
    from protocols.causal.a02_counterfactual_das import run_protocol
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

PROTOCOL_ID = "A02"
PROTOCOL_NAME = "Counterfactual DAS (Causal Abstraction)"
METRICS = ["das_iia", "iia_variants", "corrupt_restore", "multi_axis_iia",
           "counterfactual_consistency", "path_patching", "intermediate_state_prediction"]
CALIBRATIONS = CAUSAL_CALIBRATIONS
OUTPUT_DIR = Path(__file__).parent / "results" / "a02_counterfactual_das"

THRESHOLDS = {
    "das_iia": 0.8,
    "iia_variants": 0.5,
    "corrupt_restore": 0.7,
    "multi_axis_iia": 0.5,
    "counterfactual_consistency": 0.7,
    "path_patching": 0.5,
    "intermediate_state_prediction": 0.5,
}




def run_protocol(model, tasks: list[str] | None = None, n_prompts: int = 40,
                 run_cals: bool = True) -> ProtocolResult:
    """Run all A02 metrics + calibrations. Returns a ProtocolResult."""
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


def iia_analysis(result: ProtocolResult) -> list[str]:
    """Analyze results through IIA abstraction quality lens."""
    lines = ["\n  Causal Abstraction Analysis:", "  ────────────────────────────"]

    for task in result.tasks:
        lines.append(f"\n    {task}:")

        iia = _find(result.metrics.get("das_iia", []), task)
        iv = _find(result.metrics.get("iia_variants", []), task)
        cr = _find(result.metrics.get("corrupt_restore", []), task)
        ma = _find(result.metrics.get("multi_axis_iia", []), task)
        cc = _find(result.metrics.get("counterfactual_consistency", []), task)
        pp = _find(result.metrics.get("path_patching", []), task)
        isp = _find(result.metrics.get("intermediate_state_prediction", []), task)

        if iia:
            score = iia.value
            if score > 0.8:
                strength = "STRONG abstraction"
            elif score > 0.5:
                strength = "MODERATE abstraction"
            else:
                strength = "WEAK abstraction"
            lines.append(f"      DAS IIA = {score:.4f} — {strength}")

        if iv:
            lines.append(f"      IIA variants = {iv.value:.4f}")

        if cr:
            lines.append(f"      Corrupt-restore = {cr.value:.4f}")

        if ma:
            lines.append(f"      Multi-axis IIA = {ma.value:.4f}")
            if iia and ma.value < iia.value - 0.2:
                lines.append(f"        Warning: multi-axis IIA much lower than single-axis")
                lines.append(f"        → variables may not be independently manipulable")

        if cc:
            lines.append(f"      Counterfactual consistency = {cc.value:.4f}")

        if pp:
            lines.append(f"      Path patching = {pp.value:.4f}")
            top = _top_heads(pp, 5)
            if top:
                lines.append(f"        Top causal edges: {top}")

        if isp:
            lines.append(f"      Intermediate state prediction = {isp.value:.4f}")

        # Overall verdict
        scores = [r.value for r in [iia, iv, cr, ma, cc, pp, isp] if r is not None]
        if scores:
            mean_score = np.mean(scores)
            if mean_score > 0.8:
                verdict = "Strong causal abstraction — circuit faithfully mirrors model computation"
            elif mean_score > 0.5:
                verdict = "Moderate abstraction — circuit captures main causal pathways"
            else:
                verdict = "Weak abstraction — circuit hypothesis is a poor causal model"
            lines.append(f"      VERDICT: {verdict} (mean={mean_score:.4f})")

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

    lines.extend(iia_analysis(result))

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
