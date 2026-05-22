"""Protocol A06 — Mediation Analysis
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Lens:         Neuroscience
Validity Type: Internal
Framework:    https://mechanistic-validity.github.io/mechanistic-validity/framework/metrics/causal/a06-mediation/
Family:       Causal (Rung 2 — Intervention)
Validity:     Internal — I1 Necessity, I3 Mediation

References:
    Pearl (2001) "Direct and Indirect Effects", UAI — formalized NDE/NIE
    Baron & Kenny (1986) "The Moderator-Mediator Variable Distinction
        in Social Psychological Research" — original mediation framework
    Vig et al. (2020) "Causal Mediation Analysis for Interpreting
        Neural NLP" — applied mediation analysis to transformers

Question:
    Does the circuit MEDIATE the causal effect of input on output? Or does
    the effect flow through alternative pathways that bypass the circuit?

    The Natural Direct Effect (NDE) captures how much effect flows AROUND
    the circuit (bypass pathways). The Natural Indirect Effect (NIE)
    captures how much flows THROUGH the circuit. The mediation ratio
    NIE / (NDE + NIE) tells us what fraction of the total effect the
    circuit is responsible for.

    High mediation ratio means the circuit is a genuine bottleneck. Low
    mediation means important alternative pathways exist that the circuit
    hypothesis does not capture.

Metrics:
    mediation    — NDE/NIE decomposition via resample mediation
    mediation_v2 — Optimized variant with ablation-based decomposition
    pse          — Proportion of Sufficient Explanation

Calibrations:
    bootstrap           — Are metric values stable across resampled prompts?
    seed_variance       — Are results reproducible across random seeds?
    ablation_invariance — Do results change under different ablation methods?
    method_invariance   — Zero vs mean vs resample ablation agreement
    convergent_validity — Do different causal metrics agree?

Usage:
    uv run python a06_mediation.py                       # all tasks, CPU
    uv run python a06_mediation.py --device cuda          # GPU
    uv run python a06_mediation.py --tasks ioi induction  # specific tasks

    # As a callable module:
    from protocols.causal.a06_mediation import run_protocol
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

PROTOCOL_ID = "A06"
PROTOCOL_NAME = "Mediation Analysis"
METRICS = ["mediation", "mediation_v2", "pse"]
CALIBRATIONS = CAUSAL_CALIBRATIONS
OUTPUT_DIR = Path(__file__).parent / "results" / "a06_mediation"

THRESHOLDS = {
    "mediation": 0.5,
    "mediation_v2": 0.5,
    "pse": 0.7,
}




def run_protocol(model, tasks: list[str] | None = None, n_prompts: int = 40,
                 run_cals: bool = True) -> ProtocolResult:
    """Run all A06 metrics + calibrations. Returns a ProtocolResult."""
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


def mediation_analysis(result: ProtocolResult) -> list[str]:
    """Analyze results through NDE/NIE mediation decomposition."""
    lines = ["\n  Mediation Decomposition:", "  ────────────────────────"]

    for task in result.tasks:
        lines.append(f"\n    {task}:")

        med = _find(result.metrics.get("mediation", []), task)
        med2 = _find(result.metrics.get("mediation_v2", []), task)
        pse = _find(result.metrics.get("pse", []), task)

        if med:
            ratio = med.value
            nde = med.metadata.get("nde", None)
            nie = med.metadata.get("nie", None)
            lines.append(f"      Mediation ratio = {ratio:.4f}")
            if nde is not None and nie is not None:
                lines.append(f"        NDE (direct/bypass) = {nde:.4f}")
                lines.append(f"        NIE (indirect/through circuit) = {nie:.4f}")
            if ratio > 0.8:
                lines.append(f"        Strong mediation — circuit is the dominant causal pathway")
            elif ratio > 0.5:
                lines.append(f"        Moderate mediation — circuit mediates majority of effect")
            else:
                lines.append(f"        Weak mediation — significant bypass pathways exist")

        if med2:
            lines.append(f"      Mediation v2 = {med2.value:.4f}")
            if med and abs(med2.value - med.value) > 0.15:
                lines.append(f"        Note: v1 and v2 disagree by {abs(med2.value - med.value):.4f}")
                lines.append(f"        → check ablation method sensitivity")

        if pse:
            lines.append(f"      PSE = {pse.value:.4f}")
            if pse.value > 0.7:
                lines.append(f"        Circuit sufficiently explains model behavior")
            else:
                lines.append(f"        Circuit is necessary but not sufficient — missing components")

        # Combined verdict
        scores = [r.value for r in [med, med2, pse] if r is not None]
        if scores:
            mean_score = np.mean(scores)
            if mean_score > 0.7:
                verdict = "Strong mediation — circuit is a genuine causal bottleneck"
            elif mean_score > 0.5:
                verdict = "Moderate mediation — circuit captures primary pathway but bypass routes exist"
            else:
                verdict = "Weak mediation — circuit hypothesis misses important causal pathways"
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

    lines.extend(mediation_analysis(result))

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
