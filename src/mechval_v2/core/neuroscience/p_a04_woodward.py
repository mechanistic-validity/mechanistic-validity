"""Protocol A04 — Woodward Interventionism
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Lens:         Neuroscience
Validity Type: Internal
Framework:    https://mechanistic-validity.github.io/mechanistic-validity/framework/metrics/causal/a04-woodward/
Family:       Causal (Rung 2 — Intervention)
Validity:     Internal — I1 Necessity, I2 Sufficiency

References:
    Woodward (2003) "Making Things Happen: A Theory of Causal Explanation"
        — interventionist account of causation
    Craver & Bechtel (2007) "Top-down Causation Without Top-down Causes"
        — multi-level mechanistic explanation

Question:
    Does the circuit satisfy Woodward's interventionist criteria for genuine
    causal explanation? Specifically:

    (1) STABILITY: Does the causal relationship hold under a range of
        background conditions (not just cherry-picked prompts)?
    (2) PROPORTIONALITY: Is the intervention at the right level of
        abstraction (not too coarse, not too fine)?
    (3) INVARIANCE: Does the cause-effect relationship hold under
        interventions on other variables?

    Sigma ablation tests graded intervention strength. Resample complement
    tests whether the effect is specific to the circuit (complement should
    NOT recover performance). Misalignment measures how much the circuit's
    causal contribution deviates from the hypothesis.

Metrics:
    sigma_ablation      — Graded ablation (noise injection at varying sigma)
    resample_complement — Resample everything EXCEPT the circuit
    misalignment        — Deviation between predicted and observed causal effects

Calibrations:
    bootstrap           — Are metric values stable across resampled prompts?
    seed_variance       — Are results reproducible across random seeds?
    ablation_invariance — Do results change under different ablation methods?
    method_invariance   — Zero vs mean vs resample ablation agreement
    convergent_validity — Do different causal metrics agree?

Usage:
    uv run python a04_woodward.py                       # all tasks, CPU
    uv run python a04_woodward.py --device cuda          # GPU
    uv run python a04_woodward.py --tasks ioi induction  # specific tasks

    # As a callable module:
    from protocols.causal.a04_woodward import run_protocol
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

PROTOCOL_ID = "A04"
PROTOCOL_NAME = "Woodward Interventionism"
METRICS = ["sigma_ablation", "resample_complement", "misalignment"]
CALIBRATIONS = CAUSAL_CALIBRATIONS
OUTPUT_DIR = Path(__file__).parent / "results" / "a04_woodward"

THRESHOLDS = {
    "sigma_ablation": 0.5,
    "resample_complement": 0.7,
    "misalignment": 0.3,
}




def run_protocol(model, tasks: list[str] | None = None, n_prompts: int = 40,
                 run_cals: bool = True) -> ProtocolResult:
    """Run all A04 metrics + calibrations. Returns a ProtocolResult."""
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


def interventionist_analysis(result: ProtocolResult) -> list[str]:
    """Analyze results through Woodward's interventionist criteria."""
    lines = ["\n  Woodward Interventionist Criteria:", "  ──────────────────────────────────"]

    for task in result.tasks:
        lines.append(f"\n    {task}:")

        sa = _find(result.metrics.get("sigma_ablation", []), task)
        rc = _find(result.metrics.get("resample_complement", []), task)
        mi = _find(result.metrics.get("misalignment", []), task)

        criteria_met = []

        # Stability: does the effect degrade gracefully with sigma?
        if sa:
            lines.append(f"      Sigma ablation = {sa.value:.4f}")
            per_sigma = sa.metadata.get("per_sigma", {})
            if per_sigma:
                lines.append(f"        Graded response:")
                for sigma, effect in sorted(per_sigma.items(), key=lambda kv: float(kv[0])):
                    eff = effect.get("effect", effect) if isinstance(effect, dict) else effect
                    lines.append(f"          sigma={sigma}: effect={eff:.4f}")
            if sa.value > THRESHOLDS["sigma_ablation"]:
                lines.append(f"        Stability: PASS — effect degrades proportionally with noise")
                criteria_met.append("stability")
            else:
                lines.append(f"        Stability: FAIL — effect is fragile or non-monotonic")

        # Proportionality: complement resample should NOT recover performance
        if rc:
            lines.append(f"      Resample complement = {rc.value:.4f}")
            if rc.value > THRESHOLDS["resample_complement"]:
                lines.append(f"        Proportionality: PASS — complement cannot substitute for circuit")
                criteria_met.append("proportionality")
            else:
                lines.append(f"        Proportionality: FAIL — non-circuit components partially recover the effect")

        # Invariance: low misalignment means the causal model is stable
        if mi:
            lines.append(f"      Misalignment = {mi.value:.4f}")
            if mi.value < THRESHOLDS["misalignment"]:
                lines.append(f"        Invariance: PASS — causal predictions match observations")
                criteria_met.append("invariance")
            else:
                lines.append(f"        Invariance: FAIL — causal model predictions deviate from observations")

        n = len(criteria_met)
        if n == 3:
            verdict = "Full interventionist support (stability + proportionality + invariance)"
        elif n >= 1:
            verdict = f"Partial ({', '.join(criteria_met)})"
        else:
            verdict = "No interventionist support — circuit fails all three criteria"
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

    lines.extend(interventionist_analysis(result))

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
