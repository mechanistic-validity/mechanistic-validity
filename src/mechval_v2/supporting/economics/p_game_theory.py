"""Protocol GT — Game Theory Analysis
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Lens:         Economics
Validity Type: External
Framework:    https://mechanistic-validity.github.io/mechanistic-validity/framework/metrics/cross_discipline/game-theory/
Family:       Cross-discipline (Game Theory)
Validity:     External — strategic interaction analysis of circuit components

References:
    Nash (1950) "Equilibrium Points in N-Person Games"
    Shapley (1953) "A Value for N-Person Games"
    Banzhaf (1965) "Weighted Voting Doesn't Work"
    Taylor & Jonker (1978) "Evolutionary Stable Strategies and Game Dynamics"

Question:
    Do circuit components form strategic equilibria? Can heads be modeled
    as rational agents in a cooperative game? Nash equilibrium tests whether
    any head could unilaterally improve the circuit. Banzhaf/voting power
    measures each head's pivotal influence. Core stability and envy-freeness
    test whether the circuit allocation is stable. Coalition tracking and
    replicator dynamics test evolutionary stability. Nucleolus identifies
    the fairest allocation of credit.

Metrics:
    nash_equilibrium    — Is the circuit at a Nash equilibrium?
    banzhaf_power       — Banzhaf power index per head
    core_stability      — Is the head allocation in the core?
    voting_power        — Shapley-Shubik voting power
    envy_freeness       — Does any head "envy" another's allocation?
    coalition_discovery — Identify stable head coalitions
    coalition_tracking  — Track coalition stability over ablation
    replicator_dynamics — Evolutionary stability of head populations
    nucleolus           — Fairest credit allocation (nucleolus)

Usage:
    uv run python game_theory.py                       # all tasks, CPU
    uv run python game_theory.py --device cuda          # GPU
    uv run python game_theory.py --tasks ioi induction  # specific tasks

    # As a callable module:
    from protocols.cross_discipline.game_theory import run_protocol
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

PROTOCOL_ID = "GT"
PROTOCOL_NAME = "Game Theory Analysis"
METRICS = [
    "nash_equilibrium", "banzhaf_power", "core_stability", "voting_power",
    "envy_freeness", "coalition_discovery", "coalition_tracking",
    "replicator_dynamics", "nucleolus",
]
CALIBRATIONS = STRUCTURAL_CALIBRATIONS
OUTPUT_DIR = Path(__file__).parent / "results" / "gt_game_theory"

THRESHOLDS = {
    "nash_equilibrium": 0.5,
    "banzhaf_power": 0.0,
    "core_stability": 0.5,
    "voting_power": 0.0,
    "envy_freeness": 0.5,
    "coalition_discovery": 0.0,
    "coalition_tracking": 0.5,
    "replicator_dynamics": 0.5,
    "nucleolus": 0.0,
}




def run_protocol(model, tasks: list[str] | None = None, n_prompts: int = 40,
                 run_cals: bool = True) -> ProtocolResult:
    """Run all GT metrics + calibrations. Returns a ProtocolResult."""
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


def game_theory_analysis(result: ProtocolResult) -> list[str]:
    """Analyze results through game-theoretic lens."""
    lines = ["\n  Game Theory Analysis:", "  ----------------------"]

    for task in result.tasks:
        lines.append(f"\n    {task}:")

        ne = _find(result.metrics.get("nash_equilibrium", []), task)
        bp = _find(result.metrics.get("banzhaf_power", []), task)
        cs = _find(result.metrics.get("core_stability", []), task)
        vp = _find(result.metrics.get("voting_power", []), task)
        ef = _find(result.metrics.get("envy_freeness", []), task)
        cd = _find(result.metrics.get("coalition_discovery", []), task)
        ct = _find(result.metrics.get("coalition_tracking", []), task)
        rd = _find(result.metrics.get("replicator_dynamics", []), task)
        nu = _find(result.metrics.get("nucleolus", []), task)

        properties = []

        if ne:
            lines.append(f"      Nash equilibrium = {ne.value:.4f}")
            if ne.value > THRESHOLDS["nash_equilibrium"]:
                properties.append("equilibrium")

        if bp:
            lines.append(f"      Banzhaf power (Gini) = {bp.value:.4f}")

        if cs:
            lines.append(f"      Core stability = {cs.value:.4f}")
            if cs.value > THRESHOLDS["core_stability"]:
                properties.append("core-stable")

        if vp:
            lines.append(f"      Voting power (Gini) = {vp.value:.4f}")

        if ef:
            lines.append(f"      Envy-freeness = {ef.value:.4f}")
            if ef.value > THRESHOLDS["envy_freeness"]:
                properties.append("envy-free")

        if cd:
            lines.append(f"      Coalition discovery = {cd.value:.4f}")

        if ct:
            lines.append(f"      Coalition tracking = {ct.value:.4f}")
            if ct.value > THRESHOLDS["coalition_tracking"]:
                properties.append("coalition-stable")

        if rd:
            lines.append(f"      Replicator dynamics = {rd.value:.4f}")
            if rd.value > THRESHOLDS["replicator_dynamics"]:
                properties.append("evolutionarily-stable")

        if nu:
            lines.append(f"      Nucleolus = {nu.value:.4f}")

        if properties:
            verdict = f"Game-theoretic properties: {', '.join(properties)}"
        else:
            verdict = "No game-theoretic stability detected"
        lines.append(f"      VERDICT: {verdict}")

    return lines


def _find(results: list[EvalResult], task: str) -> EvalResult | None:
    return next((r for r in results if r.metadata.get("task") == task), None)


def summarize(result: ProtocolResult) -> str:
    lines = []
    lines.append(f"\n{'=' * 70}")
    lines.append(f"  PROTOCOL {result.protocol_id}: {result.protocol_name}")
    lines.append(f"{'=' * 70}\n")

    header = f"{'Task':20s}" + "".join(f"  {m:>20s}" for m in METRICS)
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
                row += f"  {v:>16.4f}{tag}"
            else:
                row += f"  {'---':>20s}"
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

    lines.extend(game_theory_analysis(result))

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
