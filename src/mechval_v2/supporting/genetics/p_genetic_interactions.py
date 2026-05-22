"""Protocol MB_GI — Genetic Interactions
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Lens:         Genetics
Validity Type: Internal
Framework:    Molecular Biology — Epistasis and Synthetic Lethality
Family:       Molecular Biology (Genetic Interactions)
Validity:     Internal — combinatorial necessity and redundancy

References:
    Norman et al. (2019) "Exploring genetic interaction manifolds
        constructed from rich single-cell phenotypes" — combinatorial Perturb-seq
    Costanzo et al. (2016) "A global genetic interaction network maps a
        wiring diagram of cellular function" — yeast GI maps

Question:
    How do circuit components interact in combination? Synthetic lethality
    (knocking out two individually tolerable components kills function)
    reveals hidden dependencies. Epistasis (one component masking another)
    reveals hierarchical relationships. Shapley interactions quantify
    marginal contributions. These map directly to pairwise ablation studies
    in mechanistic interpretability.

Metrics:
    epistasis            — Pairwise epistatic interactions between heads
    pairwise_synergy     — Synergistic vs redundant head pairs
    shapley_interactions — Shapley value decomposition of contributions

Calibrations:
    STRUCTURAL_CALIBRATIONS

Usage:
    uv run python genetic_interactions.py                       # all tasks, CPU
    uv run python genetic_interactions.py --device cuda          # GPU
    uv run python genetic_interactions.py --tasks ioi induction  # specific tasks

    # As a callable module:
    from protocols.molecular_biology.genetic_interactions import run_protocol
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

PROTOCOL_ID = "MB_GI"
PROTOCOL_NAME = "Genetic Interactions"
METRICS = ["epistasis", "pairwise_synergy", "shapley_interactions"]
CALIBRATIONS = STRUCTURAL_CALIBRATIONS
OUTPUT_DIR = Path(__file__).parent / "results" / "mb_gi_genetic_interactions"

THRESHOLDS = {
    "epistasis": 0.3,
    "pairwise_synergy": 0.3,
    "shapley_interactions": 0.2,
}


def run_protocol(model, tasks: list[str] | None = None, n_prompts: int = 40,
                 device: str = "cpu", run_cals: bool = True) -> ProtocolResult:
    """Run all MB_GI metrics + calibrations. Returns a ProtocolResult."""
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


def interaction_analysis(result: ProtocolResult) -> list[str]:
    """Analyze results through genetic interaction lens.

    Biology analogy: genetic interaction maps classify gene pairs as
    synergistic (double mutant worse than expected = synthetic lethality),
    redundant (double mutant no worse = buffering), or epistatic (one
    masks the other = hierarchical). Dense interaction networks suggest
    tightly coupled modules; sparse networks suggest modular independence.
    """
    lines = ["\n  Genetic Interaction Analysis:", "  -----------------------------"]

    for task in result.tasks:
        ep = _find(result.metrics.get("epistasis", []), task)
        ps = _find(result.metrics.get("pairwise_synergy", []), task)
        sh = _find(result.metrics.get("shapley_interactions", []), task)

        lines.append(f"\n    {task}:")

        if ep:
            if ep.value > 0.5:
                label = "strong epistasis — hierarchical circuit"
            elif ep.value > 0.2:
                label = "moderate epistasis — partial hierarchy"
            else:
                label = "weak epistasis — independent components"
            lines.append(f"      Epistasis:             {ep.value:.4f} — {label}")

        if ps:
            if ps.value > 0.3:
                label = "synergistic — synthetic lethal pairs present"
            elif ps.value > 0.0:
                label = "mixed — some synergy, some redundancy"
            else:
                label = "redundant — buffering dominates"
            lines.append(f"      Pairwise synergy:      {ps.value:.4f} — {label}")

        if sh:
            if sh.value > 0.3:
                label = "concentrated — few critical components"
            elif sh.value > 0.1:
                label = "distributed — shared contributions"
            else:
                label = "flat — no dominant contributors"
            lines.append(f"      Shapley interactions:  {sh.value:.4f} — {label}")

        if ep and ps:
            if ep.value > 0.3 and ps.value > 0.3:
                verdict = "TIGHTLY COUPLED — strong interactions, synthetic lethal pairs"
            elif ep.value < 0.2 and ps.value < 0.1:
                verdict = "MODULAR — independent components, minimal cross-talk"
            else:
                verdict = "MIXED — some coupling, partial redundancy"
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

    lines.extend(interaction_analysis(result))

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
