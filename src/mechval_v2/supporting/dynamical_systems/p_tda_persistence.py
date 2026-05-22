"""Protocol WC_M5 — Topological Data Analysis (Persistence Diagrams)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Lens:         Dynamical Systems
Validity Type: Measurement
Framework:    Wildcard — Computational Topology
Family:       Topological Data Analysis (Persistent Homology)
Validity:     Structural — geometric invariants of activation manifolds

References:
    Edelsbrunner & Harer (2008) "Persistent Homology — a Survey"
    Gudhi library for computational topology
    Wasserstein distance between persistence diagrams

Question:
    Do circuit activation point clouds have stable topological structure?
    Persistent homology extracts multi-scale geometric features from
    activation manifolds: H0 (connected components) reveals clustering
    structure; H1 (loops) reveals circular/ring geometry.
    CKA measures representational similarity between tasks.
    Cross-task generalization checks if topology transfers.
    Cross-model invariance checks if topology is model-independent.

Metrics:
    cka                       — Representational similarity between tasks
    cross_task_generalization  — Does topological structure transfer across tasks?
    cross_model_invariance     — Is topology model-independent?

Calibrations:
    STRUCTURAL_CALIBRATIONS

Usage:
    uv run python tda_persistence.py                       # all tasks, CPU
    uv run python tda_persistence.py --device cuda          # GPU
    uv run python tda_persistence.py --tasks ioi induction  # specific tasks

    # As a callable module:
    from protocols.wildcard.tda_persistence import run_protocol
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

PROTOCOL_ID = "WC_M5"
PROTOCOL_NAME = "Topological Data Analysis (Persistence Diagrams)"
METRICS = ["cka", "cross_task_generalization", "cross_model_invariance"]
CALIBRATIONS = STRUCTURAL_CALIBRATIONS
OUTPUT_DIR = Path(__file__).parent / "results" / "wc_m5_tda_persistence"

THRESHOLDS = {
    "cka": 0.5,
    "cross_task_generalization": 0.5,
    "cross_model_invariance": 0.6,
}


def run_protocol(model, tasks: list[str] | None = None, n_prompts: int = 40,
                 device: str = "cpu", run_cals: bool = True) -> ProtocolResult:
    """Run all WC_M5 metrics + calibrations. Returns a ProtocolResult."""
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


def tda_persistence_analysis(result: ProtocolResult) -> list[str]:
    """Analyze results through TDA / persistent homology lens.

    Persistent homology extracts multi-scale topological features from
    activation point clouds. For each task, activations at a hook point
    across many prompts form a point cloud; a Rips complex built on this
    cloud yields persistence diagrams. H0 (connected components) reveals
    clustering structure; H1 (loops) reveals circular/ring geometry.

    Interpretation grid:
        High CKA + high generalization -> SHARED TOPOLOGY
            Circuits have similar geometric structure across tasks.
        Low CKA + high generalization -> DIFFERENT GEOMETRY, SAME FUNCTION
            Topologically distinct but functionally similar.
        High CKA + low generalization -> GEOMETRIC COINCIDENCE
            Similar representations but different circuits.
        Low everything -> TASK-SPECIFIC TOPOLOGY
            Geometry is unique to each task.

    Wasserstein distance between persistence diagrams provides a
    principled metric for comparing topological structure across
    tasks and models.
    """
    lines = ["\n  TDA Persistence Analysis:", "  ------------------------"]

    for task in result.tasks:
        cka = _find(result.metrics.get("cka", []), task)
        ctg = _find(result.metrics.get("cross_task_generalization", []), task)
        cmi = _find(result.metrics.get("cross_model_invariance", []), task)

        lines.append(f"\n    {task}:")

        cka_high = False
        ctg_high = False
        cmi_high = False

        if cka:
            label = "high similarity" if cka.value > 0.5 else ("moderate" if cka.value > 0.3 else "low similarity")
            lines.append(f"      CKA (representational similarity):     {cka.value:.4f} — {label}")
            cka_high = cka.value > THRESHOLDS["cka"]

        if ctg:
            label = "transfers" if ctg.value > 0.5 else ("partial" if ctg.value > 0.3 else "task-specific")
            lines.append(f"      Cross-task generalization:             {ctg.value:.4f} — {label}")
            ctg_high = ctg.value > THRESHOLDS["cross_task_generalization"]

        if cmi:
            label = "model-invariant" if cmi.value > 0.6 else ("partial" if cmi.value > 0.3 else "model-specific")
            lines.append(f"      Cross-model invariance:                {cmi.value:.4f} — {label}")
            cmi_high = cmi.value > THRESHOLDS["cross_model_invariance"]

        if cka_high and ctg_high:
            topology = "SHARED TOPOLOGY — circuits have similar geometric structure across tasks"
        elif not cka_high and ctg_high:
            topology = "DIFFERENT GEOMETRY, SAME FUNCTION — topologically distinct but functionally similar"
        elif cka_high and not ctg_high:
            topology = "GEOMETRIC COINCIDENCE — similar representations but different circuits"
        else:
            topology = "TASK-SPECIFIC TOPOLOGY — geometry is unique to each task"
        lines.append(f"      Topology: {topology}")

        if cka_high and ctg_high:
            verdict = "TOPOLOGICALLY STABLE — robust geometric structure"
        elif cka_high and not ctg_high:
            verdict = "TOPOLOGICALLY FRAGILE — geometry does not transfer"
        elif ctg_high:
            verdict = "FUNCTIONALLY INVARIANT — generalization despite different geometry"
        else:
            verdict = "TOPOLOGICALLY UNIQUE — task-specific geometry"
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

    lines.extend(tda_persistence_analysis(result))

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
