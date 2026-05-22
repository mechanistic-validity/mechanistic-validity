"""Protocol MB_CO — Bayesian Colocalization
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Lens:         Genetics
Validity Type: Construct
Framework:    Molecular Biology — Bayesian Colocalization
Family:       Molecular Biology (Colocalization)
Validity:     External — shared causal component detection

References:
    Giambartolomei et al. (2014) "Bayesian test for colocalisation
        between pairs of genetic association studies"
    Wallace et al. (2021) "A more accurate method for colocalisation
        analysis"

Question:
    Do two behavioral signals (e.g., IOI performance and induction
    performance) share the same causal component? This is Part H1 of the
    Bio-Causal spec. Five hypotheses:
        H0: Neither signal is real
        H1: Only signal 1 is real
        H2: Only signal 2 is real
        H3: Both real, different causal variants (components)
        H4: Both real, same causal variant (COLOCALIZED — shared mechanism)
    PP.H4 > 0.8 means the two circuits share a component.

Metrics:
    cross_task_generalization — Same components important across tasks
    cross_task_transfer       — Causal estimates transfer between tasks
    cross_model_invariance    — Cross-model colocalization

Calibrations:
    STRUCTURAL_CALIBRATIONS

Usage:
    uv run python colocalization.py                       # all tasks, CPU
    uv run python colocalization.py --device cuda          # GPU
    uv run python colocalization.py --tasks ioi induction  # specific tasks

    # As a callable module:
    from protocols.molecular_biology.colocalization import run_protocol
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

PROTOCOL_ID = "MB_CO"
PROTOCOL_NAME = "Bayesian Colocalization"
METRICS = ["cross_task_generalization", "cross_task_transfer", "cross_model_invariance"]
CALIBRATIONS = STRUCTURAL_CALIBRATIONS
OUTPUT_DIR = Path(__file__).parent / "results" / "mb_co_colocalization"

THRESHOLDS = {
    "cross_task_generalization": 0.5,
    "cross_task_transfer": 0.4,
    "cross_model_invariance": 0.6,
}


def run_protocol(model, tasks: list[str] | None = None, n_prompts: int = 40,
                 device: str = "cpu", run_cals: bool = True) -> ProtocolResult:
    """Run all MB_CO metrics + calibrations. Returns a ProtocolResult."""
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


def colocalization_analysis(result: ProtocolResult) -> list[str]:
    """Analyze results through Bayesian Colocalization lens.

    Biology analogy: Colocalization tests whether two GWAS signals share
    the same causal variant at a genomic locus. Five hypotheses partition
    the posterior: H0 (neither real), H1/H2 (one real), H3 (both real,
    different variants), H4 (both real, same variant — colocalized).

    Approximate posterior probabilities from circuit metrics:
      - High cross_task_generalization + high cross_task_transfer
        → evidence for H4 (shared mechanism)
      - High cross_task_generalization + low cross_task_transfer
        → evidence for H3 (different mechanisms in same region)
      - Low generalization → evidence for H1/H2 (task-specific)
    """
    lines = ["\n  Bayesian Colocalization Analysis:", "  ---------------------------------"]

    for task in result.tasks:
        cg = _find(result.metrics.get("cross_task_generalization", []), task)
        ct = _find(result.metrics.get("cross_task_transfer", []), task)
        cm = _find(result.metrics.get("cross_model_invariance", []), task)

        lines.append(f"\n    {task}:")

        if cg:
            label = "shared components" if cg.value > 0.5 else ("partial overlap" if cg.value > 0.3 else "task-specific")
            lines.append(f"      Cross-task generalization:             {cg.value:.4f} — {label}")

        if ct:
            label = "transfers" if ct.value > 0.4 else ("partial" if ct.value > 0.2 else "no transfer")
            lines.append(f"      Cross-task transfer:                   {ct.value:.4f} — {label}")

        if cm:
            label = "cross-model" if cm.value > 0.6 else ("partial" if cm.value > 0.3 else "model-specific")
            lines.append(f"      Cross-model invariance:                {cm.value:.4f} — {label}")

        cg_val = cg.value if cg else 0.0
        ct_val = ct.value if ct else 0.0

        if cg_val > 0.6 and ct_val > 0.5:
            verdict = "COLOCALIZED (H4)"
        elif cg_val > 0.4 and ct_val < 0.3:
            verdict = "DISTINCT MECHANISMS (H3)"
        elif cg_val < 0.3:
            verdict = "TASK-SPECIFIC (H1/H2)"
        else:
            verdict = "INCONCLUSIVE"
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

    lines.extend(colocalization_analysis(result))

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
