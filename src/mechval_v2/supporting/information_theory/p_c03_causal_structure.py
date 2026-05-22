"""Protocol C03 — Causal Structure Discovery
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Lens:         Information Theory
Validity Type: Construct
Framework:    https://mechanistic-validity.github.io/mechanistic-validity/framework/metrics/structural/c09-notears/
Family:       Information-theoretic (causal structure learning from data)
Validity:     Internal — I1 Necessity; Construct — C1 Convergent, C2 Discriminant

References:
    Zheng et al. (2018) "DAGs with NO TEARS: Continuous Optimization for
        Structure Learning" — NOTEARS algorithm
    Rubenstein et al. (2017) "Causal Consistency of Structural Equation Models"
        — OCSE (Observational-Counterfactual Structural Equivalence)
    Peters et al. (2017) "Elements of Causal Inference"
    Spirtes et al. (2000) "Causation, Prediction, and Search"

Question:
    Can the circuit's causal structure be recovered from observational data
    alone? Does the learned DAG match the hypothesized circuit graph? Is the
    circuit structurally equivalent under observational and counterfactual
    distributions?

Metrics:
    ocse     — Observational-Counterfactual Structural Equivalence
    notears  — NOTEARS DAG recovery: does continuous optimization recover
               the circuit graph from activations?

Calibrations:
    measurement_invariance — Are structure learning results stable across prompts?
    convergent_validity    — Do discovered structures agree with causal metrics?
    discriminant_validity  — Do discovered structures distinguish circuits?

Usage:
    uv run python c03_causal_structure.py                       # all tasks, CPU
    uv run python c03_causal_structure.py --device cuda          # GPU
    uv run python c03_causal_structure.py --tasks ioi induction  # specific tasks

    # As a callable module:
    from protocols.information.c03_causal_structure import run_protocol
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

PROTOCOL_ID = "C03"
PROTOCOL_NAME = "Causal Structure Discovery"
METRICS = ["ocse", "notears"]
CALIBRATIONS = STRUCTURAL_CALIBRATIONS
OUTPUT_DIR = Path(__file__).parent / "results" / "c03_causal_structure"

THRESHOLDS = {
    "ocse": 0.5,
    "notears": 0.5,
}




def run_protocol(model, tasks: list[str] | None = None, n_prompts: int = 40,
                 run_cals: bool = True) -> ProtocolResult:
    """Run all C03 metrics + calibrations. Returns a ProtocolResult."""
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


def causal_structure_analysis(result: ProtocolResult) -> list[str]:
    """Analyze results through causal structure discovery lens."""
    lines = ["\n  Causal Structure Discovery:", "  ────────────────────────────"]

    for task in result.tasks:
        lines.append(f"\n    {task}:")

        ocse_r = _find(result.metrics.get("ocse", []), task)
        notears_r = _find(result.metrics.get("notears", []), task)

        findings = []

        if ocse_r:
            lines.append(f"      OCSE (obs-counterfactual equivalence): {ocse_r.value:.4f}")
            if ocse_r.value > THRESHOLDS["ocse"]:
                findings.append("structurally equivalent")
                lines.append(f"        → Observational and counterfactual structures agree")
            else:
                lines.append(f"        → Structure differs between observational and "
                             f"counterfactual — hidden confounders likely")

        if notears_r:
            lines.append(f"      NOTEARS DAG recovery: {notears_r.value:.4f}")
            n_edges = notears_r.metadata.get("n_edges", None)
            shd = notears_r.metadata.get("structural_hamming_distance", None)
            if n_edges is not None:
                lines.append(f"        Recovered {n_edges} edges")
            if shd is not None:
                lines.append(f"        Structural Hamming distance: {shd}")
            if notears_r.value > THRESHOLDS["notears"]:
                findings.append("DAG recoverable")
                lines.append(f"        → Circuit DAG recoverable from activations")
            else:
                lines.append(f"        → DAG recovery poor — structure may not be "
                             f"identifiable from data alone")

        if len(findings) == 2:
            verdict = "Strong causal structure: recoverable and consistent"
        elif findings:
            verdict = f"Partial structure: {', '.join(findings)}"
        else:
            verdict = "Weak causal structure — circuit graph not recoverable"
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

    lines.extend(causal_structure_analysis(result))

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
