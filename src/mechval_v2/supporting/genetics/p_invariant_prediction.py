"""Protocol MB_IP — Invariant Causal Prediction
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Lens:         Genetics
Validity Type: Internal
Framework:    Molecular Biology — Invariant Causal Prediction
Family:       Molecular Biology (Invariant Causal Prediction)
Validity:     External — stability of causal claims across environments

References:
    Peters, Buehlmann, Meinshausen (2016) "Causal inference by using
        invariant prediction: identification and confidence intervals"
    Heinze-Deml, Peters, Meinshausen (2018) "Invariant causal prediction
        for nonlinear models"

Question:
    Are circuit components stably causal across environments? The ICP
    framework identifies causal parents as variables whose conditional
    distribution remains invariant across experimental conditions. Here,
    environments are tasks, input distributions, and model variants. Truly
    causal components should show stable effects everywhere; spurious
    associations break under distribution shift.

Metrics:
    cross_task_generalization — Stability across task environments
    generalization_gap        — Performance drop under distribution shift
    cross_model_invariance    — Stability across model environments

Calibrations:
    STRUCTURAL_CALIBRATIONS

Usage:
    uv run python invariant_prediction.py                       # all tasks, CPU
    uv run python invariant_prediction.py --device cuda          # GPU
    uv run python invariant_prediction.py --tasks ioi induction  # specific tasks

    # As a callable module:
    from protocols.molecular_biology.invariant_prediction import run_protocol
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

PROTOCOL_ID = "MB_IP"
PROTOCOL_NAME = "Invariant Causal Prediction"
METRICS = ["cross_task_generalization", "generalization_gap", "cross_model_invariance"]
CALIBRATIONS = STRUCTURAL_CALIBRATIONS
OUTPUT_DIR = Path(__file__).parent / "results" / "mb_ip_invariant_prediction"

THRESHOLDS = {
    "cross_task_generalization": 0.5,
    "generalization_gap": 0.3,
    "cross_model_invariance": 0.6,
}


def run_protocol(model, tasks: list[str] | None = None, n_prompts: int = 40,
                 device: str = "cpu", run_cals: bool = True) -> ProtocolResult:
    """Run all MB_IP metrics + calibrations. Returns a ProtocolResult."""
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


def invariance_analysis(result: ProtocolResult) -> list[str]:
    """Analyze results through Invariant Causal Prediction lens.

    Biology analogy: causal genes show consistent phenotypic effects across
    genetic backgrounds, cell types, and environmental conditions. Spurious
    associations (confounders) break under these shifts. ICP identifies
    the invariant set S* of variables whose conditional distribution
    Y | X_S is stable across all environments. Stability score = variance
    across environments; lower = more invariant = more likely causal.
    """
    lines = ["\n  Invariant Causal Prediction Analysis:", "  -------------------------------------"]

    for task in result.tasks:
        ctg = _find(result.metrics.get("cross_task_generalization", []), task)
        gg = _find(result.metrics.get("generalization_gap", []), task)
        cmi = _find(result.metrics.get("cross_model_invariance", []), task)

        lines.append(f"\n    {task}:")

        invariant_count = 0

        if ctg:
            if ctg.value > 0.7:
                label = "highly stable across tasks"
            elif ctg.value > 0.4:
                label = "partially stable"
            else:
                label = "task-specific (not invariant)"
            lines.append(f"      Cross-task stability:   {ctg.value:.4f} — {label}")
            if ctg.value > 0.5:
                invariant_count += 1

        if gg:
            if gg.value < 0.1:
                label = "minimal gap — robust to distribution shift"
            elif gg.value < 0.3:
                label = "moderate gap — some sensitivity"
            else:
                label = "large gap — fragile under shift"
            lines.append(f"      Generalization gap:     {gg.value:.4f} — {label}")
            if gg.value < 0.3:
                invariant_count += 1

        if cmi:
            if cmi.value > 0.7:
                label = "model-invariant — conserved mechanism"
            elif cmi.value > 0.4:
                label = "partially conserved"
            else:
                label = "model-specific — not conserved"
            lines.append(f"      Cross-model invariance: {cmi.value:.4f} — {label}")
            if cmi.value > 0.6:
                invariant_count += 1

        if invariant_count == 3:
            verdict = "INVARIANT SET MEMBER — stable across all environments (strong causal)"
        elif invariant_count == 2:
            verdict = "LIKELY CAUSAL — stable in most environments"
        elif invariant_count == 1:
            verdict = "PARTIALLY INVARIANT — environment-sensitive"
        else:
            verdict = "NOT INVARIANT — likely spurious or confounded"
        lines.append(f"      VERDICT: {verdict}")

    return lines


def _find(results: list[EvalResult], task: str) -> EvalResult | None:
    return next((r for r in results if r.metadata.get("task") == task), None)


def summarize(result: ProtocolResult) -> str:
    lines = []
    lines.append(f"\n{'=' * 70}")
    lines.append(f"  PROTOCOL {result.protocol_id}: {result.protocol_name}")
    lines.append(f"{'=' * 70}\n")

    header = f"{'Task':20s}" + "".join(f"  {m:>28s}" for m in METRICS)
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
                row += f"  {v:>24.4f}{tag}"
            else:
                row += f"  {'---':>28s}"
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

    lines.extend(invariance_analysis(result))

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
