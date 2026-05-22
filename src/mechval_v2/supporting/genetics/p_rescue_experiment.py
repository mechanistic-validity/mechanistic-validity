"""Protocol MB_RE — Rescue Experiments
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Lens:         Genetics
Validity Type: Internal
Framework:    Molecular Biology — Knockout-then-Rescue
Family:       Molecular Biology (Rescue Experiments)
Validity:     Internal — necessity + sufficiency via rescue

References:
    Wang et al. (2023) "Interpretability in the Wild" — IOI corrupt-restore
    Geiger et al. (2021) "Causal Abstractions of Neural Networks" —
        interchange intervention / DAS

Question:
    Does re-expressing a knocked-out component rescue function? This is the
    gold standard in molecular biology: knockout shows necessity, rescue
    shows sufficiency, and the combination provides the strongest causal
    evidence. Maps directly to corrupt-then-restore activation patching.

Metrics:
    corrupt_restore            — Knockout-rescue: corrupt then restore activations
    corrupt_restore_behavioral — Behavioral rescue: task performance recovery
    das_iia                    — Interchange intervention accuracy (causal abstraction)

Calibrations:
    STRUCTURAL_CALIBRATIONS

Usage:
    uv run python rescue_experiment.py                       # all tasks, CPU
    uv run python rescue_experiment.py --device cuda          # GPU
    uv run python rescue_experiment.py --tasks ioi induction  # specific tasks

    # As a callable module:
    from protocols.molecular_biology.rescue_experiment import run_protocol
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

PROTOCOL_ID = "MB_RE"
PROTOCOL_NAME = "Rescue Experiments"
METRICS = ["corrupt_restore", "corrupt_restore_behavioral", "das_iia"]
CALIBRATIONS = STRUCTURAL_CALIBRATIONS
OUTPUT_DIR = Path(__file__).parent / "results" / "mb_re_rescue_experiment"

THRESHOLDS = {
    "corrupt_restore": 0.7,
    "corrupt_restore_behavioral": 0.6,
    "das_iia": 0.7,
}


def run_protocol(model, tasks: list[str] | None = None, n_prompts: int = 40,
                 device: str = "cpu", run_cals: bool = True) -> ProtocolResult:
    """Run all MB_RE metrics + calibrations. Returns a ProtocolResult."""
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


def rescue_analysis(result: ProtocolResult) -> list[str]:
    """Analyze results through knockout-rescue lens.

    Biology analogy: the gold standard for gene function. Knockout shows
    the gene is necessary; rescue by re-expression shows it is sufficient.
    Necessity score = corruption effect, sufficiency score = restoration,
    rescue efficiency = sufficiency / necessity ratio.
    """
    lines = ["\n  Rescue Experiment Analysis:", "  ---------------------------"]

    for task in result.tasks:
        cr = _find(result.metrics.get("corrupt_restore", []), task)
        cb = _find(result.metrics.get("corrupt_restore_behavioral", []), task)
        iia = _find(result.metrics.get("das_iia", []), task)

        lines.append(f"\n    {task}:")

        necessity = None
        sufficiency = None

        if cr:
            necessity = cr.value
            label = "strong" if cr.value > 0.7 else ("moderate" if cr.value > 0.4 else "weak")
            lines.append(f"      Necessity (corrupt_restore):      {cr.value:.4f} — {label} knockout effect")

        if cb:
            sufficiency = cb.value
            label = "full rescue" if cb.value > 0.8 else ("partial rescue" if cb.value > 0.5 else "failed rescue")
            lines.append(f"      Sufficiency (behavioral rescue):  {cb.value:.4f} — {label}")

        if iia:
            label = "aligned" if iia.value > 0.7 else ("partial" if iia.value > 0.4 else "misaligned")
            lines.append(f"      Causal abstraction (DAS IIA):     {iia.value:.4f} — {label}")

        if necessity is not None and sufficiency is not None:
            if necessity > 0.01:
                efficiency = sufficiency / necessity
                lines.append(f"      Rescue efficiency (suff/nec):     {efficiency:.4f}")
            if necessity > 0.7 and sufficiency > 0.7:
                verdict = "FULL RESCUE — strong causal evidence (both necessary and sufficient)"
            elif necessity > 0.4 and sufficiency > 0.4:
                verdict = "PARTIAL RESCUE — moderate causal evidence"
            elif necessity > 0.4 and sufficiency < 0.3:
                verdict = "NECESSARY BUT NOT SUFFICIENT — redundant rescue pathways"
            elif necessity < 0.3 and sufficiency > 0.4:
                verdict = "SUFFICIENT BUT NOT NECESSARY — alternative knockdown targets exist"
            else:
                verdict = "WEAK EVIDENCE — neither strong necessity nor sufficiency"
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

    lines.extend(rescue_analysis(result))

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
