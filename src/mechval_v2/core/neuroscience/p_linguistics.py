"""Protocol EX03 — Linguistic Probes
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Lens:         Neuroscience
Validity Type: Construct
Framework:    https://mechanistic-validity.github.io/mechanistic-validity/framework/metrics/cognitive_science/linguistics/
Family:       Cognitive Science (Linguistics)
Validity:     External — linguistic structure sensitivity of circuits

References:
    Linzen et al. (2016) "Assessing the Ability of LSTMs to Learn Syntax-Sensitive Dependencies"
    Marvin & Linzen (2018) "Targeted Syntactic Evaluation of Language Models"
    Futrell et al. (2019) "Neural language models as psycholinguistic subjects"

Question:
    Do circuit components exhibit sensitivity to linguistic structure?
    Priming tests whether circuits show facilitated processing for related
    items. Garden-path tests sensitivity to syntactic reanalysis. Binding
    theory tests knowledge of referential constraints. Animacy tests
    selectional restriction sensitivity.

Metrics:
    priming         — Semantic/syntactic priming effects in circuit activations
    garden_path     — Sensitivity to garden-path syntactic reanalysis
    binding_theory  — Binding constraint sensitivity (anaphora/pronouns)
    animacy         — Animacy selectional restriction sensitivity

Usage:
    uv run python linguistics.py                       # all tasks, CPU
    uv run python linguistics.py --device cuda          # GPU
    uv run python linguistics.py --tasks ioi induction  # specific tasks

    # As a callable module:
    from protocols.cognitive_science.linguistics import run_protocol
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
from protocols.calibration_runner import BEHAVIORAL_CALIBRATIONS, run_calibrations, summarize_calibrations

PROTOCOL_ID = "EX03"
PROTOCOL_NAME = "Linguistic Probes"
METRICS = ["priming", "garden_path", "binding_theory", "animacy"]
CALIBRATIONS = BEHAVIORAL_CALIBRATIONS
OUTPUT_DIR = Path(__file__).parent / "results" / "ex03_linguistics"

THRESHOLDS = {
    "priming": 0.0,
    "garden_path": 0.0,
    "binding_theory": 0.5,
    "animacy": 0.5,
}




def run_protocol(model, tasks: list[str] | None = None, n_prompts: int = 40,
                 run_cals: bool = True) -> ProtocolResult:
    """Run all EX03 metrics + calibrations. Returns a ProtocolResult."""
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


def linguistic_analysis(result: ProtocolResult) -> list[str]:
    """Analyze results through linguistic structure lens."""
    lines = ["\n  Linguistic Structure Analysis:", "  ------------------------------"]

    for task in result.tasks:
        lines.append(f"\n    {task}:")

        pr = _find(result.metrics.get("priming", []), task)
        gp = _find(result.metrics.get("garden_path", []), task)
        bt = _find(result.metrics.get("binding_theory", []), task)
        an = _find(result.metrics.get("animacy", []), task)

        probes_passed = 0
        probes_total = 0

        if pr:
            lines.append(f"      Priming effect = {pr.value:+.4f}")
            probes_total += 1
            if pr.value > THRESHOLDS["priming"]:
                probes_passed += 1

        if gp:
            lines.append(f"      Garden-path sensitivity = {gp.value:+.4f}")
            probes_total += 1
            if gp.value > THRESHOLDS["garden_path"]:
                probes_passed += 1

        if bt:
            lines.append(f"      Binding theory = {bt.value:.4f}")
            probes_total += 1
            if bt.value > THRESHOLDS["binding_theory"]:
                probes_passed += 1

        if an:
            lines.append(f"      Animacy selectional = {an.value:.4f}")
            probes_total += 1
            if an.value > THRESHOLDS["animacy"]:
                probes_passed += 1

        if probes_total > 0:
            if probes_passed == probes_total:
                verdict = "Full linguistic sensitivity"
            elif probes_passed > 0:
                verdict = f"Partial ({probes_passed}/{probes_total} probes)"
            else:
                verdict = "No linguistic sensitivity detected"
            lines.append(f"      VERDICT: {verdict}")

    return lines


def _find(results: list[EvalResult], task: str) -> EvalResult | None:
    return next((r for r in results if r.metadata.get("task") == task), None)


def summarize(result: ProtocolResult) -> str:
    lines = []
    lines.append(f"\n{'=' * 70}")
    lines.append(f"  PROTOCOL {result.protocol_id}: {result.protocol_name}")
    lines.append(f"{'=' * 70}\n")

    header = f"{'Task':20s}" + "".join(f"  {m:>16s}" for m in METRICS)
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
                row += f"  {v:>12.4f}{tag}"
            else:
                row += f"  {'---':>16s}"
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

    lines.extend(linguistic_analysis(result))

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
