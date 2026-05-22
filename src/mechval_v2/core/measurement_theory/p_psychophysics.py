"""Protocol EX_PP — Psychophysics
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Lens:         Measurement Theory
Validity Type: Measurement
Framework:    https://mechanistic-validity.github.io/mechanistic-validity/framework/metrics/cognitive_science/psychophysics/
Family:       Cognitive Science (Psychophysics)
Validity:     External — Hebbian learning and psychophysical law conformity

References:
    Hebb (1949) "The Organization of Behavior"
    Weber (1834/1996) "De Pulsu, Resorptione, Auditu et Tactu"
    Fechner (1860/1966) "Elements of Psychophysics"
    Stevens (1957) "On the psychophysical law"

Question:
    Does the circuit show Hebbian learning signatures — do neurons that
    fire together wire together? Does circuit sensitivity follow the
    Weber-Fechner law (logarithmic sensitivity to stimulus magnitude),
    or does it better fit Stevens' power law? These psychophysical
    relationships constrain how circuits transduce input intensity into
    representational magnitude.

Metrics:
    hebbian       — Hebbian co-activation correlation between connected circuit components
    weber_fechner — Fit to Weber-Fechner logarithmic sensitivity law

Usage:
    uv run python psychophysics.py
    uv run python psychophysics.py --tasks ioi induction --device cuda

    from protocols.cognitive_science.psychophysics import run_protocol
    result = run_protocol(model, tasks=["ioi"])
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

PROTOCOL_ID = "EX_PP"
PROTOCOL_NAME = "Psychophysics"
METRICS = ["hebbian", "weber_fechner"]
CALIBRATIONS = STRUCTURAL_CALIBRATIONS
OUTPUT_DIR = Path(__file__).parent / "results" / "ex_pp_psychophysics"

THRESHOLDS = {
    "hebbian": 0.3,
    "weber_fechner": 0.5,
}




def run_protocol(model, tasks: list[str] | None = None, n_prompts: int = 40,
                 run_cals: bool = True) -> ProtocolResult:
    """Run all EX_PP metrics + calibrations. Returns a ProtocolResult."""
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


def psychophysics_analysis(result: ProtocolResult) -> list[str]:
    """Analyze results through the psychophysics lens."""
    lines = ["\n  Psychophysics Analysis:", "  -----------------------"]

    for task in result.tasks:
        lines.append(f"\n    {task}:")

        hb = _find(result.metrics.get("hebbian", []), task)
        wf = _find(result.metrics.get("weber_fechner", []), task)

        if hb:
            v = hb.value
            lines.append(f"      Hebbian correlation = {v:.4f}")
            if v > 0.6:
                lines.append(f"        Strong Hebbian signature — co-active circuit components "
                             f"have strongly correlated weights (fire together, wire together)")
            elif v > THRESHOLDS["hebbian"]:
                lines.append(f"        Moderate Hebbian signature — some correlation between "
                             f"co-activation and connectivity strength")
            elif v > 0.0:
                lines.append(f"        Weak Hebbian signature — co-activation and connectivity "
                             f"are only loosely related")
            else:
                lines.append(f"        Anti-Hebbian or absent — no positive correlation between "
                             f"co-firing and connection strength")

        if wf:
            v = wf.value
            lines.append(f"      Weber-Fechner fit = {v:.4f}")
            if v > 0.8:
                lines.append(f"        Excellent log-law fit — circuit sensitivity follows "
                             f"Fechner's law (response = k * log(stimulus))")
            elif v > THRESHOLDS["weber_fechner"]:
                lines.append(f"        Good log-law fit — circuit approximately follows "
                             f"Weber-Fechner; may also fit Stevens' power law")
            elif v > 0.2:
                lines.append(f"        Poor log-law fit — circuit sensitivity does not follow "
                             f"Fechner's law; may follow Stevens' power law (R = k * S^n) "
                             f"or a different transduction function")
            else:
                lines.append(f"        No psychophysical law fit — circuit sensitivity is "
                             f"neither logarithmic nor power-law; response may be linear "
                             f"or discontinuous")

        if hb and wf:
            hb_pass = hb.value > THRESHOLDS["hebbian"]
            wf_pass = wf.value > THRESHOLDS["weber_fechner"]

            if hb_pass and wf_pass:
                verdict = ("Both Hebbian and Weber-Fechner signatures present — circuit "
                           "shows biologically-plausible learning and sensitivity dynamics")
            elif hb_pass:
                verdict = ("Hebbian learning without Weber-Fechner sensitivity — circuit "
                           "has associative learning structure but non-logarithmic "
                           "input transduction")
            elif wf_pass:
                verdict = ("Weber-Fechner sensitivity without Hebbian learning — circuit "
                           "has logarithmic input sensitivity but no co-activation-based "
                           "weight structure")
            else:
                verdict = ("Neither Hebbian nor Weber-Fechner signatures — circuit uses "
                           "non-biological computational primitives")
            lines.append(f"      VERDICT: {verdict}")

    # Cross-task summary
    hb_vals = [r.value for r in result.metrics.get("hebbian", [])]
    wf_vals = [r.value for r in result.metrics.get("weber_fechner", [])]

    if hb_vals and wf_vals and len(result.tasks) > 1:
        lines.append(f"\n    Cross-task consistency:")
        hb_cv = np.std(hb_vals) / max(abs(np.mean(hb_vals)), 1e-8)
        wf_cv = np.std(wf_vals) / max(abs(np.mean(wf_vals)), 1e-8)
        lines.append(f"      Hebbian CV = {hb_cv:.3f}  "
                     f"({'consistent' if hb_cv < 0.3 else 'variable'} across tasks)")
        lines.append(f"      Weber-Fechner CV = {wf_cv:.3f}  "
                     f"({'consistent' if wf_cv < 0.3 else 'variable'} across tasks)")
        if hb_cv < 0.3 and wf_cv < 0.3:
            lines.append(f"      Psychophysical properties are stable across tasks — "
                         f"likely reflect architectural rather than task-specific features")
        else:
            lines.append(f"      Psychophysical properties vary across tasks — "
                         f"may reflect task-specific computational strategies")

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

    lines.extend(psychophysics_analysis(result))

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
