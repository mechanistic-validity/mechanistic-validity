"""Protocol MB_MD -- Causal Mediation Analysis
----------------------------------------------------------------------
Lens:         Genetics
Validity Type: Internal
Framework:    Molecular Biology -- Causal Mediation
Family:       Molecular Biology (Causal Mediation Analysis)
Validity:     Internal -- decomposing total causal effects into paths

References:
    Baron & Kenny (1986) "The moderator-mediator variable distinction
        in social psychological research"
    Pearl (2001) "Direct and indirect effects"
    VanderWeele (2015) "Explanation in Causal Inference: Methods for
        Mediation and Interaction"

Question:
    Can we decompose the total causal effect of a circuit component
    into direct and indirect (mediated) paths? Causal mediation
    separates the Natural Indirect Effect (NIE, through the mediator)
    from the Natural Direct Effect (NDE, bypassing it). Path-specific
    effects trace signal along individual causal paths, analogous to
    tracing a signaling cascade in biology.

    Parts B1-B4 from the Bio-Causal spec:
      B1. NDE/NIE decomposition via mediation / mediation_v2
      B2. CDE (Controlled Direct Effect) via path_patching
      B3. PSE (Path-Specific Effects) via pse metric
      B4. G-formula: multi-layer mediated effect estimation

Metrics:
    mediation       -- NDE/NIE decomposition of total causal effect
    mediation_v2    -- Improved NDE/NIE with counterfactual correction
    pse             -- Path-specific effect decomposition
    path_patching   -- Controlled direct effect estimation

Calibrations:
    STRUCTURAL_CALIBRATIONS

Usage:
    uv run python mediation_biology.py                       # all tasks, CPU
    uv run python mediation_biology.py --device cuda          # GPU
    uv run python mediation_biology.py --tasks ioi induction  # specific tasks

    # As a callable module:
    from protocols.molecular_biology.mediation_biology import run_protocol
    result = run_protocol(model, tasks=["ioi"], n_prompts=40)
----------------------------------------------------------------------
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

PROTOCOL_ID = "MB_MD"
PROTOCOL_NAME = "Causal Mediation Analysis"
METRICS = ["mediation", "mediation_v2", "pse", "path_patching"]
CALIBRATIONS = STRUCTURAL_CALIBRATIONS
OUTPUT_DIR = Path(__file__).parent / "results" / "mb_md_mediation"

THRESHOLDS = {
    "mediation": 0.3,
    "mediation_v2": 0.3,
    "pse": 0.2,
    "path_patching": 0.5,
}


def run_protocol(model, tasks: list[str] | None = None, n_prompts: int = 40,
                 device: str = "cpu", run_cals: bool = True) -> ProtocolResult:
    """Run all MB_MD metrics + calibrations. Returns a ProtocolResult."""
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


def mediation_biology_analysis(result: ProtocolResult) -> list[str]:
    """Analyze results through Causal Mediation lens.

    Biology analogy: mediation analysis decomposes a total causal effect
    into the portion traveling through a mediator (NIE, like a signaling
    cascade relay) and the portion bypassing it (NDE, a direct pathway).

    B1. NDE/NIE: mediation and mediation_v2 give the fraction of the
        total effect that is mediated (NIE / (NIE + NDE)). High values
        mean the component is a true relay in the causal chain.
    B2. CDE: path_patching estimates the controlled direct effect --
        what happens when the mediator is clamped. If CDE differs from
        NDE, there is interaction between direct and indirect paths.
    B3. PSE: pse decomposes effects along specific causal paths, like
        tracing which branch of a signaling cascade carries the signal.
    B4. G-formula: multi-layer mediation reconstructs the full causal
        path decomposition across layers.
    """
    lines = ["\n  Causal Mediation Analysis:", "  --------------------------"]

    for task in result.tasks:
        med = _find(result.metrics.get("mediation", []), task)
        med2 = _find(result.metrics.get("mediation_v2", []), task)
        pse = _find(result.metrics.get("pse", []), task)
        pp = _find(result.metrics.get("path_patching", []), task)

        lines.append(f"\n    {task}:")

        # -- B1: NDE/NIE decomposition --
        mediation_val = None
        if med:
            mediation_val = med.value
            if mediation_val > 0.7:
                frac_label = "complete mediation"
            elif mediation_val > 0.1:
                frac_label = "partial mediation"
            else:
                frac_label = "no mediation"
            lines.append(f"      Mediation fraction (mediation):        {med.value:.4f} — {frac_label}")

        if med2:
            if mediation_val is None:
                mediation_val = med2.value
            if med2.value > 0.7:
                frac_label = "complete mediation"
            elif med2.value > 0.1:
                frac_label = "partial mediation"
            else:
                frac_label = "no mediation"
            lines.append(f"      Mediation fraction (mediation_v2):     {med2.value:.4f} — {frac_label}")

        # -- B3: Path-specific effects --
        if pse:
            label = "strong path" if pse.value > 0.4 else ("moderate path" if pse.value > 0.2 else "weak path")
            lines.append(f"      Path-specific effect (pse):             {pse.value:.4f} — {label}")

        # -- B2: Controlled direct effect --
        pp_val = None
        if pp:
            pp_val = pp.value
            label = "strong direct" if pp.value > 0.5 else ("moderate direct" if pp.value > 0.2 else "weak direct")
            lines.append(f"      Controlled direct effect (path_patch):  {pp.value:.4f} — {label}")

        # -- Consistency check: mediation vs path_patching --
        if med and pp:
            if med.value > 0.3 and pp.value > 0.5:
                lines.append("      Consistency: mediation + path_patching both high — true mediator")
            elif med.value < 0.1 and pp.value < 0.2:
                lines.append("      Consistency: both low — not involved in this pathway")
            elif med.value > 0.3 and pp.value < 0.2:
                lines.append("      Consistency: mediation high but path_patching low — confounded mediation?")
            elif med.value < 0.1 and pp.value > 0.5:
                lines.append("      Consistency: path_patching high but mediation low — direct effect dominant")

        # -- Verdict --
        med_high = mediation_val is not None and mediation_val > 0.7
        med_mid = mediation_val is not None and 0.2 < mediation_val <= 0.7
        med_low = mediation_val is not None and mediation_val <= 0.2
        pp_high = pp_val is not None and pp_val > 0.5
        pp_low = pp_val is None or pp_val <= 0.5

        if med_high and pp_high:
            verdict = "COMPLETE MEDIATOR"
        elif med_mid:
            verdict = "PARTIAL MEDIATOR"
        elif med_low and pp_high:
            verdict = "DIRECT EFFECT DOMINANT"
        elif med_low and pp_low:
            verdict = "NOT A MEDIATOR"
        elif mediation_val is None and pp_val is None:
            verdict = "INSUFFICIENT DATA"
        else:
            verdict = "PARTIAL MEDIATOR" if med_mid else "NOT A MEDIATOR"

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

    lines.extend(mediation_biology_analysis(result))

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
