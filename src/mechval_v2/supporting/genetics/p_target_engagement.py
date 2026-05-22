"""Protocol MB_TE — Target Engagement Verification
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Lens:         Genetics
Validity Type: Internal
Framework:    Molecular Biology — Target Engagement (CETSA)
Family:       Molecular Biology (Target Engagement Verification)
Validity:     Internal — intervention specificity and on-target verification

References:
    Martinez Molina et al. (2013) "Monitoring drug target engagement in
        cells and tissues using the cellular thermal shift assay"
    Jafari et al. (2014) "The cellular thermal shift assay for evaluating
        drug target interactions in cells" — MS-CETSA

Question:
    Does the intervention actually engage the intended circuit component?
    In drug discovery, CETSA verifies that the drug binds its intended
    protein target (not an off-target). Here, DAS/IIA tests whether the
    intervention engages the hypothesized causal variable. Misalignment
    measures off-target effects. Specificity is the ratio of on-target to
    off-target signal. High IIA + low misalignment + high specificity =
    confirmed target engagement.

Metrics:
    das_iia                  — On-target engagement (interchange intervention accuracy)
    intervention_specificity — Specificity ratio: on-target vs off-target effects
    misalignment             — Off-target risk: intervention hitting wrong components

Calibrations:
    STRUCTURAL_CALIBRATIONS

Usage:
    uv run python target_engagement.py                       # all tasks, CPU
    uv run python target_engagement.py --device cuda          # GPU
    uv run python target_engagement.py --tasks ioi induction  # specific tasks

    # As a callable module:
    from protocols.molecular_biology.target_engagement import run_protocol
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

PROTOCOL_ID = "MB_TE"
PROTOCOL_NAME = "Target Engagement Verification"
METRICS = ["intervention_specificity", "misalignment", "das_iia"]
CALIBRATIONS = STRUCTURAL_CALIBRATIONS
OUTPUT_DIR = Path(__file__).parent / "results" / "mb_te_target_engagement"

THRESHOLDS = {
    "das_iia": 0.7,
    "intervention_specificity": 0.6,
    "misalignment": 0.2,
}


def run_protocol(model, tasks: list[str] | None = None, n_prompts: int = 40,
                 device: str = "cpu", run_cals: bool = True) -> ProtocolResult:
    """Run all MB_TE metrics + calibrations. Returns a ProtocolResult."""
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


def engagement_analysis(result: ProtocolResult) -> list[str]:
    """Analyze results through target engagement (CETSA) lens.

    Biology analogy: CETSA measures whether a drug physically binds its
    intended protein target by monitoring thermal stability shifts.
    On-target score (IIA) = thermal shift at the target protein.
    Off-target risk (misalignment) = shifts at unrelated proteins.
    Specificity ratio = selectivity index. The ideal drug shows strong
    target binding with no off-target engagement.
    """
    lines = ["\n  Target Engagement Analysis:", "  ---------------------------"]

    for task in result.tasks:
        iia = _find(result.metrics.get("das_iia", []), task)
        spec = _find(result.metrics.get("intervention_specificity", []), task)
        mis = _find(result.metrics.get("misalignment", []), task)

        lines.append(f"\n    {task}:")

        if iia:
            if iia.value > 0.8:
                label = "strong engagement — high thermal shift"
            elif iia.value > 0.5:
                label = "moderate engagement"
            else:
                label = "weak engagement — drug may not bind target"
            lines.append(f"      On-target (DAS IIA):       {iia.value:.4f} — {label}")

        if mis:
            if mis.value < 0.1:
                label = "minimal off-target — clean selectivity"
            elif mis.value < 0.3:
                label = "some off-target — partial promiscuity"
            else:
                label = "high off-target — promiscuous binder"
            lines.append(f"      Off-target (misalignment): {mis.value:.4f} — {label}")

        if spec:
            if spec.value > 0.8:
                label = "highly selective"
            elif spec.value > 0.5:
                label = "moderately selective"
            else:
                label = "non-selective — broad spectrum"
            lines.append(f"      Specificity ratio:         {spec.value:.4f} — {label}")

        if iia and mis:
            on = iia.value
            off = mis.value
            if on > 0.7 and off < 0.2:
                verdict = "CONFIRMED ENGAGEMENT — on-target binding, minimal off-target"
            elif on > 0.7 and off > 0.3:
                verdict = "ENGAGED BUT PROMISCUOUS — strong binding with off-target risk"
            elif on < 0.4 and off < 0.2:
                verdict = "NO ENGAGEMENT — intervention does not reach target"
            elif on < 0.4 and off > 0.3:
                verdict = "OFF-TARGET DOMINANT — intervention engages wrong components"
            else:
                verdict = "PARTIAL ENGAGEMENT — moderate on-target with some off-target"
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

    lines.extend(engagement_analysis(result))

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
