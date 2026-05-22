"""Protocol MB_SA --- Sensitivity Analysis (E-Values)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Lens:         Genetics
Validity Type: Internal
Framework:    Molecular Biology --- Sensitivity Analysis
Family:       Molecular Biology (Sensitivity Analysis)
Validity:     External --- robustness of causal claims to unmeasured confounding

References:
    VanderWeele & Ding (2017) "Sensitivity analysis in observational
        research: introducing the E-value"
    Ding & VanderWeele (2016) "Sensitivity analysis without assumptions"

Question:
    How robust are our causal claims to unmeasured confounding? The E-value
    quantifies the minimum strength of association an unmeasured confounder
    would need with both the treatment (component) and outcome (behavior)
    to fully explain away an observed effect. In interpretability: RR is
    derived from the DAS-IIA score (how much intervening on a component
    changes behavior). An unmeasured confounder = a hidden variable
    (superposed feature, upstream component) that correlates with both the
    component and behavior but isn't the mechanism you think. The mediational
    E-value extends this to indirect effects through mediators.

    Parts G1-G2 from the Bio-Causal spec.

Metrics:
    das_iia              --- Distributed alignment search IIA score
    activation_patching  --- Activation patching effect
    effect_size          --- Causal effect size
    misalignment         --- Confounding indicator (misalignment between methods)

Calibrations:
    STRUCTURAL_CALIBRATIONS

Usage:
    uv run python sensitivity_analysis.py                       # all tasks, CPU
    uv run python sensitivity_analysis.py --device cuda          # GPU
    uv run python sensitivity_analysis.py --tasks ioi induction  # specific tasks

    # As a callable module:
    from protocols.molecular_biology.sensitivity_analysis import run_protocol
    result = run_protocol(model, tasks=["ioi"], n_prompts=40)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from mechval.metrics.common import CIRCUIT_TASKS, EvalResult, get_circuit_heads, load_model

from protocols import ProtocolResult, import_metric_runner
from protocols.calibration_runner import STRUCTURAL_CALIBRATIONS, run_calibrations, summarize_calibrations

PROTOCOL_ID = "MB_SA"
PROTOCOL_NAME = "Sensitivity Analysis (E-Values)"
METRICS = ["activation_patching", "effect_size", "misalignment", "das_iia"]
CALIBRATIONS = STRUCTURAL_CALIBRATIONS
OUTPUT_DIR = Path(__file__).parent / "results" / "mb_sa_sensitivity"

THRESHOLDS = {
    "das_iia": 0.6,
    "activation_patching": 0.5,
    "effect_size": 0.8,
    "misalignment": 0.2,
}


def run_protocol(model, tasks: list[str] | None = None, n_prompts: int = 40,
                 device: str = "cpu", run_cals: bool = True) -> ProtocolResult:
    """Run all MB_SA metrics + calibrations. Returns a ProtocolResult."""
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


def sensitivity_biology_analysis(result: ProtocolResult) -> list[str]:
    """Analyze results through Sensitivity Analysis (E-value) lens.

    Biology analogy: The E-value (VanderWeele & Ding, 2017) quantifies the
    minimum strength an unmeasured confounder would need to explain away an
    observed causal effect. For a relative risk RR, E = RR + sqrt(RR*(RR-1)).
    A large E-value means a very strong unmeasured confounder would be needed.

    In interpretability: RR is derived from the DAS-IIA score converted to a
    relative risk scale. An unmeasured confounder is a hidden variable
    (superposed feature, upstream component) that correlates with both the
    component and behavior but isn't the mechanism you think.

    The mediational E-value extends this to mediation: how much
    mediator-outcome confounding (Ding & VanderWeele, 2016) is needed to
    explain away the indirect effect.
    """
    lines = ["\n  Sensitivity Analysis (E-Values):", "  --------------------------------"]

    for task in result.tasks:
        iia_r = _find(result.metrics.get("das_iia", []), task)
        ap_r = _find(result.metrics.get("activation_patching", []), task)
        es_r = _find(result.metrics.get("effect_size", []), task)
        mis_r = _find(result.metrics.get("misalignment", []), task)

        lines.append(f"\n    {task}:")

        iia = iia_r.value if iia_r else 0.0
        misalignment = mis_r.value if mis_r else 1.0

        # Convert IIA to relative risk (iia_null ~ 0 for random baseline)
        iia_null = 0.0
        rr = max(1.0, (iia - iia_null) / (1.0 - iia_null + 1e-6) + 1)

        # E-value: minimum confounder strength to explain away effect
        e_val = rr + math.sqrt(rr * (rr - 1))

        lines.append(f"      DAS-IIA:               {iia:.4f}")
        if ap_r:
            lines.append(f"      Activation patching:   {ap_r.value:.4f}")
        if es_r:
            lines.append(f"      Effect size:           {es_r.value:.4f}")
        lines.append(f"      Misalignment:          {misalignment:.4f}")
        lines.append(f"      Relative risk (RR):    {rr:.4f}")
        lines.append(f"      E-value:               {e_val:.4f}")

        # Robustness level
        if e_val > 5:
            robustness = "VERY ROBUST — would need 5x confounder"
        elif e_val > 3:
            robustness = "ROBUST — would need 3x confounder"
        elif e_val > 2:
            robustness = "MODERATE — 2x confounder could explain"
        else:
            robustness = "FRAGILE — small confounder could explain away"
        lines.append(f"      Robustness:            {robustness}")

        # Verdict
        if misalignment > 0.5:
            verdict = "CONFOUNDING LIKELY"
        elif e_val < 2 or misalignment > 0.3:
            verdict = "FRAGILE — SENSITIVITY CONCERN"
        elif e_val > 3 and misalignment < 0.2:
            verdict = "ROBUST CAUSAL CLAIM"
        elif e_val > 2:
            verdict = "MODERATE ROBUSTNESS"
        else:
            verdict = "FRAGILE — SENSITIVITY CONCERN"
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

    lines.extend(sensitivity_biology_analysis(result))

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
