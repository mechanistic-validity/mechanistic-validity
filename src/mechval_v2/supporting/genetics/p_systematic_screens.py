"""Protocol MB_SS — Systematic Perturbation Screens
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Lens:         Genetics
Validity Type: Internal
Framework:    Molecular Biology — Genome-wide Screens
Family:       Molecular Biology (Systematic Perturbation Screens)
Validity:     Internal — unbiased component-level discovery

References:
    Replogle et al. (2022) "Mapping information-rich genotype-phenotype
        landscapes with genome-scale Perturb-seq"
    Conmy et al. (2023) "Towards Automated Circuit Discovery for
        Mechanistic Interpretability" — ACDC

Question:
    Can we systematically screen all circuit components to identify causal
    ones, analogous to genome-wide CRISPR screens? Automated discovery
    methods (EAP, ATP*, activation patching) each perturb every component
    and measure phenotypic effect. Agreement between methods reduces
    false discovery risk.

Metrics:
    eap                 — Edge Attribution Patching: gradient-based screen
    atp_star            — Attribution Patching (ATP*): linearized approximation
    activation_patching — Gold-standard per-head ablation screen

Calibrations:
    STRUCTURAL_CALIBRATIONS

Usage:
    uv run python systematic_screens.py                       # all tasks, CPU
    uv run python systematic_screens.py --device cuda          # GPU
    uv run python systematic_screens.py --tasks ioi induction  # specific tasks

    # As a callable module:
    from protocols.molecular_biology.systematic_screens import run_protocol
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

PROTOCOL_ID = "MB_SS"
PROTOCOL_NAME = "Systematic Perturbation Screens"
METRICS = ["eap", "atp_star", "activation_patching"]
CALIBRATIONS = STRUCTURAL_CALIBRATIONS
OUTPUT_DIR = Path(__file__).parent / "results" / "mb_ss_systematic_screens"

THRESHOLDS = {
    "eap": 0.5,
    "atp_star": 0.5,
    "activation_patching": 0.7,
}


def run_protocol(model, tasks: list[str] | None = None, n_prompts: int = 40,
                 device: str = "cpu", run_cals: bool = True) -> ProtocolResult:
    """Run all MB_SS metrics + calibrations. Returns a ProtocolResult."""
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


def screen_analysis(result: ProtocolResult) -> list[str]:
    """Analyze results through genome-wide screen lens.

    Biology analogy: CRISPR screens perturb every gene and measure
    phenotype. Hit rate = fraction of components identified as causal.
    Agreement between screen methods (like comparing CRISPR-KO vs CRISPRi
    vs shRNA) reduces false positives. Discordance flags method-specific
    artifacts.
    """
    lines = ["\n  Systematic Screen Analysis:", "  ---------------------------"]

    for task in result.tasks:
        eap = _find(result.metrics.get("eap", []), task)
        atp = _find(result.metrics.get("atp_star", []), task)
        ap = _find(result.metrics.get("activation_patching", []), task)

        lines.append(f"\n    {task}:")

        screen_results = []
        if eap:
            lines.append(f"      EAP screen (gradient):     {eap.value:.4f}")
            screen_results.append(("EAP", eap.value))

        if atp:
            lines.append(f"      ATP* screen (linearized):  {atp.value:.4f}")
            screen_results.append(("ATP*", atp.value))

        if ap:
            lines.append(f"      Patching screen (direct):  {ap.value:.4f}")
            screen_results.append(("patching", ap.value))

        if len(screen_results) >= 2:
            vals = [v for _, v in screen_results]
            mean_val = np.mean(vals)
            spread = max(vals) - min(vals)

            lines.append(f"      Mean hit score:            {mean_val:.4f}")
            lines.append(f"      Method spread:             {spread:.4f}")

            if spread < 0.15:
                agreement = "HIGH AGREEMENT — screens converge (low FDR)"
            elif spread < 0.3:
                agreement = "MODERATE AGREEMENT — some method-specific hits"
            else:
                agreement = "LOW AGREEMENT — possible method artifacts"
            lines.append(f"      VERDICT: {agreement}")

    return lines


def _find(results: list[EvalResult], task: str) -> EvalResult | None:
    return next((r for r in results if r.metadata.get("task") == task), None)


def summarize(result: ProtocolResult) -> str:
    lines = []
    lines.append(f"\n{'=' * 70}")
    lines.append(f"  PROTOCOL {result.protocol_id}: {result.protocol_name}")
    lines.append(f"{'=' * 70}\n")

    header = f"{'Task':20s}" + "".join(f"  {m:>20s}" for m in METRICS)
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
                row += f"  {v:>16.4f}{tag}"
            else:
                row += f"  {'---':>20s}"
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

    lines.extend(screen_analysis(result))

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
