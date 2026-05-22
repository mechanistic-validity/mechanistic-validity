"""Protocol MB_MRX — Extended Mendelian Randomization
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Lens:         Genetics
Validity Type: Internal
Framework:    Molecular Biology — Extended Instrumental Variable Causal Inference
Family:       Molecular Biology (Extended Mendelian Randomization)
Validity:     External — multi-method causal triangulation via MR extensions

References:
    Bowden et al. (2015) "Mendelian randomization with invalid instruments:
        effect estimation and bias detection through Egger regression"
    Verbanck et al. (2018) "Detection of widespread horizontal pleiotropy
        in causal relationships inferred from Mendelian randomization
        between complex traits and diseases"
    Burgess & Thompson (2015) "Multivariable Mendelian randomization:
        the use of pleiotropic genetic variants to estimate causal effects"
        (Network MR framework)

Question:
    Beyond basic instrument validity, do extended MR methods (IVW, Egger,
    weighted median, PRESSO, bidirectional, network) converge on the same
    causal claim? IVW aggregates effect estimates. Egger detects directional
    pleiotropy (polysemanticity). Weighted median is robust to up to 50%
    invalid instruments. PRESSO detects outlier task variants. Bidirectional
    MR tests causal direction. Network MR tests mediated cascades.

Metrics:
    path_patching          — Instrument validity: position-specific causal paths
    cross_task_transfer    — Two-sample MR: train on one task, test on another
    das_iia                — Distributed alignment search interchange accuracy
    activation_patching    — Activation-level causal intervention
    mediation              — Mediated causal effects through component chains

Calibrations:
    STRUCTURAL_CALIBRATIONS

Usage:
    uv run python mr_extended.py                       # all tasks, CPU
    uv run python mr_extended.py --device cuda          # GPU
    uv run python mr_extended.py --tasks ioi induction  # specific tasks

    # As a callable module:
    from protocols.molecular_biology.mr_extended import run_protocol
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

PROTOCOL_ID = "MB_MRX"
PROTOCOL_NAME = "Extended Mendelian Randomization"
METRICS = ["activation_patching", "mediation", "path_patching", "cross_task_transfer", "das_iia"]
CALIBRATIONS = STRUCTURAL_CALIBRATIONS
OUTPUT_DIR = Path(__file__).parent / "results" / "mb_mrx_extended_mr"

THRESHOLDS = {
    "path_patching": 0.5,
    "cross_task_transfer": 0.4,
    "das_iia": 0.6,
    "activation_patching": 0.5,
    "mediation": 0.3,
}


def run_protocol(model, tasks: list[str] | None = None, n_prompts: int = 40,
                 device: str = "cpu", run_cals: bool = True) -> ProtocolResult:
    """Run all MB_MRX metrics + calibrations. Returns a ProtocolResult."""
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


def extended_mr_analysis(result: ProtocolResult) -> list[str]:
    """Analyze results through Extended Mendelian Randomization lens.

    Biology analogy: Extended MR applies multiple complementary methods to
    triangulate causal claims, each with different assumptions:

    IVW (Inverse-Variance Weighted): Aggregate multiple path_patching results
    across tasks using inverse-variance weighting. Cochran's Q tests for
    heterogeneity — high Q indicates the causal effect is not consistent
    across instruments (task variants).

    MR-Egger (Bowden et al. 2015): Tests for directional pleiotropy. If the
    Egger intercept differs significantly from 0, the component is
    "pleiotropic" (polysemantic — affecting behavior through multiple
    mechanisms, not just the claimed one).

    Weighted Median: The median of Wald-ratio-analogues across task variants.
    Robust even if up to 50% of "instruments" (task variants) are invalid.

    MR-PRESSO (Verbanck et al. 2018): Detects outlier task variants that
    violate instrument assumptions. Flags tasks where the causal estimate
    is inconsistent with the majority.

    Bidirectional MR: Tests causal direction between pairs of components.
    Compares forward vs reverse path_patching to determine whether A->B
    or B->A.

    Network MR (Burgess & Thompson 2015): Tests mediated effects through
    a chain of components (layer cascade). Checks whether mediation values
    show layered structure consistent with a causal chain.
    """
    lines = ["\n  Extended Mendelian Randomization Analysis:", "  ------------------------------------------"]

    pp_results = result.metrics.get("path_patching", [])
    ct_results = result.metrics.get("cross_task_transfer", [])
    das_results = result.metrics.get("das_iia", [])
    ap_results = result.metrics.get("activation_patching", [])
    med_results = result.metrics.get("mediation", [])

    verdicts = []

    # --- IVW: Inverse-Variance Weighted estimate ---
    lines.append("\n    IVW (Inverse-Variance Weighted):")
    if pp_results and das_results:
        pp_vals = np.array([r.value for r in pp_results])
        das_vals = np.array([r.value for r in das_results])
        all_vals = np.concatenate([pp_vals, das_vals])
        variances = np.var(all_vals) * np.ones(len(all_vals)) if np.var(all_vals) > 0 else np.ones(len(all_vals))
        weights = 1.0 / variances
        ivw_estimate = np.sum(weights * all_vals) / np.sum(weights)
        ivw_se = np.sqrt(1.0 / np.sum(weights))

        # Cochran's Q for heterogeneity
        cochran_q = np.sum(weights * (all_vals - ivw_estimate) ** 2)
        q_df = len(all_vals) - 1
        q_pval_approx = 1.0 - _chi2_cdf_approx(cochran_q, q_df) if q_df > 0 else 1.0

        lines.append(f"      IVW estimate:    {ivw_estimate:.4f} (SE={ivw_se:.4f})")
        lines.append(f"      Cochran's Q:     {cochran_q:.2f} (df={q_df}, p~{q_pval_approx:.4f})")
        heterogeneous = q_pval_approx < 0.05
        if heterogeneous:
            lines.append("      WARNING: significant heterogeneity detected")
    else:
        ivw_estimate = None
        lines.append("      insufficient data (need path_patching + das_iia)")

    # --- MR-Egger: directional pleiotropy ---
    lines.append("\n    MR-Egger (directional pleiotropy):")
    egger_intercept_sig = False
    if pp_results and das_results:
        x = np.array([r.value for r in pp_results[:len(das_results)]])
        y = np.array([r.value for r in das_results[:len(x)]])
        if len(x) >= 3:
            x_mean = np.mean(x)
            y_mean = np.mean(y)
            beta_egger = np.sum((x - x_mean) * (y - y_mean)) / (np.sum((x - x_mean) ** 2) + 1e-12)
            intercept_egger = y_mean - beta_egger * x_mean
            residuals = y - (intercept_egger + beta_egger * x)
            se_intercept = np.sqrt(np.sum(residuals ** 2) / (len(x) - 2) / (np.sum((x - x_mean) ** 2) + 1e-12))
            t_stat = abs(intercept_egger) / (se_intercept + 1e-12)
            egger_intercept_sig = t_stat > 2.0
            lines.append(f"      Egger intercept: {intercept_egger:.4f} (SE={se_intercept:.4f}, t={t_stat:.2f})")
            lines.append(f"      Egger slope:     {beta_egger:.4f}")
            if egger_intercept_sig:
                lines.append("      WARNING: intercept significantly differs from 0 — directional pleiotropy")
                verdicts.append("PLEIOTROPY WARNING")
            else:
                lines.append("      intercept consistent with 0 — no evidence of directional pleiotropy")
        else:
            lines.append("      insufficient tasks for Egger regression (need >= 3)")
    else:
        lines.append("      insufficient data (need path_patching + das_iia)")

    # --- Weighted Median ---
    lines.append("\n    Weighted Median:")
    if pp_results:
        pp_vals = np.array([r.value for r in pp_results])
        sorted_idx = np.argsort(pp_vals)
        sorted_vals = pp_vals[sorted_idx]
        weights_wm = np.ones(len(sorted_vals)) / len(sorted_vals)
        cum_weights = np.cumsum(weights_wm)
        median_idx = np.searchsorted(cum_weights, 0.5)
        median_idx = min(median_idx, len(sorted_vals) - 1)
        weighted_median = sorted_vals[median_idx]
        lines.append(f"      Weighted median estimate: {weighted_median:.4f}")
        lines.append(f"      (robust to up to 50% invalid instruments)")
    else:
        weighted_median = None
        lines.append("      insufficient data")

    # --- MR-PRESSO: outlier detection ---
    lines.append("\n    MR-PRESSO (outlier detection):")
    outlier_tasks = []
    if pp_results:
        pp_vals = np.array([r.value for r in pp_results])
        pp_mean = np.mean(pp_vals)
        pp_std = np.std(pp_vals)
        if pp_std > 0:
            residuals_presso = np.abs(pp_vals - pp_mean) / pp_std
            for i, r in enumerate(pp_results):
                task = r.metadata.get("task", "?")
                if residuals_presso[i] > 2.0:
                    outlier_tasks.append(task)
                    lines.append(f"      OUTLIER: {task} (residual={residuals_presso[i]:.2f} SD)")
            if not outlier_tasks:
                lines.append("      no outlier tasks detected (all within 2 SD)")
            else:
                verdicts.append("OUTLIER DETECTED")
        else:
            lines.append("      zero variance — cannot compute residuals")
    else:
        lines.append("      insufficient data")

    # --- Bidirectional MR ---
    lines.append("\n    Bidirectional MR (causal direction):")
    if pp_results and ap_results:
        pp_mean_val = np.mean([r.value for r in pp_results])
        ap_mean_val = np.mean([r.value for r in ap_results])
        lines.append(f"      Forward  (path_patching mean):       {pp_mean_val:.4f}")
        lines.append(f"      Reverse  (activation_patching mean): {ap_mean_val:.4f}")
        if pp_mean_val > ap_mean_val:
            lines.append("      Direction: forward path_patching > reverse — consistent with claimed direction")
            verdicts.append("BIDIRECTIONAL EVIDENCE")
        else:
            lines.append("      Direction: reverse >= forward — causal direction ambiguous")
    else:
        lines.append("      insufficient data (need path_patching + activation_patching)")

    # --- Network MR: cascade structure ---
    lines.append("\n    Network MR (cascade structure):")
    if med_results:
        med_vals = np.array([r.value for r in med_results])
        med_tasks = [r.metadata.get("task", "?") for r in med_results]
        lines.append(f"      Mediation values: {', '.join(f'{v:.4f}' for v in med_vals)}")
        above_threshold = sum(1 for v in med_vals if v > THRESHOLDS["mediation"])
        total = len(med_vals)
        lines.append(f"      Above threshold ({THRESHOLDS['mediation']}): {above_threshold}/{total}")
        if above_threshold > total * 0.5:
            lines.append("      CASCADE: majority of tasks show mediated effects — layered causal structure")
            verdicts.append("CASCADE STRUCTURE")
        else:
            lines.append("      no clear cascade structure detected")
    else:
        lines.append("      insufficient data (need mediation)")

    # --- Convergence assessment ---
    lines.append("\n    Convergence Assessment:")
    if ivw_estimate is not None and weighted_median is not None and not egger_intercept_sig:
        ivw_wm_agree = abs(ivw_estimate - weighted_median) < 0.15
        if ivw_wm_agree:
            lines.append("      IVW, Egger (no pleiotropy), and weighted median CONVERGE")
            verdicts.append("CONVERGENT")
        else:
            lines.append(f"      IVW ({ivw_estimate:.4f}) and weighted median ({weighted_median:.4f}) diverge")
    elif egger_intercept_sig:
        lines.append("      Egger intercept significant — convergence compromised by pleiotropy")
    else:
        lines.append("      insufficient data for convergence assessment")

    # --- Per-task summary ---
    for task in result.tasks:
        pp = _find(pp_results, task)
        ct = _find(ct_results, task)
        das = _find(das_results, task)
        ap = _find(ap_results, task)
        med = _find(med_results, task)

        lines.append(f"\n    {task}:")

        if pp:
            label = "valid instrument" if pp.value > THRESHOLDS["path_patching"] else "weak/invalid"
            lines.append(f"      path_patching:       {pp.value:.4f} — {label}")
        if ct:
            label = "consistent" if ct.value > THRESHOLDS["cross_task_transfer"] else "inconsistent"
            lines.append(f"      cross_task_transfer: {ct.value:.4f} — {label}")
        if das:
            label = "aligned" if das.value > THRESHOLDS["das_iia"] else "misaligned"
            lines.append(f"      das_iia:             {das.value:.4f} — {label}")
        if ap:
            label = "causal" if ap.value > THRESHOLDS["activation_patching"] else "non-causal"
            lines.append(f"      activation_patching: {ap.value:.4f} — {label}")
        if med:
            label = "mediated" if med.value > THRESHOLDS["mediation"] else "direct/absent"
            lines.append(f"      mediation:           {med.value:.4f} — {label}")

        if task in outlier_tasks:
            lines.append(f"      ** OUTLIER per MR-PRESSO **")

    # --- Overall verdicts ---
    lines.append(f"\n    VERDICTS: {', '.join(verdicts) if verdicts else 'NO CLEAR VERDICT'}")

    return lines


def _chi2_cdf_approx(x: float, df: int) -> float:
    """Approximate chi-squared CDF using the Wilson-Hilferty normal approximation."""
    if df <= 0:
        return 0.0
    z = ((x / df) ** (1 / 3) - (1 - 2 / (9 * df))) / np.sqrt(2 / (9 * df))
    return 0.5 * (1 + _erf_approx(z / np.sqrt(2)))


def _erf_approx(x: float) -> float:
    """Approximate error function (Abramowitz & Stegun 7.1.26)."""
    sign = 1 if x >= 0 else -1
    x = abs(x)
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
    return sign * y


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

    lines.extend(extended_mr_analysis(result))

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
