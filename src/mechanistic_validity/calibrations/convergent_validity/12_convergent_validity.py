"""Convergent Validity
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     F02 — Convergent Validity
Categories:     measurement
Validity layer: Measurement
Criteria:       C5 Convergent validity
Establishes:    Different metric families agree on which heads are circuit components
Requires:       CPU, data-only
Doc:            /instruments_v2/measurement/f02-convergent-validity
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Measures Spearman correlation between rankings produced by different
metric families (C2 patching, C7 oCSE, 10 LLC) over the same set of
heads. High cross-metric agreement = construct validity for "being a
circuit component."

Optional: pip install pingouin (for ICC). Falls back to pairwise
Spearman.

Usage:
    uv run python 12_convergent_validity.py --tasks ioi sva
    uv run python 12_convergent_validity.py --metrics 02 07 10
"""
import json

import numpy as np
from scipy import stats as sp_stats

from mechanistic_validity.metrics.common import (
    CIRCUIT_TASKS,
    DATA_DIR,
    EvalResult,
    log,
    parse_common_args,
    save_results,
)

try:
    import pingouin as pg
    HAS_PINGOUIN = True
except ImportError:
    HAS_PINGOUIN = False


def load_metric_scores(metric_file: str, task: str) -> dict[str, float] | None:
    """Load per-head scores from a metric result file.

    Returns dict mapping "L{l}H{h}" -> score, or None if not available.
    """
    path = DATA_DIR / metric_file
    if not path.exists():
        return None

    with open(path) as f:
        data = json.load(f)

    if isinstance(data, list):
        for entry in data:
            meta = entry.get("metadata", {})
            if meta.get("task") == task:
                per_head = meta.get("per_head_effects") or meta.get("per_head_llc") or meta.get("per_head_iia")
                if per_head:
                    return per_head
                discovered = meta.get("discovered_parents", [])
                if discovered:
                    return {d["head"]: d.get("score", 1.0) for d in discovered}

    return None


METRIC_FILES = {
    "02": "02_activation_patching.json",
    "05": "05_mediation.json",
    "07": "07_ocse.json",
    "10": "10_llc.json",
}


def compute_convergent_validity(task: str, metrics: list[str] | None = None) -> EvalResult | None:
    """Compute cross-metric agreement for a single task."""
    if metrics is None:
        metrics = list(METRIC_FILES.keys())

    score_dicts = {}
    for metric in metrics:
        if metric not in METRIC_FILES:
            continue
        scores = load_metric_scores(METRIC_FILES[metric], task)
        if scores:
            score_dicts[metric] = scores

    if len(score_dicts) < 2:
        log(f"  {task}: need >= 2 metrics with per-head scores, got {len(score_dicts)}")
        return None

    all_heads = set()
    for scores in score_dicts.values():
        all_heads.update(scores.keys())
    all_heads = sorted(all_heads)

    metric_names = sorted(score_dicts.keys())
    n_metrics = len(metric_names)
    n_heads = len(all_heads)

    score_matrix = np.zeros((n_heads, n_metrics))
    for j, metric in enumerate(metric_names):
        for i, head in enumerate(all_heads):
            score_matrix[i, j] = score_dicts[metric].get(head, 0.0)

    spearman_pairs = {}
    for a_idx in range(n_metrics):
        for b_idx in range(a_idx + 1, n_metrics):
            a_name = metric_names[a_idx]
            b_name = metric_names[b_idx]
            rho, p = sp_stats.spearmanr(score_matrix[:, a_idx], score_matrix[:, b_idx])
            spearman_pairs[f"{a_name}_vs_{b_name}"] = {
                "rho": float(rho), "p_value": float(p),
            }

    mean_rho = float(np.mean([v["rho"] for v in spearman_pairs.values()]))

    icc_value = None
    if HAS_PINGOUIN and n_metrics >= 2 and n_heads >= 3:
        import pandas as pd
        rows = []
        for i, head in enumerate(all_heads):
            for j, metric in enumerate(metric_names):
                rows.append({"head": head, "metric": metric, "score": score_matrix[i, j]})
        df = pd.DataFrame(rows)
        try:
            icc_result = pg.intraclass_corr(data=df, targets="head",
                                             raters="metric", ratings="score")
            icc_21 = icc_result[icc_result["Type"] == "ICC2"]["ICC"].values
            if len(icc_21) > 0:
                icc_value = float(icc_21[0])
        except Exception:
            pass

    log(f"  {task}: mean_rho={mean_rho:.3f} ICC={icc_value:.3f}" if icc_value else
        f"  {task}: mean_rho={mean_rho:.3f}")

    return EvalResult(
        metric_id="C12.convergent_validity",
        value=mean_rho,
        n_samples=n_heads,
        metadata={
            "task": task,
            "spearman_pairs": spearman_pairs,
            "icc": icc_value,
            "metrics_used": metric_names,
            "n_heads": n_heads,
        },
    )


def run_convergent_validity(tasks: list[str],
                             metrics: list[str] | None = None) -> list[EvalResult]:
    results = []
    for task in tasks:
        result = compute_convergent_validity(task, metrics)
        if result:
            results.append(result)
    return results


def main():
    parser = parse_common_args("C12: Convergent Validity")
    parser.add_argument("--metrics", nargs="+", default=None,
                        help="Which metrics to compare (default: all available)")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS

    log("=" * 60)
    log("C12: CONVERGENT VALIDITY (Cross-Metric Agreement)")
    log("=" * 60)

    available = [f for f in METRIC_FILES.values() if (DATA_DIR / f).exists()]
    if len(available) < 2:
        log(f"Need >= 2 metric result files. Found: {available}")
        log(f"Run C2, C7, 10 first to generate per-head scores.")
        return

    if not HAS_PINGOUIN:
        log("NOTE: pingouin not installed. Skipping ICC, using Spearman only.")

    results = run_convergent_validity(tasks, args.metrics)

    out = args.out or "12_convergent_validity.json"
    save_results(results, out)

    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
