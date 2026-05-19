"""Measurement Invariance
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     F05 — Measurement Invariance
Categories:     measurement
Validity layer: Measurement
Criteria:       M2 Invariance
Establishes:    Circuit scores are stable across prompt templates
Requires:       GPU, model
Doc:            /instruments_v2/measurement/f05-measurement-invariance
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Tests whether circuit faithfulness scores are stable across prompt
templates -- the MI equivalent of measurement invariance in confirmatory
factor analysis.

A circuit that scores 0.87 on template A and 0.31 on template B is not
reliably measuring "the circuit." No existing MI paper tests this
systematically.

Optional: pip install pingouin (for Welch ANOVA + eta-squared).
Falls back to scipy ANOVA.

Usage:
    uv run python 13_measurement_invariance.py --tasks ioi sva
"""

import numpy as np
import torch
from scipy import stats as sp_stats

from mechanistic_validity.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    calibrate_mean_z,
    compute_faithfulness,
    generate_prompts,
    get_circuit_heads,
    get_token_ids,
    load_model,
    log,
    parse_common_args,
    save_results,
)

try:
    import pingouin as pg
    HAS_PINGOUIN = True
except ImportError:
    HAS_PINGOUIN = False


def split_by_template(prompts, correct_ids, incorrect_ids, n_groups: int = 3):
    """Split prompts into template families based on structural features.

    Groups by: prompt length (short/medium/long), or if metadata has
    template info, uses that.
    """
    lengths = [len(p.text.split()) for p in prompts]
    n = len(prompts)
    if n < n_groups * 3:
        return None

    sorted_idx = np.argsort(lengths)
    group_size = n // n_groups
    groups = {}
    labels = ["short", "medium", "long"][:n_groups]

    for g in range(n_groups):
        start = g * group_size
        end = start + group_size if g < n_groups - 1 else n
        indices = sorted_idx[start:end].tolist()
        groups[labels[g]] = {
            "prompts": [prompts[i] for i in indices],
            "correct_ids": [correct_ids[i] for i in indices],
            "incorrect_ids": [incorrect_ids[i] for i in indices],
        }

    return groups


def split_by_metadata(prompts, correct_ids, incorrect_ids):
    """Try to split by metadata-based template families."""
    groups = {}
    for i, p in enumerate(prompts):
        if i >= len(correct_ids):
            break
        meta = getattr(p, "metadata", {}) or {}
        template = meta.get("template", meta.get("type", meta.get("category", "default")))
        template = str(template)
        if template not in groups:
            groups[template] = {"prompts": [], "correct_ids": [], "incorrect_ids": []}
        groups[template]["prompts"].append(p)
        groups[template]["correct_ids"].append(correct_ids[i])
        groups[template]["incorrect_ids"].append(incorrect_ids[i])

    groups = {k: v for k, v in groups.items() if len(v["prompts"]) >= 3}
    return groups if len(groups) >= 2 else None


def run_measurement_invariance(model, tasks: list[str],
                                n_prompts: int = 60) -> list[EvalResult]:
    tokenizer = model.tokenizer
    results = []

    for task in tasks:
        circuit_heads = get_circuit_heads(task)
        if not circuit_heads:
            log(f"  {task}: no circuit, skipping")
            continue

        prompts = generate_prompts(task, tokenizer, n_prompts)
        if not prompts:
            continue

        correct_ids, incorrect_ids = get_token_ids(prompts, tokenizer)
        if not correct_ids:
            continue

        log(f"  {task} ({len(circuit_heads)} heads, {len(prompts)} prompts)...")

        groups = split_by_metadata(prompts, correct_ids, incorrect_ids)
        if groups is None:
            groups = split_by_template(prompts, correct_ids, incorrect_ids)
        if groups is None:
            log(f"    Cannot split into template groups, skipping")
            continue

        mean_z = calibrate_mean_z(model, prompts, n_calibration=min(100, len(prompts)))

        group_scores = {}
        all_scores = []
        group_labels = []

        for group_name, group_data in groups.items():
            faith = compute_faithfulness(
                model, group_data["prompts"], group_data["correct_ids"],
                group_data["incorrect_ids"], circuit_heads, mean_z,
            )
            group_scores[group_name] = {
                "faithfulness": faith,
                "n_prompts": len(group_data["prompts"]),
            }
            all_scores.append(faith)
            group_labels.append(group_name)
            log(f"    {group_name}: faith={faith:.3f} (n={len(group_data['prompts'])})")

        if len(all_scores) < 2:
            continue

        score_range = max(all_scores) - min(all_scores)
        score_std = float(np.std(all_scores))

        eta_sq = None
        f_stat = None
        p_value = None

        if HAS_PINGOUIN and len(groups) >= 2:
            import pandas as pd
            rows = []
            for group_name, group_data in groups.items():
                for j in range(len(group_data["prompts"])):
                    tokens = model.to_tokens(group_data["prompts"][j].text)
                    from mechanistic_validity.metrics.common import logit_diff_from_logits
                    clean_ld = logit_diff_from_logits(
                        model(tokens), group_data["correct_ids"][j],
                        group_data["incorrect_ids"][j])
                    rows.append({"group": group_name, "logit_diff": clean_ld})
            df = pd.DataFrame(rows)
            try:
                aov = pg.welch_anova(data=df, dv="logit_diff", between="group")
                eta_sq = float(aov["np2"].values[0])
                f_stat = float(aov["F"].values[0])
                p_value = float(aov["p-unc"].values[0])
            except Exception:
                pass
        elif len(groups) >= 2:
            group_arrays = []
            for group_data in groups.values():
                arr = []
                for j in range(len(group_data["prompts"])):
                    tokens = model.to_tokens(group_data["prompts"][j].text)
                    from mechanistic_validity.metrics.common import logit_diff_from_logits
                    ld = logit_diff_from_logits(
                        model(tokens), group_data["correct_ids"][j],
                        group_data["incorrect_ids"][j])
                    arr.append(ld)
                group_arrays.append(arr)
            f_stat_val, p_val = sp_stats.f_oneway(*group_arrays)
            f_stat = float(f_stat_val)
            p_value = float(p_val)
            ss_between = sum(len(g) * (np.mean(g) - np.mean([x for ga in group_arrays for x in ga])) ** 2
                            for g in group_arrays)
            ss_total = sum((x - np.mean([x for ga in group_arrays for x in ga])) ** 2
                          for ga in group_arrays for x in ga)
            eta_sq = ss_between / ss_total if ss_total > 0 else 0.0

        invariance = "invariant" if (eta_sq is not None and eta_sq < 0.01) else \
                     "moderate" if (eta_sq is not None and eta_sq < 0.06) else \
                     "template_sensitive"

        log(f"    eta²={eta_sq:.4f} -> {invariance}" if eta_sq else
            f"    range={score_range:.3f}")

        results.append(EvalResult(
            metric_id="C13.measurement_invariance",
            value=eta_sq if eta_sq is not None else score_range,
            n_samples=len(prompts),
            metadata={
                "task": task,
                "group_scores": group_scores,
                "eta_squared": eta_sq,
                "f_statistic": f_stat,
                "p_value": p_value,
                "invariance_verdict": invariance,
                "score_range": score_range,
                "score_std": score_std,
                "n_groups": len(groups),
                "group_names": list(groups.keys()),
            },
        ))

    return results


def main():
    parser = parse_common_args("C13: Measurement Invariance")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("C13: MEASUREMENT INVARIANCE ACROSS TEMPLATES")
    log("=" * 60)

    if not HAS_PINGOUIN:
        log("NOTE: pingouin not installed. Using scipy ANOVA fallback.")

    results = run_measurement_invariance(model, tasks, args.n_prompts)

    out = args.out or "13_measurement_invariance.json"
    save_results(results, out)

    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
