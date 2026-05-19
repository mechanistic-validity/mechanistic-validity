"""Derived Metrics (no forward passes required)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     D01 — Logit Diff Recovery
Categories:     behavioral
Validity layer: Internal
Criteria:       I2 Sufficiency
Establishes:    Computes derived ratios, d-primes, and signal-detection statistics from prior evals
Requires:       CPU, data-only
Doc:            /instruments_v2/behavioral/d01-logit-diff-recovery
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Reads existing JSON data files from 01-13 evaluations and the
unexplored_pillars experiments, then computes ~20 derived metrics
(ratios, d-primes, signal-detection statistics) that require only
arithmetic on previously computed values.

Usage:
    uv run python 14_derived_metrics.py
    uv run python 14_derived_metrics.py --tasks ioi sva
"""
import json
import math
from pathlib import Path

import numpy as np

from mechanistic_validity.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    get_circuit_heads,
    load_results,
    log,
    parse_common_args,
    save_results,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PILLAR_DATA = SCRIPT_DIR.parents[1] / "unexplored_pillars" / "data"

GPT2_TOTAL_HEADS = 144  # 12 layers x 12 heads

# Published circuit sizes from Wang et al. (2022), Hanna et al. (2023), etc.
PUBLISHED_CIRCUIT_SIZES = {
    "ioi": 26,
    "greater_than": 12,
    "induction": 5,
    "sva": 18,
    "gendered_pronoun": 7,
}

# Weight-classifier performance (from part4 experiments)
WEIGHT_CLASSIFIER_RESULTS = {
    "ioi": {"recall": 0.93, "precision": 0.87},
    "sva": {"recall": 1.0, "precision": 0.92},
    "gendered_pronoun": {"recall": 1.0, "precision": 0.83},
    "rti": {"recall": 1.0, "precision": 0.73},
    "greater_than": {"recall": 1.0, "precision": 0.86},
    "induction": {"recall": 1.0, "precision": 0.71},
    "acronym": {"recall": 1.0, "precision": 0.75},
    "copy_suppression": {"recall": 1.0, "precision": 0.86},
}


def _load_pillar_json(filename: str):
    """Load a JSON file from the unexplored_pillars/data directory."""
    path = PILLAR_DATA / filename
    if not path.exists():
        log(f"  Pillar file not found: {path.name}")
        return None
    with open(path) as f:
        return json.load(f)


def _d_prime(mean_signal: float, mean_noise: float,
             std_signal: float, std_noise: float) -> float:
    """Compute d-prime (signal detection discriminability)."""
    pooled_std = math.sqrt((std_signal ** 2 + std_noise ** 2) / 2.0)
    if pooled_std < 1e-12:
        return 0.0
    return (mean_signal - mean_noise) / pooled_std


def _compute_sparsity(tasks: list[str]) -> list[EvalResult]:
    """C14.sparsity = n_circuit_heads / 144."""
    results = []
    for task in tasks:
        heads = get_circuit_heads(task)
        if not heads:
            continue
        sparsity = len(heads) / GPT2_TOTAL_HEADS
        log(f"  {task}: sparsity = {len(heads)}/{GPT2_TOTAL_HEADS} = {sparsity:.4f}")
        results.append(EvalResult(
            metric_id="C14.sparsity",
            value=sparsity,
            n_samples=GPT2_TOTAL_HEADS,
            metadata={
                "task": task,
                "n_circuit_heads": len(heads),
                "n_total_heads": GPT2_TOTAL_HEADS,
            },
        ))
    return results


def _compute_node_overlap(tasks: list[str]) -> list[EvalResult]:
    """C14.node_overlap_jaccard = Jaccard of our circuit vs. published size."""
    results = []
    for task in tasks:
        heads = get_circuit_heads(task)
        if not heads:
            continue
        pub_size = PUBLISHED_CIRCUIT_SIZES.get(task)
        if pub_size is None:
            continue
        our_size = len(heads)
        # Upper bound on Jaccard: min(A,B) / max(A,B) -- without knowing
        # exact published head identities, report size ratio as proxy
        size_ratio = min(our_size, pub_size) / max(our_size, pub_size)
        log(f"  {task}: size_ratio = {our_size}/{pub_size} = {size_ratio:.3f}")
        results.append(EvalResult(
            metric_id="C14.node_overlap_jaccard",
            value=size_ratio,
            metadata={
                "task": task,
                "our_circuit_size": our_size,
                "published_circuit_size": pub_size,
                "note": "Size ratio proxy -- exact Jaccard requires head identity matching",
            },
        ))
    return results


def _compute_spectral_metrics(tasks: list[str]) -> list[EvalResult]:
    """C14.spectral_norm_ratio and C14.sv_entropy_ratio from effective_rank_wov.json."""
    data = _load_pillar_json("effective_rank_wov.json")
    if data is None:
        return []

    per_head = data.get("per_head", {})
    if not per_head:
        return []

    results = []
    for task in tasks:
        heads = get_circuit_heads(task)
        if not heads:
            continue

        head_keys = {f"L{l}H{h}" for l, h in heads}
        circuit_specs = []
        circuit_ents = []
        noncircuit_specs = []
        noncircuit_ents = []

        for hk, hv in per_head.items():
            sn = hv.get("spectral_norm")
            se = hv.get("sv_entropy")
            if sn is None or se is None:
                continue
            if hk in head_keys:
                circuit_specs.append(sn)
                circuit_ents.append(se)
            else:
                noncircuit_specs.append(sn)
                noncircuit_ents.append(se)

        if not circuit_specs or not noncircuit_specs:
            continue

        spec_ratio = float(np.mean(circuit_specs) / np.mean(noncircuit_specs))
        ent_ratio = float(np.mean(circuit_ents) / np.mean(noncircuit_ents))

        log(f"  {task}: spectral_norm_ratio={spec_ratio:.4f}, sv_entropy_ratio={ent_ratio:.4f}")

        results.append(EvalResult(
            metric_id="C14.spectral_norm_ratio",
            value=spec_ratio,
            metadata={
                "task": task,
                "circuit_mean_spectral_norm": float(np.mean(circuit_specs)),
                "noncircuit_mean_spectral_norm": float(np.mean(noncircuit_specs)),
                "n_circuit": len(circuit_specs),
                "n_noncircuit": len(noncircuit_specs),
            },
        ))
        results.append(EvalResult(
            metric_id="C14.sv_entropy_ratio",
            value=ent_ratio,
            metadata={
                "task": task,
                "circuit_mean_sv_entropy": float(np.mean(circuit_ents)),
                "noncircuit_mean_sv_entropy": float(np.mean(noncircuit_ents)),
                "n_circuit": len(circuit_ents),
                "n_noncircuit": len(noncircuit_ents),
            },
        ))

    return results


def _compute_patching_d_prime(tasks: list[str]) -> list[EvalResult]:
    """C14.d_prime from 02 activation patching data.

    d' = (mean_circuit_effect - mean_random_effect) / pooled_std
    """
    data_02 = load_results("02_activation_patching.json")
    if data_02 is None:
        log("  02_activation_patching.json not found, skipping d_prime")
        return []

    results = []
    for item in data_02:
        task = item["metadata"]["task"]
        if task not in tasks:
            continue

        mean_circuit = item["value"]
        mean_random = item.get("baseline_random")
        random_std = item["metadata"].get("random_std")

        if mean_random is None or random_std is None:
            log(f"  {task}: missing random baseline/std for d_prime")
            continue

        # Circuit effect std: compute from per_head_effects
        per_head = item["metadata"].get("per_head_effects", {})
        if not per_head:
            continue
        circuit_values = list(per_head.values())
        circuit_std = float(np.std(circuit_values, ddof=1)) if len(circuit_values) > 1 else random_std

        dprime = _d_prime(abs(mean_circuit), abs(mean_random), circuit_std, random_std)
        log(f"  {task}: d_prime = {dprime:.4f} (|circuit|={abs(mean_circuit):.4f}, |random|={abs(mean_random):.4f})")

        results.append(EvalResult(
            metric_id="C14.d_prime",
            value=dprime,
            metadata={
                "task": task,
                "mean_circuit_effect": mean_circuit,
                "mean_random_effect": mean_random,
                "circuit_std": circuit_std,
                "random_std": random_std,
                "source": "02_activation_patching",
            },
        ))

    return results


def _compute_llc_llc_d_prime(tasks: list[str]) -> list[EvalResult]:
    """C14.llc_d_prime from 10 LLC data.

    d' = (mean_noncircuit_llc - mean_circuit_llc) / pooled_std
    Lower circuit LLC = more specialized, so noncircuit - circuit is the signal.
    """
    data_10 = load_results("10_llc.json")
    if data_10 is None:
        log("  10_llc.json not found, skipping llc_d_prime")
        return []

    results = []
    for item in data_10:
        task = item["metadata"]["task"]
        if task not in tasks:
            continue

        mean_circuit = item["metadata"].get("mean_circuit_llc")
        mean_noncircuit = item["metadata"].get("mean_non_circuit_llc")
        if mean_circuit is None or mean_noncircuit is None:
            continue

        # Compute std from per_head_llc values
        per_head = item["metadata"].get("per_head_llc", {})
        circuit_values = list(per_head.values())
        if len(circuit_values) < 2:
            continue
        circuit_std = float(np.std(circuit_values, ddof=1))

        # Use baseline_random as noncircuit mean proxy, estimate noncircuit std
        # Since we don't have individual noncircuit values, use circuit_std as proxy
        noncircuit_std = circuit_std  # conservative estimate

        dprime = _d_prime(mean_noncircuit, mean_circuit, noncircuit_std, circuit_std)
        log(f"  {task}: llc_d_prime = {dprime:.4f} (noncircuit={mean_noncircuit:.4f}, circuit={mean_circuit:.4f})")

        results.append(EvalResult(
            metric_id="C14.llc_d_prime",
            value=dprime,
            metadata={
                "task": task,
                "mean_circuit_llc": mean_circuit,
                "mean_noncircuit_llc": mean_noncircuit,
                "circuit_std": circuit_std,
                "source": "10_llc",
            },
        ))

    return results


def _compute_weight_classifier_sdt(tasks: list[str]) -> list[EvalResult]:
    """C14.hit_rate, C14.false_alarm_rate from weight classifier results."""
    results = []
    for task in tasks:
        if task not in WEIGHT_CLASSIFIER_RESULTS:
            continue
        wc = WEIGHT_CLASSIFIER_RESULTS[task]
        recall = wc["recall"]
        precision = wc["precision"]
        false_alarm = 1.0 - precision

        log(f"  {task}: hit_rate={recall:.3f}, false_alarm_rate={false_alarm:.3f}")

        results.append(EvalResult(
            metric_id="C14.hit_rate",
            value=recall,
            metadata={"task": task, "source": "weight_classifier"},
        ))
        results.append(EvalResult(
            metric_id="C14.false_alarm_rate",
            value=false_alarm,
            metadata={"task": task, "source": "weight_classifier"},
        ))

    return results


def _compute_invariance_partial_eta_sq(tasks: list[str]) -> list[EvalResult]:
    """C14.partial_eta_sq -- alias for 13 eta_squared."""
    data_13 = load_results("13_measurement_invariance.json")
    if data_13 is None:
        log("  13_measurement_invariance.json not found, skipping partial_eta_sq")
        return []

    results = []
    for item in data_13:
        task = item["metadata"]["task"]
        if task not in tasks:
            continue

        eta_sq = item["metadata"].get("eta_squared")
        if eta_sq is None:
            continue

        log(f"  {task}: partial_eta_sq = {eta_sq:.6f}")
        results.append(EvalResult(
            metric_id="C14.partial_eta_sq",
            value=eta_sq,
            metadata={
                "task": task,
                "invariance_verdict": item["metadata"].get("invariance_verdict"),
                "source": "13_measurement_invariance",
            },
        ))

    return results


def _compute_predictive_coding_metrics(tasks: list[str]) -> list[EvalResult]:
    """C14.pc_ratio_circuit_mean, C14.pc_ratio_noncircuit_mean, C14.pc_ratio_d_prime."""
    data = _load_pillar_json("predictive_coding.json")
    if data is None:
        return []

    per_head = data.get("per_head", {})
    if not per_head:
        return []

    results = []
    for task in tasks:
        heads = get_circuit_heads(task)
        if not heads:
            continue

        head_keys = {f"L{l}H{h}" for l, h in heads}
        circuit_ratios = []
        noncircuit_ratios = []

        for hk, hv in per_head.items():
            pc_ratio = hv.get("pc_ratio")
            if pc_ratio is None:
                continue
            if hk in head_keys:
                circuit_ratios.append(pc_ratio)
            else:
                noncircuit_ratios.append(pc_ratio)

        if not circuit_ratios or not noncircuit_ratios:
            continue

        circuit_mean = float(np.mean(circuit_ratios))
        noncircuit_mean = float(np.mean(noncircuit_ratios))
        circuit_std = float(np.std(circuit_ratios, ddof=1)) if len(circuit_ratios) > 1 else 1.0
        noncircuit_std = float(np.std(noncircuit_ratios, ddof=1)) if len(noncircuit_ratios) > 1 else 1.0
        dprime = _d_prime(noncircuit_mean, circuit_mean, noncircuit_std, circuit_std)

        log(f"  {task}: pc_ratio circuit={circuit_mean:.3f}, noncircuit={noncircuit_mean:.3f}, d'={dprime:.3f}")

        results.append(EvalResult(
            metric_id="C14.pc_ratio_circuit_mean",
            value=circuit_mean,
            metadata={"task": task, "n_circuit": len(circuit_ratios)},
        ))
        results.append(EvalResult(
            metric_id="C14.pc_ratio_noncircuit_mean",
            value=noncircuit_mean,
            metadata={"task": task, "n_noncircuit": len(noncircuit_ratios)},
        ))
        results.append(EvalResult(
            metric_id="C14.pc_ratio_d_prime",
            value=dprime,
            metadata={
                "task": task,
                "circuit_mean": circuit_mean,
                "noncircuit_mean": noncircuit_mean,
                "circuit_std": circuit_std,
                "noncircuit_std": noncircuit_std,
            },
        ))

    return results


def _compute_composition_metrics(tasks: list[str]) -> list[EvalResult]:
    """Extract Q/K/OV composition p-values from composition_scores.json."""
    data = _load_pillar_json("composition_scores.json")
    if data is None:
        return []

    per_task = data.get("per_task", {})
    if not per_task:
        return []

    results = []
    for task in tasks:
        if task not in per_task:
            continue
        td = per_task[task]

        for comp_type in ["ov", "q", "k"]:
            t_pval = td.get(f"{comp_type}_t_pval")
            edge_mean = td.get(f"{comp_type}_edge_mean")
            nonedge_mean = td.get(f"{comp_type}_nonedge_mean")
            edge_std = td.get(f"{comp_type}_edge_std")
            nonedge_std = td.get(f"{comp_type}_nonedge_std")

            if t_pval is None:
                continue

            # Compute d-prime for composition
            dprime = None
            if edge_std is not None and nonedge_std is not None:
                dprime = _d_prime(edge_mean, nonedge_mean, edge_std, nonedge_std)

            log(f"  {task}: {comp_type}_composition p={t_pval:.4f}" +
                (f", d'={dprime:.3f}" if dprime is not None else ""))

            results.append(EvalResult(
                metric_id=f"C14.{comp_type}_composition_pval",
                value=t_pval,
                metadata={
                    "task": task,
                    "edge_mean": edge_mean,
                    "nonedge_mean": nonedge_mean,
                    "edge_std": edge_std,
                    "nonedge_std": nonedge_std,
                    "d_prime": dprime,
                    "n_edges": td.get("n_edges"),
                    "n_non_edges": td.get("n_non_edges"),
                },
            ))

    return results


def _compute_attribution_auroc(tasks: list[str]) -> list[EvalResult]:
    """C14.attribution_auroc from attribution_patching.json."""
    data = _load_pillar_json("attribution_patching.json")
    if data is None:
        return []

    results = []
    for task in tasks:
        if task not in data:
            continue
        td = data[task]
        auroc = td.get("auroc")
        if auroc is None:
            continue

        log(f"  {task}: attribution_auroc={auroc:.4f}")
        results.append(EvalResult(
            metric_id="C14.attribution_auroc",
            value=auroc,
            metadata={
                "task": task,
                "average_precision": td.get("average_precision"),
                "precision_at_k": td.get("precision_at_k"),
                "recall_at_k": td.get("recall_at_k"),
                "source": "attribution_patching",
            },
        ))

    return results


def _compute_faithfulness_completeness(tasks: list[str]) -> list[EvalResult]:
    """C14.faithfulness and C14.completeness from pillar data."""
    results = []

    faith_data = _load_pillar_json("faithfulness.json")
    if faith_data is not None:
        for task in tasks:
            if task not in faith_data:
                continue
            td = faith_data[task]
            faith = td.get("faithfulness")
            if faith is None:
                continue
            log(f"  {task}: faithfulness={faith:.4f}")
            results.append(EvalResult(
                metric_id="C14.faithfulness",
                value=faith,
                metadata={
                    "task": task,
                    "n_circuit_heads": td.get("n_circuit_heads"),
                    "clean_logit_diff": td.get("clean_logit_diff"),
                    "circuit_only_logit_diff": td.get("circuit_only_logit_diff"),
                },
            ))

    comp_data = _load_pillar_json("completeness.json")
    if comp_data is not None:
        for task in tasks:
            if task not in comp_data:
                continue
            td = comp_data[task]
            comp = td.get("completeness")
            if comp is None:
                continue
            log(f"  {task}: completeness={comp:.4f}")
            results.append(EvalResult(
                metric_id="C14.completeness",
                value=comp,
                metadata={
                    "task": task,
                    "n_circuit_heads": td.get("n_circuit_heads"),
                    "clean_logit_diff": td.get("clean_logit_diff"),
                    "ablate_circuit_logit_diff": td.get("ablate_circuit_logit_diff"),
                },
            ))

    return results


def _compute_minimality_summary(tasks: list[str]) -> list[EvalResult]:
    """C14.minimality_mean_importance from minimality.json."""
    data = _load_pillar_json("minimality.json")
    if data is None:
        return []

    results = []
    for task in tasks:
        if task not in data:
            continue
        td = data[task]
        head_importance = td.get("head_importance", {})
        if not head_importance:
            continue

        importances = [h["importance"] for h in head_importance.values()
                       if "importance" in h]
        if not importances:
            continue

        mean_imp = float(np.mean(importances))
        n_essential = sum(1 for imp in importances if imp > 0.05)
        log(f"  {task}: mean_importance={mean_imp:.4f}, n_essential={n_essential}/{len(importances)}")

        results.append(EvalResult(
            metric_id="C14.minimality_mean_importance",
            value=mean_imp,
            metadata={
                "task": task,
                "n_circuit_heads": len(importances),
                "n_essential_heads": n_essential,
                "max_importance": float(max(importances)),
                "min_importance": float(min(importances)),
            },
        ))

    return results


def _compute_scrubbing_recovery(tasks: list[str]) -> list[EvalResult]:
    """C14.logit_diff_recovery from 04 causal scrubbing data."""
    data_04 = load_results("04_causal_scrubbing.json")
    if data_04 is None:
        log("  04_causal_scrubbing.json not found, skipping recovery")
        return []

    results = []
    for item in data_04:
        task = item["metadata"]["task"]
        if task not in tasks:
            continue

        recovery = item["metadata"].get("logit_diff_recovery")
        if recovery is None:
            continue

        log(f"  {task}: logit_diff_recovery={recovery:.4f}")
        results.append(EvalResult(
            metric_id="C14.logit_diff_recovery",
            value=recovery,
            metadata={
                "task": task,
                "kl_divergence": item["metadata"].get("kl_divergence"),
                "n_circuit_heads": item["metadata"].get("n_circuit_heads"),
                "source": "04_causal_scrubbing",
            },
        ))

    return results


def main():
    parser = parse_common_args("C14: Derived Metrics (no forward passes)")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS

    log("=" * 60)
    log("C14: DERIVED METRICS (read-only, no forward passes)")
    log("=" * 60)
    log(f"Tasks: {tasks}")
    log(f"DATA_DIR: {SCRIPT_DIR / 'data'}")
    log(f"PILLAR_DATA: {PILLAR_DATA}")

    all_results: list[EvalResult] = []

    # --- Sparsity ---
    log("\n--- C14.sparsity ---")
    all_results.extend(_compute_sparsity(tasks))

    # --- Node overlap ---
    log("\n--- C14.node_overlap_jaccard ---")
    all_results.extend(_compute_node_overlap(tasks))

    # --- Spectral metrics from effective_rank_wov.json ---
    log("\n--- C14.spectral_norm_ratio / C14.sv_entropy_ratio ---")
    all_results.extend(_compute_spectral_metrics(tasks))

    # --- D-prime from 02 ---
    log("\n--- C14.d_prime (from 02 activation patching) ---")
    all_results.extend(_compute_patching_d_prime(tasks))

    # --- LLC d-prime from 10 ---
    log("\n--- C14.llc_d_prime (from 10 LLC) ---")
    all_results.extend(_compute_llc_llc_d_prime(tasks))

    # --- Weight classifier SDT ---
    log("\n--- C14.hit_rate / C14.false_alarm_rate ---")
    all_results.extend(_compute_weight_classifier_sdt(tasks))

    # --- Partial eta-squared from 13 ---
    log("\n--- C14.partial_eta_sq (from C13) ---")
    all_results.extend(_compute_invariance_partial_eta_sq(tasks))

    # --- Predictive coding metrics ---
    log("\n--- C14.pc_ratio (from predictive_coding.json) ---")
    all_results.extend(_compute_predictive_coding_metrics(tasks))

    # --- Composition p-values ---
    log("\n--- C14.composition (from composition_scores.json) ---")
    all_results.extend(_compute_composition_metrics(tasks))

    # --- Attribution AUROC ---
    log("\n--- C14.attribution_auroc (from attribution_patching.json) ---")
    all_results.extend(_compute_attribution_auroc(tasks))

    # --- Faithfulness / Completeness ---
    log("\n--- C14.faithfulness / C14.completeness ---")
    all_results.extend(_compute_faithfulness_completeness(tasks))

    # --- Minimality ---
    log("\n--- C14.minimality_mean_importance ---")
    all_results.extend(_compute_minimality_summary(tasks))

    # --- 04 recovery ---
    log("\n--- C14.logit_diff_recovery (from C04) ---")
    all_results.extend(_compute_scrubbing_recovery(tasks))

    # --- Summary ---
    log("\n" + "=" * 60)
    metric_ids = sorted(set(r.metric_id for r in all_results))
    log(f"Computed {len(all_results)} results across {len(metric_ids)} metric types:")
    for mid in metric_ids:
        count = sum(1 for r in all_results if r.metric_id == mid)
        log(f"  {mid}: {count} tasks")

    out = args.out or "14_derived_metrics.json"
    save_results(all_results, out)

    log(f"\nDone. {len(all_results)} total results saved.")


if __name__ == "__main__":
    main()
