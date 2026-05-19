"""Circuit Logic Gate Analysis (AND/OR/NOT Proportions)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     A05 — MDC/Glennan Mechanisms
Categories:     causal
Validity layer: Construct
Criteria:       C2 Structural plausibility
Establishes:    Circuit heads implement identifiable logic gates (AND/OR/NOT) with mechanistic structure
Requires:       GPU, model
Doc:            /instruments_v2/causal/a05-mdc-glennan
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Based on the 2025 result that circuits contain AND-gates, OR-gates, and
NOT-gates, and completeness requires both noising and denoising interventions.

For each task:
1. Compute faithfulness under NOISING (corrupt clean run by ablating
   non-circuit heads) = completeness_noising.
2. Compute faithfulness under DENOISING (restore corrupted run by patching
   in clean circuit activations) = completeness_denoising.
3. delta_completeness = completeness_noising - completeness_denoising.

Gate type detection per pair of connected circuit heads:
- AND-gate: superadditive (ablating both worse than sum of singles).
- OR-gate: redundant (ablating both no worse than worst single).
- NOT-gate: inhibitory (ablating a head *improves* performance).

Usage:
    uv run python 19_logic_gates.py --tasks ioi sva
    uv run python 19_logic_gates.py --device cuda --n-prompts 60
"""
import itertools

import numpy as np
import torch

from mechanistic_validity.instruments.common import (
    CIRCUIT_TASKS,
    EvalResult,
    calibrate_mean_z,
    compute_completeness,
    compute_faithfulness,
    generate_prompts,
    get_circuit_heads,
    get_token_ids,
    heads_to_layer_dict,
    load_model,
    log,
    logit_diff_from_logits,
    make_ablation_hook,
    parse_common_args,
    save_results,
)


@torch.no_grad()
def completeness_denoising(model, prompts, correct_ids, incorrect_ids,
                           circuit_heads, mean_z) -> float:
    """Denoising completeness: ablate ALL heads, then restore circuit heads.

    Returns (restored_ld - corrupt_ld) / (clean_ld - corrupt_ld).
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    all_heads = {(L, H) for L in range(n_layers) for H in range(n_heads)}
    all_by_layer = heads_to_layer_dict(all_heads)
    non_circuit = all_heads - circuit_heads
    non_circuit_by_layer = heads_to_layer_dict(non_circuit)

    hooks_all = make_ablation_hook(all_by_layer, mean_z, "mean")
    hooks_non_circuit = make_ablation_hook(non_circuit_by_layer, mean_z, "mean")

    numer, denom = 0.0, 0.0
    for i, p in enumerate(prompts):
        if i >= len(correct_ids):
            break
        tokens = model.to_tokens(p.text)
        clean_logits = model(tokens)
        clean_ld = logit_diff_from_logits(clean_logits, correct_ids[i], incorrect_ids[i])

        corrupt_logits = model.run_with_hooks(tokens, fwd_hooks=hooks_all)
        corrupt_ld = logit_diff_from_logits(corrupt_logits, correct_ids[i], incorrect_ids[i])

        restored_logits = model.run_with_hooks(tokens, fwd_hooks=hooks_non_circuit)
        restored_ld = logit_diff_from_logits(restored_logits, correct_ids[i], incorrect_ids[i])

        gap = clean_ld - corrupt_ld
        if abs(gap) > 1e-8:
            numer += (restored_ld - corrupt_ld)
            denom += gap

    if abs(denom) < 1e-8:
        return 0.0
    return numer / denom


@torch.no_grad()
def single_head_ablation_effect(model, prompts, correct_ids, incorrect_ids,
                                head, mean_z) -> float:
    """Effect of ablating a single head: (clean_ld - ablated_ld) / clean_ld."""
    L, H = head
    by_layer = heads_to_layer_dict({head})
    hooks = make_ablation_hook(by_layer, mean_z, "mean")

    numer, denom = 0.0, 0.0
    for i, p in enumerate(prompts):
        if i >= len(correct_ids):
            break
        tokens = model.to_tokens(p.text)
        clean_logits = model(tokens)
        clean_ld = logit_diff_from_logits(clean_logits, correct_ids[i], incorrect_ids[i])
        ablated_logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
        ablated_ld = logit_diff_from_logits(ablated_logits, correct_ids[i], incorrect_ids[i])
        numer += (clean_ld - ablated_ld)
        denom += abs(clean_ld) if abs(clean_ld) > 1e-8 else 1.0

    if abs(denom) < 1e-8:
        return 0.0
    return numer / denom


@torch.no_grad()
def pair_ablation_effect(model, prompts, correct_ids, incorrect_ids,
                         head_a, head_b, mean_z) -> float:
    """Effect of ablating two heads simultaneously."""
    by_layer = heads_to_layer_dict({head_a, head_b})
    hooks = make_ablation_hook(by_layer, mean_z, "mean")

    numer, denom = 0.0, 0.0
    for i, p in enumerate(prompts):
        if i >= len(correct_ids):
            break
        tokens = model.to_tokens(p.text)
        clean_logits = model(tokens)
        clean_ld = logit_diff_from_logits(clean_logits, correct_ids[i], incorrect_ids[i])
        ablated_logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
        ablated_ld = logit_diff_from_logits(ablated_logits, correct_ids[i], incorrect_ids[i])
        numer += (clean_ld - ablated_ld)
        denom += abs(clean_ld) if abs(clean_ld) > 1e-8 else 1.0

    if abs(denom) < 1e-8:
        return 0.0
    return numer / denom


def classify_gate(effect_i: float, effect_j: float, effect_both: float) -> str:
    """Classify the interaction between two heads as AND, OR, or NOT."""
    if effect_i < 0:
        return "NOT_i"
    if effect_j < 0:
        return "NOT_j"
    if effect_both > effect_i + effect_j + 1e-6:
        return "AND"
    if effect_both < max(effect_i, effect_j) - 1e-6:
        return "OR"
    return "ADDITIVE"


def run_logic_gates(model, tasks: list[str], n_prompts: int = 40) -> list[EvalResult]:
    tokenizer = model.tokenizer
    results = []

    for task in tasks:
        circuit_heads = get_circuit_heads(task)
        if not circuit_heads:
            log(f"  {task}: no circuit, skipping")
            continue

        prompts = generate_prompts(task, tokenizer, n_prompts)
        if not prompts:
            log(f"  {task}: no prompts, skipping")
            continue

        correct_ids, incorrect_ids = get_token_ids(prompts, tokenizer)
        if not correct_ids:
            continue

        log(f"  {task} ({len(circuit_heads)} heads, {len(prompts)} prompts)...")
        mean_z = calibrate_mean_z(model, prompts, n_calibration=min(100, len(prompts)))

        # --- Noising vs denoising completeness ---
        comp_noising = compute_completeness(
            model, prompts, correct_ids, incorrect_ids, circuit_heads, mean_z,
        )
        comp_denoising = completeness_denoising(
            model, prompts, correct_ids, incorrect_ids, circuit_heads, mean_z,
        )
        delta = comp_noising - comp_denoising

        log(f"    noising={comp_noising:.3f}  denoising={comp_denoising:.3f}  delta={delta:.3f}")

        # --- Per-head effects ---
        sorted_heads = sorted(circuit_heads)
        head_effects = {}
        for head in sorted_heads:
            eff = single_head_ablation_effect(
                model, prompts, correct_ids, incorrect_ids, head, mean_z,
            )
            head_effects[head] = eff
            log(f"    L{head[0]}H{head[1]}: effect={eff:.3f}")

        # --- Pairwise gate classification ---
        # Only consider pairs where L_i < L_j (information flows forward)
        gate_counts = {"AND": 0, "OR": 0, "NOT_i": 0, "NOT_j": 0, "ADDITIVE": 0}
        pair_details = []
        pairs = [(a, b) for a, b in itertools.combinations(sorted_heads, 2) if a[0] < b[0]]

        for head_a, head_b in pairs:
            eff_a = head_effects[head_a]
            eff_b = head_effects[head_b]
            eff_both = pair_ablation_effect(
                model, prompts, correct_ids, incorrect_ids, head_a, head_b, mean_z,
            )
            gate = classify_gate(eff_a, eff_b, eff_both)
            gate_counts[gate] += 1
            pair_details.append({
                "head_a": f"L{head_a[0]}H{head_a[1]}",
                "head_b": f"L{head_b[0]}H{head_b[1]}",
                "effect_a": eff_a,
                "effect_b": eff_b,
                "effect_both": eff_both,
                "gate_type": gate,
            })

        n_not = gate_counts["NOT_i"] + gate_counts["NOT_j"]
        n_pairs = len(pairs) if pairs else 1
        proportions = {
            "AND": gate_counts["AND"] / n_pairs,
            "OR": gate_counts["OR"] / n_pairs,
            "NOT": n_not / n_pairs,
            "ADDITIVE": gate_counts["ADDITIVE"] / n_pairs,
        }

        log(f"    gates: AND={gate_counts['AND']} OR={gate_counts['OR']} "
            f"NOT={n_not} ADDITIVE={gate_counts['ADDITIVE']} (of {len(pairs)} pairs)")

        per_head_serialized = {f"L{L}H{H}": eff for (L, H), eff in head_effects.items()}

        results.append(EvalResult(
            metric_id="C19.logic_gates",
            value=delta,
            n_samples=len(prompts),
            metadata={
                "task": task,
                "completeness_noising": comp_noising,
                "completeness_denoising": comp_denoising,
                "delta_completeness": delta,
                "gate_counts": gate_counts,
                "gate_proportions": proportions,
                "n_pairs_tested": len(pairs),
                "pair_details": pair_details,
                "per_head_effects": per_head_serialized,
                "n_circuit_heads": len(circuit_heads),
                "circuit_heads": sorted(circuit_heads),
            },
        ))

    return results


def main():
    parser = parse_common_args("C19: Circuit Logic Gate Analysis")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("C19: CIRCUIT LOGIC GATE ANALYSIS")
    log("=" * 60)

    results = run_logic_gates(model, tasks, args.n_prompts)

    out = args.out or "19_logic_gates.json"
    save_results(results, out)

    log(f"\nDone. {len(results)} tasks evaluated.")
    for r in results:
        t = r.metadata["task"]
        props = r.metadata["gate_proportions"]
        log(f"  {t}: delta={r.value:.3f}  AND={props['AND']:.0%} OR={props['OR']:.0%} "
            f"NOT={props['NOT']:.0%} ADD={props['ADDITIVE']:.0%}")


if __name__ == "__main__":
    main()
