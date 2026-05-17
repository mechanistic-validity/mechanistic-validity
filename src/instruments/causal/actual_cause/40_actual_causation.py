"""Actual Causation (Halpern-Pearl 2005/2015)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     A11 — Actual Causation
Categories:     causal
Validity layer: Internal
Criteria:       I1 Necessity
Establishes:    Circuit heads are actual causes (Halpern-Pearl AC1-AC3) with identifiable witnesses
Requires:       GPU, model
Doc:            /instruments_v2/causal/a11-actual-cause
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Implements Halpern & Pearl's definition of actual cause for circuit heads.
A head H is an actual cause of behavior B if there exists a WITNESS —
a setting W of other variables (heads) such that:
  (AC1) H = h and B = b in the actual world
  (AC2) There exists W and h' != h such that setting H=h', W=w changes B
  (AC3) H is minimal (no proper subset of {H} satisfies AC1-AC2)

The key difference from standard activation patching: standard patching
tests necessity in the ACTUAL context (all other heads at their natural
values). Actual causation searches across CONTINGENT contexts — it finds
settings of other heads that make H critical even when H is not critical
in the actual context.

This captures cases like: a head that is individually redundant (standard
patching shows low effect because a backup head compensates) but is an
actual cause because there exists a context (backup head ablated) where
it becomes critical.

Outputs per task:
  - Per-head: is_actual_cause, witness set, contingency depth
  - Comparison to standard necessity (activation patching effect)
  - Heads that are actual causes but not necessary (backup detection)

Usage:
    uv run python 40_actual_causation.py --tasks ioi sva --n-prompts 40
    uv run python 40_actual_causation.py --device cuda --max-witness-size 3
"""
import sys
from itertools import combinations
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "causal"))
from _common import (
    CIRCUIT_TASKS,
    EvalResult,
    calibrate_mean_z,
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
def compute_effect_in_context(
    model, tokens, correct_id, incorrect_id,
    target_head: tuple[int, int],
    context_ablated: set[tuple[int, int]],
    mean_z: torch.Tensor,
) -> tuple[float, float]:
    """Compute the effect of ablating target_head when context_ablated heads
    are already ablated.

    Returns (ld_with_target, ld_without_target) in the contingent world
    where context_ablated heads are already removed.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    context_by_layer = heads_to_layer_dict(context_ablated)
    context_hooks = make_ablation_hook(context_by_layer, mean_z, "mean")

    both_ablated = context_ablated | {target_head}
    both_by_layer = heads_to_layer_dict(both_ablated)
    both_hooks = make_ablation_hook(both_by_layer, mean_z, "mean")

    logits_context = model.run_with_hooks(tokens, fwd_hooks=context_hooks)
    ld_with_target = logit_diff_from_logits(logits_context, correct_id, incorrect_id)

    logits_both = model.run_with_hooks(tokens, fwd_hooks=both_hooks)
    ld_without_target = logit_diff_from_logits(logits_both, correct_id, incorrect_id)

    return ld_with_target, ld_without_target


@torch.no_grad()
def find_witness(
    model, prompts, correct_ids, incorrect_ids,
    target_head: tuple[int, int],
    other_heads: set[tuple[int, int]],
    mean_z: torch.Tensor,
    max_witness_size: int = 3,
    effect_threshold: float = 0.5,
) -> dict | None:
    """Search for a witness set W such that ablating target_head
    in the context where W is ablated produces a large effect.

    Returns the witness info if found, None otherwise.
    """
    others_list = sorted(other_heads - {target_head})

    for witness_size in range(0, min(max_witness_size + 1, len(others_list) + 1)):
        combos = list(combinations(range(len(others_list)), witness_size))
        if len(combos) > 200:
            combos = combos[:200]

        for combo in combos:
            witness_set = {others_list[i] for i in combo}

            total_effect = 0.0
            total_baseline = 0.0
            n_valid = 0

            for i, p in enumerate(prompts[:10]):
                if i >= len(correct_ids):
                    break
                tokens = model.to_tokens(p.text)

                ld_with, ld_without = compute_effect_in_context(
                    model, tokens, correct_ids[i], incorrect_ids[i],
                    target_head, witness_set, mean_z,
                )

                effect = ld_with - ld_without
                total_effect += abs(effect)

                clean_logits = model(tokens)
                clean_ld = logit_diff_from_logits(clean_logits, correct_ids[i], incorrect_ids[i])
                total_baseline += abs(clean_ld)
                n_valid += 1

            if n_valid == 0 or total_baseline < 1e-8:
                continue

            normalized_effect = total_effect / total_baseline

            if normalized_effect >= effect_threshold:
                return {
                    "witness": sorted(list(witness_set)),
                    "witness_size": witness_size,
                    "normalized_effect": normalized_effect,
                    "raw_effect": total_effect / n_valid,
                }

    return None


@torch.no_grad()
def compute_standard_necessity(
    model, prompts, correct_ids, incorrect_ids,
    head: tuple[int, int], mean_z: torch.Tensor,
) -> float:
    """Standard activation patching necessity (no witness context)."""
    total_effect = 0.0
    total_baseline = 0.0

    head_by_layer = heads_to_layer_dict({head})
    hooks = make_ablation_hook(head_by_layer, mean_z, "mean")

    for i, p in enumerate(prompts[:10]):
        if i >= len(correct_ids):
            break
        tokens = model.to_tokens(p.text)
        clean_logits = model(tokens)
        clean_ld = logit_diff_from_logits(clean_logits, correct_ids[i], incorrect_ids[i])

        ablated_logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
        ablated_ld = logit_diff_from_logits(ablated_logits, correct_ids[i], incorrect_ids[i])

        total_effect += abs(clean_ld - ablated_ld)
        total_baseline += abs(clean_ld)

    if total_baseline < 1e-8:
        return 0.0
    return total_effect / total_baseline


@torch.no_grad()
def main():
    parser = parse_common_args("A11 — Actual Causation (Halpern-Pearl)")
    parser.add_argument("--max-witness-size", type=int, default=3,
                        help="Max number of heads in witness set")
    parser.add_argument("--effect-threshold", type=float, default=0.3,
                        help="Min normalized effect to count as actual cause")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    all_results = {}

    for task in tasks:
        log(f"\n{'='*60}")
        log(f"Task: {task}")
        circuit_heads = get_circuit_heads(task)
        if not circuit_heads:
            log(f"  No circuit for {task}, skipping")
            continue

        prompts = generate_prompts(task, model.tokenizer, args.n_prompts)
        if not prompts:
            log(f"  No prompts for {task}, skipping")
            continue

        correct_ids, incorrect_ids = get_token_ids(prompts, model.tokenizer)
        mean_z = calibrate_mean_z(model, prompts)

        log(f"  Circuit size: {len(circuit_heads)} heads")
        log(f"  Testing actual causation (max witness size={args.max_witness_size})...")

        head_results = {}
        n_actual_causes = 0
        n_necessary = 0
        n_actual_not_necessary = 0

        for head in sorted(circuit_heads):
            necessity = compute_standard_necessity(
                model, prompts, correct_ids, incorrect_ids, head, mean_z
            )
            is_necessary = necessity >= args.effect_threshold

            witness = find_witness(
                model, prompts, correct_ids, incorrect_ids,
                head, circuit_heads, mean_z,
                max_witness_size=args.max_witness_size,
                effect_threshold=args.effect_threshold,
            )
            is_actual_cause = witness is not None or is_necessary

            if is_actual_cause:
                n_actual_causes += 1
            if is_necessary:
                n_necessary += 1
            if is_actual_cause and not is_necessary:
                n_actual_not_necessary += 1

            head_results[f"L{head[0]}H{head[1]}"] = {
                "is_actual_cause": is_actual_cause,
                "is_necessary": is_necessary,
                "is_actual_not_necessary": is_actual_cause and not is_necessary,
                "standard_necessity": necessity,
                "witness": witness,
            }

            status = "ACTUAL CAUSE" if is_actual_cause else "not cause"
            witness_info = f" (witness: {witness['witness']}, effect={witness['normalized_effect']:.3f})" if witness else ""
            log(f"    {head}: necessity={necessity:.3f}, {status}{witness_info}")

        log(f"\n  Summary: {n_actual_causes}/{len(circuit_heads)} actual causes, "
            f"{n_necessary} necessary, {n_actual_not_necessary} actual-but-not-necessary (backup detection)")

        all_results[task] = {
            "n_circuit_heads": len(circuit_heads),
            "n_actual_causes": n_actual_causes,
            "n_necessary": n_necessary,
            "n_actual_not_necessary": n_actual_not_necessary,
            "heads": head_results,
            "max_witness_size": args.max_witness_size,
            "effect_threshold": args.effect_threshold,
        }

    save_results(all_results, "a11_actual_causation.json")
    log("\nDone.")


if __name__ == "__main__":
    main()
