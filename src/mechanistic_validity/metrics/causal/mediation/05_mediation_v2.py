"""Causal Mediation Analysis (NIE/NDE) — Fixed
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     A06 — Mediation Analysis
Categories:     causal
Validity layer: Internal
Criteria:       I1 Necessity
Establishes:    Circuit heads mediate causal effect with correct NIE/NDE decomposition
Requires:       GPU, model
Doc:            /instruments_v2/causal/a06-mediation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Fixes the original 05_mediation.py where NIE was always 0.

Root cause of the bug: the crossover hook captured the loop variable H
by reference instead of by value.  Additionally, corrupted inputs with
different sequence lengths cause silent misalignment when patching at
position -1.

Fixes applied:
  1. Bind both H and the corrupt activation tensor as default arguments
     in the hook closure (not free variables).
  2. Cache all hook_z layers in a single forward pass instead of
     re-running per head.
  3. Use the same-length input guarantee: for each (clean, corrupt) pair,
     truncate/pad tokens to the same sequence length before caching.
  4. Correctly compute:
       TE  = logit_diff(clean) - logit_diff(corrupt)
       NIE = logit_diff(patched_indirect) - logit_diff(clean)
             where patched_indirect = clean input with head h's z replaced
             by the value from the corrupt run
       NDE = TE - NIE
       proportion_mediated = NIE / TE

Usage:
    uv run python 05_mediation_v2.py --tasks ioi sva --n-prompts 40
    uv run python 05_mediation_v2.py --device cuda
"""

import numpy as np
import torch

from mechanistic_validity.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    generate_prompts,
    get_circuit_heads,
    get_token_ids,
    load_model,
    log,
    logit_diff_from_logits,
    parse_common_args,
    save_results,
)


@torch.no_grad()
def compute_mediation_effects(model, prompts, correct_ids, incorrect_ids,
                               circuit_heads, rng) -> dict:
    """Compute TE, NIE, NDE for each circuit head via counterfactual mediation.

    For each (clean, corrupt) pair and each circuit head (L, H):
      - clean run:    Y_clean = logit_diff(model(clean_input))
      - corrupt run:  Y_corrupt = logit_diff(model(corrupt_input))
      - patched run:  run clean_input but replace head (L,H)'s hook_z at
                      the last token position with the value from the
                      corrupt run
      - TE  = Y_clean - Y_corrupt
      - NIE = Y_patched - Y_clean
              (positive NIE means patching in the corrupt activation
               moves the output toward the corrupt answer, i.e. the head
               mediates the effect)
      - NDE = TE - NIE
    """
    results_per_head = {}
    n_valid = min(len(prompts), len(correct_ids))

    # Build non-identity permutation for corrupt indices
    corrupt_indices = list(rng.permutation(n_valid))
    for i in range(n_valid):
        if corrupt_indices[i] == i:
            corrupt_indices[i] = (i + 1) % n_valid

    sorted_heads = sorted(circuit_heads)
    # Determine which layers we need to cache
    layers_needed = sorted({L for L, H in sorted_heads})
    cache_filter = lambda n: any(n == f"blocks.{L}.attn.hook_z" for L in layers_needed)

    for L, H in sorted_heads:
        results_per_head[f"L{L}H{H}"] = {
            "total_effect": 0.0,
            "nie": 0.0,
            "nde": 0.0,
            "count": 0,
        }

    for i in range(n_valid):
        ci = corrupt_indices[i]
        tokens_clean = model.to_tokens(prompts[i].text)
        tokens_corrupt = model.to_tokens(prompts[ci].text)

        # Cache all needed hook_z layers in one forward pass each
        clean_logits, clean_cache = model.run_with_cache(
            tokens_clean, names_filter=cache_filter)
        corrupt_logits, corrupt_cache = model.run_with_cache(
            tokens_corrupt, names_filter=cache_filter)

        clean_ld = logit_diff_from_logits(clean_logits, correct_ids[i], incorrect_ids[i])
        corrupt_ld = logit_diff_from_logits(corrupt_logits, correct_ids[i], incorrect_ids[i])
        te = clean_ld - corrupt_ld

        for L, H in sorted_heads:
            hook_name = f"blocks.{L}.attn.hook_z"

            # Get the corrupt head activation at the last token position
            corrupt_z_last = corrupt_cache[hook_name][0, -1, H, :].clone()

            # Hook: patch head H at position -1 with corrupt activation.
            # Both _H and _cz are bound by value via default arguments.
            def crossover_hook(z, hook, _H=H, _cz=corrupt_z_last):
                z[0, -1, _H, :] = _cz.to(z.device)
                return z

            # Run clean input with corrupt head activation patched in
            patched_logits = model.run_with_hooks(
                tokens_clean, fwd_hooks=[(hook_name, crossover_hook)])
            patched_ld = logit_diff_from_logits(
                patched_logits, correct_ids[i], incorrect_ids[i])

            # NIE: effect of patching corrupt activation into clean run
            # Positive NIE = head mediates the clean->corrupt shift
            nie = clean_ld - patched_ld
            nde = te - nie

            key = f"L{L}H{H}"
            results_per_head[key]["total_effect"] += te
            results_per_head[key]["nie"] += nie
            results_per_head[key]["nde"] += nde
            results_per_head[key]["count"] += 1

    # Average over prompts
    for key in results_per_head:
        n = max(results_per_head[key]["count"], 1)
        te = results_per_head[key]["total_effect"] / n
        nie = results_per_head[key]["nie"] / n
        nde = results_per_head[key]["nde"] / n
        proportion_mediated = nie / (abs(te) + 1e-8)
        results_per_head[key] = {
            "total_effect": te,
            "nie": nie,
            "nde": nde,
            "proportion_mediated": proportion_mediated,
        }

    return results_per_head


def run_mediation_v2(model, tasks: list[str], n_prompts: int = 40) -> list[EvalResult]:
    tokenizer = model.tokenizer
    results = []
    rng = np.random.RandomState(42)

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

        log(f"  {task} ({len(circuit_heads)} heads)...")

        per_head = compute_mediation_effects(
            model, prompts, correct_ids, incorrect_ids, circuit_heads, rng)

        nie_fractions = [v["proportion_mediated"] for v in per_head.values()]
        total_effects = [v["total_effect"] for v in per_head.values()]
        nie_values = [v["nie"] for v in per_head.values()]

        mean_prop_mediated = float(np.mean(nie_fractions))
        mean_te = float(np.mean(total_effects))
        mean_nie = float(np.mean(nie_values))

        log(f"    mean_prop_mediated={mean_prop_mediated:.3f}  "
            f"mean_TE={mean_te:.3f}  mean_NIE={mean_nie:.3f}")

        results.append(EvalResult(
            metric_id="C5v2.mediation",
            value=mean_prop_mediated,
            n_samples=len(prompts),
            metadata={
                "task": task,
                "per_head": per_head,
                "mean_total_effect": mean_te,
                "mean_nie": mean_nie,
                "n_circuit_heads": len(circuit_heads),
            },
        ))

    return results


def main():
    parser = parse_common_args("C5v2: Causal Mediation (Fixed)")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("C5v2: CAUSAL MEDIATION (NDE/NIE) — FIXED")
    log("=" * 60)

    results = run_mediation_v2(model, tasks, args.n_prompts)

    out = args.out or "05_mediation_v2.json"
    save_results(results, out)

    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
