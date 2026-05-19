"""Do-Calculus Mediation (Natural Direct/Indirect Effect)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     A06 — Mediation Analysis
Categories:     causal
Validity layer: Internal
Criteria:       I1 Necessity
Establishes:    Circuit heads mediate causal effect (NDE/NIE decomposition via Pearl's formula)
Requires:       GPU, model
Doc:            /instruments_v2/causal/a06-mediation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Measures NDE and NIE of each circuit head as a mediator between input
and output using Pearl's mediation formula.

Optional: pip install dowhy (for formal SCM). Falls back to manual
two-stage regression if unavailable.

Usage:
    uv run python 05_mediation.py --tasks ioi sva --n-prompts 40
"""

import numpy as np
import torch

from mechanistic_validity.instruments.common import (
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

try:
    from dowhy import CausalModel
    HAS_DOWHY = True
except ImportError:
    HAS_DOWHY = False


@torch.no_grad()
def compute_mediation_effects(model, prompts, correct_ids, incorrect_ids,
                               circuit_heads, rng) -> dict:
    """Compute NIE and NDE for each circuit head via resample mediation.

    Treatment T = clean vs corrupt input (resample from another prompt).
    Mediator M = head H activation at last position.
    Outcome Y = logit diff.

    NIE = Y(clean, M_clean) - Y(clean, M_corrupt)  [effect through head H]
    NDE = Y(clean, M_corrupt) - Y(corrupt)          [effect bypassing head H]
    TE  = NIE + NDE = Y(clean) - Y(corrupt)
    """
    results_per_head = {}
    n_valid = min(len(prompts), len(correct_ids))

    corrupt_indices = list(rng.permutation(n_valid))
    for i in range(n_valid):
        if corrupt_indices[i] == i:
            corrupt_indices[i] = (i + 1) % n_valid

    for L, H in sorted(circuit_heads):
        hook_name = f"blocks.{L}.attn.hook_z"

        total_effect_sum = 0.0
        nie_sum = 0.0
        nde_sum = 0.0
        n_computed = 0

        for i in range(n_valid):
            ci = corrupt_indices[i]
            tokens_clean = model.to_tokens(prompts[i].text)
            tokens_corrupt = model.to_tokens(prompts[ci].text)

            clean_logits, clean_cache = model.run_with_cache(
                tokens_clean, names_filter=lambda n: n == hook_name)
            clean_ld = logit_diff_from_logits(clean_logits, correct_ids[i], incorrect_ids[i])

            corrupt_logits, corrupt_cache = model.run_with_cache(
                tokens_corrupt, names_filter=lambda n: n == hook_name)
            corrupt_ld = logit_diff_from_logits(corrupt_logits, correct_ids[i], incorrect_ids[i])

            total_effect = clean_ld - corrupt_ld

            corrupt_z_last = corrupt_cache[hook_name][0, -1, H, :].clone()
            def crossover_hook(z, hook, _cz=corrupt_z_last):
                z[0, -1, H, :] = _cz
                return z

            crossover_logits = model.run_with_hooks(
                tokens_clean, fwd_hooks=[(hook_name, crossover_hook)])
            crossover_ld = logit_diff_from_logits(crossover_logits, correct_ids[i], incorrect_ids[i])

            nie = clean_ld - crossover_ld
            nde = crossover_ld - corrupt_ld

            total_effect_sum += total_effect
            nie_sum += nie
            nde_sum += nde
            n_computed += 1

        n = max(n_computed, 1)
        te = total_effect_sum / n
        results_per_head[f"L{L}H{H}"] = {
            "total_effect": te,
            "nie": nie_sum / n,
            "nde": nde_sum / n,
            "nie_fraction": (nie_sum / n) / (abs(te) + 1e-8),
        }

    return results_per_head


def run_mediation(model, tasks: list[str], n_prompts: int = 40) -> list[EvalResult]:
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
            continue

        correct_ids, incorrect_ids = get_token_ids(prompts, tokenizer)
        if not correct_ids:
            continue

        log(f"  {task} ({len(circuit_heads)} heads)...")

        per_head = compute_mediation_effects(model, prompts, correct_ids, incorrect_ids,
                                              circuit_heads, rng)

        mean_nie_frac = float(np.mean([v["nie_fraction"] for v in per_head.values()]))
        mean_total = float(np.mean([v["total_effect"] for v in per_head.values()]))

        log(f"    mean_NIE_frac={mean_nie_frac:.3f}  mean_total_effect={mean_total:.3f}")

        results.append(EvalResult(
            metric_id="C5.mediation",
            value=mean_nie_frac,
            n_samples=len(prompts),
            metadata={
                "task": task,
                "per_head": per_head,
                "mean_total_effect": mean_total,
                "n_circuit_heads": len(circuit_heads),
            },
        ))

    return results


def main():
    parser = parse_common_args("C5: Do-Calculus Mediation")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("C5: DO-CALCULUS MEDIATION (NDE/NIE)")
    log("=" * 60)

    if not HAS_DOWHY:
        log("NOTE: dowhy not installed. Using manual mediation implementation.")

    results = run_mediation(model, tasks, args.n_prompts)

    out = args.out or "05_mediation.json"
    save_results(results, out)

    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
