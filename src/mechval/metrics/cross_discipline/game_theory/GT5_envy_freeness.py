"""Envy-Freeness Test for Circuit Credit Allocation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Metric:         GT5 — Envy-Freeness (Fair Division)
Categories:     causal, game_theory
Evidence family: causal
Description mode: implementational-functional

Tests whether the circuit's credit allocation (Shapley values) is
"envy-free" — does any head prefer another head's credit share,
given its own contribution profile?

Background:
    Fair division theory (Steinhaus 1948, "The Problem of Fair Division",
    Econometrica 16:101-104; Foley 1967, "Resource Allocation and the
    Public Sector", Yale Economic Essays 7:45-98) studies how to divide
    goods among agents so no one envies another's share.

    An allocation is ENVY-FREE if no agent prefers another agent's
    bundle to their own. Applied to circuits:
    - The "goods" being divided are credit for the circuit's output
    - Each head's "share" is its Shapley value (marginal contribution)
    - Head A envies head B if swapping their roles (positions in the
      circuit) would give A a higher Shapley value than it currently has

    In practice, we operationalize envy as: head A envies head B if
    |Shapley(B)| > |Shapley(A)| but head A's mean marginal contribution
    (Banzhaf-style, without ordering weights) exceeds head B's.
    This means head A "does more work" but gets "less credit" — the
    Shapley weighting by coalition ordering unfairly disadvantages A.

    When envy is high, the credit allocation is sensitive to the
    specific cooperative game formulation — different methods would
    assign credit differently.

    Connections:
    - Steinhaus (1948) — first formal treatment of fair division
    - Foley (1967) — envy-freeness criterion
    - Budish (2011) "The Combinatorial Assignment Problem", JPE
      119(6):1061-1103 — approximate envy-freeness
    - Varian (1974) "Equity, Envy, and Efficiency", JET 9:63-91
    - Moulin (2003) "Fair Division and Collective Welfare", MIT Press
    - shapiq (Fumagalli et al. NeurIPS 2024)

Method:
    1. Compute Shapley values for each head (from marginal contributions)
    2. Compute Banzhaf values for each head (uniform coalition weighting)
    3. For each pair (A, B):
       - Shapley says |phi_A| vs |phi_B|
       - Banzhaf says |beta_A| vs |beta_B|
       - If Shapley ranks A below B but Banzhaf ranks A above B (or vice
         versa), this is an "envy pair" — the ranking is method-dependent
    4. Envy-freeness ratio = 1 - (envy_pairs / total_pairs)

Pass condition: envy-freeness ratio > 0.8 (< 20% of pairs show envy).

Usage:
    mechval.run("envy_freeness", tasks=["ioi"], device="cpu")
"""

import itertools
import math

import numpy as np
import torch
from scipy.stats import spearmanr

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    InstrumentInfo,
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
    save_incremental,
    save_results,
)

INSTRUMENT_INFO = InstrumentInfo(
    name="Envy-Freeness Test",
    paper_ref="Foley 1967; Steinhaus 1948; Varian 1974; Moulin 2003",
    paper_cite="Foley 1967, envy-freeness; Steinhaus 1948, fair division",
    description="Tests whether credit allocation (Shapley vs Banzhaf) produces ranking disagreements (envy)",
    category="causal",
    tier="cross_discipline",
    origin="established",
)

ENVY_THRESHOLD = 0.8
MAX_ENUMERATION_HEADS = 10
N_SAMPLES_LARGE = 200


@torch.no_grad()
def _coalition_value(model, prompts, correct_ids, incorrect_ids,
                     active: set[tuple[int, int]],
                     all_heads: set[tuple[int, int]],
                     mean_z: torch.Tensor) -> float:
    ablated = all_heads - active
    if not ablated:
        hooks = []
    else:
        hooks = make_ablation_hook(heads_to_layer_dict(ablated), mean_z, "mean")

    lds = []
    for idx, p in enumerate(prompts):
        if idx >= len(correct_ids):
            break
        tokens = model.to_tokens(p.text)
        if hooks:
            logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
        else:
            logits = model(tokens)
        ld = logit_diff_from_logits(logits, correct_ids[idx], incorrect_ids[idx])
        lds.append(ld)
    return float(np.mean(lds)) if lds else 0.0


@torch.no_grad()
def run_envy_freeness(model, tasks: list[str],
                      n_prompts: int = 20) -> list[EvalResult]:
    tokenizer = model.tokenizer
    results = []

    for task in tasks:
        circuit_heads = get_circuit_heads(task)
        if not circuit_heads or len(circuit_heads) < 2:
            log(f"  {task}: <2 circuit heads, skipping")
            continue

        prompts = generate_prompts(task, tokenizer, n_prompts)
        if not prompts:
            continue

        correct_ids, incorrect_ids = get_token_ids(prompts, tokenizer)
        if not correct_ids:
            continue

        heads = sorted(circuit_heads)
        n_heads = len(heads)
        log(f"  {task}: {n_heads} heads, {len(prompts)} prompts")

        mean_z = calibrate_mean_z(model, prompts, n_calibration=min(100, len(prompts)))

        shapley = {}
        banzhaf = {}

        for h in heads:
            others = [oh for oh in heads if oh != h]
            shapley_marginals = []
            banzhaf_marginals = []

            if n_heads <= MAX_ENUMERATION_HEADS:
                for size in range(len(others) + 1):
                    for coalition in itertools.combinations(others, size):
                        s = set(coalition)
                        v_without = _coalition_value(
                            model, prompts, correct_ids, incorrect_ids,
                            s, circuit_heads, mean_z)
                        v_with = _coalition_value(
                            model, prompts, correct_ids, incorrect_ids,
                            s | {h}, circuit_heads, mean_z)
                        mc = v_with - v_without

                        banzhaf_marginals.append(mc)

                        n = n_heads
                        s_size = len(s)
                        weight = math.factorial(s_size) * math.factorial(n - s_size - 1) / math.factorial(n)
                        shapley_marginals.append(weight * mc)
            else:
                rng = np.random.default_rng(42)

                for _ in range(N_SAMPLES_LARGE):
                    perm = list(rng.permutation(others))
                    idx = rng.integers(0, len(perm) + 1)
                    s = set(perm[:idx])
                    v_without = _coalition_value(
                        model, prompts, correct_ids, incorrect_ids,
                        s, circuit_heads, mean_z)
                    v_with = _coalition_value(
                        model, prompts, correct_ids, incorrect_ids,
                        s | {h}, circuit_heads, mean_z)
                    shapley_marginals.append(v_with - v_without)

                for _ in range(N_SAMPLES_LARGE):
                    mask = rng.integers(0, 2, size=len(others))
                    s = {others[i] for i in range(len(others)) if mask[i]}
                    v_without = _coalition_value(
                        model, prompts, correct_ids, incorrect_ids,
                        s, circuit_heads, mean_z)
                    v_with = _coalition_value(
                        model, prompts, correct_ids, incorrect_ids,
                        s | {h}, circuit_heads, mean_z)
                    banzhaf_marginals.append(v_with - v_without)

            key = f"L{h[0]}H{h[1]}"
            shapley[key] = float(np.mean(shapley_marginals)) if shapley_marginals else 0.0
            banzhaf[key] = float(np.mean(banzhaf_marginals)) if banzhaf_marginals else 0.0

        shapley_rank = sorted(shapley.keys(), key=lambda k: abs(shapley[k]), reverse=True)
        banzhaf_rank = sorted(banzhaf.keys(), key=lambda k: abs(banzhaf[k]), reverse=True)

        shapley_order = {k: i for i, k in enumerate(shapley_rank)}
        banzhaf_order = {k: i for i, k in enumerate(banzhaf_rank)}

        n_pairs = 0
        n_envy = 0
        envy_pairs = []

        for a, b in itertools.combinations(shapley.keys(), 2):
            n_pairs += 1
            shapley_a_better = shapley_order[a] < shapley_order[b]
            banzhaf_a_better = banzhaf_order[a] < banzhaf_order[b]

            if shapley_a_better != banzhaf_a_better:
                n_envy += 1
                envy_pairs.append({
                    "head_a": a, "head_b": b,
                    "shapley_a": shapley[a], "shapley_b": shapley[b],
                    "banzhaf_a": banzhaf[a], "banzhaf_b": banzhaf[b],
                    "shapley_rank_a": shapley_order[a],
                    "shapley_rank_b": shapley_order[b],
                    "banzhaf_rank_a": banzhaf_order[a],
                    "banzhaf_rank_b": banzhaf_order[b],
                })

        envy_freeness = 1.0 - (n_envy / n_pairs) if n_pairs > 0 else 1.0
        passed = envy_freeness >= ENVY_THRESHOLD

        if len(shapley_rank) >= 3:
            rho, pval = spearmanr(
                [shapley_order[k] for k in shapley.keys()],
                [banzhaf_order[k] for k in shapley.keys()],
            )
        else:
            rho, pval = 1.0, 0.0

        log(f"    envy_freeness={envy_freeness:.3f}  "
            f"envy_pairs={n_envy}/{n_pairs}  "
            f"spearman_rho={rho:.3f}")
        log(f"    Shapley rank: {' > '.join(shapley_rank[:5])}")
        log(f"    Banzhaf rank: {' > '.join(banzhaf_rank[:5])}")
        log(f"    [{'PASS (fair)' if passed else 'FAIL (envious)'}]")

        results.append(EvalResult(
            metric_id="GT5.envy_freeness",
            value=envy_freeness,
            n_samples=len(prompts),
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": task,
                "n_heads": n_heads,
                "shapley_values": shapley,
                "banzhaf_values": banzhaf,
                "shapley_ranking": shapley_rank,
                "banzhaf_ranking": banzhaf_rank,
                "spearman_rho": rho,
                "spearman_pval": pval,
                "n_pairs": n_pairs,
                "n_envy_pairs": n_envy,
                "envy_freeness": envy_freeness,
                "envy_pairs": envy_pairs[:10],
                "passed": passed,
                "threshold": ENVY_THRESHOLD,
            },
        ))

    return results


def main():
    parser = parse_common_args("GT5: Envy-Freeness Test")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("GT5: ENVY-FREENESS TEST")
    log("=" * 60)

    out = args.out or "GT5_envy_freeness.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_envy_freeness(model, [task], args.n_prompts)
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)

    save_results(results, out, args=args)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
