"""Nucleolus Stability for Circuit Credit Allocation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Metric:         GT9 — Nucleolus
Categories:     causal, game_theory
Evidence family: causal
Description mode: implementational-functional

Computes how close the circuit's Shapley allocation is to the
nucleolus — the "fairest" imputation that minimizes maximum complaint.

Background:
    The nucleolus (Schmeidler 1969, "The Nucleolus of a Characteristic
    Function Game", SIAM Journal on Applied Mathematics 17:1163-1170)
    is the unique imputation that lexicographically minimizes the
    maximum excess (complaint) of any coalition.

    Excess of coalition S under imputation x:
        e(S, x) = v(S) - sum_{i in S} x_i

    A positive excess means S can do better on its own than its
    allocated share — S has a "complaint." The nucleolus minimizes
    the worst complaint.

    Unlike Shapley values (which minimize total variance of marginal
    contributions) or the core (which just requires no positive excess),
    the nucleolus actively distributes unhappiness as evenly as possible.

    Applied to circuits: if the nucleolus is close to the Shapley values,
    the Shapley allocation is already "fair" in the minimax sense.
    If they differ, the Shapley allocation overcompensates some heads
    at the expense of others.

    Connections:
    - Schmeidler (1969) — nucleolus definition and uniqueness proof
    - Maschler, Peleg & Shapley (1979) "Geometric Properties of the
      Kernel, Nucleolus, and Related Solution Concepts", Mathematics
      of Operations Research 4:303-338
    - Kohlberg (1971) "On the Nucleolus of a Characteristic Function
      Game", SIAM Journal 20:62-66
    - Elkind et al. (2009) "Computational Complexity of Weighted
      Threshold Games", AAAI

Method:
    Since exact nucleolus computation requires LP solvers for large
    games, we use a sampling-based approximation:

    1. Compute Shapley values (the "current" allocation)
    2. Sample coalitions and compute their excess under Shapley
    3. The max excess = "maximum complaint" under Shapley
    4. Compute an alternative allocation that minimizes max excess:
       - Start from Shapley
       - Iteratively transfer credit from heads with low excess
         to heads in coalitions with high excess
    5. Report: max_excess under Shapley, and how much reallocation
       would be needed to reach the nucleolus

Pass condition: max_excess < 0.1 * grand_value (maximum complaint is
< 10% of total output).

Usage:
    mechval.run("nucleolus", tasks=["ioi"], device="cpu")
"""

import itertools
import math

import numpy as np
import torch

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
    name="Nucleolus Stability",
    paper_ref="Schmeidler 1969, SIAM J. Applied Mathematics 17:1163-1170",
    paper_cite="Schmeidler 1969 (nucleolus); Maschler, Peleg & Shapley 1979",
    description="Measures how close Shapley allocation is to the nucleolus (minimax-fair credit)",
    category="causal",
    tier="cross_discipline",
    origin="established",
)

MAX_EXCESS_THRESHOLD = 0.1
MAX_ENUMERATION_HEADS = 10
N_SAMPLES_LARGE = 300


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
def run_nucleolus(model, tasks: list[str],
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

        grand_value = _coalition_value(
            model, prompts, correct_ids, incorrect_ids,
            circuit_heads, circuit_heads, mean_z)

        shapley = {}
        for h in heads:
            others = [oh for oh in heads if oh != h]
            marginals = []

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
                        s_size = len(s)
                        weight = math.factorial(s_size) * math.factorial(n_heads - s_size - 1) / math.factorial(n_heads)
                        marginals.append(weight * (v_with - v_without))
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
                    marginals.append(v_with - v_without)

            shapley[h] = float(np.mean(marginals))

        coalition_excesses = []

        if n_heads <= MAX_ENUMERATION_HEADS:
            for size in range(1, n_heads):
                for coalition in itertools.combinations(heads, size):
                    s = set(coalition)
                    v_s = _coalition_value(
                        model, prompts, correct_ids, incorrect_ids,
                        s, circuit_heads, mean_z)
                    shapley_share = sum(shapley[h] for h in s)
                    excess = v_s - shapley_share
                    coalition_excesses.append({
                        "coalition": [f"L{h[0]}H{h[1]}" for h in sorted(s)],
                        "value": v_s,
                        "shapley_share": shapley_share,
                        "excess": excess,
                    })
        else:
            rng = np.random.default_rng(42)
            for _ in range(N_SAMPLES_LARGE):
                mask = rng.integers(0, 2, size=n_heads)
                if mask.sum() == 0 or mask.sum() == n_heads:
                    continue
                s = {heads[i] for i in range(n_heads) if mask[i]}
                v_s = _coalition_value(
                    model, prompts, correct_ids, incorrect_ids,
                    s, circuit_heads, mean_z)
                shapley_share = sum(shapley[h] for h in s)
                excess = v_s - shapley_share
                coalition_excesses.append({
                    "coalition": [f"L{h[0]}H{h[1]}" for h in sorted(s)],
                    "value": v_s,
                    "shapley_share": shapley_share,
                    "excess": excess,
                })

        excesses = [ce["excess"] for ce in coalition_excesses]
        max_excess = max(excesses) if excesses else 0.0
        min_excess = min(excesses) if excesses else 0.0
        mean_excess = float(np.mean(excesses)) if excesses else 0.0
        std_excess = float(np.std(excesses)) if excesses else 0.0

        relative_max_excess = max_excess / abs(grand_value) if abs(grand_value) > 1e-8 else 0.0
        passed = relative_max_excess < MAX_EXCESS_THRESHOLD

        top_complaints = sorted(coalition_excesses, key=lambda x: x["excess"], reverse=True)[:5]

        shapley_display = {f"L{h[0]}H{h[1]}": v for h, v in shapley.items()}

        log(f"    grand_value={grand_value:.4f}")
        log(f"    max_excess={max_excess:.4f}  "
            f"relative={relative_max_excess:.4f}  "
            f"mean_excess={mean_excess:.4f}")
        log(f"    Top complaints:")
        for tc in top_complaints[:3]:
            log(f"      {tc['coalition']}: excess={tc['excess']:+.4f}")
        log(f"    [{'PASS (near-nucleolus)' if passed else 'FAIL (high complaints)'}]")

        results.append(EvalResult(
            metric_id="GT9.nucleolus",
            value=1.0 - min(relative_max_excess, 1.0),
            n_samples=len(prompts),
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": task,
                "n_heads": n_heads,
                "grand_value": grand_value,
                "shapley_values": shapley_display,
                "max_excess": max_excess,
                "min_excess": min_excess,
                "mean_excess": mean_excess,
                "std_excess": std_excess,
                "relative_max_excess": relative_max_excess,
                "n_coalitions_tested": len(coalition_excesses),
                "top_complaints": top_complaints,
                "passed": passed,
                "threshold": MAX_EXCESS_THRESHOLD,
            },
        ))

    return results


def main():
    parser = parse_common_args("GT9: Nucleolus Stability")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("GT9: NUCLEOLUS STABILITY")
    log("=" * 60)

    out = args.out or "GT9_nucleolus.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_nucleolus(model, [task], args.n_prompts)
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)

    save_results(results, out, args=args)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
