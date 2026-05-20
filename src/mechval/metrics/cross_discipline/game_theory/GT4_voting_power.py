"""Voting Power Analysis for Circuit Components
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Metric:         GT4 — Voting Power (Social Choice)
Categories:     causal, game_theory
Evidence family: causal
Description mode: implementational-functional

Treats each circuit head as a "voter" that casts a weighted ballot
for the correct vs incorrect output token, then analyzes the voting
structure using social choice theory.

Background:
    Social choice theory (Arrow 1951, "Social Choice and Individual
    Values"; Sen 1970, "Collective Choice and Social Welfare") studies
    how individual preferences aggregate into collective decisions.
    The Condorcet jury theorem (1785) shows that majority voting
    among imperfect-but-better-than-chance voters converges to the
    correct answer as N grows.

    Applied to circuits: each head contributes a "vote" toward the
    correct or incorrect token via its output logit contribution.
    This metric measures:
    - Whether individual heads are "better than chance" (positive
      contribution to the correct logit direction)
    - Whether the aggregation is majority-rule-like or dictatorial
    - The effective number of voters (inverse HHI concentration)
    - Condorcet efficiency: does the winner match what a simple
      majority of heads would choose?

    Connections:
    - Arrow (1951) impossibility theorem — no voting system satisfies
      all desirable properties simultaneously
    - Condorcet (1785) jury theorem — majority rule among independent,
      better-than-chance voters converges to truth
    - Herfindahl-Hirschman Index (HHI) — concentration measure from
      industrial organization (Herfindahl 1950; Hirschman 1964)
    - Penrose (1946) "The Elementary Statistics of Majority Voting",
      Journal of the Royal Statistical Society 109:53-57
    - shapiq (Fumagalli et al. NeurIPS 2024) for interaction indices

Method:
    For each prompt:
      1. Run the model and cache hook_z at every attention layer
      2. For each circuit head h at (layer, head):
         - Extract its output contribution: z_h @ W_O[h] @ W_U
         - Compute the head's "vote": logit(correct) - logit(incorrect)
           from that contribution alone
      3. Tally: how many heads vote correct? How many vote incorrect?
      4. Does the majority match the model's actual output?
    Aggregate:
      - voting_accuracy: fraction of prompts where majority rule gives
        the correct answer
      - condorcet_efficiency: fraction where majority matches actual output
      - effective_voters: 1/HHI of absolute vote magnitudes
      - dictator_ratio: max |vote| / total |votes|

Pass condition: effective_voters > 2 (not dominated by a single head).

Usage:
    mechval.run("voting_power", tasks=["ioi"], device="cpu")
"""

import numpy as np
import torch

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    InstrumentInfo,
    generate_prompts,
    get_circuit_heads,
    get_token_ids,
    load_model,
    log,
    parse_common_args,
    save_incremental,
    save_results,
)

INSTRUMENT_INFO = InstrumentInfo(
    name="Voting Power Analysis",
    paper_ref="Arrow 1951; Condorcet 1785; Penrose 1946",
    paper_cite="Arrow 1951, Social Choice and Individual Values; Condorcet 1785, jury theorem",
    description="Social choice analysis — treats heads as voters and measures concentration, majority accuracy, Condorcet efficiency",
    category="causal",
    tier="cross_discipline",
    origin="established",
)

MIN_EFFECTIVE_VOTERS = 2.0


@torch.no_grad()
def run_voting_power(model, tasks: list[str],
                     n_prompts: int = 40) -> list[EvalResult]:
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

        W_U = model.W_U.detach()

        per_prompt_votes = []
        majority_correct_count = 0
        condorcet_match_count = 0
        total_prompts = 0

        for idx, p in enumerate(prompts):
            if idx >= len(correct_ids):
                break
            tokens = model.to_tokens(p.text)
            logits, cache = model.run_with_cache(
                tokens,
                names_filter=lambda n: "hook_z" in n or "hook_result" in n,
            )

            actual_ld = (logits[0, -1, correct_ids[idx]] - logits[0, -1, incorrect_ids[idx]]).item()

            votes = {}
            for layer, head in heads:
                z = cache[f"blocks.{layer}.attn.hook_z"][0, -1, head]
                W_O_h = model.W_O[layer, head]
                contribution = z @ W_O_h @ W_U
                vote = (contribution[correct_ids[idx]] - contribution[incorrect_ids[idx]]).item()
                votes[f"L{layer}H{head}"] = vote

            per_prompt_votes.append(votes)

            n_positive = sum(1 for v in votes.values() if v > 0)
            majority_positive = n_positive > n_heads / 2
            total_prompts += 1

            if majority_positive:
                majority_correct_count += 1

            actual_correct = actual_ld > 0
            majority_matches_actual = majority_positive == actual_correct
            if majority_matches_actual:
                condorcet_match_count += 1

        if total_prompts == 0:
            continue

        mean_votes = {}
        for key in per_prompt_votes[0]:
            mean_votes[key] = float(np.mean([pv[key] for pv in per_prompt_votes]))

        abs_votes = np.array([abs(v) for v in mean_votes.values()])
        total_abs = abs_votes.sum()
        if total_abs > 0:
            shares = abs_votes / total_abs
            hhi = float(np.sum(shares ** 2))
            effective_voters = 1.0 / hhi if hhi > 0 else n_heads
            dictator_ratio = float(shares.max())
        else:
            effective_voters = 0.0
            dictator_ratio = 0.0
            hhi = 1.0

        voting_accuracy = majority_correct_count / total_prompts
        condorcet_efficiency = condorcet_match_count / total_prompts
        passed = effective_voters >= MIN_EFFECTIVE_VOTERS

        positive_voter_frac = float(np.mean([
            sum(1 for v in pv.values() if v > 0) / n_heads
            for pv in per_prompt_votes
        ]))

        ranking = sorted(mean_votes.items(), key=lambda x: x[1], reverse=True)

        log(f"    effective_voters={effective_voters:.2f}  "
            f"voting_accuracy={voting_accuracy:.3f}  "
            f"condorcet_eff={condorcet_efficiency:.3f}  "
            f"dictator_ratio={dictator_ratio:.3f}")
        log(f"    vote ranking: {' > '.join(f'{k}({v:+.3f})' for k, v in ranking[:5])}")
        log(f"    [{'PASS (distributed)' if passed else 'FAIL (concentrated)'}]")

        results.append(EvalResult(
            metric_id="GT4.voting_power",
            value=effective_voters,
            n_samples=len(prompts),
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": task,
                "n_heads": n_heads,
                "mean_votes": mean_votes,
                "effective_voters": effective_voters,
                "hhi": hhi,
                "dictator_ratio": dictator_ratio,
                "voting_accuracy": voting_accuracy,
                "condorcet_efficiency": condorcet_efficiency,
                "positive_voter_fraction": positive_voter_frac,
                "ranking": [k for k, _ in ranking],
                "passed": passed,
                "threshold": MIN_EFFECTIVE_VOTERS,
            },
        ))

    return results


def main():
    parser = parse_common_args("GT4: Voting Power Analysis")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("GT4: VOTING POWER ANALYSIS")
    log("=" * 60)

    out = args.out or "GT4_voting_power.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_voting_power(model, [task], args.n_prompts)
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)

    save_results(results, out, args=args)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
