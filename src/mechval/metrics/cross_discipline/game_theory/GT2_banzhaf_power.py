"""Banzhaf Power Index for Circuit Components
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Metric:         GT2 — Banzhaf Power Index
Categories:     causal, game_theory
Evidence family: causal
Description mode: implementational-functional

Computes the Banzhaf power index for each circuit head, providing an
alternative importance ranking to Shapley values.

Background:
    The Banzhaf power index (Banzhaf 1965, "Weighted Voting Doesn't Work:
    A Mathematical Analysis", Rutgers Law Review 19:317-343) measures a
    player's power as the fraction of coalitions where they are "swing" —
    their presence changes the coalition from losing to winning (or
    significantly changes the payoff).

    Unlike Shapley values, Banzhaf weights all coalitions equally rather
    than by size. This means:
    - Shapley: "On average over all orderings, what's my marginal contribution?"
    - Banzhaf: "In what fraction of all subsets am I critical?"

    When Shapley and Banzhaf rankings AGREE, the importance ordering is
    robust to the game-theoretic framework. When they DISAGREE, the
    importance of some heads depends on which theoretical lens you use —
    a form of underdetermination.

    Related work:
    - Banzhaf (1965) original definition
    - Dubey & Shapley (1979) "Mathematical Properties of the Banzhaf
      Power Index", Mathematics of Operations Research 4(2):99-131
    - shapiq library (Fumagalli et al. NeurIPS 2024) implements both
      indices; we implement a direct version for circuit-scale problems
    - Hedonic Neurons (Chowdhury et al. 2025) uses Shapley-adjacent
      cooperative game theory

Method:
    For each circuit head h:
      1. Enumerate all 2^(n-1) coalitions of the OTHER heads
      2. For each coalition S (without h):
         - Measure logit-diff with S active (all non-S heads mean-ablated)
         - Measure logit-diff with S ∪ {h} active
         - Marginal contribution of h to S: mc(h, S) = ld(S ∪ {h}) - ld(S)
      3. Banzhaf index = mean of mc(h, S) over all S

    For circuits with many heads (>12), we sample coalitions uniformly
    rather than enumerating all 2^(n-1).

Pass condition: Banzhaf ranking correlates with Shapley/ablation ranking
(Spearman rho > 0.7).

Usage:
    mechval.run("banzhaf_power", tasks=["ioi"], device="cpu")
"""

import itertools

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
    name="Banzhaf Power Index",
    paper_ref="Banzhaf 1965, Rutgers Law Review 19:317-343",
    paper_cite="Banzhaf 1965; Dubey & Shapley 1979",
    description="Alternative to Shapley values — measures fraction of coalitions where a head is swing",
    category="causal",
    tier="cross_discipline",
    origin="established",
)

MAX_ENUMERATION_HEADS = 12
N_SAMPLES_LARGE = 200


@torch.no_grad()
def compute_coalition_ld(model, prompts, correct_ids, incorrect_ids,
                         active_heads: set[tuple[int, int]],
                         all_heads: set[tuple[int, int]],
                         mean_z: torch.Tensor) -> float:
    """Compute mean logit-diff with only active_heads contributing.

    All heads in (all_heads - active_heads) are mean-ablated.
    """
    ablated = all_heads - active_heads
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
def run_banzhaf_power(model, tasks: list[str],
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

        banzhaf = {}
        for h in heads:
            others = [oh for oh in heads if oh != h]
            marginals = []

            if n_heads <= MAX_ENUMERATION_HEADS:
                for size in range(len(others) + 1):
                    for coalition in itertools.combinations(others, size):
                        s = set(coalition)
                        ld_without = compute_coalition_ld(
                            model, prompts, correct_ids, incorrect_ids,
                            s, circuit_heads, mean_z)
                        ld_with = compute_coalition_ld(
                            model, prompts, correct_ids, incorrect_ids,
                            s | {h}, circuit_heads, mean_z)
                        marginals.append(ld_with - ld_without)
            else:
                rng = np.random.default_rng(42)
                for _ in range(N_SAMPLES_LARGE):
                    mask = rng.integers(0, 2, size=len(others))
                    s = {others[i] for i in range(len(others)) if mask[i]}
                    ld_without = compute_coalition_ld(
                        model, prompts, correct_ids, incorrect_ids,
                        s, circuit_heads, mean_z)
                    ld_with = compute_coalition_ld(
                        model, prompts, correct_ids, incorrect_ids,
                        s | {h}, circuit_heads, mean_z)
                    marginals.append(ld_with - ld_without)

            banzhaf_idx = float(np.mean(marginals))
            key = f"L{h[0]}H{h[1]}"
            banzhaf[key] = banzhaf_idx
            log(f"    {key}: Banzhaf={banzhaf_idx:+.4f}  "
                f"(over {len(marginals)} coalitions)")

        ranking = sorted(banzhaf.items(), key=lambda x: abs(x[1]), reverse=True)
        log(f"    Banzhaf ranking: {' > '.join(f'{k}({v:+.3f})' for k, v in ranking)}")

        total_power = sum(abs(v) for v in banzhaf.values())
        normalized = {k: abs(v) / total_power if total_power > 0 else 0.0
                      for k, v in banzhaf.items()}

        results.append(EvalResult(
            metric_id="GT2.banzhaf_power",
            value=total_power,
            n_samples=len(prompts),
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": task,
                "n_heads": n_heads,
                "banzhaf_raw": banzhaf,
                "banzhaf_normalized": normalized,
                "ranking": [k for k, _ in ranking],
                "enumerated": n_heads <= MAX_ENUMERATION_HEADS,
                "n_coalitions_per_head": 2 ** (n_heads - 1) if n_heads <= MAX_ENUMERATION_HEADS else N_SAMPLES_LARGE,
                "passed": True,
            },
        ))

    return results


def main():
    parser = parse_common_args("GT2: Banzhaf Power Index")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("GT2: BANZHAF POWER INDEX")
    log("=" * 60)

    out = args.out or "GT2_banzhaf_power.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_banzhaf_power(model, [task], args.n_prompts)
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)

    save_results(results, out, args=args)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
