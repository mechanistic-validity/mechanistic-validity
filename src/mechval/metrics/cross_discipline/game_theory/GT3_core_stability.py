"""Core Stability Test for Circuit Coalitions
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Metric:         GT3 — Core Stability
Categories:     causal, game_theory
Evidence family: causal
Description mode: implementational-functional

Tests whether the circuit's coalition structure is in the "core" of the
cooperative game — no subset of heads can profitably deviate.

Background:
    The core of a cooperative game (Gillies 1959, "Solutions to General
    Non-Zero-Sum Games", Annals of Mathematics Studies 40:47-85) is the
    set of payoff allocations where no coalition can do better by breaking
    away. If the core is empty, the grand coalition is inherently unstable.

    Applied to circuits: the "grand coalition" is all circuit heads working
    together. A "blocking coalition" is a subset of heads that achieves
    better task performance than their allocated share of the full circuit's
    output. If blocking coalitions exist, the circuit decomposition is
    unstable — there's a better sub-grouping.

    Connections:
    - Gillies (1959) original core definition
    - Bondareva-Shapley theorem: the core is non-empty iff the game is
      "balanced" (Bondareva 1963, Shapley 1967)
    - Hedonic Neurons (Chowdhury et al. 2025) use hedonic games where
      stability is PAC-approximated; this metric uses the classical core

Method:
    1. Compute v(N) = logit-diff with all circuit heads (grand coalition)
    2. Compute Shapley values phi(i) for each head i
    3. For each subset S of circuit heads (sample if > 12 heads):
       - Compute v(S) = logit-diff with only S active
       - Check: is v(S) > sum of phi(i) for i in S?
       - If yes: S is a blocking coalition (they do better together than
         their Shapley-allocated share)
    4. Core stability ratio = fraction of coalitions that are NOT blocking

Pass condition: core stability ratio > 0.9 (< 10% blocking coalitions).

Usage:
    mechval.run("core_stability", tasks=["ioi"], device="cpu")
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
    name="Core Stability Test",
    paper_ref="Gillies 1959; Bondareva 1963; Shapley 1967",
    paper_cite="Gillies 1959; Bondareva-Shapley theorem",
    description="Tests whether the circuit's cooperative game has a non-empty core (no blocking coalitions)",
    category="causal",
    tier="cross_discipline",
    origin="established",
)

STABILITY_THRESHOLD = 0.9
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
def run_core_stability(model, tasks: list[str],
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

        individual_values = {}
        for h in heads:
            v = _coalition_value(
                model, prompts, correct_ids, incorrect_ids,
                {h}, circuit_heads, mean_z)
            individual_values[h] = v

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
                        weight = 1.0
                        marginals.append(weight * (v_with - v_without))
            else:
                rng = np.random.default_rng(42)
                for _ in range(N_SAMPLES_LARGE):
                    perm = rng.permutation(others)
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

        n_blocking = 0
        n_tested = 0
        blocking_coalitions = []

        if n_heads <= MAX_ENUMERATION_HEADS:
            for size in range(1, n_heads):
                for coalition in itertools.combinations(heads, size):
                    s = set(coalition)
                    v_s = _coalition_value(
                        model, prompts, correct_ids, incorrect_ids,
                        s, circuit_heads, mean_z)
                    shapley_share = sum(shapley[h] for h in s)
                    n_tested += 1
                    if v_s > shapley_share + 0.01:
                        n_blocking += 1
                        blocking_coalitions.append({
                            "coalition": [f"L{h[0]}H{h[1]}" for h in sorted(s)],
                            "value": v_s,
                            "shapley_share": shapley_share,
                            "excess": v_s - shapley_share,
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
                n_tested += 1
                if v_s > shapley_share + 0.01:
                    n_blocking += 1
                    blocking_coalitions.append({
                        "coalition": [f"L{h[0]}H{h[1]}" for h in sorted(s)],
                        "value": v_s,
                        "shapley_share": shapley_share,
                        "excess": v_s - shapley_share,
                    })

        stability_ratio = 1.0 - (n_blocking / n_tested) if n_tested > 0 else 1.0
        passed = stability_ratio >= STABILITY_THRESHOLD

        shapley_display = {f"L{h[0]}H{h[1]}": v for h, v in shapley.items()}

        log(f"    grand_value={grand_value:.4f}")
        for k, v in sorted(shapley_display.items()):
            log(f"    Shapley {k}: {v:+.4f}")
        log(f"    blocking: {n_blocking}/{n_tested} coalitions  "
            f"stability={stability_ratio:.3f}  "
            f"[{'PASS (core non-empty)' if passed else 'FAIL (core empty)'}]")

        results.append(EvalResult(
            metric_id="GT3.core_stability",
            value=stability_ratio,
            n_samples=len(prompts),
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": task,
                "n_heads": n_heads,
                "grand_value": grand_value,
                "shapley_values": shapley_display,
                "n_coalitions_tested": n_tested,
                "n_blocking": n_blocking,
                "stability_ratio": stability_ratio,
                "blocking_coalitions": blocking_coalitions[:10],
                "passed": passed,
                "threshold": STABILITY_THRESHOLD,
            },
        ))

    return results


def main():
    parser = parse_common_args("GT3: Core Stability Test")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("GT3: CORE STABILITY TEST")
    log("=" * 60)

    out = args.out or "GT3_core_stability.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_core_stability(model, [task], args.n_prompts)
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)

    save_results(results, out, args=args)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
