"""Nash Equilibrium Test for Circuit Components
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Metric:         GT1 — Nash Equilibrium
Categories:     causal, game_theory
Evidence family: causal
Description mode: implementational-functional

Tests whether a circuit's component configuration is a Nash equilibrium:
can any single head unilaterally improve the circuit's output by changing
its contribution, while all other heads remain fixed?

Background:
    In game theory, a Nash equilibrium (Nash 1950, "Equilibrium Points in
    N-Person Games", PNAS 36(1):48-49) is a strategy profile where no
    player can improve their payoff by unilaterally changing their strategy.

    Applied to circuits: each head is a "player" whose "strategy" is its
    current weight configuration. The "payoff" is the circuit's task
    performance (logit diff). If no head can unilaterally improve the
    circuit by scaling/rotating its contribution, the circuit is at a
    Nash equilibrium — a stable fixed point.

    Connections:
    - Nash (1950) original definition
    - Nisan et al. (2007) "Algorithmic Game Theory", Ch. 1-3
    - Hedonic Neurons (Chowdhury et al. 2025, arXiv:2509.23684) use
      cooperative game theory; this metric uses non-cooperative framing

Method:
    For each circuit head h:
      1. Compute baseline logit-diff with all heads at their current values
      2. Scale head h's output by factors [0.0, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
      3. For each scaling, measure logit-diff with all other heads fixed
      4. If any scaling != 1.0 improves logit-diff, head h has a profitable
         deviation — the circuit is NOT at Nash equilibrium for this head

    A circuit at Nash equilibrium has each head at its locally optimal
    contribution level. Heads with profitable deviations are "misallocated"
    — the circuit could be improved by rescaling them.

Pass condition: No head has a profitable deviation > 5% of baseline logit-diff.

Usage:
    mechval.run("nash_equilibrium", tasks=["ioi"], device="cpu")
"""

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
    load_model,
    log,
    logit_diff_from_logits,
    parse_common_args,
    save_incremental,
    save_results,
)

INSTRUMENT_INFO = InstrumentInfo(
    name="Nash Equilibrium Test",
    paper_ref="Nash 1950, PNAS 36(1):48-49; Nisan et al. 2007, Algorithmic Game Theory",
    paper_cite="Nash 1950; adapted for circuits",
    description="Tests whether circuit heads are at a Nash equilibrium (no profitable unilateral deviation)",
    category="causal",
    tier="cross_discipline",
    origin="established",
)

DEVIATION_THRESHOLD = 0.05
SCALE_FACTORS = [0.0, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]


@torch.no_grad()
def run_nash_equilibrium(model, tasks: list[str],
                         n_prompts: int = 40) -> list[EvalResult]:
    tokenizer = model.tokenizer
    results = []

    for task in tasks:
        circuit_heads = get_circuit_heads(task)
        if not circuit_heads:
            log(f"  {task}: no circuit heads, skipping")
            continue

        prompts = generate_prompts(task, tokenizer, n_prompts)
        if not prompts:
            log(f"  {task}: no prompts, skipping")
            continue

        correct_ids, incorrect_ids = get_token_ids(prompts, tokenizer)
        if not correct_ids:
            continue

        log(f"  {task}: {len(circuit_heads)} heads, {len(prompts)} prompts")

        baseline_lds = []
        for idx, p in enumerate(prompts):
            if idx >= len(correct_ids):
                break
            tokens = model.to_tokens(p.text)
            logits = model(tokens)
            ld = logit_diff_from_logits(logits, correct_ids[idx], incorrect_ids[idx])
            baseline_lds.append(ld)

        baseline_mean_ld = float(np.mean(baseline_lds))
        if abs(baseline_mean_ld) < 1e-6:
            log(f"  {task}: baseline logit-diff near zero, skipping")
            continue

        per_head = {}
        has_deviation = False

        for layer, head in sorted(circuit_heads):
            best_scale = 1.0
            best_ld = baseline_mean_ld
            scale_results = {}

            for scale in SCALE_FACTORS:
                def make_scale_hook(target_layer, target_head, scale_factor):
                    def hook_fn(value, hook):
                        value[:, :, target_head] = value[:, :, target_head] * scale_factor
                        return value
                    return (f"blocks.{target_layer}.attn.hook_z", hook_fn)

                hook = make_scale_hook(layer, head, scale)
                scaled_lds = []
                for idx, p in enumerate(prompts):
                    if idx >= len(correct_ids):
                        break
                    tokens = model.to_tokens(p.text)
                    logits = model.run_with_hooks(tokens, fwd_hooks=[hook])
                    ld = logit_diff_from_logits(logits, correct_ids[idx], incorrect_ids[idx])
                    scaled_lds.append(ld)

                mean_scaled_ld = float(np.mean(scaled_lds))
                scale_results[scale] = mean_scaled_ld

                if mean_scaled_ld > best_ld:
                    best_ld = mean_scaled_ld
                    best_scale = scale

            improvement = (best_ld - baseline_mean_ld) / abs(baseline_mean_ld) if baseline_mean_ld != 0 else 0.0
            is_deviation = improvement > DEVIATION_THRESHOLD
            if is_deviation:
                has_deviation = True

            key = f"L{layer}H{head}"
            per_head[key] = {
                "best_scale": best_scale,
                "best_ld": best_ld,
                "improvement_frac": improvement,
                "has_deviation": is_deviation,
                "scale_curve": {str(s): v for s, v in scale_results.items()},
            }
            label = "DEVIATION" if is_deviation else "equilibrium"
            log(f"    {key}: best_scale={best_scale:.1f}  "
                f"improvement={improvement:+.3f}  [{label}]")

        passed = not has_deviation
        n_deviations = sum(1 for h in per_head.values() if h["has_deviation"])

        log(f"    Nash: {n_deviations}/{len(per_head)} heads with profitable deviations  "
            f"[{'PASS (equilibrium)' if passed else 'FAIL (not equilibrium)'}]")

        results.append(EvalResult(
            metric_id="GT1.nash_equilibrium",
            value=1.0 - (n_deviations / len(per_head)) if per_head else 0.0,
            n_samples=len(prompts),
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": task,
                "n_heads": len(circuit_heads),
                "baseline_mean_ld": baseline_mean_ld,
                "n_deviations": n_deviations,
                "is_equilibrium": passed,
                "per_head": per_head,
                "passed": passed,
                "threshold": DEVIATION_THRESHOLD,
            },
        ))

    return results


def main():
    parser = parse_common_args("GT1: Nash Equilibrium Test")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("GT1: NASH EQUILIBRIUM TEST")
    log("=" * 60)

    out = args.out or "GT1_nash_equilibrium.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_nash_equilibrium(model, [task], args.n_prompts)
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)

    save_results(results, out, args=args)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
