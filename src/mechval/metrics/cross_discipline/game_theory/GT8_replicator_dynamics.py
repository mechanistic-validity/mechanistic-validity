"""Evolutionary Game Theory / Replicator Dynamics for Circuits
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Metric:         GT8 — Replicator Dynamics
Categories:     causal, game_theory
Evidence family: causal
Description mode: implementational-functional

Simulates evolutionary dynamics on the circuit: if heads could
"replicate" in proportion to their fitness (task contribution),
which heads would dominate and which would go extinct?

Background:
    The replicator equation (Taylor & Jonker 1978, "Evolutionary Stable
    Strategies and Game Dynamics", Mathematical Biosciences 40:145-156)
    models population dynamics where strategies that perform above
    average grow, and those below average shrink:

        dx_i/dt = x_i * (f_i(x) - f_bar(x))

    where x_i is the frequency of strategy i, f_i is its fitness, and
    f_bar is the population mean fitness.

    Applied to circuits: each head has a "population share" (its
    relative contribution to the output). Fitness is measured as the
    head's marginal contribution when scaled up/down. The replicator
    equation predicts whether the current allocation is evolutionarily
    stable — resistant to invasion by mutant strategies.

    An Evolutionarily Stable Strategy (ESS, Maynard Smith & Price 1973,
    "The Logic of Animal Conflict", Nature 246:15-18) is a strategy
    that, once established, cannot be invaded by any small mutation.

    Connections:
    - Taylor & Jonker (1978) — replicator equation
    - Maynard Smith & Price (1973) — ESS definition
    - Weibull (1995) "Evolutionary Game Theory", MIT Press
    - Hofbauer & Sigmund (1998) "Evolutionary Games and Population
      Dynamics", Cambridge University Press
    - Nowak (2006) "Evolutionary Dynamics", Harvard University Press

Method:
    1. Measure each head's "fitness" = logit-diff contribution (via
       leave-one-out: ld_full - ld_without_h)
    2. Compute mean fitness across heads
    3. Simulate replicator dynamics for T steps:
       - x_i(t+1) = x_i(t) * (1 + eta * (f_i - f_bar))
       - Normalize so sum(x_i) = 1
       - eta = learning rate (default 0.1)
    4. Track convergence: do population shares stabilize?
    5. At equilibrium: which heads dominate (x_i > 2/n)?
       Which go extinct (x_i < 0.01)?
    6. ESS check: at the fixed point, is any head's fitness above
       the mean? If all fitnesses equal the mean, it's an ESS.

Pass condition: dynamics converge (population shares stabilize within
50 steps) AND no head dominates > 50% of total share.

Usage:
    mechval.run("replicator_dynamics", tasks=["ioi"], device="cpu")
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
    name="Replicator Dynamics",
    paper_ref="Taylor & Jonker 1978; Maynard Smith & Price 1973; Weibull 1995",
    paper_cite="Taylor & Jonker 1978 (replicator equation); Maynard Smith & Price 1973 (ESS)",
    description="Simulates evolutionary dynamics on head contributions — tests for evolutionary stability",
    category="causal",
    tier="cross_discipline",
    origin="established",
)

REPLICATOR_STEPS = 50
LEARNING_RATE = 0.1
CONVERGENCE_THRESHOLD = 1e-4
DOMINANCE_THRESHOLD = 0.5
EXTINCTION_THRESHOLD = 0.01


@torch.no_grad()
def run_replicator_dynamics(model, tasks: list[str],
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

        full_ld = _compute_mean_ld(model, prompts, correct_ids, incorrect_ids)

        fitness = np.zeros(n_heads)
        for i, h in enumerate(heads):
            ablate_hooks = make_ablation_hook(
                heads_to_layer_dict({h}), mean_z, "mean")
            without_ld = _compute_mean_ld_with_hooks(
                model, prompts, correct_ids, incorrect_ids, ablate_hooks)
            fitness[i] = full_ld - without_ld

        fitness = np.maximum(fitness, 0.0)

        x = np.ones(n_heads) / n_heads
        trajectory = [x.copy()]
        converged = False
        convergence_step = REPLICATOR_STEPS

        for step in range(REPLICATOR_STEPS):
            f_bar = np.dot(x, fitness)

            dx = x * (fitness - f_bar)
            x_new = x + LEARNING_RATE * dx

            x_new = np.maximum(x_new, 1e-10)
            x_new /= x_new.sum()

            delta = np.max(np.abs(x_new - x))
            x = x_new
            trajectory.append(x.copy())

            if delta < CONVERGENCE_THRESHOLD:
                converged = True
                convergence_step = step + 1
                break

        dominant = [i for i in range(n_heads) if x[i] > DOMINANCE_THRESHOLD]
        extinct = [i for i in range(n_heads) if x[i] < EXTINCTION_THRESHOLD]

        max_share = float(x.max())
        has_monopoly = max_share > DOMINANCE_THRESHOLD
        passed = converged and not has_monopoly

        final_f_bar = np.dot(x, fitness)
        fitness_deviations = fitness - final_f_bar
        is_ess = bool(np.all(np.abs(fitness_deviations[x > EXTINCTION_THRESHOLD]) < 0.01 * abs(final_f_bar + 1e-8)))

        head_results = {}
        for i, h in enumerate(heads):
            key = f"L{h[0]}H{h[1]}"
            head_results[key] = {
                "fitness": float(fitness[i]),
                "initial_share": 1.0 / n_heads,
                "final_share": float(x[i]),
                "status": ("dominant" if i in dominant
                          else "extinct" if i in extinct
                          else "surviving"),
            }

        ranking = sorted(head_results.items(),
                        key=lambda kv: kv[1]["final_share"], reverse=True)

        log(f"    full_ld={full_ld:.4f}  "
            f"converged={converged} (step {convergence_step})  "
            f"ESS={is_ess}")
        for k, v in ranking[:5]:
            log(f"    {k}: fitness={v['fitness']:+.4f}  "
                f"share={v['final_share']:.3f} [{v['status']}]")
        log(f"    dominant={len(dominant)}  extinct={len(extinct)}  "
            f"max_share={max_share:.3f}  "
            f"[{'PASS (stable ecosystem)' if passed else 'FAIL (monopoly or unstable)'}]")

        traj_sampled = []
        sample_steps = [0, 1, 2, 5, 10, 25, min(len(trajectory) - 1, 49)]
        for s in sorted(set(sample_steps)):
            if s < len(trajectory):
                traj_sampled.append({
                    "step": s,
                    "shares": {f"L{heads[i][0]}H{heads[i][1]}": float(trajectory[s][i])
                              for i in range(n_heads)},
                })

        results.append(EvalResult(
            metric_id="GT8.replicator_dynamics",
            value=1.0 - max_share,
            n_samples=len(prompts),
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": task,
                "n_heads": n_heads,
                "full_logit_diff": full_ld,
                "per_head": head_results,
                "ranking": [k for k, _ in ranking],
                "converged": converged,
                "convergence_step": convergence_step,
                "is_ess": is_ess,
                "max_share": max_share,
                "n_dominant": len(dominant),
                "n_extinct": len(extinct),
                "trajectory_samples": traj_sampled,
                "passed": passed,
                "dominance_threshold": DOMINANCE_THRESHOLD,
            },
        ))

    return results


@torch.no_grad()
def _compute_mean_ld(model, prompts, correct_ids, incorrect_ids) -> float:
    lds = []
    for idx, p in enumerate(prompts):
        if idx >= len(correct_ids):
            break
        tokens = model.to_tokens(p.text)
        logits = model(tokens)
        ld = logit_diff_from_logits(logits, correct_ids[idx], incorrect_ids[idx])
        lds.append(ld)
    return float(np.mean(lds)) if lds else 0.0


@torch.no_grad()
def _compute_mean_ld_with_hooks(model, prompts, correct_ids, incorrect_ids,
                                hooks) -> float:
    lds = []
    for idx, p in enumerate(prompts):
        if idx >= len(correct_ids):
            break
        tokens = model.to_tokens(p.text)
        logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
        ld = logit_diff_from_logits(logits, correct_ids[idx], incorrect_ids[idx])
        lds.append(ld)
    return float(np.mean(lds)) if lds else 0.0


def main():
    parser = parse_common_args("GT8: Replicator Dynamics")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("GT8: REPLICATOR DYNAMICS (EVOLUTIONARY GAME THEORY)")
    log("=" * 60)

    out = args.out or "GT8_replicator_dynamics.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_replicator_dynamics(model, [task], args.n_prompts)
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)

    save_results(results, out, args=args)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
