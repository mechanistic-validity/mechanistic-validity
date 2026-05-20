"""PAC-Stable Coalition Discovery for Circuit Components
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Metric:         GT6 — Coalition Discovery (PAC-Top-Cover)
Categories:     causal, game_theory
Evidence family: causal
Description mode: implementational-functional

Discovers stable coalitions of heads using an approach inspired by
PAC-Top-Cover from hedonic game theory, adapted for neural circuits.

Background:
    In hedonic games (Drèze & Greenberg 1980, "Hedonic Coalitions:
    Optimality and Stability", Econometrica 48:987-1003), agents form
    coalitions based on preferences over coalition membership. A
    partition is STABLE if no agent wants to switch coalitions.

    PAC-Top-Cover (Sliwinski & Zick 2017, "Learning Hedonic Games",
    IJCAI) learns stable partitions from samples by iteratively:
    1. Finding each agent's most-preferred coalition
    2. Checking stability via pairwise blocking

    Chowdhury et al. (2025, "Hedonic Neurons", arXiv:2509.23684)
    applied this to neural network components, using PAS (Pairwise
    Additive Synergy) to define preferences.

    This metric adapts the approach for transformer circuits:
    - Agents = circuit heads
    - Preference = coalition performance (logit-diff when only
      coalition members are active)
    - Stability = no head wants to leave its coalition for another

    The algorithm:
    1. Compute pairwise synergy matrix S_ij for all head pairs
    2. Use agglomerative clustering on synergy to form initial coalitions
    3. Check stability: for each head, is it in its most-preferred
       coalition? If not, reassign.
    4. Iterate until convergence or max_iter

    Connections:
    - Drèze & Greenberg (1980) — hedonic game foundations
    - Bogomolnaia & Jackson (2002) "The Stability of Hedonic Coalition
      Structures", Games and Economic Behavior 38:201-230
    - Sliwinski & Zick (2017) — PAC-Top-Cover algorithm
    - Chowdhury et al. (2025) — Hedonic Neurons (PAS, OCA)
    - Banerjee et al. (2001) "Core in a Simple Coalition Formation
      Game", Social Choice and Welfare 18:135-153

Method:
    1. Compute pairwise synergy S_ij for all head pairs:
       S_ij = v({i,j}) - v({i}) - v({j})
       where v(S) = logit-diff with only S active
    2. Build initial partition via greedy: each head joins the
       coalition that maximizes its total synergy with members
    3. Stability refinement: iterate up to 10 rounds:
       - For each head, check if moving to another coalition improves
         its total synergy with coalition members
       - If yes, move it
    4. Report: partition, stability status, coalition-level statistics

Pass condition: partition converges to a stable state (no head wants
to move) within max_iter iterations.

Usage:
    mechval.run("coalition_discovery", tasks=["ioi"], device="cpu")
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
    name="Coalition Discovery (PAC-Top-Cover)",
    paper_ref="Drèze & Greenberg 1980; Sliwinski & Zick 2017; Chowdhury et al. 2025",
    paper_cite="Chowdhury et al. 2025 (Hedonic Neurons); Sliwinski & Zick 2017 (PAC-Top-Cover)",
    description="Discovers stable head coalitions via hedonic game theory and pairwise synergy",
    category="causal",
    tier="cross_discipline",
    origin="established",
)

MAX_STABILITY_ITER = 10


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


def _head_key(h: tuple[int, int]) -> str:
    return f"L{h[0]}H{h[1]}"


@torch.no_grad()
def run_coalition_discovery(model, tasks: list[str],
                            n_prompts: int = 20) -> list[EvalResult]:
    tokenizer = model.tokenizer
    results = []

    for task in tasks:
        circuit_heads = get_circuit_heads(task)
        if not circuit_heads or len(circuit_heads) < 3:
            log(f"  {task}: <3 circuit heads, skipping (need >=3 for coalitions)")
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

        individual_values = {}
        for h in heads:
            individual_values[h] = _coalition_value(
                model, prompts, correct_ids, incorrect_ids,
                {h}, circuit_heads, mean_z)

        log(f"    Computing pairwise synergy matrix...")
        synergy = np.zeros((n_heads, n_heads))
        pair_values = {}
        for i in range(n_heads):
            for j in range(i + 1, n_heads):
                v_pair = _coalition_value(
                    model, prompts, correct_ids, incorrect_ids,
                    {heads[i], heads[j]}, circuit_heads, mean_z)
                pair_values[(i, j)] = v_pair
                s_ij = v_pair - individual_values[heads[i]] - individual_values[heads[j]]
                synergy[i, j] = s_ij
                synergy[j, i] = s_ij

        partition: dict[int, int] = {}
        coalitions: dict[int, set[int]] = {}
        next_coalition_id = 0

        for i in range(n_heads):
            best_coalition = -1
            best_synergy = float('-inf')

            for cid, members in coalitions.items():
                total_syn = sum(synergy[i, m] for m in members)
                if total_syn > best_synergy:
                    best_synergy = total_syn
                    best_coalition = cid

            if best_coalition >= 0 and best_synergy > 0:
                partition[i] = best_coalition
                coalitions[best_coalition].add(i)
            else:
                partition[i] = next_coalition_id
                coalitions[next_coalition_id] = {i}
                next_coalition_id += 1

        converged = False
        for iteration in range(MAX_STABILITY_ITER):
            moves = 0
            for i in range(n_heads):
                current_cid = partition[i]
                current_members = coalitions[current_cid] - {i}
                current_synergy = sum(synergy[i, m] for m in current_members)

                best_cid = current_cid
                best_synergy = current_synergy

                for cid, members in coalitions.items():
                    if cid == current_cid:
                        continue
                    candidate_synergy = sum(synergy[i, m] for m in members)
                    if candidate_synergy > best_synergy:
                        best_synergy = candidate_synergy
                        best_cid = cid

                if best_synergy > 0 and best_cid == -1:
                    best_cid = next_coalition_id
                    coalitions[next_coalition_id] = set()
                    next_coalition_id += 1

                if best_cid != current_cid:
                    coalitions[current_cid].discard(i)
                    if not coalitions[current_cid]:
                        del coalitions[current_cid]
                    partition[i] = best_cid
                    coalitions.setdefault(best_cid, set()).add(i)
                    moves += 1

            if moves == 0:
                converged = True
                log(f"    Converged after {iteration + 1} iterations")
                break

        if not converged:
            log(f"    Did not converge after {MAX_STABILITY_ITER} iterations")

        coalition_stats = []
        for cid in sorted(coalitions.keys()):
            members = sorted(coalitions[cid])
            if not members:
                continue
            member_heads = {heads[m] for m in members}
            coalition_value = _coalition_value(
                model, prompts, correct_ids, incorrect_ids,
                member_heads, circuit_heads, mean_z)
            sum_individual = sum(individual_values[heads[m]] for m in members)
            internal_synergy = coalition_value - sum_individual

            member_keys = [_head_key(heads[m]) for m in members]
            log(f"    Coalition {cid}: {member_keys}  "
                f"value={coalition_value:.4f}  "
                f"synergy={internal_synergy:+.4f}")
            coalition_stats.append({
                "coalition_id": cid,
                "members": member_keys,
                "size": len(members),
                "coalition_value": coalition_value,
                "sum_individual_values": sum_individual,
                "internal_synergy": internal_synergy,
            })

        synergy_display = {}
        for i in range(n_heads):
            for j in range(i + 1, n_heads):
                key = f"{_head_key(heads[i])}<->{_head_key(heads[j])}"
                synergy_display[key] = float(synergy[i, j])

        n_coalitions = len([c for c in coalitions.values() if c])
        log(f"    {n_coalitions} coalitions found  "
            f"converged={converged}  "
            f"[{'PASS (stable)' if converged else 'FAIL (unstable)'}]")

        results.append(EvalResult(
            metric_id="GT6.coalition_discovery",
            value=float(n_coalitions),
            n_samples=len(prompts),
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": task,
                "n_heads": n_heads,
                "n_coalitions": n_coalitions,
                "converged": converged,
                "coalition_stats": coalition_stats,
                "pairwise_synergy": synergy_display,
                "individual_values": {_head_key(h): v for h, v in individual_values.items()},
                "passed": converged,
            },
        ))

    return results


def main():
    parser = parse_common_args("GT6: Coalition Discovery")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("GT6: COALITION DISCOVERY (PAC-TOP-COVER)")
    log("=" * 60)

    out = args.out or "GT6_coalition_discovery.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_coalition_discovery(model, [task], args.n_prompts)
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)

    save_results(results, out, args=args)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
