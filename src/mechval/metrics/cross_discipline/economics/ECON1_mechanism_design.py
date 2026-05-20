"""Mechanism Design: Incentive Compatibility of Circuit Heads
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Metric:         ECON1 — Mechanism Design
Categories:     cross_discipline, economics
Tier:           cross_discipline
Origin:         established

Tests whether credit allocation is incentive-compatible. Compares
each head's "declared contribution" (activation norm) with its
"true contribution" (logit diff change upon ablation). High Spearman
correlation means heads cannot free-ride.

Pass: Spearman correlation > 0.5
Ref: Hurwicz 1960 "Optimality and Informational Efficiency in
     Resource Allocation", Mathematical Methods in the Social Sciences

Usage:
    uv run python ECON1_mechanism_design.py --tasks ioi sva --n-prompts 40
"""

import numpy as np
import torch
from scipy import stats

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    InstrumentInfo,
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
    name="Mechanism Design",
    paper_ref="Hurwicz 1960",
    paper_cite="Hurwicz 1960, Optimality and Informational Efficiency in Resource Allocation, Mathematical Methods in the Social Sciences",
    description="Tests whether circuit credit allocation is incentive-compatible via declaration vs true contribution correlation",
    category="mi",
    tier="core",
    origin="established",
)

CORRELATION_THRESHOLD = 0.5


@torch.no_grad()
def run_mechanism_design(model, tasks: list[str],
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
            continue

        correct_ids, incorrect_ids = get_token_ids(prompts, tokenizer)
        if not correct_ids:
            continue

        n_valid = min(len(prompts), len(correct_ids))
        head_list = sorted(circuit_heads)

        if len(head_list) < 3:
            log(f"  {task}: need >= 3 heads for correlation, skipping")
            continue

        log(f"  {task}: {len(head_list)} heads, {n_valid} prompts")

        # Compute baseline logit diff
        baseline_lds = []
        for i in range(n_valid):
            tokens = model.to_tokens(prompts[i].text)
            logits = model(tokens)
            baseline_lds.append(logit_diff_from_logits(logits, correct_ids[i], incorrect_ids[i]))
        baseline_ld = float(np.mean(baseline_lds))

        if abs(baseline_ld) < 1e-8:
            log(f"    baseline LD ~ 0, skipping")
            continue

        # For each head: compute declared (activation norm) and true (ablation impact)
        declared = []  # mean activation norm
        true_contrib = []  # LD drop upon zero-ablation

        for L, H in head_list:
            norms = []
            ablated_lds = []

            for i in range(n_valid):
                tokens = model.to_tokens(prompts[i].text)

                # Declared: activation norm
                _, cache = model.run_with_cache(
                    tokens, names_filter=lambda n: "hook_z" in n)
                z = cache[f"blocks.{L}.attn.hook_z"]
                norms.append(z[0, -1, H].norm().item())

                # True: logit diff with this head zero-ablated
                def _ablate(z_val, hook, _H=H):
                    z_val[0, :, _H, :] = 0.0
                    return z_val

                logits = model.run_with_hooks(
                    tokens,
                    fwd_hooks=[(f"blocks.{L}.attn.hook_z", _ablate)])
                ablated_lds.append(
                    logit_diff_from_logits(logits, correct_ids[i], incorrect_ids[i]))

            declared.append(float(np.mean(norms)))
            # True contribution = how much LD drops when ablated
            true_contrib.append(baseline_ld - float(np.mean(ablated_lds)))

        declared_arr = np.array(declared)
        true_arr = np.array(true_contrib)

        spearman_r, spearman_p = stats.spearmanr(declared_arr, true_arr)
        spearman_r = float(spearman_r)
        passed = spearman_r > CORRELATION_THRESHOLD

        head_details = []
        for idx, (L, H) in enumerate(head_list):
            head_details.append({
                "head": f"L{L}H{H}",
                "declared_norm": declared[idx],
                "true_contribution": true_contrib[idx],
            })

        log(f"    spearman_r={spearman_r:.4f} (p={spearman_p:.4f})  [{'PASS' if passed else 'FAIL'}]")

        results.append(EvalResult(
            metric_id="ECON1.mechanism_design",
            value=spearman_r,
            n_samples=len(prompts),
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": task,
                "n_heads": len(head_list),
                "spearman_r": spearman_r,
                "spearman_p": float(spearman_p),
                "baseline_ld": baseline_ld,
                "per_head": head_details,
                "threshold": CORRELATION_THRESHOLD,
                "passed": passed,
            },
        ))

    return results


def main():
    parser = parse_common_args("ECON1: Mechanism Design")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("ECON1: MECHANISM DESIGN")
    log("=" * 60)

    out = args.out or "ECON1_mechanism_design.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_mechanism_design(model, [task], args.n_prompts)
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)

    save_results(results, out, args=args)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
