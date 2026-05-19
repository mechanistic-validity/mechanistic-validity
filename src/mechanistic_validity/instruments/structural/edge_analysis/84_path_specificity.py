"""Path Specificity (Graph Structure G3)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     G03 — Graph Structure
Categories:     structural
Validity layer: Internal
Criteria:       G3 Path Specificity
Establishes:    Whether different conditions activate different circuit paths
Requires:       CPU or GPU, model
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Compute edge activation patterns (edge effect magnitudes across all edges)
for task-relevant vs control conditions. Compare using Spearman correlation.

If the circuit uses genuinely different paths for different conditions,
the edge activation patterns should differ (low correlation).

Pass condition: Spearman rho < 0.5 (different conditions use different paths).

Usage:
    uv run python 84_path_specificity.py --tasks ioi --n-prompts 40
    uv run python 84_path_specificity.py --tasks ioi sva --device cpu
"""

import numpy as np
import torch
from scipy.stats import spearmanr

from mechanistic_validity.instruments.common import (
    CIRCUIT_TASKS,
    EvalResult,
    calibrate_mean_z,
    generate_prompts,
    get_circuit_info,
    get_token_ids,
    load_model,
    log,
    logit_diff_from_logits,
    parse_common_args,
    save_incremental,
    save_results,
)


@torch.no_grad()
def compute_edge_effects_for_prompt(model, tokens, correct_id, incorrect_id,
                                    edges, mean_z):
    """Compute the effect magnitude of ablating each edge for one prompt.

    Returns dict mapping edge -> effect magnitude.
    """
    clean_logits, clean_cache = model.run_with_cache(tokens)
    clean_ld = logit_diff_from_logits(clean_logits, correct_id, incorrect_id)

    effects = {}
    for (up_l, up_h, down_l, down_h) in edges:
        W_O = model.W_O[up_l]
        up_z = clean_cache[f"blocks.{up_l}.attn.hook_z"][0, -1, up_h, :]
        up_mean = mean_z[up_l, up_h].to(up_z.device)
        diff = (up_z - up_mean) @ W_O[up_h]

        def ablate_hook(activation, hook, _diff=diff):
            activation[0, -1, :] -= _diff
            return activation

        ablated_logits = model.run_with_hooks(
            tokens, fwd_hooks=[(f"blocks.{down_l}.hook_resid_pre", ablate_hook)])
        ablated_ld = logit_diff_from_logits(ablated_logits, correct_id, incorrect_id)
        effects[(up_l, up_h, down_l, down_h)] = clean_ld - ablated_ld

    return effects


@torch.no_grad()
def run_path_specificity(model, tasks: list[str],
                         n_prompts: int = 40) -> list[EvalResult]:
    tokenizer = model.tokenizer
    results = []

    for task in tasks:
        circuit, all_heads, all_edges = get_circuit_info(task)
        if circuit is None or not all_edges:
            log(f"  {task}: no circuit edges, skipping")
            continue

        prompts = generate_prompts(task, tokenizer, n_prompts)
        if not prompts:
            log(f"  {task}: no prompts, skipping")
            continue

        correct_ids, incorrect_ids = get_token_ids(prompts, tokenizer)
        if not correct_ids:
            continue

        sorted_edges = sorted(all_edges)
        if len(sorted_edges) < 2:
            log(f"  {task}: need >=2 edges for correlation, skipping")
            continue

        log(f"  {task}: {len(sorted_edges)} edges, {len(prompts)} prompts")

        mean_z = calibrate_mean_z(model, prompts, n_calibration=min(100, len(prompts)))

        # Split prompts into two halves for condition comparison
        half = len(correct_ids) // 2
        if half < 2:
            log(f"  {task}: not enough prompts for split, skipping")
            continue

        # Condition A: first half of prompts (task-relevant direction)
        pattern_a = np.zeros(len(sorted_edges))
        for i in range(half):
            tokens = model.to_tokens(prompts[i].text)
            effects = compute_edge_effects_for_prompt(
                model, tokens, correct_ids[i], incorrect_ids[i],
                sorted_edges, mean_z)
            for j, edge in enumerate(sorted_edges):
                pattern_a[j] += abs(effects[edge])
        pattern_a /= half

        # Condition B: second half with swapped correct/incorrect
        # (control condition: measure edges when task signal is reversed)
        pattern_b = np.zeros(len(sorted_edges))
        for i in range(half, min(2 * half, len(correct_ids))):
            tokens = model.to_tokens(prompts[i].text)
            effects = compute_edge_effects_for_prompt(
                model, tokens, incorrect_ids[i], correct_ids[i],
                sorted_edges, mean_z)
            for j, edge in enumerate(sorted_edges):
                pattern_b[j] += abs(effects[edge])
        pattern_b /= half

        # Spearman correlation between edge patterns
        if np.std(pattern_a) < 1e-10 or np.std(pattern_b) < 1e-10:
            rho = 1.0  # degenerate case: all equal
            p_val = 1.0
        else:
            rho, p_val = spearmanr(pattern_a, pattern_b)
            rho = float(rho)
            p_val = float(p_val)

        passed = bool(rho < 0.5)

        log(f"    Spearman rho={rho:.4f}  p={p_val:.4f}  [{'PASS' if passed else 'FAIL'}]")

        results.append(EvalResult(
            metric_id="G3.path_specificity",
            value=rho,
            n_samples=len(prompts),
            metadata={
                "task": task,
                "n_edges": len(sorted_edges),
                "spearman_rho": float(rho),
                "spearman_p": float(p_val),
                "pattern_task": pattern_a.tolist(),
                "pattern_control": pattern_b.tolist(),
                "edge_names": [f"L{e[0]}H{e[1]}->L{e[2]}H{e[3]}" for e in sorted_edges],
                "passed": passed,
                "threshold_rho": 0.5,
            },
        ))

    return results


def main():
    parser = parse_common_args("G3: Path Specificity")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("G3: PATH SPECIFICITY")
    log("=" * 60)

    out = args.out or "84_path_specificity.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_path_specificity(model, [task], args.n_prompts)
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)
            p = "PASS" if r.metadata["passed"] else "FAIL"
            log(f"  {task}: rho={r.value:.4f}  [{p}]")

    save_results(results, out)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
