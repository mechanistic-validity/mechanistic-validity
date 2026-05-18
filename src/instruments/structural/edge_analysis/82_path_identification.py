"""Path Identification (Graph Structure G1)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     G01 — Graph Structure
Categories:     structural
Validity layer: Internal
Criteria:       G1 Path Identification
Establishes:    Whether specific information flow paths can be traced through the circuit
Requires:       CPU or GPU, model
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For each edge (upstream -> downstream head pair from circuit pathways),
compute path-patching: extract upstream head's output on clean vs corrupted
input, subtract that difference at the downstream layer, measure change
in logit diff.

Compute specificity: ratio of edge effect magnitude on task-relevant pairs
vs control pairs (random baseline with shuffled token answers).

Pass condition: at least 1 path with specificity > 5x.

Usage:
    uv run python 82_path_identification.py --tasks ioi --n-prompts 40
    uv run python 82_path_identification.py --tasks ioi sva --device cpu
"""
import sys
from pathlib import Path

import numpy as np
import torch

_INSTRUMENTS = Path(__file__).resolve().parents[2]  # up to src/instruments/
sys.path.insert(0, str(_INSTRUMENTS))
from _common import (
    CIRCUIT_TASKS,
    EvalResult,
    calibrate_mean_z,
    generate_prompts,
    get_circuit_info,
    get_all_edges,
    get_token_ids,
    load_model,
    log,
    logit_diff_from_logits,
    parse_common_args,
    save_incremental,
    save_results,
)


@torch.no_grad()
def compute_edge_effect(model, tokens, correct_id, incorrect_id,
                        up_layer, up_head, down_layer, cache_clean, cache_corrupt):
    """Measure path-patching effect for one edge on one prompt.

    Replaces upstream head's contribution at the downstream layer with
    the corrupted version and measures change in logit diff.
    """
    W_O = model.W_O[up_layer]
    up_z_clean = cache_clean[f"blocks.{up_layer}.attn.hook_z"][0, -1, up_head, :]
    up_z_corrupt = cache_corrupt[f"blocks.{up_layer}.attn.hook_z"][0, -1, up_head, :]
    diff = (up_z_corrupt @ W_O[up_head]) - (up_z_clean @ W_O[up_head])

    def patch_hook(activation, hook, _diff=diff):
        activation[0, -1, :] += _diff
        return activation

    patched_logits = model.run_with_hooks(
        tokens, fwd_hooks=[(f"blocks.{down_layer}.hook_resid_pre", patch_hook)])
    clean_logits = model(tokens)

    clean_ld = logit_diff_from_logits(clean_logits, correct_id, incorrect_id)
    patched_ld = logit_diff_from_logits(patched_logits, correct_id, incorrect_id)
    return clean_ld - patched_ld


@torch.no_grad()
def run_path_identification(model, tasks: list[str],
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

        log(f"  {task}: {len(all_edges)} edges, {len(prompts)} prompts")

        # Compute edge effects on task-relevant prompts
        edge_effects_task = {e: [] for e in sorted(all_edges)}
        edge_effects_ctrl = {e: [] for e in sorted(all_edges)}

        # Build shuffled answer-token indices for the control condition.
        # For each prompt i, the control uses answer tokens from a different
        # prompt so the edge effect on unrelated tokens should be smaller.
        n_usable = min(len(prompts), len(correct_ids))
        shuffled_idx = [(i + n_usable // 2) % n_usable for i in range(n_usable)]

        for i, p in enumerate(prompts):
            if i >= n_usable:
                break
            tokens = model.to_tokens(p.text)
            _, cache_clean = model.run_with_cache(tokens)

            # Control: use answer tokens from a different prompt
            j = shuffled_idx[i]
            ctrl_correct = correct_ids[j]
            ctrl_incorrect = incorrect_ids[j]

            # Generate a corrupted cache by using a different prompt
            corrupt_idx = (i + 1) % n_usable
            corrupt_tokens = model.to_tokens(prompts[corrupt_idx].text)
            _, cache_corrupt = model.run_with_cache(corrupt_tokens)

            for (up_l, up_h, down_l, down_h) in sorted(all_edges):
                effect = compute_edge_effect(
                    model, tokens, correct_ids[i], incorrect_ids[i],
                    up_l, up_h, down_l, cache_clean, cache_corrupt)
                edge_effects_task[(up_l, up_h, down_l, down_h)].append(effect)

                ctrl_effect = compute_edge_effect(
                    model, tokens, ctrl_correct, ctrl_incorrect,
                    up_l, up_h, down_l, cache_clean, cache_corrupt)
                edge_effects_ctrl[(up_l, up_h, down_l, down_h)].append(ctrl_effect)

        # Compute specificity per edge
        edge_specificities = {}
        max_spec = 0.0
        for edge in sorted(all_edges):
            task_mean = abs(np.mean(edge_effects_task[edge]))
            ctrl_mean = abs(np.mean(edge_effects_ctrl[edge]))
            spec = task_mean / max(ctrl_mean, 1e-8)
            edge_specificities[edge] = spec
            max_spec = max(max_spec, spec)

            up_l, up_h, down_l, down_h = edge
            name = f"L{up_l}H{up_h}->L{down_l}H{down_h}"
            log(f"    {name}: task={task_mean:.5f} ctrl={ctrl_mean:.5f} spec={spec:.2f}x")

        n_specific = sum(1 for s in edge_specificities.values() if s > 5.0)
        passed = n_specific >= 1

        log(f"    max_specificity={max_spec:.2f}x  n_specific(>5x)={n_specific}  "
            f"[{'PASS' if passed else 'FAIL'}]")

        results.append(EvalResult(
            metric_id="G1.path_identification",
            value=max_spec,
            n_samples=len(prompts),
            metadata={
                "task": task,
                "n_edges": len(all_edges),
                "max_specificity": max_spec,
                "n_specific_edges": n_specific,
                "edge_specificities": {
                    f"L{e[0]}H{e[1]}->L{e[2]}H{e[3]}": s
                    for e, s in edge_specificities.items()
                },
                "passed": passed,
                "threshold_specificity": 5.0,
            },
        ))

    return results


def main():
    parser = parse_common_args("G1: Path Identification")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("G1: PATH IDENTIFICATION")
    log("=" * 60)

    out = args.out or "82_path_identification.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_path_identification(model, [task], args.n_prompts)
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)
            p = "PASS" if r.metadata["passed"] else "FAIL"
            log(f"  {task}: max_spec={r.value:.2f}x  [{p}]")

    save_results(results, out)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
