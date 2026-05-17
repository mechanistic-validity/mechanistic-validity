"""Intervention Specificity (Metric #29)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     A03 — Rubin CATE
Categories:     causal
Validity layer: Internal
Criteria:       I3 Specificity
Establishes:    Circuit interventions affect target task specifically (high target/non-target ratio)
Requires:       GPU, model
Doc:            /instruments_v2/causal/a03-rubin-cate
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For each task, patch the circuit heads (activation patching from
clean->corrupt via mean ablation). Measure the effect on the TARGET
task metric (logit diff) AND on non-target metrics. Specificity is
target_effect / mean_nontarget_effect.

High specificity means the circuit is task-specific rather than
affecting general model capabilities.

Usage:
    uv run python 25_intervention_specificity.py --tasks ioi sva
    uv run python 25_intervention_specificity.py --device cuda --n-prompts 60
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    CIRCUIT_TASKS,
    EvalResult,
    calibrate_mean_z,
    compute_logit_diffs,
    generate_prompts,
    get_circuit_heads,
    get_token_ids,
    heads_to_layer_dict,
    load_model,
    log,
    logit_diff_from_logits,
    make_ablation_hook,
    parse_common_args,
    save_results,
)


@torch.no_grad()
def compute_ablation_effect(model, prompts, correct_ids, incorrect_ids,
                            ablation_hooks: list) -> float:
    """Compute fractional logit diff change when ablation hooks are applied.

    Returns (clean_ld - ablated_ld) / clean_ld averaged over prompts.
    """
    total_effect = 0.0
    total_clean = 0.0
    count = 0

    for i, p in enumerate(prompts):
        if i >= len(correct_ids):
            break
        tokens = model.to_tokens(p.text)
        clean_logits = model(tokens)
        clean_ld = logit_diff_from_logits(clean_logits, correct_ids[i], incorrect_ids[i])

        ablated_logits = model.run_with_hooks(tokens, fwd_hooks=ablation_hooks)
        ablated_ld = logit_diff_from_logits(ablated_logits, correct_ids[i], incorrect_ids[i])

        total_effect += abs(clean_ld - ablated_ld)
        total_clean += abs(clean_ld)
        count += 1

    if count == 0 or total_clean < 1e-8:
        return 0.0
    return total_effect / total_clean


@torch.no_grad()
def run_intervention_specificity(model, tasks: list[str],
                                 n_prompts: int = 40) -> list[EvalResult]:
    tokenizer = model.tokenizer
    results = []

    # Pre-generate prompts and token IDs for all tasks
    task_data = {}
    for task in tasks:
        prompts = generate_prompts(task, tokenizer, n_prompts)
        if not prompts:
            continue
        correct_ids, incorrect_ids = get_token_ids(prompts, tokenizer)
        if not correct_ids:
            continue
        task_data[task] = (prompts, correct_ids, incorrect_ids)

    if len(task_data) < 2:
        log("  Need at least 2 tasks for specificity analysis")
        return results

    for target_task in tasks:
        circuit_heads = get_circuit_heads(target_task)
        if not circuit_heads:
            log(f"  {target_task}: no circuit, skipping")
            continue
        if target_task not in task_data:
            log(f"  {target_task}: no prompts, skipping")
            continue

        log(f"  {target_task} ({len(circuit_heads)} heads)...")

        # Calibrate mean_z from target task prompts
        target_prompts, target_correct, target_incorrect = task_data[target_task]
        mean_z = calibrate_mean_z(model, target_prompts,
                                  n_calibration=min(100, len(target_prompts)))

        # Build ablation hooks for the target circuit heads
        circuit_by_layer = heads_to_layer_dict(circuit_heads)
        hooks = make_ablation_hook(circuit_by_layer, mean_z, "mean")

        # Effect on target task
        target_effect = compute_ablation_effect(
            model, target_prompts, target_correct, target_incorrect, hooks,
        )
        log(f"    target_effect={target_effect:.4f}")

        # Effect on non-target tasks
        nontarget_effects = {}
        for other_task in tasks:
            if other_task == target_task:
                continue
            if other_task not in task_data:
                continue
            other_prompts, other_correct, other_incorrect = task_data[other_task]
            effect = compute_ablation_effect(
                model, other_prompts, other_correct, other_incorrect, hooks,
            )
            nontarget_effects[other_task] = effect
            log(f"    {other_task}: effect={effect:.4f}")

        if not nontarget_effects:
            continue

        mean_nontarget = float(np.mean(list(nontarget_effects.values())))
        specificity = target_effect / mean_nontarget if mean_nontarget > 1e-8 else float("inf")

        log(f"    specificity={specificity:.4f} "
            f"(target={target_effect:.4f}, mean_nontarget={mean_nontarget:.4f})")

        results.append(EvalResult(
            metric_id="C25.intervention_specificity",
            value=specificity,
            n_samples=len(target_prompts),
            metadata={
                "task": target_task,
                "target_effect": target_effect,
                "mean_nontarget_effect": mean_nontarget,
                "nontarget_effects": nontarget_effects,
                "specificity": specificity,
                "n_circuit_heads": len(circuit_heads),
                "n_nontarget_tasks": len(nontarget_effects),
            },
        ))

    return results


def main():
    parser = parse_common_args("C25: Intervention Specificity")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("C25: INTERVENTION SPECIFICITY")
    log("=" * 60)

    results = run_intervention_specificity(model, tasks, args.n_prompts)

    out = args.out or "25_intervention_specificity.json"
    save_results(results, out)

    log(f"\nDone. {len(results)} tasks evaluated.")
    for r in results:
        t = r.metadata["task"]
        log(f"  {t}: specificity={r.value:.4f}")


if __name__ == "__main__":
    main()
