"""Hyperparameter Sensitivity (Metric #87)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     A09 — MDL/SLT
Categories:     causal, structural
Validity layer: Construct
Criteria:       C4 Minimality
Establishes:    Faithfulness is robust to hyperparameter variation (low CV across settings)
Requires:       GPU, model
Doc:            /instruments_v2/causal/a09-mdl-slt
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Test sensitivity of faithfulness to hyperparameters. For each task,
compute faithfulness with different settings:
  - n_prompts: 20, 40, 80
  - ablation_type: "zero", "mean", "mean_last"

Reports CV (coefficient of variation) across settings as the primary
metric. Low CV = robust; high CV = hyperparameter-dependent.

Usage:
    uv run python 29_hyperparam_sensitivity.py --tasks ioi sva
    uv run python 29_hyperparam_sensitivity.py --device cuda
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
    compute_faithfulness,
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

N_PROMPTS_SETTINGS = [20, 40, 80]
ABLATION_SETTINGS = ["zero", "mean", "mean_last"]


@torch.no_grad()
def faithfulness_with_ablation_type(model, prompts, correct_ids, incorrect_ids,
                                    circuit_heads: set[tuple[int, int]],
                                    mean_z: torch.Tensor,
                                    ablation_type: str) -> float:
    """Compute faithfulness with a specific ablation type."""
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    non_circuit = {(L, H) for L in range(n_layers) for H in range(n_heads)} - circuit_heads
    non_circuit_by_layer = heads_to_layer_dict(non_circuit)
    hooks = make_ablation_hook(non_circuit_by_layer, mean_z, ablation_type)

    faith_num, faith_den = 0.0, 0.0
    for i, p in enumerate(prompts):
        if i >= len(correct_ids):
            break
        tokens = model.to_tokens(p.text)
        clean_logits = model(tokens)
        clean_ld = logit_diff_from_logits(clean_logits, correct_ids[i], incorrect_ids[i])
        ablated_logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
        ablated_ld = logit_diff_from_logits(ablated_logits, correct_ids[i], incorrect_ids[i])
        faith_num += ablated_ld
        faith_den += clean_ld

    if abs(faith_den) < 1e-8:
        return 0.0
    return faith_num / faith_den


@torch.no_grad()
def run_hyperparam_sensitivity(model, tasks: list[str],
                               n_prompts: int = 40) -> list[EvalResult]:
    tokenizer = model.tokenizer
    results = []

    for task in tasks:
        circuit_heads = get_circuit_heads(task)
        if not circuit_heads:
            log(f"  {task}: no circuit, skipping")
            continue

        # Generate the maximum number of prompts we need
        max_prompts = max(N_PROMPTS_SETTINGS)
        all_prompts = generate_prompts(task, tokenizer, max_prompts)
        if not all_prompts:
            log(f"  {task}: no prompts, skipping")
            continue

        all_correct, all_incorrect = get_token_ids(all_prompts, tokenizer)
        if not all_correct:
            continue

        log(f"  {task} ({len(circuit_heads)} heads)...")

        # Calibrate mean_z from full set
        mean_z = calibrate_mean_z(model, all_prompts,
                                  n_calibration=min(100, len(all_prompts)))

        setting_scores = {}

        # Vary n_prompts with default ablation (mean)
        for np_setting in N_PROMPTS_SETTINGS:
            subset_prompts = all_prompts[:np_setting]
            subset_correct = all_correct[:np_setting]
            subset_incorrect = all_incorrect[:np_setting]

            if not subset_correct:
                continue

            faith = compute_faithfulness(
                model, subset_prompts, subset_correct, subset_incorrect,
                circuit_heads, mean_z,
            )
            key = f"n{np_setting}_mean"
            setting_scores[key] = faith
            log(f"    {key}: {faith:.4f}")

        # Vary ablation type with default n_prompts
        default_prompts = all_prompts[:n_prompts]
        default_correct = all_correct[:n_prompts]
        default_incorrect = all_incorrect[:n_prompts]

        for abl in ABLATION_SETTINGS:
            key = f"n{n_prompts}_{abl}"
            if key in setting_scores:
                continue  # Already computed (n40_mean)
            faith = faithfulness_with_ablation_type(
                model, default_prompts, default_correct, default_incorrect,
                circuit_heads, mean_z, abl,
            )
            setting_scores[key] = faith
            log(f"    {key}: {faith:.4f}")

        scores = list(setting_scores.values())
        mean_f = float(np.mean(scores))
        std_f = float(np.std(scores))
        cv = std_f / abs(mean_f) if abs(mean_f) > 1e-8 else float("inf")

        log(f"    CV={cv:.4f} (mean={mean_f:.4f}, std={std_f:.4f})")

        results.append(EvalResult(
            metric_id="C29.hyperparam_sensitivity",
            value=cv,
            n_samples=len(scores),
            metadata={
                "task": task,
                "setting_scores": setting_scores,
                "mean_faithfulness": mean_f,
                "std_faithfulness": std_f,
                "cv": cv,
                "n_circuit_heads": len(circuit_heads),
                "n_settings_tested": len(scores),
            },
        ))

    return results


def main():
    parser = parse_common_args("C29: Hyperparameter Sensitivity")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("C29: HYPERPARAMETER SENSITIVITY")
    log("=" * 60)

    results = run_hyperparam_sensitivity(model, tasks, args.n_prompts)

    out = args.out or "29_hyperparam_sensitivity.json"
    save_results(results, out)

    log(f"\nDone. {len(results)} tasks evaluated.")
    for r in results:
        t = r.metadata["task"]
        log(f"  {t}: CV={r.value:.4f}")


if __name__ == "__main__":
    main()
