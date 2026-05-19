"""Counterfactual Consistency (Metric #30)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     A02 — Counterfactual DAS/IIA
Categories:     causal, representational
Validity layer: Internal + Representational
Criteria:       I2 Sufficiency
Establishes:    Circuit captures invariant computation across paraphrased prompts
Requires:       GPU, model
Doc:            /instruments_v2/causal/a02-counterfactual-das
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Test whether the circuit gives consistent results across paraphrased
prompts. For each task, generate prompts with different seeds to get
variant templates. Compute faithfulness and logit diff on each variant.
Report consistency = 1 - CV across paraphrases.

High consistency = the circuit captures the invariant computation,
not surface features.

Usage:
    uv run python 34_counterfactual_consistency.py --tasks ioi sva --n-prompts 40
    uv run python 34_counterfactual_consistency.py --device cuda
"""

import numpy as np
import torch

from mechanistic_validity.instruments.common import (
    CIRCUIT_TASKS,
    EvalResult,
    calibrate_mean_z,
    compute_faithfulness,
    generate_prompts,
    get_circuit_heads,
    get_token_ids,
    load_model,
    log,
    logit_diff_from_logits,
    parse_common_args,
    save_results,
)
from mechanistic_validity.registry import load_task

PARAPHRASE_SEEDS = [42, 100, 200, 300, 400]


def generate_prompts_with_seed(task_name: str, tokenizer, n_prompts: int, seed: int):
    try:
        t = load_task(task_name)
    except ValueError:
        return []
    return t.get_prompts(tokenizer, n_prompts=n_prompts, seed=seed)


@torch.no_grad()
def compute_logit_diffs_for_prompts(model, prompts, correct_ids, incorrect_ids):
    """Return list of per-prompt logit diffs."""
    diffs = []
    for i, p in enumerate(prompts):
        if i >= len(correct_ids):
            break
        tokens = model.to_tokens(p.text)
        logits = model(tokens)
        diffs.append(logit_diff_from_logits(logits, correct_ids[i], incorrect_ids[i]))
    return diffs


@torch.no_grad()
def run_counterfactual_consistency(model, tasks: list[str],
                                   n_prompts: int = 40) -> list[EvalResult]:
    tokenizer = model.tokenizer
    results = []

    for task in tasks:
        circuit_heads = get_circuit_heads(task)
        if not circuit_heads:
            log(f"  {task}: no circuit, skipping")
            continue

        log(f"  {task} ({len(circuit_heads)} heads, {len(PARAPHRASE_SEEDS)} seeds)...")

        per_seed_faithfulness = []
        per_seed_mean_ld = []
        seed_details = {}

        for seed in PARAPHRASE_SEEDS:
            prompts = generate_prompts_with_seed(task, tokenizer, n_prompts, seed)
            if not prompts:
                continue

            correct_ids, incorrect_ids = get_token_ids(prompts, tokenizer)
            if not correct_ids:
                continue

            # Calibrate mean_z for this seed's prompts
            mean_z = calibrate_mean_z(model, prompts, n_calibration=min(50, len(prompts)))

            # Compute faithfulness
            faith = compute_faithfulness(
                model, prompts, correct_ids, incorrect_ids, circuit_heads, mean_z)

            # Compute mean logit diff
            ld_list = compute_logit_diffs_for_prompts(model, prompts, correct_ids, incorrect_ids)
            mean_ld = float(np.mean(ld_list)) if ld_list else 0.0

            per_seed_faithfulness.append(faith)
            per_seed_mean_ld.append(mean_ld)
            seed_details[f"seed_{seed}"] = {
                "faithfulness": faith,
                "mean_logit_diff": mean_ld,
                "n_prompts": len(prompts),
            }
            log(f"    seed={seed}: faith={faith:.3f}, mean_ld={mean_ld:.3f}")

        if len(per_seed_faithfulness) < 2:
            log(f"    not enough valid seeds, skipping")
            continue

        # Consistency = 1 - CV (coefficient of variation)
        faith_arr = np.array(per_seed_faithfulness)
        faith_mean = float(np.mean(faith_arr))
        faith_std = float(np.std(faith_arr))
        faith_cv = faith_std / abs(faith_mean) if abs(faith_mean) > 1e-8 else 1.0
        faith_consistency = max(0.0, 1.0 - faith_cv)

        ld_arr = np.array(per_seed_mean_ld)
        ld_mean = float(np.mean(ld_arr))
        ld_std = float(np.std(ld_arr))
        ld_cv = ld_std / abs(ld_mean) if abs(ld_mean) > 1e-8 else 1.0
        ld_consistency = max(0.0, 1.0 - ld_cv)

        overall_consistency = (faith_consistency + ld_consistency) / 2.0

        log(f"    faith_consistency={faith_consistency:.3f}, "
            f"ld_consistency={ld_consistency:.3f}, "
            f"overall={overall_consistency:.3f}")

        results.append(EvalResult(
            metric_id="C34.counterfactual_consistency",
            value=overall_consistency,
            n_samples=len(per_seed_faithfulness),
            metadata={
                "task": task,
                "faithfulness_mean": faith_mean,
                "faithfulness_std": faith_std,
                "faithfulness_cv": faith_cv,
                "faithfulness_consistency": faith_consistency,
                "logit_diff_mean": ld_mean,
                "logit_diff_std": ld_std,
                "logit_diff_cv": ld_cv,
                "logit_diff_consistency": ld_consistency,
                "seeds": PARAPHRASE_SEEDS,
                "per_seed": seed_details,
                "n_circuit_heads": len(circuit_heads),
            },
        ))

    return results


def main():
    parser = parse_common_args("C34: Counterfactual Consistency")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("C34: COUNTERFACTUAL CONSISTENCY (Metric #30)")
    log("=" * 60)

    results = run_counterfactual_consistency(model, tasks, args.n_prompts)

    out = args.out or "34_counterfactual_consistency.json"
    save_results(results, out, args=args)

    log(f"\nDone. {len(results)} tasks evaluated.")
    for r in results:
        t = r.metadata["task"]
        log(f"  {t}: consistency={r.value:.3f}")


if __name__ == "__main__":
    main()
