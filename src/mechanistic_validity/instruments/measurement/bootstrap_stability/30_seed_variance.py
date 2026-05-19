"""Seed Variance
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     F01 — Bootstrap Stability
Categories:     measurement
Validity layer: Measurement
Criteria:       M1 Reliability
Establishes:    Faithfulness scores are stable across different random seeds
Requires:       GPU, model
Doc:            /instruments_v2/measurement/f01-bootstrap-stability
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Test seed variance of the evaluation pipeline. Since we cannot retrain
the weight classifier here, we instead measure variance of faithfulness
across different prompt subsets (simulating seed variance). For each
task, generate prompts with seeds 42, 123, 456, 789, 1337, compute
faithfulness on each subset, and report CV across seeds.

Usage:
    uv run python 30_seed_variance.py --tasks ioi sva
    uv run python 30_seed_variance.py --device cuda --n-prompts 40
"""

import numpy as np
import torch

from mechanistic_validity.instruments.common import (
    CIRCUIT_TASKS,
    EvalResult,
    calibrate_mean_z,
    compute_faithfulness,
    get_circuit_heads,
    get_token_ids,
    load_model,
    log,
    parse_common_args,
    save_results,
)

# We cannot call generate_prompts with different seeds directly since
# the seed is hardcoded. Instead, we generate a large pool and subsample.
from mechanistic_validity.instruments.common import generate_prompts

SEEDS = [42, 123, 456, 789, 1337]


def _subsample_prompts(prompts, correct_ids, incorrect_ids,
                       n: int, seed: int):
    """Subsample n prompts with the given seed."""
    rng = np.random.RandomState(seed)
    total = min(len(prompts), len(correct_ids), len(incorrect_ids))
    if total <= n:
        return prompts[:total], correct_ids[:total], incorrect_ids[:total]

    indices = rng.choice(total, size=n, replace=False)
    indices.sort()
    sub_prompts = [prompts[i] for i in indices]
    sub_correct = [correct_ids[i] for i in indices]
    sub_incorrect = [incorrect_ids[i] for i in indices]
    return sub_prompts, sub_correct, sub_incorrect


@torch.no_grad()
def run_seed_variance(model, tasks: list[str], n_prompts: int = 40) -> list[EvalResult]:
    tokenizer = model.tokenizer
    results = []

    for task in tasks:
        circuit_heads = get_circuit_heads(task)
        if not circuit_heads:
            log(f"  {task}: no circuit, skipping")
            continue

        # Generate a large pool of prompts (3x to allow diverse subsampling)
        pool_size = max(n_prompts * 3, 120)
        all_prompts = generate_prompts(task, tokenizer, pool_size)
        if not all_prompts:
            log(f"  {task}: no prompts, skipping")
            continue

        all_correct, all_incorrect = get_token_ids(all_prompts, tokenizer)
        if not all_correct:
            continue

        log(f"  {task} ({len(circuit_heads)} heads, pool={len(all_prompts)} prompts)...")

        # Calibrate mean_z from full pool
        mean_z = calibrate_mean_z(model, all_prompts,
                                  n_calibration=min(100, len(all_prompts)))

        seed_scores = {}
        for seed in SEEDS:
            sub_prompts, sub_correct, sub_incorrect = _subsample_prompts(
                all_prompts, all_correct, all_incorrect, n_prompts, seed,
            )

            faith = compute_faithfulness(
                model, sub_prompts, sub_correct, sub_incorrect,
                circuit_heads, mean_z,
            )
            seed_scores[seed] = faith
            log(f"    seed={seed}: faithfulness={faith:.4f} (n={len(sub_prompts)})")

        scores = list(seed_scores.values())
        mean_f = float(np.mean(scores))
        std_f = float(np.std(scores))
        cv = std_f / abs(mean_f) if abs(mean_f) > 1e-8 else float("inf")
        score_range = float(max(scores) - min(scores))

        log(f"    CV={cv:.4f} (mean={mean_f:.4f}, std={std_f:.4f}, "
            f"range={score_range:.4f})")

        results.append(EvalResult(
            metric_id="C30.seed_variance",
            value=cv,
            n_samples=len(SEEDS),
            metadata={
                "task": task,
                "seed_scores": {str(k): v for k, v in seed_scores.items()},
                "seeds": SEEDS,
                "mean_faithfulness": mean_f,
                "std_faithfulness": std_f,
                "cv": cv,
                "range": score_range,
                "n_prompts_per_seed": n_prompts,
                "pool_size": len(all_prompts),
                "n_circuit_heads": len(circuit_heads),
            },
        ))

    return results


def main():
    parser = parse_common_args("C30: Seed Variance")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("C30: SEED VARIANCE")
    log("=" * 60)

    results = run_seed_variance(model, tasks, args.n_prompts)

    out = args.out or "30_seed_variance.json"
    save_results(results, out)

    log(f"\nDone. {len(results)} tasks evaluated.")
    for r in results:
        t = r.metadata["task"]
        log(f"  {t}: CV={r.value:.4f}  mean={r.metadata['mean_faithfulness']:.4f}")


if __name__ == "__main__":
    main()
