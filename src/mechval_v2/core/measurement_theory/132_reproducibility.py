"""Metric: Reproducibility Check --- execution-grounded reproducibility diagnostic

Paper: Bai, Baumgartner, Sun, Holtzman, Tan (2026). "The Story is Not
the Science: Execution-Grounded Evaluation of Mechanistic
Interpretability Research." arXiv:2602.18458

Inspired by MechEvalAgent's finding that 93% of MI research outputs
fail reproducibility when code is actually executed. This metric
operationalizes reproducibility as a first-class validity criterion:
for a given metric computation, run it multiple times and measure output
deviation. High deviation indicates the metric is not reproducible and
its results should not be trusted.

Reproducibility Check (Evaluation EX25)
=============================================
Instrument:     EX25 --- Reproducibility Check
Categories:     evaluation
Validity layer: Measurement
Criteria:       M1 Reliability (test-retest)
Establishes:    Whether a metric computation pipeline produces
                reproducible results across runs with different seeds
Requires:       CPU or GPU, model
=============================================

Core logic:
1. Select a base metric (logit-diff, probe accuracy, or ablation recovery).
2. Run it N times on the same model+prompts with different random seeds.
3. Compute deviation_rate = fraction of outputs differing by > threshold.
4. Compute max_deviation = max |score_i - score_j| / |mean|.
5. Compute coherence_score = Spearman rank correlation between runs.

Pass condition: deviation_rate < 0.05; max_deviation < 0.08; coherence > 0.9

Usage:
    uv run python 132_reproducibility.py --model gpt2 --device cpu
    uv run python 132_reproducibility.py --n-runs 10 --n-prompts 50
"""

import numpy as np
import torch

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    InstrumentInfo,
    compute_logit_diffs,
    generate_prompts,
    get_token_ids,
    load_model,
    log,
    parse_common_args,
    save_results,
)

INSTRUMENT_INFO = InstrumentInfo(
    name="Reproducibility Check",
    paper_ref="Bai et al. arXiv:2602.18458 (Feb 2026)",
    paper_cite=(
        "Bai, Baumgartner, Sun, Holtzman, Tan 2026, "
        "The Story is Not the Science: Execution-Grounded Evaluation "
        "of Mechanistic Interpretability Research "
        "(arXiv:2602.18458)"
    ),
    description=(
        "Runs a metric computation pipeline multiple times with "
        "different random seeds and measures output deviation. "
        "Reports deviation_rate, max_deviation, and rank coherence. "
        "Inspired by MechEvalAgent's 93% reproducibility failure rate."
    ),
    category="evaluation",
    tier="established",
    origin="external",
)

DEVIATION_THRESHOLD = 0.05
MAX_DEVIATION_THRESHOLD = 0.08
COHERENCE_THRESHOLD = 0.9


def _spearman_rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Spearman rank correlation between two arrays."""
    n = len(a)
    if n < 3:
        return 1.0
    rank_a = np.argsort(np.argsort(a)).astype(float)
    rank_b = np.argsort(np.argsort(b)).astype(float)
    d = rank_a - rank_b
    return float(1.0 - 6.0 * np.sum(d ** 2) / (n * (n ** 2 - 1)))


@torch.no_grad()
def _run_logit_diff_once(
    model, prompts, correct_ids: list[int], incorrect_ids: list[int],
    seed: int,
) -> list[float]:
    """Run logit-diff computation with a seeded prompt subsample.

    Returns per-prompt logit-diffs for the selected subsample.
    """
    rng = np.random.RandomState(seed)
    n = len(correct_ids)
    if n == 0:
        return []
    indices = rng.permutation(n)
    subsample = min(n, max(2, n * 3 // 4))
    selected = sorted(indices[:subsample].tolist())

    lds = []
    for idx in selected:
        if idx >= len(prompts):
            continue
        tokens = model.to_tokens(prompts[idx].text)
        logits = model(tokens)
        last = logits[0, -1]
        ld = (last[correct_ids[idx]] - last[incorrect_ids[idx]]).item()
        lds.append(ld)

    return lds


def run_reproducibility_check(
    model,
    tasks: list[str] | None = None,
    n_prompts: int = 40,
    n_runs: int = 5,
) -> list[EvalResult]:
    """Run the reproducibility check diagnostic.

    For each task, runs logit-diff computation multiple times and
    measures cross-run deviation and rank coherence.

    Args:
        model: HookedTransformer instance.
        tasks: list of task names.
        n_prompts: number of prompts per task.
        n_runs: number of independent runs.

    Returns:
        List of EvalResult, one per task plus aggregate.
    """
    if tasks is None:
        tasks = CIRCUIT_TASKS

    log(f"  Reproducibility Check: n_runs={n_runs}, n_prompts={n_prompts}")

    results = []
    all_devs = []

    for task in tasks:
        prompts = generate_prompts(task, model.tokenizer, n_prompts=n_prompts)
        if not prompts:
            log(f"    {task}: no prompts, skipping")
            continue

        correct_ids, incorrect_ids = get_token_ids(prompts, model.tokenizer)
        if not correct_ids:
            log(f"    {task}: no valid token ids, skipping")
            continue

        # Run multiple times
        run_means = []
        all_run_lds = []
        for seed in range(n_runs):
            lds = _run_logit_diff_once(
                model, prompts, correct_ids, incorrect_ids, seed
            )
            if lds:
                run_means.append(float(np.mean(lds)))
                all_run_lds.append(lds)

        if len(run_means) < 2:
            log(f"    {task}: insufficient runs, skipping")
            continue

        run_means_arr = np.array(run_means)
        global_mean = float(np.mean(run_means_arr))
        global_abs = max(abs(global_mean), 1e-8)

        # Deviation rate: fraction of run pairs with relative deviation > threshold
        n_pairs = 0
        n_deviating = 0
        for i in range(len(run_means)):
            for j in range(i + 1, len(run_means)):
                n_pairs += 1
                rel_dev = abs(run_means[i] - run_means[j]) / global_abs
                if rel_dev > MAX_DEVIATION_THRESHOLD:
                    n_deviating += 1

        deviation_rate = n_deviating / max(n_pairs, 1)

        # Max deviation
        max_dev = float(np.max(np.abs(run_means_arr - global_mean))) / global_abs

        # Coherence: mean pairwise Spearman correlation of per-prompt rankings
        coherences = []
        min_len = min(len(r) for r in all_run_lds)
        if min_len >= 3:
            for i in range(len(all_run_lds)):
                for j in range(i + 1, len(all_run_lds)):
                    rho = _spearman_rank_corr(
                        np.array(all_run_lds[i][:min_len]),
                        np.array(all_run_lds[j][:min_len]),
                    )
                    coherences.append(rho)

        coherence_score = float(np.mean(coherences)) if coherences else 1.0

        all_devs.append(deviation_rate)

        passed_dev = deviation_rate < DEVIATION_THRESHOLD
        passed_max = max_dev < MAX_DEVIATION_THRESHOLD
        passed_coh = coherence_score > COHERENCE_THRESHOLD
        passed = passed_dev and passed_max and passed_coh

        log(f"    {task}: dev_rate={deviation_rate:.4f}, max_dev={max_dev:.4f}, "
            f"coherence={coherence_score:.4f} ({'PASS' if passed else 'FAIL'})")

        results.append(EvalResult(
            metric_id="EX25.reproducibility_check",
            value=deviation_rate,
            n_samples=len(correct_ids),
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": task,
                "deviation_rate": deviation_rate,
                "max_deviation": max_dev,
                "coherence_score": coherence_score,
                "run_means": run_means,
                "global_mean": global_mean,
                "n_runs": n_runs,
                "passed": passed,
                "threshold_deviation": DEVIATION_THRESHOLD,
                "threshold_max_dev": MAX_DEVIATION_THRESHOLD,
                "threshold_coherence": COHERENCE_THRESHOLD,
            },
        ))

    # Aggregate
    if all_devs:
        agg_dev = float(np.mean(all_devs))
        agg_passed = agg_dev < DEVIATION_THRESHOLD
        log(f"  Aggregate: deviation_rate={agg_dev:.4f} "
            f"({'PASS' if agg_passed else 'FAIL'})")

        results.append(EvalResult(
            metric_id="EX25.reproducibility_check",
            value=agg_dev,
            n_samples=sum(r.n_samples for r in results),
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": "aggregate",
                "mean_deviation_rate": agg_dev,
                "n_tasks": len(all_devs),
                "passed": agg_passed,
                "threshold": DEVIATION_THRESHOLD,
            },
        ))

    return results


def main():
    parser = parse_common_args("EX25: Reproducibility Check")
    parser.add_argument("--n-runs", type=int, default=5,
                        help="Number of independent runs")
    args = parser.parse_args()

    model = load_model(args.model, args.device)

    log("=" * 60)
    log("EX25: REPRODUCIBILITY CHECK")
    log("=" * 60)

    tasks = args.tasks or CIRCUIT_TASKS
    results = run_reproducibility_check(
        model,
        tasks=tasks,
        n_prompts=args.n_prompts,
        n_runs=args.n_runs,
    )

    out = args.out or "132_reproducibility.json"
    save_results(results, out)
    log(f"\nDone. {len(results)} results.")


if __name__ == "__main__":
    main()
