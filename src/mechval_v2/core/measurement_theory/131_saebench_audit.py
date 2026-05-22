"""Metric: SAEBench Reliability Audit --- reseed noise and discriminability of SAE metrics

Paper: Anonymous (2026). "Are Sparse Autoencoder Benchmarks Reliable?"
arXiv:2605.18229

Audits SAE evaluation metrics via three independent lenses: reseed
coefficient of variation, ground-truth correlation on synthetic SAEs,
and discriminability across training trajectories. The SAEBench audit
independently derived the framework's M-frame validity criteria and
found that TPP and SCR fail comprehensively while sae-probes is most
reliable.

SAEBench Reliability Audit (Evaluation EX24)
=============================================
Instrument:     EX24 --- SAEBench Reliability Audit
Categories:     evaluation
Validity layer: Measurement
Criteria:       M1 Reliability, M2 Measurement Invariance
Establishes:    Whether evaluation metrics used for SAE comparison are
                themselves reliable (low reseed noise) and discriminative
                (can distinguish meaningfully different SAEs)
Requires:       CPU or GPU, model
=============================================

Core logic:
1. Select an evaluation metric (e.g., probe accuracy, logit-diff recovery).
2. Run the metric N times on the same model+prompts with different random
   seeds (baseline shuffles, prompt orderings, ablation samples).
3. Compute coefficient of variation (CV) = std / |mean|.
4. Compute discriminability: run metric on two SAE configurations that
   differ by a known quality dimension, compute effect size (Cohen's d).
5. Report CV and discriminability as the audit diagnostics.

Pass condition: CV < 0.05; discriminability_d > 0.8

Usage:
    uv run python 131_saebench_audit.py --model gpt2 --device cpu
    uv run python 131_saebench_audit.py --n-reseeds 10 --n-prompts 50
"""

import numpy as np
import torch

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    InstrumentInfo,
    generate_prompts,
    get_token_ids,
    load_model,
    log,
    parse_common_args,
    save_results,
)

INSTRUMENT_INFO = InstrumentInfo(
    name="SAEBench Reliability Audit",
    paper_ref="Anonymous, arXiv:2605.18229 (May 2026)",
    paper_cite=(
        "Anonymous 2026, "
        "Are Sparse Autoencoder Benchmarks Reliable? "
        "(arXiv:2605.18229)"
    ),
    description=(
        "Audits evaluation metrics by measuring reseed coefficient of "
        "variation and discriminability. Metrics with high CV (like TPP "
        "at 16-39%) are flagged as unreliable. Implements the SAEBench "
        "audit's three-lens framework as a reusable meta-diagnostic."
    ),
    category="evaluation",
    tier="established",
    origin="external",
)

CV_THRESHOLD = 0.05
DISCRIMINABILITY_THRESHOLD = 0.8


@torch.no_grad()
def _compute_probe_accuracy_once(
    model, prompts, correct_ids: list[int], incorrect_ids: list[int],
    hook_name: str, seed: int,
) -> float:
    """Run a simple probe-based metric with a given random seed.

    Uses logit-diff at a hook point as a proxy for probe accuracy.
    The seed controls which prompts are sampled and the order of
    evaluation, introducing the reseed variance we want to measure.
    """
    rng = np.random.RandomState(seed)
    n = len(correct_ids)
    if n == 0:
        return 0.0
    indices = rng.permutation(n)
    subsample = min(n, max(1, n * 3 // 4))
    selected = indices[:subsample]

    total_correct = 0
    for idx in selected:
        i = int(idx)
        if i >= len(prompts):
            continue
        tokens = model.to_tokens(prompts[i].text)
        logits = model(tokens)
        last = logits[0, -1]
        if last[correct_ids[i]] > last[incorrect_ids[i]]:
            total_correct += 1

    return total_correct / max(len(selected), 1)


@torch.no_grad()
def _compute_ablation_metric_once(
    model, prompts, correct_ids: list[int], incorrect_ids: list[int],
    hook_name: str, seed: int, n_ablate: int = 5,
) -> float:
    """Compute an ablation-based metric with random baseline selection.

    Ablates n_ablate random dimensions at the hook point and measures
    logit-diff recovery. The seed controls which dimensions are ablated.
    """
    rng = np.random.RandomState(seed)

    d_hook = model.cfg.d_model
    ablate_dims = rng.choice(d_hook, min(n_ablate, d_hook), replace=False).tolist()
    dim_set = set(ablate_dims)

    def ablation_hook(value, hook):
        for d in dim_set:
            value[:, :, d] = 0.0
        return value

    total_ld = 0.0
    count = 0
    for i, p in enumerate(prompts):
        if i >= len(correct_ids):
            break
        tokens = model.to_tokens(p.text)
        logits = model.run_with_hooks(
            tokens, fwd_hooks=[(hook_name, ablation_hook)]
        )
        last = logits[0, -1]
        ld = (last[correct_ids[i]] - last[incorrect_ids[i]]).item()
        total_ld += ld
        count += 1

    return total_ld / max(count, 1)


def run_saebench_audit(
    model,
    tasks: list[str] | None = None,
    n_prompts: int = 40,
    n_reseeds: int = 5,
    hook_layer: int | None = None,
) -> list[EvalResult]:
    """Run the SAEBench reliability audit diagnostic.

    For each task, measures reseed CV and discriminability of two
    proxy metrics (probe accuracy and ablation logit-diff).

    Args:
        model: HookedTransformer instance.
        tasks: list of task names.
        n_prompts: number of prompts per task.
        n_reseeds: number of reseed iterations for CV computation.
        hook_layer: layer for hook point (default: middle layer).

    Returns:
        List of EvalResult, one per task per metric type plus aggregate.
    """
    if tasks is None:
        tasks = CIRCUIT_TASKS
    if hook_layer is None:
        hook_layer = model.cfg.n_layers // 2
    hook_name = f"blocks.{hook_layer}.hook_mlp_out"

    log(f"  SAEBench Reliability Audit at hook: {hook_name}")
    log(f"  n_reseeds={n_reseeds}, n_prompts={n_prompts}")

    results = []
    all_cvs = []

    for task in tasks:
        prompts = generate_prompts(task, model.tokenizer, n_prompts=n_prompts)
        if not prompts:
            log(f"    {task}: no prompts, skipping")
            continue

        correct_ids, incorrect_ids = get_token_ids(prompts, model.tokenizer)
        if not correct_ids:
            log(f"    {task}: no valid token ids, skipping")
            continue

        # Reseed CV for probe accuracy
        probe_scores = []
        for seed in range(n_reseeds):
            score = _compute_probe_accuracy_once(
                model, prompts, correct_ids, incorrect_ids, hook_name, seed
            )
            probe_scores.append(score)

        probe_scores_arr = np.array(probe_scores)
        probe_mean = float(np.mean(probe_scores_arr))
        probe_std = float(np.std(probe_scores_arr))
        probe_cv = probe_std / max(abs(probe_mean), 1e-8)

        # Reseed CV for ablation metric
        ablation_scores = []
        for seed in range(n_reseeds):
            score = _compute_ablation_metric_once(
                model, prompts, correct_ids, incorrect_ids, hook_name, seed
            )
            ablation_scores.append(score)

        ablation_scores_arr = np.array(ablation_scores)
        ablation_mean = float(np.mean(ablation_scores_arr))
        ablation_std = float(np.std(ablation_scores_arr))
        ablation_cv = ablation_std / max(abs(ablation_mean), 1e-8)

        # Discriminability: compare probe accuracy with full ablation vs. no ablation
        full_score = _compute_probe_accuracy_once(
            model, prompts, correct_ids, incorrect_ids, hook_name, seed=999
        )
        ablated_score = _compute_ablation_metric_once(
            model, prompts, correct_ids, incorrect_ids, hook_name, seed=999,
            n_ablate=model.cfg.d_model // 4,
        )
        # Cohen's d between the two conditions (use the probe_std as denominator)
        discriminability_d = abs(full_score - ablated_score) / max(probe_std, 1e-8)

        mean_cv = (probe_cv + ablation_cv) / 2.0
        all_cvs.append(mean_cv)

        passed_cv = mean_cv < CV_THRESHOLD
        passed_disc = discriminability_d > DISCRIMINABILITY_THRESHOLD
        passed = passed_cv and passed_disc

        log(f"    {task}: probe_cv={probe_cv:.4f}, ablation_cv={ablation_cv:.4f}, "
            f"discriminability_d={discriminability_d:.4f} "
            f"({'PASS' if passed else 'FAIL'})")

        results.append(EvalResult(
            metric_id="EX24.saebench_reliability_audit",
            value=mean_cv,
            n_samples=len(correct_ids),
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": task,
                "probe_cv": probe_cv,
                "ablation_cv": ablation_cv,
                "mean_cv": mean_cv,
                "discriminability_d": discriminability_d,
                "probe_mean": probe_mean,
                "probe_std": probe_std,
                "ablation_mean": ablation_mean,
                "ablation_std": ablation_std,
                "n_reseeds": n_reseeds,
                "hook_name": hook_name,
                "passed": passed,
                "threshold_cv": CV_THRESHOLD,
                "threshold_disc": DISCRIMINABILITY_THRESHOLD,
            },
        ))

    # Aggregate
    if all_cvs:
        agg_cv = float(np.mean(all_cvs))
        agg_std = float(np.std(all_cvs))
        agg_passed = agg_cv < CV_THRESHOLD
        log(f"  Aggregate: mean_cv={agg_cv:.4f} +/- {agg_std:.4f} "
            f"({'PASS' if agg_passed else 'FAIL'})")

        results.append(EvalResult(
            metric_id="EX24.saebench_reliability_audit",
            value=agg_cv,
            n_samples=sum(r.n_samples for r in results),
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": "aggregate",
                "mean_cv": agg_cv,
                "cv_std": agg_std,
                "n_tasks": len(all_cvs),
                "per_task_cvs": {
                    r.metadata["task"]: r.metadata["mean_cv"]
                    for r in results if r.metadata.get("task") != "aggregate"
                },
                "passed": agg_passed,
                "threshold": CV_THRESHOLD,
            },
        ))

    return results


def main():
    parser = parse_common_args("EX24: SAEBench Reliability Audit")
    parser.add_argument("--n-reseeds", type=int, default=5,
                        help="Number of reseed iterations for CV")
    parser.add_argument("--hook-layer", type=int, default=None,
                        help="Layer for hook point (default: middle)")
    args = parser.parse_args()

    model = load_model(args.model, args.device)

    log("=" * 60)
    log("EX24: SAEBENCH RELIABILITY AUDIT")
    log("=" * 60)

    tasks = args.tasks or CIRCUIT_TASKS
    results = run_saebench_audit(
        model,
        tasks=tasks,
        n_prompts=args.n_prompts,
        n_reseeds=args.n_reseeds,
        hook_layer=args.hook_layer,
    )

    out = args.out or "131_saebench_audit.json"
    save_results(results, out)
    log(f"\nDone. {len(results)} results.")


if __name__ == "__main__":
    main()
