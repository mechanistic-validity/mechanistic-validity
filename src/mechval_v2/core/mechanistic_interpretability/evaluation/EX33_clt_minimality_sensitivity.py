"""CLT Graph Minimality Sensitivity (Evaluation EX33)
Paper: Ameisen, Lindsey et al. (Anthropic, 2025). "Circuit Tracing:
Revealing Computational Graphs in Language Models."
transformer-circuits.pub
=============================================
Instrument:     EX33 --- CLT Graph Minimality Sensitivity
Categories:     evaluation
Validity layer: Construct
Criteria:       C4 Minimality, M4 Sensitivity
Establishes:    Whether the pruned attribution graph's size is stable
                across pruning thresholds, or whether small threshold
                changes cause large graph size changes (fragile
                minimality)
Requires:       CPU or GPU, model
=============================================

CLT attribution graphs are pruned by removing edges (heads) below a
researcher-chosen attribution threshold. If the resulting graph size
is highly sensitive to this threshold, the minimality claim is fragile
--- the "minimal sufficient circuit" depends critically on an arbitrary
hyperparameter.

For each task:
1. Score every attention head via activation patching (same procedure
   as EX29 graph faithfulness).
2. Normalize scores to [0, 1] range.
3. Sweep pruning thresholds from aggressive (keep few heads) to
   permissive (keep many heads).
4. At each threshold, count the number of heads retained.
5. Compute minimality_stability = 1 - normalized_variance, where
   normalized_variance = var(graph_sizes) / (max_possible_variance).
   Max possible variance occurs when graph sizes are bimodal
   (all-or-nothing), so we normalize by (N/2)^2 where N is total heads.

High stability means the graph size changes smoothly across thresholds
(no cliff effects). Low stability means small threshold changes cause
large jumps in graph size.

Pass condition: minimality_stability > 0.5

Usage:
    uv run python EX33_clt_minimality_sensitivity.py --n-prompts 30
    uv run python EX33_clt_minimality_sensitivity.py --model gpt2 --device cpu
"""

import numpy as np
import torch

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    InstrumentInfo,
    calibrate_mean_z,
    generate_prompts,
    get_token_ids,
    load_model,
    log,
    logit_diff_from_logits,
    make_ablation_hook,
    parse_common_args,
    save_results,
)

INSTRUMENT_INFO = InstrumentInfo(
    name="CLT Graph Minimality Sensitivity",
    paper_ref="Ameisen, Lindsey et al. (Anthropic, 2025)",
    paper_cite=(
        "Ameisen, Lindsey et al. 2025, "
        "Circuit Tracing: Revealing Computational Graphs in Language Models "
        "(Anthropic, transformer-circuits.pub)"
    ),
    description=(
        "Sweeps pruning thresholds on a head-attribution graph and "
        "measures how stable the resulting graph size is. Fragile "
        "minimality (large size jumps at small threshold changes) "
        "undermines claims about 'the minimal circuit'."
    ),
    category="evaluation",
    tier="established",
    origin="external",
)

STABILITY_THRESHOLD = 0.5

# Threshold sweep: 20 evenly spaced points in (0, 1)
N_THRESHOLD_STEPS = 20


@torch.no_grad()
def _score_heads(
    model,
    tokens: torch.Tensor,
    correct_id: int,
    incorrect_id: int,
    mean_z: torch.Tensor,
) -> list[float]:
    """Score every (layer, head) by logit-diff change under mean-ablation.

    Returns a flat list of absolute logit-diff changes, ordered by
    (layer, head) in row-major order.
    """
    baseline_logits = model(tokens)
    baseline_ld = logit_diff_from_logits(baseline_logits, correct_id, incorrect_id)

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    scores = []

    for layer in range(n_layers):
        for head in range(n_heads):
            hooks = make_ablation_hook(
                {layer: [head]}, mean_z, ablation_type="mean"
            )
            ablated_logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
            ablated_ld = logit_diff_from_logits(
                ablated_logits, correct_id, incorrect_id
            )
            scores.append(abs(baseline_ld - ablated_ld))

    return scores


def _compute_minimality_stability(
    scores: list[float],
    n_steps: int = N_THRESHOLD_STEPS,
) -> tuple[float, list[dict]]:
    """Sweep thresholds and compute stability of graph size.

    Args:
        scores: flat list of head attribution scores.
        n_steps: number of threshold points to sweep.

    Returns:
        (minimality_stability, sweep_details) where sweep_details is a
        list of dicts with threshold, n_retained, fraction_retained.
    """
    if not scores:
        return 1.0, []

    scores_arr = np.array(scores)
    max_score = scores_arr.max()

    if max_score < 1e-10:
        # All scores near zero: graph is trivially stable (always empty)
        return 1.0, []

    # Normalize to [0, 1]
    normalized = scores_arr / max_score

    n_total = len(scores)
    thresholds = np.linspace(0.0, 1.0, n_steps + 2)[1:-1]  # exclude 0 and 1

    graph_sizes = []
    sweep_details = []

    for thresh in thresholds:
        n_retained = int((normalized >= thresh).sum())
        fraction = n_retained / n_total
        graph_sizes.append(n_retained)
        sweep_details.append({
            "threshold": float(thresh),
            "n_retained": n_retained,
            "fraction_retained": fraction,
        })

    graph_sizes_arr = np.array(graph_sizes, dtype=float)

    # Normalized variance: divide by the maximum possible variance
    # Max variance for N items: occurs at bimodal (0, N) distribution
    # which gives variance = (N/2)^2 = N^2/4
    max_var = (n_total / 2.0) ** 2
    if max_var < 1e-10:
        return 1.0, sweep_details

    actual_var = float(np.var(graph_sizes_arr))
    normalized_var = actual_var / max_var

    stability = 1.0 - min(normalized_var, 1.0)
    return float(stability), sweep_details


def run_clt_minimality_sensitivity(
    model,
    tasks: list[str] | None = None,
    n_prompts: int = 30,
    n_threshold_steps: int = N_THRESHOLD_STEPS,
) -> list[EvalResult]:
    """Compute CLT graph minimality sensitivity across tasks.

    For each task, scores heads via activation patching, sweeps pruning
    thresholds, and measures how stable the graph size is.

    Args:
        model: HookedTransformer instance.
        tasks: list of task names (default: CIRCUIT_TASKS).
        n_prompts: number of prompts per task.
        n_threshold_steps: number of threshold sweep points.

    Returns:
        List of EvalResult, one per task plus an aggregate.
    """
    if tasks is None:
        tasks = CIRCUIT_TASKS
    log(f"  CLT Minimality Sensitivity: n_prompts={n_prompts}, "
        f"n_threshold_steps={n_threshold_steps}")

    results = []
    all_stability = []

    for task in tasks:
        prompts = generate_prompts(task, model.tokenizer, n_prompts=n_prompts)
        if not prompts:
            log(f"    {task}: no prompts, skipping")
            continue

        correct_ids, incorrect_ids = get_token_ids(prompts, model.tokenizer)
        if not correct_ids:
            log(f"    {task}: no valid token ids, skipping")
            continue

        mean_z = calibrate_mean_z(model, prompts, n_calibration=min(20, len(prompts)))

        task_stability = []
        per_prompt_details = []

        for i, prompt in enumerate(prompts):
            if i >= len(correct_ids):
                break

            tokens = model.to_tokens(prompt.text)

            try:
                scores = _score_heads(
                    model, tokens, correct_ids[i], incorrect_ids[i], mean_z
                )
                stability, sweep = _compute_minimality_stability(
                    scores, n_steps=n_threshold_steps
                )
            except Exception as e:
                log(f"    {task} prompt {i}: error {e}")
                continue

            task_stability.append(stability)
            per_prompt_details.append({
                "prompt_index": i,
                "minimality_stability": stability,
                "n_heads_total": len(scores),
                "max_score": float(max(scores)) if scores else 0.0,
                "sweep": sweep,
            })

        if not task_stability:
            log(f"    {task}: no valid results")
            continue

        mean_stability = float(np.mean(task_stability))
        std_stability = float(np.std(task_stability))
        passed = mean_stability > STABILITY_THRESHOLD
        all_stability.append(mean_stability)

        log(f"    {task}: minimality_stability={mean_stability:.4f} "
            f"+/- {std_stability:.4f} ({'PASS' if passed else 'FAIL'})")

        results.append(EvalResult(
            metric_id="EX33.clt_minimality_sensitivity",
            value=mean_stability,
            n_samples=len(task_stability),
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": task,
                "minimality_stability": mean_stability,
                "minimality_stability_std": std_stability,
                "n_prompts_evaluated": len(task_stability),
                "n_threshold_steps": n_threshold_steps,
                "passed": passed,
                "threshold": STABILITY_THRESHOLD,
                "per_prompt": per_prompt_details,
            },
        ))

    # Aggregate result
    if all_stability:
        agg_mean = float(np.mean(all_stability))
        agg_std = float(np.std(all_stability))
        agg_passed = agg_mean > STABILITY_THRESHOLD
        log(f"  Aggregate: minimality_stability={agg_mean:.4f} "
            f"+/- {agg_std:.4f} ({'PASS' if agg_passed else 'FAIL'})")

        results.append(EvalResult(
            metric_id="EX33.clt_minimality_sensitivity",
            value=agg_mean,
            n_samples=sum(r.n_samples for r in results),
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": "aggregate",
                "minimality_stability": agg_mean,
                "minimality_stability_std": agg_std,
                "n_tasks_evaluated": len(all_stability),
                "per_task_stability": {
                    r.metadata["task"]: r.metadata["minimality_stability"]
                    for r in results if r.metadata.get("task") != "aggregate"
                },
                "n_threshold_steps": n_threshold_steps,
                "passed": agg_passed,
                "threshold": STABILITY_THRESHOLD,
            },
        ))

    return results


def main():
    parser = parse_common_args("EX33: CLT Graph Minimality Sensitivity")
    parser.add_argument("--n-threshold-steps", type=int, default=N_THRESHOLD_STEPS,
                        help="Number of threshold sweep points (default: 20)")
    args = parser.parse_args()

    model = load_model(args.model, args.device)

    log("=" * 60)
    log("EX33: CLT GRAPH MINIMALITY SENSITIVITY")
    log("=" * 60)

    tasks = args.tasks or CIRCUIT_TASKS
    results = run_clt_minimality_sensitivity(
        model,
        tasks=tasks,
        n_prompts=args.n_prompts,
        n_threshold_steps=args.n_threshold_steps,
    )

    out = args.out or "EX33_clt_minimality_sensitivity.json"
    save_results(results, out)
    log(f"\nDone. {len(results)} results.")


if __name__ == "__main__":
    main()
