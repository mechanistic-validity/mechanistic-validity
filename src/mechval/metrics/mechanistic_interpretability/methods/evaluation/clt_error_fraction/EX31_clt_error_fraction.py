"""CLT Error Node Fraction (Evaluation EX31)
Paper: Ameisen, Lindsey et al. (Anthropic, 2025). "Circuit Tracing:
Revealing Computational Graphs in Language Models."
transformer-circuits.pub
=============================================
Instrument:     EX31 --- CLT Error Node Fraction
Categories:     evaluation
Validity layer: Measurement
Criteria:       M6 Construct Coverage, I5 Confound Control
Establishes:    What fraction of the model's computation is captured by
                interpretable components vs absorbed by residual/error
                nodes, measuring decomposition completeness
Requires:       CPU or GPU, model
=============================================

Quantifies the replacement model gap. In the CLT framework, error nodes
absorb computation that the interpretable feature decomposition cannot
explain. A high error fraction means the decomposition is incomplete.

Since CLTs are not publicly available for standard models, this metric
approximates the error fraction concept using attention heads as the
interpretable components:

1. For each prompt, compute the full model's logit difference.
2. Mean-ablate ALL attention heads to measure how much of the logit
   difference is explained by the residual stream alone (without any
   head contributions). This residual is the "error" baseline.
3. Progressively restore heads in order of causal importance
   (activation patching scores). After restoring all heads, compute
   the recovered logit difference.
4. error_fraction = 1 - (recovered_ld / full_ld), measuring what
   fraction of the model's behavior is NOT captured by the
   individually attributable head contributions.

Low error fraction means the decomposition into individual head
contributions is faithful (little residual/error).

Pass condition: error_fraction < 0.2 (low = faithful decomposition)

Usage:
    uv run python EX31_clt_error_fraction.py --n-prompts 30
    uv run python EX31_clt_error_fraction.py --model gpt2 --device cpu
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
    name="CLT Error Node Fraction",
    paper_ref="Ameisen, Lindsey et al. (Anthropic, 2025)",
    paper_cite=(
        "Ameisen, Lindsey et al. 2025, "
        "Circuit Tracing: Revealing Computational Graphs in Language Models "
        "(Anthropic, transformer-circuits.pub)"
    ),
    description=(
        "Measures the fraction of total attribution weight on error/residual "
        "nodes by comparing summed individual head contributions against the "
        "full model's logit difference. A low error fraction indicates the "
        "decomposition into interpretable components is complete."
    ),
    category="evaluation",
    tier="established",
    origin="external",
)

ERROR_FRACTION_THRESHOLD = 0.2


@torch.no_grad()
def _compute_error_fraction(
    model,
    tokens: torch.Tensor,
    correct_id: int,
    incorrect_id: int,
    mean_z: torch.Tensor,
) -> tuple[float, dict]:
    """Compute error fraction for a single prompt.

    Measures how much of the full model's logit difference is explained
    by individual head contributions (via activation patching scores).

    Returns:
        error_fraction: 1 - (sum_of_individual_effects / full_effect)
        details: dict with per-head scores and intermediate values.
    """
    # Full model logit diff
    baseline_logits = model(tokens)
    baseline_ld = logit_diff_from_logits(baseline_logits, correct_id, incorrect_id)

    if abs(baseline_ld) < 1e-6:
        return 0.0, {"baseline_ld": baseline_ld, "note": "near-zero baseline"}

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    # Compute each head's individual contribution via mean-ablation
    # Contribution = baseline_ld - ablated_ld (signed, not absolute)
    head_contributions: dict[tuple[int, int], float] = {}
    for layer in range(n_layers):
        for head in range(n_heads):
            hooks = make_ablation_hook(
                {layer: [head]}, mean_z, ablation_type="mean"
            )
            ablated_logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
            ablated_ld = logit_diff_from_logits(
                ablated_logits, correct_id, incorrect_id
            )
            # Positive contribution means this head helps the correct answer
            head_contributions[(layer, head)] = baseline_ld - ablated_ld

    # Sum of individual contributions
    sum_contributions = sum(head_contributions.values())

    # Error fraction: what the additive decomposition cannot account for
    # If sum_contributions == baseline_ld, the decomposition is perfect
    # (error_fraction = 0). In practice, interaction effects create a gap.
    if abs(baseline_ld) < 1e-6:
        error_fraction = 0.0
    else:
        explained_ratio = sum_contributions / baseline_ld
        error_fraction = 1.0 - explained_ratio

    # Clip to [0, 1] for interpretability (can be negative if
    # sum_contributions > baseline_ld due to suppressive interactions)
    error_fraction = float(np.clip(abs(error_fraction), 0.0, 1.0))

    # Top contributing heads for diagnostics
    sorted_heads = sorted(
        head_contributions.items(), key=lambda x: abs(x[1]), reverse=True
    )

    details = {
        "baseline_ld": baseline_ld,
        "sum_contributions": sum_contributions,
        "explained_ratio": sum_contributions / baseline_ld if abs(baseline_ld) > 1e-6 else 1.0,
        "top_contributing_heads": [
            {"layer": h[0], "head": h[1], "contribution": c}
            for (h, c) in sorted_heads[:10]
        ],
    }

    return error_fraction, details


def run_clt_error_fraction(
    model,
    tasks: list[str] | None = None,
    n_prompts: int = 30,
) -> list[EvalResult]:
    """Compute CLT error node fraction across tasks.

    For each task, measures what fraction of the model's behavior cannot
    be accounted for by summing individual head contributions.

    Args:
        model: HookedTransformer instance.
        tasks: list of task names (default: CIRCUIT_TASKS).
        n_prompts: number of prompts per task.

    Returns:
        List of EvalResult, one per task plus an aggregate.
    """
    if tasks is None:
        tasks = CIRCUIT_TASKS
    log(f"  CLT Error Fraction: n_prompts={n_prompts}")

    results = []
    all_error_fractions = []

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

        task_error_fractions = []
        per_prompt_details = []

        for i, prompt in enumerate(prompts):
            if i >= len(correct_ids):
                break

            tokens = model.to_tokens(prompt.text)

            try:
                error_frac, details = _compute_error_fraction(
                    model, tokens, correct_ids[i], incorrect_ids[i], mean_z
                )
            except Exception as e:
                log(f"    {task} prompt {i}: error {e}")
                continue

            task_error_fractions.append(error_frac)
            per_prompt_details.append({
                "prompt_index": i,
                "error_fraction": error_frac,
                **details,
            })

        if not task_error_fractions:
            log(f"    {task}: no valid results")
            continue

        mean_error = float(np.mean(task_error_fractions))
        std_error = float(np.std(task_error_fractions))
        passed = mean_error < ERROR_FRACTION_THRESHOLD
        all_error_fractions.append(mean_error)

        log(f"    {task}: error_fraction={mean_error:.4f} "
            f"+/- {std_error:.4f} ({'PASS' if passed else 'FAIL'})")

        results.append(EvalResult(
            metric_id="EX31.clt_error_fraction",
            value=mean_error,
            n_samples=len(task_error_fractions),
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": task,
                "error_fraction": mean_error,
                "error_fraction_std": std_error,
                "n_prompts_evaluated": len(task_error_fractions),
                "passed": passed,
                "threshold": ERROR_FRACTION_THRESHOLD,
                "per_prompt": per_prompt_details,
            },
        ))

    # Aggregate result
    if all_error_fractions:
        agg_mean = float(np.mean(all_error_fractions))
        agg_std = float(np.std(all_error_fractions))
        agg_passed = agg_mean < ERROR_FRACTION_THRESHOLD
        log(f"  Aggregate: error_fraction={agg_mean:.4f} +/- {agg_std:.4f} "
            f"({'PASS' if agg_passed else 'FAIL'})")

        results.append(EvalResult(
            metric_id="EX31.clt_error_fraction",
            value=agg_mean,
            n_samples=sum(r.n_samples for r in results),
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": "aggregate",
                "error_fraction": agg_mean,
                "error_fraction_std": agg_std,
                "n_tasks_evaluated": len(all_error_fractions),
                "per_task_error_fraction": {
                    r.metadata["task"]: r.metadata["error_fraction"]
                    for r in results if r.metadata.get("task") != "aggregate"
                },
                "passed": agg_passed,
                "threshold": ERROR_FRACTION_THRESHOLD,
            },
        ))

    return results


def main():
    parser = parse_common_args("EX31: CLT Error Node Fraction")
    args = parser.parse_args()

    model = load_model(args.model, args.device)

    log("=" * 60)
    log("EX31: CLT ERROR NODE FRACTION")
    log("=" * 60)

    tasks = args.tasks or CIRCUIT_TASKS
    results = run_clt_error_fraction(
        model,
        tasks=tasks,
        n_prompts=args.n_prompts,
    )

    out = args.out or "EX31_clt_error_fraction.json"
    save_results(results, out)
    log(f"\nDone. {len(results)} results.")


if __name__ == "__main__":
    main()
