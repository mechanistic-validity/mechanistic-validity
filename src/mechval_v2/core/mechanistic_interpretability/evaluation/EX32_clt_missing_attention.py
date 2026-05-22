"""CLT Missing Attention Quantification (Evaluation EX32)
Paper: Ameisen, Lindsey et al. (Anthropic, 2025). "Circuit Tracing:
Revealing Computational Graphs in Language Models."
transformer-circuits.pub
=============================================
Instrument:     EX32 --- CLT Missing Attention Quantification
Categories:     evaluation
Validity layer: Internal
Criteria:       I5 Confound Control, M6 Construct Coverage
Establishes:    What fraction of a task's causal effect comes from
                attention heads (QK circuits) that CLT attribution
                graphs omit entirely, quantifying the systematic
                explanatory gap in MLP-only circuit tracing
Requires:       CPU or GPU, model
=============================================

CLT attribution graphs represent only MLP-mediated feature interactions.
Attention mechanisms (QK circuits) are frozen and excluded from the graph.
This metric quantifies the resulting gap.

For each task:
1. Run the full model to get the baseline logit difference.
2. Activation-patch every attention head (mean-ablation of attention
   output z) to measure each head's causal contribution.
3. Activation-patch every MLP layer (mean-ablation of MLP output) to
   measure each MLP layer's causal contribution.
4. Compute attention_gap_fraction = sum(attn_effects) /
   (sum(attn_effects) + sum(mlp_effects)).

A high attention_gap_fraction means a large share of the model's
computation is causally driven by attention mechanisms that CLT
attribution graphs cannot represent.

Pass condition: attention_gap_fraction < 0.3

Usage:
    uv run python EX32_clt_missing_attention.py --n-prompts 30
    uv run python EX32_clt_missing_attention.py --model gpt2 --device cpu
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
    name="CLT Missing Attention Quantification",
    paper_ref="Ameisen, Lindsey et al. (Anthropic, 2025)",
    paper_cite=(
        "Ameisen, Lindsey et al. 2025, "
        "Circuit Tracing: Revealing Computational Graphs in Language Models "
        "(Anthropic, transformer-circuits.pub)"
    ),
    description=(
        "Quantifies the fraction of a task's causal effect attributable "
        "to attention heads (QK circuits) that CLT attribution graphs "
        "omit. Uses activation patching to separately measure attention "
        "and MLP causal contributions, then reports the attention gap."
    ),
    category="evaluation",
    tier="established",
    origin="external",
)

ATTENTION_GAP_THRESHOLD = 0.3


@torch.no_grad()
def _patch_attention_heads(
    model,
    tokens: torch.Tensor,
    correct_id: int,
    incorrect_id: int,
    mean_z: torch.Tensor,
) -> dict[tuple[int, int], float]:
    """Measure causal effect of each attention head via mean-ablation.

    Returns dict mapping (layer, head) -> absolute logit-diff change.
    """
    baseline_logits = model(tokens)
    baseline_ld = logit_diff_from_logits(baseline_logits, correct_id, incorrect_id)

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    effects: dict[tuple[int, int], float] = {}

    for layer in range(n_layers):
        for head in range(n_heads):
            hooks = make_ablation_hook(
                {layer: [head]}, mean_z, ablation_type="mean"
            )
            ablated_logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
            ablated_ld = logit_diff_from_logits(
                ablated_logits, correct_id, incorrect_id
            )
            effects[(layer, head)] = abs(baseline_ld - ablated_ld)

    return effects


@torch.no_grad()
def _patch_mlp_layers(
    model,
    tokens: torch.Tensor,
    correct_id: int,
    incorrect_id: int,
) -> dict[int, float]:
    """Measure causal effect of each MLP layer via mean-ablation.

    Mean-ablates hook_mlp_out at each layer independently.
    Returns dict mapping layer -> absolute logit-diff change.
    """
    baseline_logits = model(tokens)
    baseline_ld = logit_diff_from_logits(baseline_logits, correct_id, incorrect_id)

    n_layers = model.cfg.n_layers
    effects: dict[int, float] = {}

    # Compute mean MLP activations for ablation
    _, cache = model.run_with_cache(tokens)

    for layer in range(n_layers):
        hook_name = f"blocks.{layer}.hook_mlp_out"
        mean_mlp = cache[hook_name].mean(dim=1, keepdim=True)

        def mlp_ablation_hook(value, hook, _mean=mean_mlp):
            return _mean.expand_as(value)

        ablated_logits = model.run_with_hooks(
            tokens, fwd_hooks=[(hook_name, mlp_ablation_hook)]
        )
        ablated_ld = logit_diff_from_logits(
            ablated_logits, correct_id, incorrect_id
        )
        effects[layer] = abs(baseline_ld - ablated_ld)

    return effects


def _compute_attention_gap(
    attn_effects: dict[tuple[int, int], float],
    mlp_effects: dict[int, float],
) -> float:
    """Compute fraction of total causal effect from attention heads.

    Returns attention_gap_fraction in [0, 1].
    """
    total_attn = sum(attn_effects.values())
    total_mlp = sum(mlp_effects.values())
    total = total_attn + total_mlp

    if total < 1e-10:
        return 0.0

    return total_attn / total


def run_clt_missing_attention(
    model,
    tasks: list[str] | None = None,
    n_prompts: int = 30,
) -> list[EvalResult]:
    """Compute CLT missing attention gap across tasks.

    For each task, patches attention heads and MLP layers independently,
    then reports what fraction of total causal effect comes from attention
    (which CLT graphs miss).

    Args:
        model: HookedTransformer instance.
        tasks: list of task names (default: CIRCUIT_TASKS).
        n_prompts: number of prompts per task.

    Returns:
        List of EvalResult, one per task plus an aggregate.
    """
    if tasks is None:
        tasks = CIRCUIT_TASKS
    log(f"  CLT Missing Attention: n_prompts={n_prompts}")

    results = []
    all_gaps = []

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

        task_gaps = []
        per_prompt_details = []

        for i, prompt in enumerate(prompts):
            if i >= len(correct_ids):
                break

            tokens = model.to_tokens(prompt.text)

            try:
                attn_effects = _patch_attention_heads(
                    model, tokens, correct_ids[i], incorrect_ids[i], mean_z
                )
                mlp_effects = _patch_mlp_layers(
                    model, tokens, correct_ids[i], incorrect_ids[i]
                )
                gap = _compute_attention_gap(attn_effects, mlp_effects)
            except Exception as e:
                log(f"    {task} prompt {i}: error {e}")
                continue

            task_gaps.append(gap)

            total_attn = sum(attn_effects.values())
            total_mlp = sum(mlp_effects.values())

            # Top-5 attention heads by causal effect
            top_attn = sorted(
                attn_effects.items(), key=lambda x: x[1], reverse=True
            )[:5]

            per_prompt_details.append({
                "prompt_index": i,
                "attention_gap_fraction": gap,
                "total_attn_effect": total_attn,
                "total_mlp_effect": total_mlp,
                "top_attn_heads": [
                    {"layer": h[0], "head": h[1], "effect": s}
                    for (h, s) in top_attn
                ],
            })

        if not task_gaps:
            log(f"    {task}: no valid results")
            continue

        mean_gap = float(np.mean(task_gaps))
        std_gap = float(np.std(task_gaps))
        passed = mean_gap < ATTENTION_GAP_THRESHOLD
        all_gaps.append(mean_gap)

        log(f"    {task}: attention_gap={mean_gap:.4f} "
            f"+/- {std_gap:.4f} ({'PASS' if passed else 'FAIL'})")

        results.append(EvalResult(
            metric_id="EX32.clt_missing_attention",
            value=mean_gap,
            n_samples=len(task_gaps),
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": task,
                "attention_gap_fraction": mean_gap,
                "attention_gap_std": std_gap,
                "n_prompts_evaluated": len(task_gaps),
                "passed": passed,
                "threshold": ATTENTION_GAP_THRESHOLD,
                "per_prompt": per_prompt_details,
            },
        ))

    # Aggregate result
    if all_gaps:
        agg_mean = float(np.mean(all_gaps))
        agg_std = float(np.std(all_gaps))
        agg_passed = agg_mean < ATTENTION_GAP_THRESHOLD
        log(f"  Aggregate: attention_gap={agg_mean:.4f} +/- {agg_std:.4f} "
            f"({'PASS' if agg_passed else 'FAIL'})")

        results.append(EvalResult(
            metric_id="EX32.clt_missing_attention",
            value=agg_mean,
            n_samples=sum(r.n_samples for r in results),
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": "aggregate",
                "attention_gap_fraction": agg_mean,
                "attention_gap_std": agg_std,
                "n_tasks_evaluated": len(all_gaps),
                "per_task_gaps": {
                    r.metadata["task"]: r.metadata["attention_gap_fraction"]
                    for r in results if r.metadata.get("task") != "aggregate"
                },
                "passed": agg_passed,
                "threshold": ATTENTION_GAP_THRESHOLD,
            },
        ))

    return results


def main():
    parser = parse_common_args("EX32: CLT Missing Attention Quantification")
    args = parser.parse_args()

    model = load_model(args.model, args.device)

    log("=" * 60)
    log("EX32: CLT MISSING ATTENTION QUANTIFICATION")
    log("=" * 60)

    tasks = args.tasks or CIRCUIT_TASKS
    results = run_clt_missing_attention(
        model,
        tasks=tasks,
        n_prompts=args.n_prompts,
    )

    out = args.out or "EX32_clt_missing_attention.json"
    save_results(results, out)
    log(f"\nDone. {len(results)} results.")


if __name__ == "__main__":
    main()
