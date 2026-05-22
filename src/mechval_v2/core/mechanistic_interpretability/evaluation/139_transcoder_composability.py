"""Transcoder Circuit Composability (Evaluation EX32)
Paper: Dunefsky, Chanin, Neel Nanda (2024). "Transcoders Find Interpretable
LLM Feature Circuits." arXiv:2406.11944
=============================================
Instrument:     EX32 --- Transcoder Circuit Composability
Categories:     evaluation
Validity layer: Internal / Construct
Criteria:       C2 Structural Plausibility, I2 Compositional Sufficiency
Establishes:    Whether transcoder decoder directions compose correctly
                with downstream weight matrices for weights-based circuit
                analysis --- i.e., whether the effect predicted by
                projecting a decoder direction through downstream weights
                matches the effect measured by activation patching
Requires:       CPU or GPU, model
=============================================

Transcoders decompose MLP computation as f(x) ~ sum_i a_i(x) * d_i + b,
where d_i is the decoder direction for feature i. For circuit analysis,
d_i must compose with downstream weight matrices: the contribution of
feature i to a downstream component should be predictable from
d_i @ W_downstream.

This metric tests composability by:
1. At a source MLP layer, identify the top-k activating transcoder features
   for each prompt (using the model's own MLP weights as a proxy transcoder).
2. For each feature, predict its downstream effect by projecting the decoder
   direction through the next layer's input weight matrix (weight-predicted).
3. Measure the actual effect via activation patching: zero the feature's
   contribution and observe the change in downstream activations (patching-measured).
4. Compute correlation between weight-predicted and patching-measured effects
   across features and prompts.

High correlation means weights-based circuit analysis is reliable for
transcoder features; low correlation means the weight composition is
misleading (e.g., due to nonlinearities, LayerNorm, or residual stream
interference).

Pass condition: composability_score > 0.5

Usage:
    uv run python 139_transcoder_composability.py --model gpt2 --device cpu
    uv run python 139_transcoder_composability.py --n-prompts 30 --top-k 10
"""

import numpy as np
import torch
import torch.nn.functional as F

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
    name="Transcoder Circuit Composability",
    paper_ref="Dunefsky et al. 2024",
    paper_cite=(
        "Dunefsky, Chanin, Nanda 2024, "
        "Transcoders Find Interpretable LLM Feature Circuits "
        "(arXiv:2406.11944)"
    ),
    description=(
        "Tests whether transcoder decoder directions compose correctly "
        "with downstream weight matrices. Compares weight-predicted "
        "effects (d_i @ W_downstream) against activation-patching "
        "measured effects for the same features."
    ),
    category="evaluation",
    tier="established",
    origin="external",
)

COMPOSABILITY_THRESHOLD = 0.5


def _get_mlp_weights(model, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract W_in and W_out for a given MLP layer.

    Returns:
        W_in: (d_model, d_mlp) input projection.
        W_out: (d_mlp, d_model) output projection (decoder directions as rows).
    """
    W_in = model.blocks[layer].mlp.W_in.detach()   # (d_model, d_mlp)
    W_out = model.blocks[layer].mlp.W_out.detach()  # (d_mlp, d_model)
    return W_in, W_out


def _compute_composability(
    model,
    prompt_text: str,
    source_layer: int,
    target_layer: int,
    top_k: int = 10,
) -> dict | None:
    """Compute composability between source MLP features and downstream weights.

    For each of the top-k MLP neurons (proxy transcoder features) at
    source_layer, compare:
      - weight_predicted: decoder_direction @ W_in_target (norm)
      - patching_measured: change in target MLP input when feature is zeroed

    Returns dict with per-feature scores, or None on failure.
    """
    tokens = model.to_tokens(prompt_text)

    # Get decoder directions (rows of W_out) at source layer
    _, W_out_source = _get_mlp_weights(model, source_layer)  # (d_mlp, d_model)
    W_in_target, _ = _get_mlp_weights(model, target_layer)   # (d_model, d_mlp)

    # Run clean forward, capture MLP activations at source and target input
    source_hook = f"blocks.{source_layer}.mlp.hook_post"
    target_hook = f"blocks.{target_layer}.hook_resid_mid"

    _, cache_clean = model.run_with_cache(
        tokens,
        names_filter=[source_hook, target_hook],
    )

    source_acts = cache_clean[source_hook][0, -1]  # (d_mlp,)
    target_clean = cache_clean[target_hook][0, -1]  # (d_model,)

    # Identify top-k features by activation magnitude
    k = min(top_k, source_acts.shape[0])
    top_indices = torch.topk(source_acts.abs(), k).indices  # (k,)

    weight_predicted = []
    patching_measured = []

    for idx in top_indices:
        feat_idx = idx.item()
        decoder_dir = W_out_source[feat_idx]  # (d_model,)

        # Weight-predicted: how much this decoder direction projects into
        # the target layer's input space
        wp = (decoder_dir @ W_in_target).norm().item()
        weight_predicted.append(wp)

        # Patching-measured: zero this feature and observe change at target
        def ablate_hook(value, hook, _idx=feat_idx):
            value[0, -1, _idx] = 0.0
            return value

        _, cache_ablated = model.run_with_cache(
            tokens,
            fwd_hooks=[(source_hook, ablate_hook)],
            names_filter=[target_hook],
        )
        target_ablated = cache_ablated[target_hook][0, -1]  # (d_model,)

        delta = (target_clean - target_ablated).norm().item()
        patching_measured.append(delta)

    wp_arr = np.array(weight_predicted)
    pm_arr = np.array(patching_measured)

    # Correlation between weight-predicted and patching-measured
    if wp_arr.std() < 1e-10 or pm_arr.std() < 1e-10:
        return None

    corr = float(np.corrcoef(wp_arr, pm_arr)[0, 1])

    return {
        "correlation": corr,
        "weight_predicted": wp_arr.tolist(),
        "patching_measured": pm_arr.tolist(),
        "top_feature_indices": top_indices.tolist(),
    }


def run_transcoder_composability(
    model,
    tasks: list[str] | None = None,
    n_prompts: int = 30,
    top_k: int = 10,
) -> list[EvalResult]:
    """Compute transcoder composability across tasks.

    For each task, generates prompts and measures the correlation between
    weight-predicted downstream effects and activation-patching measured
    effects for top transcoder features (using MLP neurons as proxy).

    Args:
        model: HookedTransformer instance.
        tasks: list of task names (default: CIRCUIT_TASKS).
        n_prompts: number of prompts per task.
        top_k: number of top features to evaluate per prompt.

    Returns:
        List of EvalResult, one per task plus an aggregate.
    """
    if tasks is None:
        tasks = CIRCUIT_TASKS

    n_layers = model.cfg.n_layers
    source_layer = n_layers // 2
    target_layer = min(source_layer + 1, n_layers - 1)

    log(f"  Transcoder composability: layer {source_layer} -> {target_layer}")
    log(f"  top_k={top_k}, n_prompts={n_prompts}")

    results = []
    all_scores = []

    for task in tasks:
        prompts = generate_prompts(task, model.tokenizer, n_prompts=n_prompts)
        if not prompts:
            log(f"    {task}: no prompts, skipping")
            continue

        task_correlations = []
        per_prompt_details = []

        for i, prompt in enumerate(prompts):
            try:
                result = _compute_composability(
                    model, prompt.text, source_layer, target_layer, top_k
                )
            except Exception as e:
                log(f"    {task} prompt {i}: error {e}")
                continue

            if result is None:
                continue

            task_correlations.append(result["correlation"])
            per_prompt_details.append({
                "prompt_index": i,
                **result,
            })

        if not task_correlations:
            log(f"    {task}: no valid results")
            continue

        mean_corr = float(np.mean(task_correlations))
        std_corr = float(np.std(task_correlations))
        passed = mean_corr > COMPOSABILITY_THRESHOLD
        all_scores.append(mean_corr)

        log(f"    {task}: composability={mean_corr:.4f} "
            f"+/- {std_corr:.4f} "
            f"({'PASS' if passed else 'FAIL'})")

        results.append(EvalResult(
            metric_id="EX32.transcoder_composability",
            value=mean_corr,
            n_samples=len(task_correlations),
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": task,
                "composability_score": mean_corr,
                "composability_std": std_corr,
                "source_layer": source_layer,
                "target_layer": target_layer,
                "n_prompts_evaluated": len(task_correlations),
                "top_k": top_k,
                "passed": passed,
                "threshold": COMPOSABILITY_THRESHOLD,
                "per_prompt": per_prompt_details,
            },
        ))

    # Aggregate result across all tasks
    if all_scores:
        agg_mean = float(np.mean(all_scores))
        agg_std = float(np.std(all_scores))
        agg_passed = agg_mean > COMPOSABILITY_THRESHOLD
        log(f"  Aggregate: composability={agg_mean:.4f} +/- {agg_std:.4f} "
            f"({'PASS' if agg_passed else 'FAIL'})")

        results.append(EvalResult(
            metric_id="EX32.transcoder_composability",
            value=agg_mean,
            n_samples=sum(r.n_samples for r in results),
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": "aggregate",
                "composability_score": agg_mean,
                "composability_std": agg_std,
                "n_tasks_evaluated": len(all_scores),
                "per_task_scores": {
                    r.metadata["task"]: r.metadata["composability_score"]
                    for r in results if r.metadata.get("task") != "aggregate"
                },
                "source_layer": source_layer,
                "target_layer": target_layer,
                "top_k": top_k,
                "passed": agg_passed,
                "threshold": COMPOSABILITY_THRESHOLD,
            },
        ))

    return results


def main():
    parser = parse_common_args("EX32: Transcoder Circuit Composability")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of top features per prompt (default: 10)")
    args = parser.parse_args()

    model = load_model(args.model, args.device)

    log("=" * 60)
    log("EX32: TRANSCODER CIRCUIT COMPOSABILITY")
    log("=" * 60)

    tasks = args.tasks or CIRCUIT_TASKS
    results = run_transcoder_composability(
        model,
        tasks=tasks,
        n_prompts=args.n_prompts,
        top_k=args.top_k,
    )

    out = args.out or "139_transcoder_composability.json"
    save_results(results, out)
    log(f"\nDone. {len(results)} results.")


if __name__ == "__main__":
    main()
