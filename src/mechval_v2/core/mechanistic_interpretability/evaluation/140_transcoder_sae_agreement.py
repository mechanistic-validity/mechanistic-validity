"""Transcoder vs SAE Feature Agreement (Evaluation EX33)
Paper: Dunefsky, Chanin, Neel Nanda (2024). "Transcoders Find Interpretable
LLM Feature Circuits." arXiv:2406.11944
=============================================
Instrument:     EX33 --- Transcoder vs SAE Feature Agreement
Categories:     evaluation
Validity layer: Construct
Criteria:       C5 Convergent Validity
Establishes:    Whether transcoder features and SAE features at the same
                layer identify similar computational roles, providing
                convergent evidence from two independent decomposition
                methods
Requires:       CPU or GPU, model
=============================================

Transcoders decompose MLP computation (input -> output function), while
SAEs decompose MLP activations (output state). If both methods identify
genuine structure, their features should partially overlap: features that
respond to similar inputs and point in similar directions in weight space.

This metric compares transcoder and SAE features at the same layer by:
1. Extracting "pseudo-transcoder" features (MLP W_out rows) and
   "pseudo-SAE" features (PCA directions of MLP output activations)
   as lightweight proxies that capture the structural distinction.
2. For each transcoder feature, finding the most similar SAE feature
   by cosine similarity of decoder directions.
3. For matched pairs, computing:
   (a) Direction similarity: cosine similarity of decoder directions.
   (b) Token overlap: Jaccard similarity of top activating tokens.
4. Agreement = mean of direction similarity and token overlap across
   matched feature pairs.

Moderate agreement (>0.3) provides convergent validity evidence; very
high agreement would be surprising since the methods decompose different
quantities (computation vs state).

Pass condition: transcoder_sae_agreement > 0.3

Usage:
    uv run python 140_transcoder_sae_agreement.py --model gpt2 --device cpu
    uv run python 140_transcoder_sae_agreement.py --n-prompts 30 --n-features 50
"""

import numpy as np
import torch
import torch.nn.functional as F

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    InstrumentInfo,
    generate_prompts,
    load_model,
    log,
    parse_common_args,
    save_results,
)

INSTRUMENT_INFO = InstrumentInfo(
    name="Transcoder vs SAE Feature Agreement",
    paper_ref="Dunefsky et al. 2024",
    paper_cite=(
        "Dunefsky, Chanin, Nanda 2024, "
        "Transcoders Find Interpretable LLM Feature Circuits "
        "(arXiv:2406.11944)"
    ),
    description=(
        "C5 convergent validity test. Compares transcoder features with "
        "SAE features at the same layer by measuring decoder direction "
        "cosine similarity and top activating token overlap between "
        "matched feature pairs from the two decomposition methods."
    ),
    category="evaluation",
    tier="established",
    origin="external",
)

AGREEMENT_THRESHOLD = 0.3


def _extract_pseudo_transcoder_features(model, layer: int) -> torch.Tensor:
    """Extract MLP W_out rows as proxy transcoder decoder directions.

    Returns: (d_mlp, d_model) tensor of decoder directions.
    """
    return model.blocks[layer].mlp.W_out.detach()  # (d_mlp, d_model)


@torch.no_grad()
def _extract_pseudo_sae_features(
    model,
    layer: int,
    prompts: list,
    n_features: int,
) -> torch.Tensor:
    """Extract PCA directions of MLP output as proxy SAE features.

    Collects MLP output activations across prompts, then computes
    the top-n_features principal components as SAE-like directions.

    Returns: (n_features, d_model) tensor of SAE-like directions.
    """
    hook_name = f"blocks.{layer}.hook_mlp_out"
    all_acts = []

    for prompt in prompts:
        tokens = model.to_tokens(prompt.text)
        _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
        acts = cache[hook_name][0]  # (seq, d_model)
        all_acts.append(acts)

    if not all_acts:
        return torch.zeros(n_features, model.cfg.d_model)

    all_acts = torch.cat(all_acts, dim=0)  # (total_tokens, d_model)
    all_acts = all_acts - all_acts.mean(dim=0, keepdim=True)

    # PCA via SVD
    n_components = min(n_features, all_acts.shape[0], all_acts.shape[1])
    _, _, Vh = torch.linalg.svd(all_acts, full_matrices=False)
    return Vh[:n_components]  # (n_components, d_model)


@torch.no_grad()
def _compute_top_activating_tokens(
    model,
    layer: int,
    directions: torch.Tensor,
    prompts: list,
    top_k: int = 20,
) -> list[set[int]]:
    """For each direction, find the top-k token positions by activation magnitude.

    Returns list of sets of token indices (flattened across prompts).
    """
    hook_name = f"blocks.{layer}.hook_mlp_out"
    all_acts = []

    for prompt in prompts:
        tokens = model.to_tokens(prompt.text)
        _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
        acts = cache[hook_name][0]  # (seq, d_model)
        all_acts.append(acts)

    if not all_acts:
        return [set() for _ in range(directions.shape[0])]

    all_acts = torch.cat(all_acts, dim=0)  # (total_tokens, d_model)

    # Project activations onto each direction
    directions_normed = F.normalize(directions, dim=-1)  # (n_dirs, d_model)
    projections = all_acts @ directions_normed.T  # (total_tokens, n_dirs)

    result = []
    for d_idx in range(directions.shape[0]):
        proj = projections[:, d_idx].abs()
        k = min(top_k, proj.shape[0])
        top_indices = torch.topk(proj, k).indices
        result.append(set(top_indices.tolist()))

    return result


def _compute_agreement(
    transcoder_dirs: torch.Tensor,
    sae_dirs: torch.Tensor,
    transcoder_tokens: list[set[int]],
    sae_tokens: list[set[int]],
    n_features: int,
) -> dict:
    """Compute agreement between transcoder and SAE features.

    For each transcoder feature, find the best-matching SAE feature
    by cosine similarity, then compute direction similarity and
    token overlap.

    Returns dict with per-feature and aggregate scores.
    """
    # Normalize directions
    tc_normed = F.normalize(transcoder_dirs, dim=-1)  # (n_tc, d_model)
    sae_normed = F.normalize(sae_dirs, dim=-1)        # (n_sae, d_model)

    # Cosine similarity matrix
    cos_sim = tc_normed @ sae_normed.T  # (n_tc, n_sae)

    n_tc = min(n_features, tc_normed.shape[0])
    direction_sims = []
    token_overlaps = []
    match_indices = []

    for i in range(n_tc):
        # Best match by absolute cosine similarity
        best_idx = cos_sim[i].abs().argmax().item()
        dir_sim = cos_sim[i, best_idx].abs().item()
        direction_sims.append(dir_sim)
        match_indices.append(best_idx)

        # Token overlap (Jaccard)
        if i < len(transcoder_tokens) and best_idx < len(sae_tokens):
            tc_set = transcoder_tokens[i]
            sae_set = sae_tokens[best_idx]
            if tc_set or sae_set:
                jaccard = len(tc_set & sae_set) / len(tc_set | sae_set)
            else:
                jaccard = 0.0
            token_overlaps.append(jaccard)
        else:
            token_overlaps.append(0.0)

    mean_dir_sim = float(np.mean(direction_sims)) if direction_sims else 0.0
    mean_token_overlap = float(np.mean(token_overlaps)) if token_overlaps else 0.0
    agreement = (mean_dir_sim + mean_token_overlap) / 2.0

    return {
        "transcoder_sae_agreement": agreement,
        "mean_direction_similarity": mean_dir_sim,
        "mean_token_overlap": mean_token_overlap,
        "n_features_compared": n_tc,
        "direction_similarities": direction_sims,
        "token_overlaps": token_overlaps,
        "match_indices": match_indices,
    }


def run_transcoder_sae_agreement(
    model,
    tasks: list[str] | None = None,
    n_prompts: int = 30,
    n_features: int = 50,
    top_k_tokens: int = 20,
) -> list[EvalResult]:
    """Compute transcoder vs SAE feature agreement across tasks.

    Args:
        model: HookedTransformer instance.
        tasks: list of task names (default: CIRCUIT_TASKS).
        n_prompts: number of prompts per task.
        n_features: number of features to compare.
        top_k_tokens: number of top activating tokens per feature.

    Returns:
        List of EvalResult, one per task plus an aggregate.
    """
    if tasks is None:
        tasks = CIRCUIT_TASKS

    mid_layer = model.cfg.n_layers // 2
    log(f"  Transcoder vs SAE agreement at layer {mid_layer}")
    log(f"  n_features={n_features}, top_k_tokens={top_k_tokens}")

    # Extract pseudo-transcoder features (same for all tasks)
    transcoder_dirs = _extract_pseudo_transcoder_features(model, mid_layer)

    results = []
    all_scores = []

    for task in tasks:
        prompts = generate_prompts(task, model.tokenizer, n_prompts=n_prompts)
        if not prompts:
            log(f"    {task}: no prompts, skipping")
            continue

        try:
            # Extract pseudo-SAE features from this task's activations
            sae_dirs = _extract_pseudo_sae_features(
                model, mid_layer, prompts, n_features
            )

            # Get top activating tokens for both sets of features
            tc_tokens = _compute_top_activating_tokens(
                model, mid_layer, transcoder_dirs[:n_features],
                prompts, top_k_tokens
            )
            sae_tokens = _compute_top_activating_tokens(
                model, mid_layer, sae_dirs,
                prompts, top_k_tokens
            )

            # Compute agreement
            agreement = _compute_agreement(
                transcoder_dirs, sae_dirs,
                tc_tokens, sae_tokens,
                n_features,
            )
        except Exception as e:
            log(f"    {task}: error {e}")
            continue

        score = agreement["transcoder_sae_agreement"]
        passed = score > AGREEMENT_THRESHOLD
        all_scores.append(score)

        log(f"    {task}: agreement={score:.4f} "
            f"(dir={agreement['mean_direction_similarity']:.4f}, "
            f"tok={agreement['mean_token_overlap']:.4f}) "
            f"{'PASS' if passed else 'FAIL'}")

        results.append(EvalResult(
            metric_id="EX33.transcoder_sae_agreement",
            value=score,
            n_samples=agreement["n_features_compared"],
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": task,
                "transcoder_sae_agreement": score,
                "mean_direction_similarity": agreement["mean_direction_similarity"],
                "mean_token_overlap": agreement["mean_token_overlap"],
                "n_features_compared": agreement["n_features_compared"],
                "layer": mid_layer,
                "n_prompts": len(prompts),
                "top_k_tokens": top_k_tokens,
                "passed": passed,
                "threshold": AGREEMENT_THRESHOLD,
            },
        ))

    # Aggregate result across all tasks
    if all_scores:
        agg_mean = float(np.mean(all_scores))
        agg_std = float(np.std(all_scores))
        agg_passed = agg_mean > AGREEMENT_THRESHOLD
        log(f"  Aggregate: agreement={agg_mean:.4f} +/- {agg_std:.4f} "
            f"({'PASS' if agg_passed else 'FAIL'})")

        results.append(EvalResult(
            metric_id="EX33.transcoder_sae_agreement",
            value=agg_mean,
            n_samples=sum(r.n_samples for r in results),
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": "aggregate",
                "transcoder_sae_agreement": agg_mean,
                "agreement_std": agg_std,
                "n_tasks_evaluated": len(all_scores),
                "per_task_scores": {
                    r.metadata["task"]: r.metadata["transcoder_sae_agreement"]
                    for r in results if r.metadata.get("task") != "aggregate"
                },
                "layer": mid_layer,
                "n_features": n_features,
                "passed": agg_passed,
                "threshold": AGREEMENT_THRESHOLD,
            },
        ))

    return results


def main():
    parser = parse_common_args("EX33: Transcoder vs SAE Feature Agreement")
    parser.add_argument("--n-features", type=int, default=50,
                        help="Number of features to compare (default: 50)")
    parser.add_argument("--top-k-tokens", type=int, default=20,
                        help="Top-k tokens per feature for overlap (default: 20)")
    args = parser.parse_args()

    model = load_model(args.model, args.device)

    log("=" * 60)
    log("EX33: TRANSCODER VS SAE FEATURE AGREEMENT")
    log("=" * 60)

    tasks = args.tasks or CIRCUIT_TASKS
    results = run_transcoder_sae_agreement(
        model,
        tasks=tasks,
        n_prompts=args.n_prompts,
        n_features=args.n_features,
        top_k_tokens=args.top_k_tokens,
    )

    out = args.out or "140_transcoder_sae_agreement.json"
    save_results(results, out)
    log(f"\nDone. {len(results)} results.")


if __name__ == "__main__":
    main()
