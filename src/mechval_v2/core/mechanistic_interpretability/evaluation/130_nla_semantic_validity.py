"""Metric: NLA Semantic Validity Gap --- round-trip reconstruction vs. semantic prediction

Paper: Anthropic (2026). "Natural Language Autoencoders Produce Unsupervised
Explanations of LLM Activations." transformer-circuits.pub/2026/nla/

Tests whether NLA-style round-trip reconstruction accuracy implies semantic
validity of the natural language description. High reconstruction + low
prediction = the text encodes activation information non-semantically,
exposing an E1 Content Validity gap.

NLA Semantic Validity Gap (Evaluation EX31)
=============================================
Instrument:     EX31 --- NLA Semantic Validity Gap
Categories:     evaluation
Validity layer: Construct
Criteria:       E1 Content Validity, C5 Convergent Validity
Establishes:    Whether features whose activations survive a compress-
                decompress cycle also have semantically predictive
                descriptions (i.e., whether informational validity
                implies semantic validity)
Requires:       CPU or GPU, model
=============================================

Core logic:
1. For each feature direction (from mid-layer residual stream), identify
   top-k activating positions across a corpus.
2. Compute PCA reconstruction fidelity: how well the top-k activation
   pattern reconstructs the original feature direction (proxy for NLA
   round-trip accuracy).
3. Compute semantic prediction accuracy: given the top-k token contexts,
   does the reconstructed direction predict which held-out tokens also
   activate the feature? (proxy for semantic validity).
4. Semantic validity gap = reconstruction_fidelity - prediction_accuracy.
5. Large gap = informational but not semantic validity.

Pass condition: mean_semantic_validity_gap < 0.3

Usage:
    uv run python 130_nla_semantic_validity.py --model gpt2 --device cpu
    uv run python 130_nla_semantic_validity.py --n-features 50 --top-k 20
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
    name="NLA Semantic Validity Gap",
    paper_ref="Anthropic, transformer-circuits.pub/2026/nla/ (May 2026)",
    paper_cite=(
        "Anthropic 2026, "
        "Natural Language Autoencoders Produce Unsupervised Explanations "
        "of LLM Activations (transformer-circuits.pub/2026/nla/)"
    ),
    description=(
        "Tests whether round-trip reconstruction accuracy implies semantic "
        "validity. Features with high reconstruction fidelity but low "
        "held-out prediction accuracy have informational but not semantic "
        "validity --- the E1 Content Validity gap NLAs leave open."
    ),
    category="evaluation",
    tier="established",
    origin="external",
)

SEMANTIC_GAP_THRESHOLD = 0.3


@torch.no_grad()
def _collect_activations(
    model, prompts, hook_name: str,
) -> torch.Tensor:
    """Collect residual stream activations at hook_name across prompts.

    Returns tensor of shape (total_tokens, d_model).
    """
    all_acts = []
    for p in prompts:
        tokens = model.to_tokens(p.text)
        captured = {}

        def fwd_hook(value, hook, _c=captured):
            _c["act"] = value.detach()
            return value

        model.run_with_hooks(tokens, fwd_hooks=[(hook_name, fwd_hook)])
        if "act" in captured:
            # (1, seq, d_model) -> (seq, d_model)
            all_acts.append(captured["act"].squeeze(0))

    if not all_acts:
        return torch.zeros(0, model.cfg.d_model, device=model.cfg.device)
    return torch.cat(all_acts, dim=0)


def _pca_top_direction(activations: torch.Tensor) -> torch.Tensor:
    """Compute the first principal component of activations.

    Args:
        activations: (n_tokens, d_model)

    Returns:
        direction: (d_model,) unit vector
    """
    centered = activations - activations.mean(dim=0, keepdim=True)
    # Use SVD for numerical stability
    _, _, Vh = torch.linalg.svd(centered, full_matrices=False)
    return Vh[0]  # First right singular vector


def _reconstruction_fidelity(
    direction: torch.Tensor,
    activations: torch.Tensor,
    top_k: int,
) -> float:
    """Compute reconstruction fidelity: how well top-k activations
    reconstruct the feature direction via PCA.

    Projects all activations onto the direction, selects top-k by
    projection magnitude, runs PCA on those top-k, and measures
    cosine similarity between the PCA direction and original.
    """
    # Project activations onto direction
    projections = activations @ direction  # (n_tokens,)
    k = min(top_k, len(projections))
    if k < 3:
        return 0.0

    # Select top-k by absolute projection magnitude
    _, top_indices = torch.topk(projections.abs(), k)
    top_acts = activations[top_indices]

    # Reconstruct direction from top-k via PCA
    reconstructed = _pca_top_direction(top_acts)

    # Cosine similarity
    cos = F.cosine_similarity(direction.unsqueeze(0), reconstructed.unsqueeze(0))
    return abs(cos.item())


def _prediction_accuracy(
    direction: torch.Tensor,
    activations: torch.Tensor,
    top_k: int,
) -> float:
    """Compute held-out prediction accuracy: does the feature direction
    predict which held-out tokens are high-activation?

    Splits tokens into train (top-k used to define the feature) and
    held-out. For held-out tokens, checks whether the direction's
    projection ranking matches the actual activation pattern.
    """
    projections = activations @ direction  # (n_tokens,)
    n = len(projections)
    k = min(top_k, n // 2)
    if k < 3 or n < k + 3:
        return 0.0

    # Top-k as "known feature examples"
    _, top_indices = torch.topk(projections.abs(), k)
    top_set = set(top_indices.tolist())

    # Held-out tokens
    held_out_mask = torch.ones(n, dtype=torch.bool, device=activations.device)
    held_out_mask[top_indices] = False
    held_out_indices = held_out_mask.nonzero(as_tuple=True)[0]

    if len(held_out_indices) < 3:
        return 0.0

    # Reconstruct direction from top-k
    top_acts = activations[top_indices]
    reconstructed = _pca_top_direction(top_acts)

    # Predict: rank held-out tokens by reconstructed direction projection
    held_out_acts = activations[held_out_indices]
    pred_scores = (held_out_acts @ reconstructed).abs()
    true_scores = (held_out_acts @ direction).abs()

    # Compute rank correlation as prediction accuracy proxy
    pred_rank = torch.argsort(torch.argsort(pred_scores, descending=True)).float()
    true_rank = torch.argsort(torch.argsort(true_scores, descending=True)).float()

    n_held = len(pred_rank)
    d = pred_rank - true_rank
    spearman = 1.0 - 6.0 * (d ** 2).sum().item() / (n_held * (n_held ** 2 - 1))

    # Convert to [0, 1] range
    return max(0.0, (spearman + 1.0) / 2.0)


def run_nla_semantic_validity(
    model,
    tasks: list[str] | None = None,
    n_prompts: int = 40,
    n_features: int = 20,
    top_k: int = 15,
    hook_layer: int | None = None,
) -> list[EvalResult]:
    """Run the NLA semantic validity gap diagnostic.

    For each task, extracts feature directions from residual stream
    activations, measures reconstruction fidelity and prediction accuracy,
    and reports the semantic validity gap.

    Args:
        model: HookedTransformer instance.
        tasks: list of task names.
        n_prompts: number of prompts per task.
        n_features: number of feature directions to test (via random
            projection + PCA on subsets).
        top_k: number of top activating positions for each feature.
        hook_layer: layer for hook point (default: middle layer).

    Returns:
        List of EvalResult, one per task plus aggregate.
    """
    if tasks is None:
        tasks = CIRCUIT_TASKS
    if hook_layer is None:
        hook_layer = model.cfg.n_layers // 2
    hook_name = f"blocks.{hook_layer}.hook_resid_pre"

    log(f"  NLA Semantic Validity Gap at hook: {hook_name}")
    log(f"  n_features={n_features}, top_k={top_k}, n_prompts={n_prompts}")

    results = []
    all_gaps = []

    for task in tasks:
        prompts = generate_prompts(task, model.tokenizer, n_prompts=n_prompts)
        if not prompts:
            log(f"    {task}: no prompts, skipping")
            continue

        activations = _collect_activations(model, prompts, hook_name)
        n_tokens = activations.shape[0]
        if n_tokens < top_k + 10:
            log(f"    {task}: insufficient tokens ({n_tokens}), skipping")
            continue

        # Generate feature directions via random projections
        d_model = activations.shape[1]
        feature_gaps = []
        feature_details = []

        for f_idx in range(n_features):
            # Random direction in activation space
            rand_dir = torch.randn(d_model, device=activations.device)
            rand_dir = F.normalize(rand_dir, dim=0)

            # Compute reconstruction fidelity and prediction accuracy
            recon = _reconstruction_fidelity(rand_dir, activations, top_k)
            pred = _prediction_accuracy(rand_dir, activations, top_k)
            gap = max(0.0, recon - pred)

            feature_gaps.append(gap)
            feature_details.append({
                "feature_index": f_idx,
                "reconstruction_fidelity": recon,
                "prediction_accuracy": pred,
                "semantic_validity_gap": gap,
            })

        if not feature_gaps:
            log(f"    {task}: no valid features")
            continue

        mean_gap = float(np.mean(feature_gaps))
        std_gap = float(np.std(feature_gaps))
        mean_recon = float(np.mean([d["reconstruction_fidelity"] for d in feature_details]))
        mean_pred = float(np.mean([d["prediction_accuracy"] for d in feature_details]))
        passed = mean_gap < SEMANTIC_GAP_THRESHOLD
        all_gaps.append(mean_gap)

        log(f"    {task}: gap={mean_gap:.4f} +/- {std_gap:.4f} "
            f"(recon={mean_recon:.4f}, pred={mean_pred:.4f}) "
            f"({'PASS' if passed else 'FAIL'})")

        results.append(EvalResult(
            metric_id="EX31.nla_semantic_validity",
            value=mean_gap,
            n_samples=n_tokens,
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": task,
                "mean_semantic_validity_gap": mean_gap,
                "std_semantic_validity_gap": std_gap,
                "mean_reconstruction_fidelity": mean_recon,
                "mean_prediction_accuracy": mean_pred,
                "n_features_tested": len(feature_gaps),
                "n_tokens": n_tokens,
                "top_k": top_k,
                "hook_name": hook_name,
                "passed": passed,
                "threshold": SEMANTIC_GAP_THRESHOLD,
                "per_feature": feature_details[:10],  # Top 10 for brevity
            },
        ))

    # Aggregate
    if all_gaps:
        agg_gap = float(np.mean(all_gaps))
        agg_std = float(np.std(all_gaps))
        agg_passed = agg_gap < SEMANTIC_GAP_THRESHOLD
        log(f"  Aggregate: gap={agg_gap:.4f} +/- {agg_std:.4f} "
            f"({'PASS' if agg_passed else 'FAIL'})")

        results.append(EvalResult(
            metric_id="EX31.nla_semantic_validity",
            value=agg_gap,
            n_samples=sum(r.n_samples for r in results),
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": "aggregate",
                "mean_semantic_validity_gap": agg_gap,
                "gap_std": agg_std,
                "n_tasks": len(all_gaps),
                "per_task_gaps": {
                    r.metadata["task"]: r.metadata["mean_semantic_validity_gap"]
                    for r in results if r.metadata.get("task") != "aggregate"
                },
                "passed": agg_passed,
                "threshold": SEMANTIC_GAP_THRESHOLD,
            },
        ))

    return results


def main():
    parser = parse_common_args("EX31: NLA Semantic Validity Gap")
    parser.add_argument("--n-features", type=int, default=20,
                        help="Number of feature directions to test (default: 20)")
    parser.add_argument("--top-k", type=int, default=15,
                        help="Top-k activating positions per feature (default: 15)")
    parser.add_argument("--hook-layer", type=int, default=None,
                        help="Layer for hook point (default: middle)")
    args = parser.parse_args()

    model = load_model(args.model, args.device)

    log("=" * 60)
    log("EX31: NLA SEMANTIC VALIDITY GAP")
    log("=" * 60)

    tasks = args.tasks or CIRCUIT_TASKS
    results = run_nla_semantic_validity(
        model,
        tasks=tasks,
        n_prompts=args.n_prompts,
        n_features=args.n_features,
        top_k=args.top_k,
        hook_layer=args.hook_layer,
    )

    out = args.out or "130_nla_semantic_validity.json"
    save_results(results, out)
    log(f"\nDone. {len(results)} results.")


if __name__ == "__main__":
    main()
