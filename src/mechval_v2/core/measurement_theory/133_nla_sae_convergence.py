"""Metric: NLA-SAE Convergent Validity --- MTMM test across description methods

Paper: Derived from Anthropic (2026). "Natural Language Autoencoders."
transformer-circuits.pub/2026/nla/ --- cross-referenced with SAE-based
feature descriptions via SAELens.

Tests agreement between NLA-style descriptions (top-k activation pattern
reconstructed via PCA) and SAE-style descriptions (decoder direction
projected through unembedding). Convergence of two independent description
methods on the same feature is C5 Convergent Validity evidence.

NLA-SAE Convergent Validity (Evaluation EX34)
=============================================
Instrument:     EX34 --- NLA-SAE Convergent Validity
Categories:     evaluation
Validity layer: Construct
Criteria:       C5 Convergent Validity
Establishes:    Whether two independent feature description methods (NLA-
                style activation patterns and weight-based decoder
                directions) converge on the same feature characterization
Requires:       CPU or GPU, model
=============================================

Core logic:
1. For each feature direction at a hook point, compute two independent
   characterizations:
   a. Activation-based: identify top-k activating tokens, compute PCA
      direction from their activation patterns (NLA proxy).
   b. Weight-based: project the feature direction through the unembedding
      matrix to get top promoted/suppressed tokens (SAE-style).
2. Compute agreement:
   a. Token overlap: Jaccard similarity of top promoted tokens.
   b. Direction cosine: cosine similarity between the PCA-reconstructed
      direction and the original feature direction.
3. High agreement = convergent validity; low agreement = the two methods
   are measuring different constructs.

Pass condition: mean_token_overlap > 0.3; mean_direction_cosine > 0.5

Usage:
    uv run python 133_nla_sae_convergence.py --model gpt2 --device cpu
    uv run python 133_nla_sae_convergence.py --n-features 30 --top-k 20
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
    name="NLA-SAE Convergent Validity",
    paper_ref="Anthropic NLA (May 2026) + SAELens",
    paper_cite=(
        "Anthropic 2026, Natural Language Autoencoders "
        "(transformer-circuits.pub/2026/nla/) cross-referenced with "
        "SAE decoder-based feature descriptions"
    ),
    description=(
        "MTMM test: compares NLA-style activation-based feature "
        "descriptions with SAE-style weight-based descriptions. "
        "Agreement between two independent methods is C5 Convergent "
        "Validity evidence. Divergence indicates the feature's meaning "
        "is method-dependent."
    ),
    category="evaluation",
    tier="established",
    origin="derived",
)

TOKEN_OVERLAP_THRESHOLD = 0.3
DIRECTION_COSINE_THRESHOLD = 0.5


@torch.no_grad()
def _collect_token_activations(
    model, prompts, hook_name: str,
) -> tuple[torch.Tensor, list[list[int]]]:
    """Collect activations and corresponding token IDs at hook_name.

    Returns:
        activations: (total_tokens, d_model) tensor
        token_ids: list of token ID lists, one per prompt
    """
    all_acts = []
    all_token_ids = []

    for p in prompts:
        tokens = model.to_tokens(p.text)
        captured = {}

        def fwd_hook(value, hook, _c=captured):
            _c["act"] = value.detach()
            return value

        model.run_with_hooks(tokens, fwd_hooks=[(hook_name, fwd_hook)])
        if "act" in captured:
            act = captured["act"].squeeze(0)  # (seq, d_model)
            all_acts.append(act)
            all_token_ids.append(tokens.squeeze(0).tolist())

    if not all_acts:
        return (
            torch.zeros(0, model.cfg.d_model, device=model.cfg.device),
            [],
        )
    return torch.cat(all_acts, dim=0), all_token_ids


def _activation_based_top_tokens(
    direction: torch.Tensor,
    activations: torch.Tensor,
    all_token_ids: list[list[int]],
    top_k: int,
) -> set[int]:
    """Get top tokens by activation projection onto direction (NLA proxy).

    Maps each activation position back to its token ID, then returns
    the set of unique token IDs in the top-k positions.
    """
    projections = activations @ direction  # (total_tokens,)
    k = min(top_k, len(projections))
    if k == 0:
        return set()

    _, top_indices = torch.topk(projections.abs(), k)

    # Map activation indices back to token IDs
    flat_token_ids = []
    for tids in all_token_ids:
        flat_token_ids.extend(tids)

    top_tokens = set()
    for idx in top_indices.tolist():
        if idx < len(flat_token_ids):
            top_tokens.add(flat_token_ids[idx])

    return top_tokens


@torch.no_grad()
def _weight_based_top_tokens(
    direction: torch.Tensor,
    model,
    top_k: int,
) -> set[int]:
    """Get top tokens by projecting direction through unembedding (SAE proxy).

    Computes direction @ W_U to get logit-space projections, then
    returns the top-k promoted token IDs.
    """
    W_U = model.W_U  # (d_model, d_vocab)
    logit_proj = direction @ W_U  # (d_vocab,)

    k = min(top_k, logit_proj.shape[0])
    _, top_indices = torch.topk(logit_proj.abs(), k)

    return set(top_indices.tolist())


def _jaccard_similarity(set_a: set, set_b: set) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def _pca_reconstruct_direction(
    direction: torch.Tensor,
    activations: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """Reconstruct feature direction from top-k activating positions via PCA.

    Returns the first principal component of the top-k activation subset.
    """
    projections = activations @ direction
    k = min(top_k, len(projections))
    if k < 3:
        return direction

    _, top_indices = torch.topk(projections.abs(), k)
    top_acts = activations[top_indices]

    centered = top_acts - top_acts.mean(dim=0, keepdim=True)
    _, _, Vh = torch.linalg.svd(centered, full_matrices=False)
    return Vh[0]


def run_nla_sae_convergence(
    model,
    tasks: list[str] | None = None,
    n_prompts: int = 40,
    n_features: int = 20,
    top_k: int = 20,
    hook_layer: int | None = None,
) -> list[EvalResult]:
    """Run the NLA-SAE convergent validity diagnostic.

    For each task, generates random feature directions in the residual
    stream and measures agreement between activation-based (NLA proxy)
    and weight-based (SAE proxy) descriptions.

    Args:
        model: HookedTransformer instance.
        tasks: list of task names.
        n_prompts: number of prompts per task.
        n_features: number of feature directions to test.
        top_k: number of top tokens for each method.
        hook_layer: layer for hook point (default: middle layer).

    Returns:
        List of EvalResult, one per task plus aggregate.
    """
    if tasks is None:
        tasks = CIRCUIT_TASKS
    if hook_layer is None:
        hook_layer = model.cfg.n_layers // 2
    hook_name = f"blocks.{hook_layer}.hook_resid_pre"

    log(f"  NLA-SAE Convergence at hook: {hook_name}")
    log(f"  n_features={n_features}, top_k={top_k}, n_prompts={n_prompts}")

    results = []
    all_overlaps = []

    for task in tasks:
        prompts = generate_prompts(task, model.tokenizer, n_prompts=n_prompts)
        if not prompts:
            log(f"    {task}: no prompts, skipping")
            continue

        activations, all_token_ids = _collect_token_activations(
            model, prompts, hook_name
        )
        n_tokens = activations.shape[0]
        if n_tokens < top_k + 5:
            log(f"    {task}: insufficient tokens ({n_tokens}), skipping")
            continue

        d_model = activations.shape[1]
        feature_overlaps = []
        feature_cosines = []
        feature_details = []

        for f_idx in range(n_features):
            # Random direction
            rand_dir = torch.randn(d_model, device=activations.device)
            rand_dir = F.normalize(rand_dir, dim=0)

            # NLA proxy: top tokens from activation projections
            nla_tokens = _activation_based_top_tokens(
                rand_dir, activations, all_token_ids, top_k
            )

            # SAE proxy: top tokens from weight-based projection
            sae_tokens = _weight_based_top_tokens(rand_dir, model, top_k)

            # Token overlap (Jaccard)
            overlap = _jaccard_similarity(nla_tokens, sae_tokens)
            feature_overlaps.append(overlap)

            # Direction cosine (PCA reconstruction vs. original)
            reconstructed = _pca_reconstruct_direction(
                rand_dir, activations, top_k
            )
            cosine = abs(F.cosine_similarity(
                rand_dir.unsqueeze(0), reconstructed.unsqueeze(0)
            ).item())
            feature_cosines.append(cosine)

            feature_details.append({
                "feature_index": f_idx,
                "token_overlap": overlap,
                "direction_cosine": cosine,
                "n_nla_tokens": len(nla_tokens),
                "n_sae_tokens": len(sae_tokens),
                "n_shared_tokens": len(nla_tokens & sae_tokens),
            })

        if not feature_overlaps:
            log(f"    {task}: no valid features")
            continue

        mean_overlap = float(np.mean(feature_overlaps))
        mean_cosine = float(np.mean(feature_cosines))
        passed_overlap = mean_overlap > TOKEN_OVERLAP_THRESHOLD
        passed_cosine = mean_cosine > DIRECTION_COSINE_THRESHOLD
        passed = passed_overlap and passed_cosine
        all_overlaps.append(mean_overlap)

        log(f"    {task}: overlap={mean_overlap:.4f}, cosine={mean_cosine:.4f} "
            f"({'PASS' if passed else 'FAIL'})")

        results.append(EvalResult(
            metric_id="EX34.nla_sae_convergence",
            value=mean_overlap,
            n_samples=n_tokens,
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": task,
                "mean_token_overlap": mean_overlap,
                "mean_direction_cosine": mean_cosine,
                "std_token_overlap": float(np.std(feature_overlaps)),
                "std_direction_cosine": float(np.std(feature_cosines)),
                "n_features_tested": len(feature_overlaps),
                "n_tokens": n_tokens,
                "top_k": top_k,
                "hook_name": hook_name,
                "passed_overlap": passed_overlap,
                "passed_cosine": passed_cosine,
                "passed": passed,
                "threshold_overlap": TOKEN_OVERLAP_THRESHOLD,
                "threshold_cosine": DIRECTION_COSINE_THRESHOLD,
                "per_feature": feature_details[:10],  # Top 10 for brevity
            },
        ))

    # Aggregate
    if all_overlaps:
        agg_overlap = float(np.mean(all_overlaps))
        agg_passed = agg_overlap > TOKEN_OVERLAP_THRESHOLD
        log(f"  Aggregate: overlap={agg_overlap:.4f} "
            f"({'PASS' if agg_passed else 'FAIL'})")

        results.append(EvalResult(
            metric_id="EX34.nla_sae_convergence",
            value=agg_overlap,
            n_samples=sum(r.n_samples for r in results),
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": "aggregate",
                "mean_token_overlap": agg_overlap,
                "n_tasks": len(all_overlaps),
                "per_task_overlaps": {
                    r.metadata["task"]: r.metadata["mean_token_overlap"]
                    for r in results if r.metadata.get("task") != "aggregate"
                },
                "passed": agg_passed,
                "threshold": TOKEN_OVERLAP_THRESHOLD,
            },
        ))

    return results


def main():
    parser = parse_common_args("EX34: NLA-SAE Convergent Validity")
    parser.add_argument("--n-features", type=int, default=20,
                        help="Number of feature directions (default: 20)")
    parser.add_argument("--top-k", type=int, default=20,
                        help="Top-k tokens per method (default: 20)")
    parser.add_argument("--hook-layer", type=int, default=None,
                        help="Layer for hook point (default: middle)")
    args = parser.parse_args()

    model = load_model(args.model, args.device)

    log("=" * 60)
    log("EX34: NLA-SAE CONVERGENT VALIDITY")
    log("=" * 60)

    tasks = args.tasks or CIRCUIT_TASKS
    results = run_nla_sae_convergence(
        model,
        tasks=tasks,
        n_prompts=args.n_prompts,
        n_features=args.n_features,
        top_k=args.top_k,
        hook_layer=args.hook_layer,
    )

    out = args.out or "133_nla_sae_convergence.json"
    save_results(results, out)
    log(f"\nDone. {len(results)} results.")


if __name__ == "__main__":
    main()
