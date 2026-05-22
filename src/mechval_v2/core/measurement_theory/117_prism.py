"""PRISM Polysemanticity Score (Measurement M10)
Paper: Kopf, Feldhus, Bykov, Bommer, Hedstrom, Hohne, Eberle (2025). NeurIPS 2025. arXiv:2506.15538
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     M10 — PRISM Polysemanticity
Categories:     measurement
Validity layer: Measurement
Criteria:       M6 Artifact Quality, E1 Predictive Validity
Establishes:    What fraction of SAE features are polysemantic, i.e.,
                activate on multiple semantically distinct clusters of
                contexts. Standard autointerp pipelines are architecturally
                incapable of reliably describing polysemantic features.
Requires:       An artifact adapter with activations() and directions()
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Based on Kopf, Feldhus, Bykov, Bommer, Hedstrom, Hohne, Eberle;
NeurIPS 2025 (arXiv:2506.15538).

PRISM clusters a feature's top-activating examples by semantic
similarity using embedding cosine similarity. If multiple distinct
clusters form (>1 semantic group), the feature is polysemantic. The
score measures the fraction of sampled features that are polysemantic.

Method:
    1. Collect feature activations across prompts via the artifact adapter.
    2. For each sampled feature, find the top-activating contexts.
    3. Embed those contexts using the model's residual stream (or a
       simple bag-of-embeddings).
    4. Compute pairwise cosine similarity among context embeddings.
    5. Apply agglomerative clustering with a distance threshold.
    6. A feature is polysemantic if it has >1 cluster.
    7. polysemanticity_rate = fraction of sampled features that are
       polysemantic.

Pass condition: Report only (diagnostic). Trivial pass: value >= 0.

Usage:
    uv run python 117_prism.py --artifact-path <release> --sae-id <id>
    uv run python 117_prism.py --device cpu
"""

import numpy as np
import torch
from scipy.cluster.hierarchy import fcluster, linkage

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    InstrumentInfo,
    generate_prompts,
    load_model,
    log,
    parse_common_args,
    save_incremental,
    save_results,
)

INSTRUMENT_INFO = InstrumentInfo(
    name="PRISM Polysemanticity Score (Kopf et al. 2025)",
    paper_ref="Kopf et al., NeurIPS 2025 (arXiv:2506.15538)",
    paper_cite="Kopf et al. 2025, PRISM: Polysemanticity via Semantic Clustering",
    description=(
        "Clusters a feature's top-activating contexts by semantic similarity. "
        "If multiple distinct clusters form, the feature is polysemantic. "
        "Reports the fraction of sampled features that are polysemantic."
    ),
    category="measurement",
    tier="measurement_theory",
    origin="established",
)


def _embed_contexts(
    model,
    context_texts: list[str],
) -> np.ndarray:
    """Embed context strings via mean-pooled residual stream embeddings.

    Uses the model's token embeddings (W_E) as a lightweight,
    training-free embedding. Each context is tokenized, looked up in
    W_E, and mean-pooled to produce a single vector.

    Args:
        model: HookedTransformer.
        context_texts: List of text strings.

    Returns:
        (n_contexts, d_model) numpy array of L2-normalized embeddings.
    """
    embeddings = []
    for text in context_texts:
        tokens = model.to_tokens(text, prepend_bos=False)
        with torch.no_grad():
            tok_embeds = model.embed(tokens)  # (1, seq, d_model)
            mean_embed = tok_embeds[0].mean(dim=0)  # (d_model,)
            norm = mean_embed.norm().clamp(min=1e-8)
            embeddings.append((mean_embed / norm).cpu().numpy())
    return np.stack(embeddings)


def _count_semantic_clusters(
    embeddings: np.ndarray,
    threshold: float,
) -> int:
    """Count distinct semantic clusters among context embeddings.

    Uses agglomerative clustering with average linkage and cosine
    distance. The threshold controls how similar contexts must be to
    belong to the same cluster.

    Args:
        embeddings: (n, d) L2-normalized embedding vectors.
        threshold: Cosine distance threshold for cluster merging.
            Lower = stricter (more clusters). The cosine distance is
            1 - cosine_similarity, so threshold=0.5 means clusters
            merge if their average cosine similarity > 0.5.

    Returns:
        Number of clusters.
    """
    n = embeddings.shape[0]
    if n <= 1:
        return 1

    # Cosine distance = 1 - cosine_similarity for normalized vectors
    cos_sim = embeddings @ embeddings.T
    cos_dist = 1.0 - cos_sim
    np.fill_diagonal(cos_dist, 0.0)
    cos_dist = np.clip(cos_dist, 0.0, 2.0)

    # Convert to condensed distance matrix for scipy
    condensed = cos_dist[np.triu_indices(n, k=1)]

    Z = linkage(condensed, method="average")
    labels = fcluster(Z, t=threshold, criterion="distance")
    return int(labels.max())


@torch.no_grad()
def run_prism_polysemanticity(
    model,
    tasks: list[str] | None = None,
    n_prompts: int = 100,
    artifact=None,
    hook_name: str | None = None,
    n_features: int = 100,
    cluster_threshold: float = 0.5,
) -> list[EvalResult]:
    """Run PRISM polysemanticity analysis on an artifact's features.

    For each sampled feature, collects top-activating contexts, embeds
    them, and clusters by semantic similarity. Features with >1 cluster
    are polysemantic.

    Args:
        model: HookedTransformer model.
        tasks: List of task names for prompt generation. If None, uses
            CIRCUIT_TASKS.
        n_prompts: Number of prompts to generate per task.
        artifact: ArtifactAdapter with activations() method.
        hook_name: Hook point override (defaults to artifact's hook).
        n_features: Number of features to sample for polysemanticity
            analysis.
        cluster_threshold: Cosine distance threshold for cluster
            merging. Lower = stricter (more polysemantic detections).

    Returns:
        List of EvalResult with polysemanticity_rate scores.
    """
    if tasks is None:
        tasks = CIRCUIT_TASKS

    tokenizer = model.tokenizer
    effective_hook = hook_name
    if artifact is not None and not effective_hook:
        effective_hook = getattr(artifact.manifest, "hook_point", None)
    if not effective_hook:
        effective_hook = "blocks.0.hook_resid_pre"

    # Collect all prompts across tasks
    all_prompts = []
    for task in tasks:
        prompts = generate_prompts(task, tokenizer, n_prompts)
        all_prompts.extend(prompts)
    if not all_prompts:
        log("  WARNING: no prompts generated, skipping PRISM")
        return []

    log(f"  Collected {len(all_prompts)} prompts across {len(tasks)} tasks")

    # Collect feature activations across all prompts
    # feature_acts_by_prompt[i] = (n_features_total,) max activation per feature
    # context_texts[i] = prompt text
    context_texts = []
    feature_max_acts = []

    for prompt in all_prompts:
        text = prompt.text
        tokens = model.to_tokens(text)

        if artifact is not None:
            acts = artifact.activations(model, tokens, effective_hook)
        else:
            # Fallback: use raw residual stream activations projected
            # onto a random subspace (useful for testing without an SAE)
            _, cache = model.run_with_cache(
                tokens, names_filter=[effective_hook]
            )
            acts = cache[effective_hook]

        # Max activation across sequence positions: (n_features,)
        max_acts = acts[0].abs().max(dim=0).values
        feature_max_acts.append(max_acts.cpu())
        context_texts.append(text)

    if not feature_max_acts:
        log("  WARNING: no activations collected")
        return []

    # Stack: (n_prompts_total, n_features_total)
    all_max_acts = torch.stack(feature_max_acts)
    n_features_total = all_max_acts.shape[1]

    # Sample features: pick the top-n_features by mean activation
    # (active features are more interesting than dead ones)
    mean_acts = all_max_acts.mean(dim=0)
    n_sample = min(n_features, n_features_total)
    sampled_feature_ids = mean_acts.topk(n_sample).indices.tolist()

    log(f"  Sampling {n_sample} features (of {n_features_total} total)")

    # For each sampled feature, find top-activating contexts and cluster
    top_k_contexts = min(20, len(context_texts))
    polysemantic_count = 0
    per_feature_results = []

    for feat_idx in sampled_feature_ids:
        feat_acts = all_max_acts[:, feat_idx]

        # Skip dead features (zero activation everywhere)
        if feat_acts.max().item() < 1e-8:
            per_feature_results.append({
                "feature_idx": feat_idx,
                "n_clusters": 0,
                "polysemantic": False,
                "dead": True,
            })
            continue

        # Top-k activating contexts for this feature
        topk_vals, topk_idxs = feat_acts.topk(top_k_contexts)
        top_texts = [context_texts[i] for i in topk_idxs.tolist()]

        # Embed top contexts
        embeddings = _embed_contexts(model, top_texts)

        # Cluster
        n_clusters = _count_semantic_clusters(embeddings, cluster_threshold)
        is_poly = n_clusters > 1

        if is_poly:
            polysemantic_count += 1

        per_feature_results.append({
            "feature_idx": feat_idx,
            "n_clusters": n_clusters,
            "polysemantic": is_poly,
            "dead": False,
            "top_activation": topk_vals[0].item(),
        })

    n_alive = sum(1 for r in per_feature_results if not r.get("dead", False))
    if n_alive == 0:
        polysemanticity_rate = 0.0
    else:
        polysemanticity_rate = polysemantic_count / n_alive

    n_clusters_list = [
        r["n_clusters"] for r in per_feature_results if not r.get("dead", False)
    ]
    mean_clusters = float(np.mean(n_clusters_list)) if n_clusters_list else 0.0

    log(f"  polysemanticity_rate={polysemanticity_rate:.4f}  "
        f"({polysemantic_count}/{n_alive} polysemantic features)  "
        f"mean_clusters={mean_clusters:.2f}")

    # Trivial pass: value >= 0 (diagnostic metric, always passes)
    passed = polysemanticity_rate >= 0

    results = [EvalResult(
        metric_id="M10.prism_polysemanticity",
        value=polysemanticity_rate,
        n_samples=n_alive,
        instrument_info=INSTRUMENT_INFO,
        metadata={
            "polysemanticity_rate": polysemanticity_rate,
            "polysemantic_count": polysemantic_count,
            "n_features_sampled": n_sample,
            "n_features_alive": n_alive,
            "n_features_dead": n_sample - n_alive,
            "n_features_total": n_features_total,
            "mean_clusters": mean_clusters,
            "cluster_threshold": cluster_threshold,
            "n_prompts_total": len(context_texts),
            "top_k_contexts": top_k_contexts,
            "hook_name": effective_hook,
            "tasks": tasks,
            "passed": passed,
            "per_feature": per_feature_results[:20],  # truncate for JSON
        },
    )]

    return results


def main():
    parser = parse_common_args("M10: PRISM Polysemanticity Score")
    parser.add_argument("--hook", default=None,
                        help="Hook point for feature activations")
    parser.add_argument("--artifact-type", default="sae",
                        choices=["sae", "transcoder", "crosscoder", "factor_bank"],
                        help="Artifact adapter type")
    parser.add_argument("--artifact-path", default=None,
                        help="Path or release ID for artifact")
    parser.add_argument("--sae-id", default=None,
                        help="SAE ID (for SAELens artifacts)")
    parser.add_argument("--n-features", type=int, default=100,
                        help="Number of features to sample")
    parser.add_argument("--cluster-threshold", type=float, default=0.5,
                        help="Cosine distance threshold for cluster merging")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    artifact = None
    if args.artifact_path:
        artifact = _load_artifact(
            args.artifact_type, args.artifact_path,
            args.sae_id, args.hook,
        )

    log("=" * 60)
    log("M10: PRISM POLYSEMANTICITY SCORE")
    log("=" * 60)

    out = args.out or "117_prism.json"
    jsonl_out = out.replace(".json", ".jsonl")

    results = run_prism_polysemanticity(
        model, tasks,
        n_prompts=args.n_prompts,
        artifact=artifact,
        hook_name=args.hook,
        n_features=args.n_features,
        cluster_threshold=args.cluster_threshold,
    )

    for r in results:
        save_incremental(r, jsonl_out)
        p = "PASS" if r.metadata.get("passed", False) else "FAIL"
        log(f"  polysemanticity_rate={r.value:.4f}  [{p}]")

    save_results(results, out, args=args)
    log(f"\nDone. {len(results)} results.")


def _load_artifact(artifact_type: str, artifact_path: str | None,
                   sae_id: str | None, hook: str | None):
    if artifact_path is None:
        return None

    if artifact_type == "sae":
        from mechval.lib.artifacts import SAEAdapter
        return SAEAdapter.from_pretrained(
            release=artifact_path,
            sae_id=sae_id or "",
            hook_point=hook or "",
        )
    elif artifact_type == "transcoder":
        from mechval.lib.artifacts import TranscoderAdapter
        return TranscoderAdapter.from_pretrained(
            release=artifact_path,
            sae_id=sae_id or "",
            hook_point=hook or "",
        )
    else:
        log(f"  WARNING: unsupported artifact type {artifact_type}")
        return None


if __name__ == "__main__":
    main()
