"""Matryoshka Cross-Scale Consistency (Measurement M11)
Paper: Multiple authors (2025). NeurIPS 2025. arXiv:2503.17547
-----------------------------------------------------
Instrument:     M11 -- Matryoshka Cross-Scale Consistency
Categories:     measurement
Validity layer: Measurement
Criteria:       M1 Reliability, M2 Hyperparameter Sensitivity
Establishes:    Whether features at dictionary width k correspond to coherent
                feature clusters at width 2k -- a measurement consistency check
                across SAE scales
Requires:       Two artifact adapters at different dictionary widths (small and
                large), or Matryoshka SAEs with nested dictionaries
-----------------------------------------------------

Based on arXiv:2503.17547 (NeurIPS 2025): for SAEs at different dictionary
widths (or Matryoshka SAEs with nested dictionaries), measures whether
features at width k correspond to coherent feature clusters at width 2k.

Two failure modes are tracked:
  - Splitting rate: one feature at width k maps to multiple *unrelated*
    features at width 2k (the small feature "splits" incoherently).
  - Absorption rate: multiple features at width k collapse into a single
    feature at width 2k (the large dictionary fails to differentiate).

Method:
    1. Collect activations at a shared hook point from the model.
    2. Encode activations through both artifact adapters (small and large).
    3. Compute per-feature correspondence via activation correlation:
       for each small feature, find the top-k most correlated large features.
    4. Splitting rate: fraction of small features whose top-k large-feature
       correlates have low pairwise cosine similarity (incoherent cluster).
    5. Absorption rate: fraction of large features that are the top match
       for multiple small features (many-to-one collapse).
    6. cross_scale_consistency = 1 - (splitting_rate + absorption_rate) / 2

Pass condition: cross_scale_consistency > 0.7

Usage:
    uv run python 118_matryoshka.py --artifact-small-path <release> --artifact-large-path <release>
    uv run python 118_matryoshka.py --device cpu
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
    save_incremental,
    save_results,
)

INSTRUMENT_INFO = InstrumentInfo(
    name="Matryoshka Cross-Scale Consistency (arXiv:2503.17547)",
    paper_ref="arXiv:2503.17547, NeurIPS 2025",
    paper_cite="arXiv:2503.17547, Matryoshka Cross-Scale Consistency",
    description=(
        "Tests whether features at SAE dictionary width k correspond to "
        "coherent feature clusters at width 2k -- a measurement consistency "
        "check for cross-scale feature stability"
    ),
    category="measurement",
    tier="measurement_theory",
    origin="established",
)

CONSISTENCY_THRESHOLD = 0.7
SPLITTING_COSINE_THRESHOLD = 0.5


def compute_activation_correlation(
    acts_small: torch.Tensor,
    acts_large: torch.Tensor,
) -> torch.Tensor:
    """Correlation between each small feature and each large feature.

    Args:
        acts_small: (n_positions, n_features_small) flattened activations.
        acts_large: (n_positions, n_features_large) flattened activations.

    Returns:
        (n_features_small, n_features_large) correlation matrix.
    """
    # Center and normalize
    small = acts_small.float()
    large = acts_large.float()

    small = small - small.mean(dim=0, keepdim=True)
    large = large - large.mean(dim=0, keepdim=True)

    small_norm = small.norm(dim=0, keepdim=True).clamp(min=1e-8)
    large_norm = large.norm(dim=0, keepdim=True).clamp(min=1e-8)

    small = small / small_norm
    large = large / large_norm

    # (n_features_small, n_features_large)
    return (small.T @ large)


def compute_splitting_rate(
    corr: torch.Tensor,
    dirs_large: torch.Tensor,
    top_k: int = 5,
    cosine_threshold: float = SPLITTING_COSINE_THRESHOLD,
) -> tuple[float, dict]:
    """Fraction of small features that split into incoherent large-feature clusters.

    For each small feature, take its top-k most correlated large features.
    If the pairwise cosine similarity among those top-k directions is low
    (mean < cosine_threshold), the feature has "split" incoherently.

    Args:
        corr: (n_features_small, n_features_large) correlation matrix.
        dirs_large: (n_features_large, d_model) decoder directions of large SAE.
        top_k: Number of top correlated large features to consider.
        cosine_threshold: Below this mean pairwise cosine, a feature is "split".

    Returns:
        (splitting_rate, details_dict)
    """
    n_small = corr.shape[0]
    effective_k = min(top_k, corr.shape[1])

    # Normalize large directions
    dirs_large = dirs_large.float()
    norms = dirs_large.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    dirs_normed = dirs_large / norms

    split_count = 0
    per_feature_coherence = []

    for i in range(n_small):
        _, top_indices = corr[i].abs().topk(effective_k)
        top_dirs = dirs_normed[top_indices]  # (k, d_model)

        # Pairwise cosine similarity among top-k directions
        cos_matrix = top_dirs @ top_dirs.T  # (k, k)

        # Extract upper triangle (exclude diagonal)
        if effective_k > 1:
            mask = torch.triu(torch.ones(effective_k, effective_k, device=cos_matrix.device), diagonal=1).bool()
            pairwise_cos = cos_matrix[mask].abs().mean().item()
        else:
            pairwise_cos = 1.0

        per_feature_coherence.append(pairwise_cos)
        if pairwise_cos < cosine_threshold:
            split_count += 1

    splitting_rate = split_count / max(n_small, 1)
    details = {
        "n_split_features": split_count,
        "n_small_features": n_small,
        "mean_cluster_coherence": float(np.mean(per_feature_coherence)) if per_feature_coherence else 0.0,
    }
    return splitting_rate, details


def compute_absorption_rate(
    corr: torch.Tensor,
) -> tuple[float, dict]:
    """Fraction of large features that absorb multiple small features.

    For each small feature, find its best-matching large feature. A large
    feature that is the best match for more than one small feature has
    "absorbed" them.

    Args:
        corr: (n_features_small, n_features_large) correlation matrix.

    Returns:
        (absorption_rate, details_dict)
    """
    n_small = corr.shape[0]
    n_large = corr.shape[1]

    # Best large feature for each small feature
    best_large = corr.abs().argmax(dim=1)  # (n_small,)

    # Count how many small features map to each large feature
    large_counts = torch.zeros(n_large, device=corr.device)
    for idx in best_large:
        large_counts[idx] += 1

    # A large feature "absorbs" if it is the best match for > 1 small feature
    absorbing = (large_counts > 1).sum().item()
    # Rate is fraction of *active* large features (those matched at least once)
    # that absorbed multiple small features
    active_large = (large_counts > 0).sum().item()
    absorption_rate = absorbing / max(active_large, 1)

    details = {
        "n_absorbing_features": int(absorbing),
        "n_active_large_features": int(active_large),
        "n_large_features": n_large,
        "max_absorption": int(large_counts.max().item()),
    }
    return absorption_rate, details


@torch.no_grad()
def run_matryoshka_consistency(
    model,
    tasks: list[str] | None = None,
    n_prompts: int = 50,
    artifact_small=None,
    artifact_large=None,
    hook_name: str | None = None,
    top_k: int = 20,
) -> list[EvalResult]:
    """Run Matryoshka cross-scale consistency analysis between two SAE widths.

    Args:
        model: HookedTransformer model.
        tasks: List of task names to evaluate on.
        n_prompts: Number of prompts per task.
        artifact_small: Artifact adapter for the smaller dictionary width.
        artifact_large: Artifact adapter for the larger dictionary width.
        hook_name: Hook point override (defaults to artifact's hook point).
        top_k: Number of top correlated features for splitting analysis.

    Returns:
        List of EvalResult with cross_scale_consistency scores.
    """
    if tasks is None:
        tasks = CIRCUIT_TASKS

    if artifact_small is None or artifact_large is None:
        log("  WARNING: two artifact adapters required (small and large width), "
            "skipping matryoshka consistency")
        return []

    tokenizer = model.tokenizer
    results = []

    effective_hook = hook_name or artifact_small.manifest.hook_point
    if not effective_hook:
        effective_hook = "blocks.0.hook_resid_pre"

    # Get decoder directions for coherence analysis
    dirs_small = artifact_small.directions()
    dirs_large = artifact_large.directions()
    if dirs_small.ndim == 3:
        dirs_small = dirs_small.mean(dim=0)
    if dirs_large.ndim == 3:
        dirs_large = dirs_large.mean(dim=0)

    n_features_small = dirs_small.shape[0]
    n_features_large = dirs_large.shape[0]
    log(f"  Small SAE: {n_features_small} features, Large SAE: {n_features_large} features")

    for task in tasks:
        prompts = generate_prompts(task, tokenizer, n_prompts)
        if not prompts:
            log(f"  {task}: no prompts, skipping")
            continue

        correct_ids, incorrect_ids = get_token_ids(prompts, tokenizer)
        if not correct_ids:
            log(f"  {task}: no token IDs, skipping")
            continue

        log(f"  {task}: {len(prompts)} prompts")

        # Collect feature activations from both artifacts, accumulate
        all_acts_small = []
        all_acts_large = []
        n = min(len(prompts), len(correct_ids))

        for i in range(n):
            tokens = model.to_tokens(prompts[i].text)

            acts_s = artifact_small.activations(model, tokens, effective_hook)
            acts_l = artifact_large.activations(model, tokens, effective_hook)

            # Flatten batch and seq dimensions
            all_acts_small.append(acts_s.reshape(-1, acts_s.shape[-1]))
            all_acts_large.append(acts_l.reshape(-1, acts_l.shape[-1]))

            if (i + 1) % 10 == 0:
                log(f"    processed {i+1}/{n} prompts")

        if not all_acts_small:
            continue

        # Concatenate all activations: (total_positions, n_features)
        acts_small_cat = torch.cat(all_acts_small, dim=0)
        acts_large_cat = torch.cat(all_acts_large, dim=0)

        # Compute correlation matrix
        corr = compute_activation_correlation(acts_small_cat, acts_large_cat)

        # Compute splitting and absorption rates
        splitting_rate, split_details = compute_splitting_rate(
            corr, dirs_large, top_k=top_k,
        )
        absorption_rate, absorb_details = compute_absorption_rate(corr)

        cross_scale_consistency = 1.0 - (splitting_rate + absorption_rate) / 2.0
        passed = cross_scale_consistency > CONSISTENCY_THRESHOLD

        log(f"    splitting_rate={splitting_rate:.4f}  "
            f"absorption_rate={absorption_rate:.4f}  "
            f"consistency={cross_scale_consistency:.4f}  "
            f"[{'PASS' if passed else 'FAIL'}]")

        results.append(EvalResult(
            metric_id="M11.matryoshka_consistency",
            value=cross_scale_consistency,
            n_samples=n,
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": task,
                "cross_scale_consistency": cross_scale_consistency,
                "splitting_rate": splitting_rate,
                "absorption_rate": absorption_rate,
                "passed": passed,
                "threshold": CONSISTENCY_THRESHOLD,
                "top_k": top_k,
                "n_features_small": n_features_small,
                "n_features_large": n_features_large,
                "hook_name": effective_hook,
                **{f"split_{k}": v for k, v in split_details.items()},
                **{f"absorb_{k}": v for k, v in absorb_details.items()},
            },
        ))

    return results


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


def main():
    parser = parse_common_args("M11: Matryoshka Cross-Scale Consistency")
    parser.add_argument("--hook", default=None,
                        help="Hook point for artifact activations")
    parser.add_argument("--artifact-small-type", default="sae",
                        choices=["sae", "transcoder", "crosscoder", "factor_bank"],
                        help="Artifact adapter type for small dictionary")
    parser.add_argument("--artifact-small-path", default=None,
                        help="Path or release ID for small dictionary artifact")
    parser.add_argument("--sae-small-id", default=None,
                        help="SAE ID for small dictionary artifact")
    parser.add_argument("--artifact-large-type", default="sae",
                        choices=["sae", "transcoder", "crosscoder", "factor_bank"],
                        help="Artifact adapter type for large dictionary")
    parser.add_argument("--artifact-large-path", default=None,
                        help="Path or release ID for large dictionary artifact")
    parser.add_argument("--sae-large-id", default=None,
                        help="SAE ID for large dictionary artifact")
    parser.add_argument("--top-k", type=int, default=20,
                        help="Number of top correlated features for splitting analysis")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    artifact_small = _load_artifact(args.artifact_small_type, args.artifact_small_path,
                                    args.sae_small_id, args.hook)
    artifact_large = _load_artifact(args.artifact_large_type, args.artifact_large_path,
                                    args.sae_large_id, args.hook)

    log("=" * 60)
    log("M11: MATRYOSHKA CROSS-SCALE CONSISTENCY")
    log("=" * 60)

    out = args.out or "118_matryoshka.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_matryoshka_consistency(
            model, [task],
            n_prompts=args.n_prompts,
            artifact_small=artifact_small,
            artifact_large=artifact_large,
            hook_name=args.hook,
            top_k=args.top_k,
        )
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)
            p = "PASS" if r.metadata["passed"] else "FAIL"
            log(f"  {task}: consistency={r.value:.4f}  [{p}]")

    save_results(results, out, args=args)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
