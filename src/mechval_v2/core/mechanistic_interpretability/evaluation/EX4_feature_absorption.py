"""Feature Absorption Detection (Measurement EX4)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     EX4 — Feature Absorption
Categories:     measurement
Validity layer: Measurement (M4 Sensitivity)
Criteria:       EX4 Feature Absorption
Establishes:    Whether artifact features suffer from absorption — a pathology
                where a parent feature fails to fire because a more specific
                child feature absorbs its activation
Requires:       Model, artifact adapter
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Implements feature absorption detection following Klindt, Bloom et al.
(NeurIPS 2025 Oral). Feature absorption is an SAE pathology where a
monosemantic feature fails to fire on inputs where it should because a
more specific (child) feature absorbs the signal. Example: an "A" feature
fails to fire on "Attention" because an "Attention"-concept feature
captures the activation instead.

Detection algorithm:
1. Get decoder directions from the artifact.
2. For each feature, find related features via cosine similarity of
   decoder directions (above a configurable threshold).
3. Compute feature activations on a corpus of tokens.
4. For each (parent, child) pair among related features:
   - Find positions where the child fires but the parent does not.
   - Compare against positions where both fire or neither fires.
   - Absorption score = fraction of the child's active positions where
     the parent is inactive, normalized by how often the parent fires
     elsewhere (to avoid penalizing genuinely unrelated pairs).
5. Per-feature absorption = max absorption score across all children.
6. Aggregates: absorption_rate (mean across features),
   affected_feature_fraction (fraction above threshold).

Pass condition: absorption_rate < 0.15 (lower is better — less absorption).

References:
    - Klindt, Bloom et al. (2025) "Feature Absorption in Sparse
      Autoencoders", NeurIPS 2025 Oral.
    - Templeton et al. (2024) "Scaling Monosemanticity"

Usage:
    mechval.run("feature_absorption", artifact=adapter, hook_name="blocks.5.hook_resid_pre")
"""

import numpy as np
import torch
import torch.nn.functional as F

from mechval.metrics.common import (
    EvalResult,
    InstrumentInfo,
    load_model,
    log,
    parse_common_args,
    save_results,
)

INSTRUMENT_INFO = InstrumentInfo(
    name="Feature Absorption Detection",
    paper_ref="Klindt, Bloom et al. NeurIPS 2025 Oral",
    paper_cite="Klindt, Bloom et al. 2025, Feature Absorption in Sparse Autoencoders",
    description=(
        "Detects feature absorption: a pathology where a parent feature "
        "fails to fire because a child feature absorbs its activation"
    ),
    category="measurement",
    tier="cogsci",
    origin="established",
)

ABSORPTION_THRESHOLD = 0.15
SIMILARITY_THRESHOLD = 0.3
ACTIVATION_THRESHOLD = 0.01
AFFECTED_THRESHOLD = 0.1


def _cosine_similarity_matrix(directions: torch.Tensor) -> torch.Tensor:
    """Pairwise cosine similarity between decoder directions.

    Args:
        directions: (n_features, d_model) decoder direction matrix.

    Returns:
        (n_features, n_features) cosine similarity matrix.
    """
    normed = F.normalize(directions, dim=-1)
    return normed @ normed.T


def _find_related_pairs(
    sim_matrix: torch.Tensor, threshold: float,
) -> list[tuple[int, int]]:
    """Find pairs of features with cosine similarity above threshold.

    Returns pairs (i, j) with i < j and sim(i, j) > threshold.
    """
    n = sim_matrix.shape[0]
    # Zero the diagonal and lower triangle
    mask = torch.triu(torch.ones(n, n, device=sim_matrix.device, dtype=torch.bool), diagonal=1)
    above = (sim_matrix.abs() > threshold) & mask
    indices = above.nonzero(as_tuple=False)
    return [(int(row[0]), int(row[1])) for row in indices]


def _compute_absorption_scores(
    feature_acts: torch.Tensor,
    related_pairs: list[tuple[int, int]],
    act_threshold: float,
) -> dict[int, float]:
    """Compute per-feature absorption scores.

    For each related pair, check if one feature's activation is suppressed
    when the other is active. The absorption score for a potential parent
    feature is the maximum suppression ratio across all its children.

    Args:
        feature_acts: (n_positions, n_features) flattened activations.
        related_pairs: list of (i, j) feature index pairs.
        act_threshold: activation threshold for "firing".

    Returns:
        Dict mapping feature index to its absorption score (0 to 1).
    """
    per_feature_scores: dict[int, float] = {}

    for i, j in related_pairs:
        acts_i = feature_acts[:, i]
        acts_j = feature_acts[:, j]

        active_i = acts_i.abs() > act_threshold
        active_j = acts_j.abs() > act_threshold

        # Check if j absorbs i: j fires but i doesn't
        j_active_i_not = active_j & ~active_i
        # Check if i absorbs j: i fires but j doesn't
        i_active_j_not = active_i & ~active_j

        n_j_active = int(active_j.sum())
        n_i_active = int(active_i.sum())

        # Absorption of i by j: fraction of j's firings where i is silent
        if n_j_active > 0:
            # Only count as absorption if i fires at least sometimes
            # (otherwise they're just unrelated despite similar directions)
            i_base_rate = float(active_i.float().mean())
            if i_base_rate > 0.001:
                suppression_rate = float(j_active_i_not.sum()) / n_j_active
                # Normalize: suppression_rate - (1 - base_rate) gives excess suppression
                # A feature that fires 10% of the time is expected to be absent 90%
                expected_absent = 1.0 - i_base_rate
                excess = max(0.0, suppression_rate - expected_absent)
                # Scale to [0, 1]: divide by maximum possible excess
                absorption_ij = excess / (i_base_rate + 1e-10)
                absorption_ij = min(absorption_ij, 1.0)
                per_feature_scores[i] = max(per_feature_scores.get(i, 0.0), absorption_ij)

        # Absorption of j by i: fraction of i's firings where j is silent
        if n_i_active > 0:
            j_base_rate = float(active_j.float().mean())
            if j_base_rate > 0.001:
                suppression_rate = float(i_active_j_not.sum()) / n_i_active
                expected_absent = 1.0 - j_base_rate
                excess = max(0.0, suppression_rate - expected_absent)
                absorption_ji = excess / (j_base_rate + 1e-10)
                absorption_ji = min(absorption_ji, 1.0)
                per_feature_scores[j] = max(per_feature_scores.get(j, 0.0), absorption_ji)

    return per_feature_scores


@torch.no_grad()
def run_feature_absorption(
    model,
    artifact=None,
    hook_name: str = "blocks.5.hook_resid_pre",
    n_tokens: int = 2048,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    activation_threshold: float = ACTIVATION_THRESHOLD,
    affected_threshold: float = AFFECTED_THRESHOLD,
) -> list[EvalResult]:
    if artifact is None:
        log("  WARNING: no artifact adapter provided, skipping feature absorption")
        return []

    log(f"  Computing feature absorption at {hook_name}...")

    # Step 1: get decoder directions
    try:
        directions = artifact.directions()
    except (NotImplementedError, TypeError):
        log("  WARNING: artifact does not expose directions(), skipping")
        return []

    if directions is None or directions.numel() == 0:
        log("  WARNING: empty directions from artifact, skipping")
        return []

    directions = directions.float()
    n_features = directions.shape[0]
    log(f"    {n_features} features, d_model={directions.shape[1]}")

    # Step 2: find related pairs via cosine similarity
    sim_matrix = _cosine_similarity_matrix(directions)
    related_pairs = _find_related_pairs(sim_matrix, similarity_threshold)
    log(f"    {len(related_pairs)} related pairs (sim > {similarity_threshold})")

    if not related_pairs:
        log("    No related pairs found; absorption_rate = 0")
        return [EvalResult(
            metric_id="EX4.feature_absorption",
            value=0.0,
            n_samples=0,
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "hook_name": hook_name,
                "n_features": n_features,
                "n_related_pairs": 0,
                "absorption_rate": 0.0,
                "affected_feature_fraction": 0.0,
                "worst_absorbed_features": [],
                "passed": True,
                "threshold": ABSORPTION_THRESHOLD,
                "similarity_threshold": similarity_threshold,
            },
        )]

    # Step 3: compute activations
    sample_text = "The quick brown fox jumps over the lazy dog. " * (n_tokens // 10)
    tokens = model.to_tokens(sample_text)
    tokens = tokens[:, :min(tokens.shape[1], n_tokens)]

    feature_acts = artifact.activations(model, tokens, hook_name)
    # Flatten batch and seq dims: (batch, seq, n_features) -> (n_positions, n_features)
    feature_acts = feature_acts.reshape(-1, feature_acts.shape[-1]).float()
    n_positions = feature_acts.shape[0]
    log(f"    {n_positions} positions, computing absorption scores...")

    # Step 4: compute per-feature absorption scores
    absorption_scores = _compute_absorption_scores(
        feature_acts, related_pairs, activation_threshold,
    )

    # Step 5: aggregate
    if absorption_scores:
        all_scores = np.array(list(absorption_scores.values()))
        absorption_rate = float(all_scores.mean())
        affected_count = int((all_scores > affected_threshold).sum())
        affected_fraction = affected_count / n_features

        # Top worst-absorbed features
        sorted_features = sorted(absorption_scores.items(), key=lambda x: x[1], reverse=True)
        worst_absorbed = [
            {"feature_idx": idx, "absorption_score": float(score)}
            for idx, score in sorted_features[:10]
        ]
    else:
        absorption_rate = 0.0
        affected_fraction = 0.0
        worst_absorbed = []

    passed = absorption_rate < ABSORPTION_THRESHOLD

    log(f"    absorption_rate={absorption_rate:.4f}")
    log(f"    affected_fraction={affected_fraction:.4f} "
        f"({int(affected_fraction * n_features)}/{n_features})")
    log(f"    [{'PASS' if passed else 'FAIL'}]")

    return [EvalResult(
        metric_id="EX4.feature_absorption",
        value=absorption_rate,
        n_samples=n_positions,
        instrument_info=INSTRUMENT_INFO,
        metadata={
            "hook_name": hook_name,
            "n_features": n_features,
            "n_related_pairs": len(related_pairs),
            "n_features_with_scores": len(absorption_scores),
            "absorption_rate": absorption_rate,
            "affected_feature_fraction": affected_fraction,
            "worst_absorbed_features": worst_absorbed,
            "passed": passed,
            "threshold": ABSORPTION_THRESHOLD,
            "similarity_threshold": similarity_threshold,
            "activation_threshold": activation_threshold,
            "affected_threshold": affected_threshold,
        },
    )]


def main():
    parser = parse_common_args("EX4: Feature Absorption Detection")
    parser.add_argument("--hook", default="blocks.5.hook_resid_pre", help="Hook point")
    parser.add_argument("--n-tokens", type=int, default=2048, help="Tokens to evaluate")
    parser.add_argument("--artifact-path", default=None, help="Artifact release ID")
    parser.add_argument("--sae-id", default=None, help="SAE ID")
    parser.add_argument("--similarity-threshold", type=float, default=SIMILARITY_THRESHOLD,
                        help="Cosine similarity threshold for related pairs")
    args = parser.parse_args()

    model = load_model(args.model, args.device)

    artifact = None
    if args.artifact_path:
        from mechval.lib.artifacts import SAEAdapter
        artifact = SAEAdapter.from_pretrained(
            release=args.artifact_path,
            sae_id=args.sae_id or "",
            hook_point=args.hook,
        )

    log("=" * 60)
    log("EX4: FEATURE ABSORPTION DETECTION")
    log("=" * 60)

    results = run_feature_absorption(
        model, artifact=artifact, hook_name=args.hook,
        n_tokens=args.n_tokens,
        similarity_threshold=args.similarity_threshold,
    )

    out = args.out or "EX4_feature_absorption.json"
    save_results(results, out)
    log(f"\nDone. {len(results)} results.")


if __name__ == "__main__":
    main()
