"""RoPE Massive Value Filter (Measurement EX8)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     EX8 — RoPE Massive Value Filter
Categories:     measurement
Validity layer: Measurement (M4 Sensitivity)
Criteria:       EX8 RoPE Massive Value Filter
Establishes:    Whether artifact features are contaminated by RoPE positional
                encoding artifacts (massive values in Q/K) rather than
                representing genuine semantic features
Requires:       Model, artifact adapter
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Implements RoPE massive value detection following Jin et al. (ICML 2025),
"Rope with LLM". RoPE positional encoding creates concentrated massive
values in Q and K (but NOT V) projections across all RoPE-using models
(Llama, Qwen, Gemma, Mistral). These massive values are crucial for
contextual knowledge but are NOT semantic features -- they are positional
artifacts of the rotation-based encoding scheme.

SAEs trained on attention Q/K representations of RoPE models will have
their first few features dominated by these positional artifacts. This
metric detects and flags such features so they can be excluded from
downstream interpretability analysis.

Detection algorithm:
1. Check if the model uses RoPE (positional_embedding_type == "rotary").
   If not, all features are clean -- return 1.0.
2. Construct the RoPE frequency basis vectors for the model's d_head:
   theta_k = rotary_base^(-2k/d) for k = 0..rotary_dim//2-1.
   The first few frequency dimensions (lowest frequency, largest
   wavelength) produce the massive values.
3. For each feature direction in the artifact:
   a. Alignment check: compute cosine similarity with each RoPE
      frequency basis pair (sin/cos). Flag if |cos_sim| > 0.5 with
      any of the first few RoPE dimensions.
   b. Position-dependence check: compute feature activations and
      measure ratio of variance-across-positions to variance-across-
      inputs. Positional features vary more by position than by input.
   c. Magnitude check: compare feature activation magnitude against
      the median. Features capturing massive values have activations
      >10x the median feature magnitude.
4. A feature is flagged as a RoPE positional artifact if it meets ANY
   of the three criteria above.

Pass condition: clean_fraction > 0.9 (less than 10% of features are
positional artifacts).

References:
    - Jin et al. (2025) "Rope with LLM: Compressing Context Through
      Rotational Positional Embeddings", ICML 2025.
      github.com/MingyuJ666/Rope_with_LLM
    - Su et al. (2024) "RoFormer: Enhanced Transformer with Rotary
      Position Embedding"

Usage:
    mechval.run("rope_massive_value_filter", artifact=adapter,
                hook_name="blocks.5.attn.hook_q")
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
    name="RoPE Massive Value Filter",
    paper_ref="Jin et al. ICML 2025",
    paper_cite="Jin et al. 2025, Rope with LLM",
    description=(
        "Detects RoPE positional encoding artifacts: massive values in "
        "Q/K projections that contaminate SAE features with non-semantic "
        "positional information"
    ),
    category="measurement",
    tier="cogsci",
    origin="established",
)

# Thresholds
CLEAN_FRACTION_THRESHOLD = 0.9
ALIGNMENT_THRESHOLD = 0.5
MAGNITUDE_MULTIPLIER = 10.0
N_ROPE_DIMS_TO_CHECK = 8  # Check first 8 frequency dimensions


def _build_rope_frequency_basis(
    d_model: int, rotary_dim: int, rotary_base: float = 10000.0,
) -> torch.Tensor:
    """Build the RoPE frequency basis vectors.

    RoPE applies rotation at frequency theta_k = base^(-2k/d) to pairs
    of dimensions. The first few frequencies (lowest k) have the largest
    wavelength and produce the biggest magnitude contributions.

    Returns:
        (n_freq_pairs, d_model) tensor where each row is a unit vector
        aligned with one of the RoPE sin/cos dimension pairs. We return
        2 * n_check vectors: one for the cosine component and one for the
        sine component of each frequency.
    """
    n_freq = min(rotary_dim // 2, N_ROPE_DIMS_TO_CHECK)
    basis_vectors = torch.zeros(2 * n_freq, d_model)

    for k in range(n_freq):
        # Each frequency uses a pair of dimensions: (2k, 2k+1)
        dim_cos = 2 * k
        dim_sin = 2 * k + 1
        if dim_cos < d_model:
            basis_vectors[2 * k, dim_cos] = 1.0
        if dim_sin < d_model:
            basis_vectors[2 * k + 1, dim_sin] = 1.0

    return basis_vectors


def _check_rope_alignment(
    directions: torch.Tensor,
    rope_basis: torch.Tensor,
    threshold: float,
) -> tuple[list[int], list[dict]]:
    """Check which features align with RoPE frequency dimensions.

    Args:
        directions: (n_features, d_model) feature directions.
        rope_basis: (n_rope_vecs, d_model) RoPE basis vectors.
        threshold: cosine similarity threshold for flagging.

    Returns:
        Tuple of (flagged_indices, flag_details).
    """
    # Normalize both
    dirs_normed = F.normalize(directions.float(), dim=-1)
    basis_normed = F.normalize(rope_basis.float(), dim=-1)

    # (n_features, n_rope_vecs)
    similarities = dirs_normed @ basis_normed.T

    # Flag features where any RoPE dimension has |cos_sim| > threshold
    max_abs_sim, max_dim = similarities.abs().max(dim=-1)
    flagged_mask = max_abs_sim > threshold

    flagged_indices = flagged_mask.nonzero(as_tuple=False).squeeze(-1).tolist()
    if isinstance(flagged_indices, int):
        flagged_indices = [flagged_indices]

    details = []
    for idx in flagged_indices:
        details.append({
            "feature_idx": idx,
            "reason": "rope_alignment",
            "max_cosine_sim": float(max_abs_sim[idx]),
            "aligned_rope_dim": int(max_dim[idx]),
        })

    return flagged_indices, details


def _check_position_dependence(
    feature_acts: torch.Tensor,
) -> tuple[list[int], list[dict]]:
    """Check which features vary more across positions than across inputs.

    For positional features, the activation pattern is determined primarily
    by the token's position rather than its content. We measure the ratio
    of position-axis variance to input-axis variance.

    Args:
        feature_acts: (batch, seq, n_features) activations.

    Returns:
        Tuple of (flagged_indices, flag_details).
    """
    batch, seq, n_features = feature_acts.shape

    if batch < 2 or seq < 2:
        return [], []

    # Variance across positions (averaged over batch): how much does
    # activation change with position for a given input?
    # (batch, n_features) -> mean over batch -> (n_features,)
    var_across_pos = feature_acts.var(dim=1).mean(dim=0)

    # Variance across inputs (averaged over positions): how much does
    # activation change with input content for a given position?
    var_across_inputs = feature_acts.var(dim=0).mean(dim=0)

    # Position-dependence ratio: features dominated by position have
    # var_pos >> var_input
    ratio = var_across_pos / (var_across_inputs + 1e-10)

    # Flag features where position variance dominates (ratio > 1 means
    # position matters more than content)
    flagged_mask = ratio > 1.0
    flagged_indices = flagged_mask.nonzero(as_tuple=False).squeeze(-1).tolist()
    if isinstance(flagged_indices, int):
        flagged_indices = [flagged_indices]

    details = []
    for idx in flagged_indices:
        details.append({
            "feature_idx": idx,
            "reason": "position_dependence",
            "pos_to_input_variance_ratio": float(ratio[idx]),
        })

    return flagged_indices, details


def _check_massive_magnitude(
    feature_acts: torch.Tensor,
    multiplier: float,
) -> tuple[list[int], list[dict]]:
    """Check which features have abnormally large activation magnitudes.

    RoPE massive values produce activations 10-100x larger than typical
    features. We flag features whose mean absolute activation exceeds
    multiplier * median feature magnitude.

    Args:
        feature_acts: (batch, seq, n_features) or (n_positions, n_features).
        multiplier: threshold multiplier over median magnitude.

    Returns:
        Tuple of (flagged_indices, flag_details).
    """
    # Flatten to (n_positions, n_features) if needed
    if feature_acts.dim() == 3:
        acts_flat = feature_acts.reshape(-1, feature_acts.shape[-1])
    else:
        acts_flat = feature_acts

    # Mean absolute activation per feature
    mean_abs = acts_flat.abs().mean(dim=0)  # (n_features,)
    median_mag = float(mean_abs.median())

    if median_mag < 1e-10:
        return [], []

    threshold = multiplier * median_mag
    flagged_mask = mean_abs > threshold
    flagged_indices = flagged_mask.nonzero(as_tuple=False).squeeze(-1).tolist()
    if isinstance(flagged_indices, int):
        flagged_indices = [flagged_indices]

    details = []
    for idx in flagged_indices:
        details.append({
            "feature_idx": idx,
            "reason": "massive_magnitude",
            "mean_abs_activation": float(mean_abs[idx]),
            "median_magnitude": median_mag,
            "magnitude_ratio": float(mean_abs[idx] / median_mag),
        })

    return flagged_indices, details


def _model_uses_rope(model) -> bool:
    """Check whether the model uses rotary positional embeddings."""
    cfg = model.cfg
    return getattr(cfg, "positional_embedding_type", "standard") == "rotary"


def _get_rope_params(model) -> tuple[int, int, float]:
    """Extract RoPE-relevant config from the model.

    Returns:
        (d_model, rotary_dim, rotary_base)
    """
    cfg = model.cfg
    d_model = cfg.d_model
    rotary_dim = getattr(cfg, "rotary_dim", None) or cfg.d_head
    rotary_base = getattr(cfg, "rotary_base", 10000)
    return d_model, rotary_dim, float(rotary_base)


@torch.no_grad()
def run_rope_massive_value(
    model,
    artifact=None,
    hook_name: str | None = None,
    n_tokens: int = 200,
) -> list[EvalResult]:
    """Detect RoPE positional artifacts in SAE/artifact features.

    Args:
        model: HookedTransformer model.
        artifact: Artifact adapter with directions() and activations() methods.
        hook_name: Hook point for collecting activations.
        n_tokens: Number of tokens to evaluate on.

    Returns:
        List containing a single EvalResult with clean_fraction as value.
    """
    # Check if model uses RoPE
    has_rope = _model_uses_rope(model)

    if not has_rope:
        log("  Model does not use RoPE -- all features are clean by default")
        return [EvalResult(
            metric_id="EX8.rope_massive_value_filter",
            value=1.0,
            n_samples=0,
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "has_rope": False,
                "n_flagged": 0,
                "flagged_indices": [],
                "flag_reasons": {},
                "clean_fraction": 1.0,
                "passed": True,
                "threshold": CLEAN_FRACTION_THRESHOLD,
                "note": "model does not use RoPE",
            },
        )]

    if artifact is None:
        log("  WARNING: no artifact adapter provided, skipping RoPE filter")
        return []

    if hook_name is None:
        hook_name = "blocks.0.attn.hook_q"
        log(f"  No hook_name specified, defaulting to {hook_name}")

    log(f"  Running RoPE massive value filter at {hook_name}...")

    # Get feature directions
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
    d_dir = directions.shape[1]
    log(f"    {n_features} features, d_direction={d_dir}")

    # Build RoPE frequency basis
    d_model, rotary_dim, rotary_base = _get_rope_params(model)
    rope_basis = _build_rope_frequency_basis(d_dir, rotary_dim, rotary_base)
    rope_basis = rope_basis.to(directions.device)

    # Check 1: alignment with RoPE frequency dimensions
    aligned_indices, aligned_details = _check_rope_alignment(
        directions, rope_basis, ALIGNMENT_THRESHOLD,
    )
    log(f"    Alignment check: {len(aligned_indices)} flagged")

    # Collect activations for checks 2 and 3
    sample_text = "The quick brown fox jumps over the lazy dog. " * (n_tokens // 10)
    tokens = model.to_tokens(sample_text)
    tokens = tokens[:, :min(tokens.shape[1], n_tokens)]

    feature_acts = artifact.activations(model, tokens, hook_name)

    position_indices: list[int] = []
    position_details: list[dict] = []
    magnitude_indices: list[int] = []
    magnitude_details: list[dict] = []

    if feature_acts is not None and feature_acts.numel() > 0:
        feature_acts = feature_acts.float()

        # Check 2: position dependence
        if feature_acts.dim() == 3 and feature_acts.shape[0] >= 2:
            position_indices, position_details = _check_position_dependence(
                feature_acts,
            )
            log(f"    Position-dependence check: {len(position_indices)} flagged")
        else:
            log("    Position-dependence check: skipped (need batch >= 2)")

        # Check 3: massive magnitude
        magnitude_indices, magnitude_details = _check_massive_magnitude(
            feature_acts, MAGNITUDE_MULTIPLIER,
        )
        log(f"    Magnitude check: {len(magnitude_indices)} flagged")
    else:
        log("    WARNING: no activations available, skipping checks 2 and 3")

    # Combine all flagged features (union)
    all_flagged = set(aligned_indices) | set(position_indices) | set(magnitude_indices)
    n_flagged = len(all_flagged)
    clean_fraction = 1.0 - (n_flagged / n_features) if n_features > 0 else 1.0

    # Build per-feature reason map
    flag_reasons: dict[int, list[str]] = {}
    for detail in aligned_details:
        idx = detail["feature_idx"]
        flag_reasons.setdefault(idx, []).append("rope_alignment")
    for detail in position_details:
        idx = detail["feature_idx"]
        flag_reasons.setdefault(idx, []).append("position_dependence")
    for detail in magnitude_details:
        idx = detail["feature_idx"]
        flag_reasons.setdefault(idx, []).append("massive_magnitude")

    passed = clean_fraction > CLEAN_FRACTION_THRESHOLD

    log(f"    Total flagged: {n_flagged}/{n_features}")
    log(f"    Clean fraction: {clean_fraction:.4f}")
    log(f"    [{'PASS' if passed else 'FAIL'}]")

    n_samples = int(feature_acts.shape[0] * feature_acts.shape[1]) if (
        feature_acts is not None and feature_acts.dim() >= 2
    ) else 0

    return [EvalResult(
        metric_id="EX8.rope_massive_value_filter",
        value=clean_fraction,
        n_samples=n_samples,
        instrument_info=INSTRUMENT_INFO,
        metadata={
            "has_rope": True,
            "n_features": n_features,
            "n_flagged": n_flagged,
            "flagged_indices": sorted(all_flagged),
            "flag_reasons": {str(k): v for k, v in flag_reasons.items()},
            "n_flagged_by_alignment": len(aligned_indices),
            "n_flagged_by_position": len(position_indices),
            "n_flagged_by_magnitude": len(magnitude_indices),
            "alignment_details": aligned_details[:20],
            "magnitude_details": magnitude_details[:20],
            "clean_fraction": clean_fraction,
            "passed": passed,
            "threshold": CLEAN_FRACTION_THRESHOLD,
            "alignment_threshold": ALIGNMENT_THRESHOLD,
            "magnitude_multiplier": MAGNITUDE_MULTIPLIER,
            "n_rope_dims_checked": min(
                (getattr(model.cfg, "rotary_dim", None) or model.cfg.d_head) // 2,
                N_ROPE_DIMS_TO_CHECK,
            ),
            "hook_name": hook_name,
        },
    )]


def main():
    parser = parse_common_args("EX8: RoPE Massive Value Filter")
    parser.add_argument("--hook", default=None, help="Hook point for activations")
    parser.add_argument("--n-tokens", type=int, default=200, help="Tokens to evaluate")
    parser.add_argument("--artifact-path", default=None, help="Artifact release ID")
    parser.add_argument("--sae-id", default=None, help="SAE ID")
    args = parser.parse_args()

    model = load_model(args.model, args.device)

    artifact = None
    if args.artifact_path:
        from mechval.lib.artifacts import SAEAdapter
        artifact = SAEAdapter.from_pretrained(
            release=args.artifact_path,
            sae_id=args.sae_id or "",
            hook_point=args.hook or "blocks.0.attn.hook_q",
        )

    log("=" * 60)
    log("EX8: ROPE MASSIVE VALUE FILTER")
    log("=" * 60)

    results = run_rope_massive_value(
        model, artifact=artifact, hook_name=args.hook,
        n_tokens=args.n_tokens,
    )

    out = args.out or "EX8_rope_massive_value.json"
    save_results(results, out)
    log(f"\nDone. {len(results)} results.")


if __name__ == "__main__":
    main()
