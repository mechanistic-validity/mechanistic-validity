"""NLA Reconstruction Fidelity (Measurement EX5)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     EX5 — NLA Reconstruction Fidelity
Categories:     measurement
Validity layer: Measurement (M4 Sensitivity)
Criteria:       EX5 Roundtrip Reconstruction Fidelity
Establishes:    Whether artifact features are coherent enough that their
                activation pattern survives a compress-decompress cycle
Requires:       Model, artifact adapter with directions() and activations()
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Proxy implementation of the Natural Language Autoencoder (NLA) roundtrip
fidelity metric from Anthropic (transformer-circuits.pub/2026/nla/,
github.com/kitft/natural_language_autoencoders).

NLAs train a model to translate internal activations into natural language
explanations, then reconstruct the activation from that text. The
reconstruction loss is a built-in validity test: features that survive
the roundtrip are coherent enough to be captured in language.

Since calling the actual NLA model requires Claude, this implements a
proxy that tests the same property -- whether a feature's activation
pattern is coherent enough to survive a compress-decompress cycle:

1. For each feature, identify top-k activating token positions (the
   "explanation" -- which tokens this feature responds to).
2. From those positions' residual-stream activations, reconstruct a
   "concept direction" via PCA (first principal component).
3. Compare the reconstructed direction to the original feature direction
   via cosine similarity.

High roundtrip fidelity means the feature captures a coherent pattern
in the data that can be recovered from its top activations alone --
the same property NLA roundtrip tests at a higher level of abstraction.

Pass condition: mean_roundtrip_fidelity > 0.3

References:
    - Anthropic (2026) "Natural Language Autoencoders",
      transformer-circuits.pub/2026/nla/
    - github.com/kitft/natural_language_autoencoders

Usage:
    mechval.run("natural_language_autoencoder", artifact=adapter,
                hook_name="blocks.5.hook_resid_pre")
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
    name="NLA Reconstruction Fidelity",
    paper_ref="Anthropic 2026, Natural Language Autoencoders",
    paper_cite="Anthropic 2026, transformer-circuits.pub/2026/nla/",
    description=(
        "Proxy for NLA roundtrip fidelity: measures whether a feature's "
        "activation pattern is coherent enough that its top-activating "
        "positions reconstruct the original feature direction"
    ),
    category="measurement",
    tier="cogsci",
    origin="established",
)

FIDELITY_THRESHOLD = 0.3
TOP_K_POSITIONS = 20
RANDOM_BASELINE_SAMPLES = 50


def _reconstruct_direction_pca(activations: torch.Tensor) -> torch.Tensor:
    """Extract the first principal component from a set of activation vectors.

    Args:
        activations: (n_positions, d_model) activation matrix.

    Returns:
        (d_model,) unit vector -- the first principal component.
    """
    # Center the activations
    centered = activations - activations.mean(dim=0, keepdim=True)
    # SVD to get first principal component
    _, _, Vt = torch.linalg.svd(centered, full_matrices=False)
    return F.normalize(Vt[0], dim=0)


def _compute_roundtrip_fidelity(
    feature_direction: torch.Tensor,
    top_activations: torch.Tensor,
) -> float:
    """Compute roundtrip fidelity for a single feature.

    Args:
        feature_direction: (d_model,) the feature's decoder direction.
        top_activations: (n_positions, d_model) residual-stream activations
            at the feature's top-activating positions.

    Returns:
        Cosine similarity between original direction and PCA-reconstructed
        direction (absolute value, since sign is arbitrary in PCA).
    """
    if top_activations.shape[0] < 2:
        return 0.0

    reconstructed = _reconstruct_direction_pca(top_activations)
    cos_sim = F.cosine_similarity(
        feature_direction.unsqueeze(0),
        reconstructed.unsqueeze(0),
    )
    # Absolute value: PCA sign is arbitrary
    return float(cos_sim.abs().item())


def _compute_random_baseline(
    all_activations: torch.Tensor,
    n_positions: int,
    n_samples: int,
) -> float:
    """Compute baseline: PCA of random positions vs random direction.

    This measures what roundtrip fidelity you get by chance.
    """
    n_total = all_activations.shape[0]
    d_model = all_activations.shape[1]

    if n_total < n_positions or n_positions < 2:
        return 0.0

    scores = []
    for _ in range(n_samples):
        # Random direction
        rand_dir = F.normalize(torch.randn(d_model, device=all_activations.device), dim=0)
        # Random subset of positions
        indices = torch.randperm(n_total, device=all_activations.device)[:n_positions]
        subset = all_activations[indices]
        reconstructed = _reconstruct_direction_pca(subset)
        cos_sim = float(F.cosine_similarity(
            rand_dir.unsqueeze(0), reconstructed.unsqueeze(0),
        ).abs().item())
        scores.append(cos_sim)

    return float(np.mean(scores))


@torch.no_grad()
def run_nla_reconstruction(
    model,
    artifact=None,
    hook_name: str = "blocks.5.hook_resid_pre",
    n_features: int = 50,
    n_tokens: int = 500,
    top_k: int = TOP_K_POSITIONS,
) -> list[EvalResult]:
    if artifact is None:
        log("  WARNING: no artifact adapter provided, skipping NLA reconstruction")
        return []

    log(f"  Computing NLA reconstruction fidelity at {hook_name}...")

    # Step 1: get feature directions (decoder weights)
    try:
        directions = artifact.directions()
    except (NotImplementedError, TypeError):
        log("  WARNING: artifact does not expose directions(), skipping")
        return []

    if directions is None or directions.numel() == 0:
        log("  WARNING: empty directions from artifact, skipping")
        return []

    directions = directions.float()
    total_features = directions.shape[0]
    d_model = directions.shape[1]
    n_eval = min(n_features, total_features)
    log(f"    {total_features} total features (evaluating {n_eval}), d_model={d_model}")

    # Step 2: get activations on sample text
    sample_text = "The quick brown fox jumps over the lazy dog. " * (n_tokens // 10)
    tokens = model.to_tokens(sample_text)
    tokens = tokens[:, :min(tokens.shape[1], n_tokens)]

    feature_acts = artifact.activations(model, tokens, hook_name)
    # (batch, seq, n_features) -> (n_positions, n_features)
    feature_acts_flat = feature_acts.reshape(-1, feature_acts.shape[-1]).float()
    n_positions = feature_acts_flat.shape[0]

    # Step 3: get residual-stream activations at the hook point
    _, cache = model.run_with_cache(tokens, names_filter=hook_name)
    resid = cache[hook_name]
    # (batch, seq, d_model) -> (n_positions, d_model)
    resid_flat = resid.reshape(-1, d_model).float()

    log(f"    {n_positions} token positions collected")

    # Step 4: for each feature, compute roundtrip fidelity
    per_feature_fidelity = []
    per_feature_details = []

    for feat_idx in range(n_eval):
        feat_direction = F.normalize(directions[feat_idx], dim=0)
        feat_vals = feature_acts_flat[:, feat_idx]

        # Get top-k activating positions
        k = min(top_k, n_positions)
        top_indices = feat_vals.topk(k).indices
        top_resid = resid_flat[top_indices]

        fidelity = _compute_roundtrip_fidelity(feat_direction, top_resid)
        per_feature_fidelity.append(fidelity)

        per_feature_details.append({
            "feature_idx": feat_idx,
            "roundtrip_fidelity": fidelity,
            "max_activation": float(feat_vals.max()),
            "mean_top_activation": float(feat_vals[top_indices].mean()),
        })

    # Step 5: compute random baseline
    random_baseline = _compute_random_baseline(
        resid_flat, top_k, RANDOM_BASELINE_SAMPLES,
    )

    # Step 6: aggregate
    fidelity_array = np.array(per_feature_fidelity)
    mean_fidelity = float(fidelity_array.mean())
    median_fidelity = float(np.median(fidelity_array))
    std_fidelity = float(fidelity_array.std())
    frac_above_threshold = float((fidelity_array > FIDELITY_THRESHOLD).mean())

    passed = mean_fidelity > FIDELITY_THRESHOLD

    log(f"    mean_fidelity={mean_fidelity:.4f} (median={median_fidelity:.4f}, "
        f"std={std_fidelity:.4f})")
    log(f"    random_baseline={random_baseline:.4f}")
    log(f"    frac_above_threshold={frac_above_threshold:.4f}")
    log(f"    [{'PASS' if passed else 'FAIL'}]")

    # Sort for top/bottom features
    sorted_details = sorted(per_feature_details, key=lambda x: x["roundtrip_fidelity"],
                            reverse=True)

    return [EvalResult(
        metric_id="EX5.nla_reconstruction_fidelity",
        value=mean_fidelity,
        baseline_random=random_baseline,
        n_samples=n_eval,
        instrument_info=INSTRUMENT_INFO,
        metadata={
            "hook_name": hook_name,
            "n_features_evaluated": n_eval,
            "n_features_total": total_features,
            "n_positions": n_positions,
            "top_k": top_k,
            "mean_fidelity": mean_fidelity,
            "median_fidelity": median_fidelity,
            "std_fidelity": std_fidelity,
            "frac_above_threshold": frac_above_threshold,
            "random_baseline": random_baseline,
            "lift_over_random": mean_fidelity - random_baseline,
            "passed": passed,
            "threshold": FIDELITY_THRESHOLD,
            "top_features": sorted_details[:10],
            "bottom_features": sorted_details[-5:],
        },
    )]


def main():
    parser = parse_common_args("EX5: NLA Reconstruction Fidelity")
    parser.add_argument("--hook", default="blocks.5.hook_resid_pre", help="Hook point")
    parser.add_argument("--n-features", type=int, default=50,
                        help="Number of features to evaluate")
    parser.add_argument("--n-tokens", type=int, default=500, help="Tokens to evaluate")
    parser.add_argument("--top-k", type=int, default=TOP_K_POSITIONS,
                        help="Top-k positions for reconstruction")
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
            hook_point=args.hook,
        )

    log("=" * 60)
    log("EX5: NLA RECONSTRUCTION FIDELITY")
    log("=" * 60)

    results = run_nla_reconstruction(
        model, artifact=artifact, hook_name=args.hook,
        n_features=args.n_features, n_tokens=args.n_tokens,
        top_k=args.top_k,
    )

    out = args.out or "EX5_nla_reconstruction.json"
    save_results(results, out)
    log(f"\nDone. {len(results)} results.")


if __name__ == "__main__":
    main()
