"""TopK SAE Scaling Metrics (Measurement EX7)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     EX7 — TopK SAE Scaling
Categories:     measurement
Validity layer: Measurement (M4 Sensitivity)
Criteria:       EX7a Feature Recovery, EX7b Downstream Sparsity,
                EX7c Activation Explainability
Establishes:    SAE feature quality via three complementary sub-metrics:
                planted feature recovery, downstream effect sparsity,
                and activation pattern explainability
Requires:       Model, artifact adapter
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Implements three SAE feature quality metrics from Gao, Dupre la Tour
et al. (ICLR 2025 Oral), "Scaling and Evaluating Sparse Autoencoders":

EX7a — Hypothesized Feature Recovery
    Plant synthetic random unit directions, check whether any SAE decoder
    direction achieves high cosine similarity with each planted direction.
    A well-trained SAE should have directions spanning the residual stream
    and thus recover arbitrary planted features.

EX7b — Downstream Effect Sparsity
    Project each decoder direction through the unembedding matrix W_U to
    obtain per-feature logit effects. Measure the L0 sparsity of these
    effects — a good feature should affect few output logits.

EX7c — Activation Explainability
    Proxy autointerp score computed from activation statistics: kurtosis
    (peakedness) and sparsity (fraction of zero activations). Features
    with sharp, sparse activation patterns are more interpretable.

Pass thresholds: recovery > 0.3, sparsity > 0.8, explainability > 0.5.

References:
    - Gao, Dupre la Tour et al. (2025) "Scaling and Evaluating Sparse
      Autoencoders", ICLR 2025 Oral.

Usage:
    mechval.run("topk_scaling", artifact=adapter, hook_name="blocks.5.hook_resid_pre")
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

INSTRUMENT_INFO_RECOVERY = InstrumentInfo(
    name="Hypothesized Feature Recovery",
    paper_ref="Gao, Dupre la Tour et al. ICLR 2025 Oral",
    paper_cite="Gao, Dupre la Tour et al. 2025, Scaling and Evaluating Sparse Autoencoders",
    description=(
        "Tests whether SAE decoder directions recover planted synthetic features "
        "via max cosine similarity"
    ),
    category="measurement",
    tier="cogsci",
    origin="established",
)

INSTRUMENT_INFO_SPARSITY = InstrumentInfo(
    name="Downstream Effect Sparsity",
    paper_ref="Gao, Dupre la Tour et al. ICLR 2025 Oral",
    paper_cite="Gao, Dupre la Tour et al. 2025, Scaling and Evaluating Sparse Autoencoders",
    description=(
        "Measures how sparsely each SAE feature affects model output logits "
        "via decoder-unembedding projection"
    ),
    category="measurement",
    tier="cogsci",
    origin="established",
)

INSTRUMENT_INFO_EXPLAINABILITY = InstrumentInfo(
    name="Activation Explainability",
    paper_ref="Gao, Dupre la Tour et al. ICLR 2025 Oral",
    paper_cite="Gao, Dupre la Tour et al. 2025, Scaling and Evaluating Sparse Autoencoders",
    description=(
        "Proxy autointerp score from activation kurtosis and sparsity — "
        "sharp, sparse activations are more interpretable"
    ),
    category="measurement",
    tier="cogsci",
    origin="established",
)

# Pass thresholds
RECOVERY_THRESHOLD = 0.3
SPARSITY_THRESHOLD = 0.8
EXPLAINABILITY_THRESHOLD = 0.5

# Algorithm parameters
COSINE_RECOVERY_CUTOFF = 0.5
DOWNSTREAM_EFFECT_THRESHOLD = 0.1


def _feature_recovery(
    directions: torch.Tensor,
    n_synthetic: int,
    device: torch.device,
) -> tuple[float, dict]:
    """Plant synthetic features and measure recovery via cosine similarity.

    Args:
        directions: (n_features, d_model) decoder directions from the artifact.
        n_synthetic: number of random synthetic directions to plant.
        device: torch device.

    Returns:
        (recovery_rate, detail_dict) where recovery_rate is the fraction of
        synthetic directions with max cosine sim > COSINE_RECOVERY_CUTOFF.
    """
    d_model = directions.shape[1]

    # Generate random unit directions
    synthetic = torch.randn(n_synthetic, d_model, device=device)
    synthetic = F.normalize(synthetic, dim=-1)

    # Normalize decoder directions
    normed_dirs = F.normalize(directions.to(device).float(), dim=-1)

    # Cosine similarity: (n_synthetic, n_features)
    cosine_sims = synthetic @ normed_dirs.T

    # Max cosine similarity per synthetic direction
    max_cosims, best_feature_idxs = cosine_sims.abs().max(dim=1)
    max_cosims_list = max_cosims.cpu().tolist()

    recovered = sum(1 for c in max_cosims_list if c > COSINE_RECOVERY_CUTOFF)
    recovery_rate = recovered / n_synthetic

    return recovery_rate, {
        "n_synthetic": n_synthetic,
        "n_recovered": recovered,
        "cosine_cutoff": COSINE_RECOVERY_CUTOFF,
        "max_cosims": max_cosims_list,
        "best_feature_idxs": best_feature_idxs.cpu().tolist(),
        "mean_max_cosim": float(np.mean(max_cosims_list)),
        "median_max_cosim": float(np.median(max_cosims_list)),
    }


def _downstream_sparsity(
    directions: torch.Tensor,
    model,
    n_features_to_check: int,
) -> tuple[float, dict]:
    """Measure L0 sparsity of feature effects on logits via W_U projection.

    Args:
        directions: (n_sae, d_model) decoder directions.
        model: HookedTransformer with W_U attribute.
        n_features_to_check: number of features to sample for evaluation.

    Returns:
        (mean_downstream_sparsity, detail_dict) where sparsity is
        1.0 - mean(L0 / d_vocab).
    """
    device = directions.device
    d_sae = directions.shape[0]

    # Get unembedding matrix: (d_model, d_vocab)
    W_U = model.W_U.detach().to(device).float()
    d_vocab = W_U.shape[1]

    # Sample features if there are too many
    n_check = min(n_features_to_check, d_sae)
    if n_check < d_sae:
        indices = torch.randperm(d_sae, device=device)[:n_check]
        sampled_dirs = directions[indices].float()
    else:
        indices = torch.arange(d_sae, device=device)
        sampled_dirs = directions.float()

    # Project decoder directions through unembedding: (n_check, d_vocab)
    logit_effects = sampled_dirs @ W_U

    # L0 per feature: count entries above threshold
    l0_per_feature = (logit_effects.abs() > DOWNSTREAM_EFFECT_THRESHOLD).float().sum(dim=1)
    l0_fractions = l0_per_feature / d_vocab

    mean_l0_fraction = float(l0_fractions.mean())
    mean_downstream_sparsity = 1.0 - mean_l0_fraction

    return mean_downstream_sparsity, {
        "n_features_checked": n_check,
        "d_vocab": d_vocab,
        "effect_threshold": DOWNSTREAM_EFFECT_THRESHOLD,
        "mean_l0": float(l0_per_feature.mean()),
        "median_l0": float(l0_per_feature.median()),
        "mean_l0_fraction": mean_l0_fraction,
        "min_sparsity": float(1.0 - l0_fractions.max()),
        "max_sparsity": float(1.0 - l0_fractions.min()),
    }


def _activation_explainability(
    feature_acts: torch.Tensor,
    n_features_to_check: int,
) -> tuple[float, dict]:
    """Compute proxy explainability from kurtosis and sparsity of activations.

    Args:
        feature_acts: (n_positions, n_features) activation matrix.
        n_features_to_check: number of features to sample.

    Returns:
        (mean_explainability, detail_dict) where explainability combines
        normalized kurtosis and activation sparsity.
    """
    n_positions, n_features = feature_acts.shape

    n_check = min(n_features_to_check, n_features)
    if n_check < n_features:
        indices = torch.randperm(n_features, device=feature_acts.device)[:n_check]
        acts = feature_acts[:, indices].float()
    else:
        acts = feature_acts.float()

    explainability_scores = []
    kurtosis_values = []
    sparsity_values = []

    for i in range(n_check):
        col = acts[:, i]

        # Sparsity: fraction of zero (or near-zero) activations
        sparsity = float((col.abs() < 1e-6).float().mean())

        # Kurtosis: (m4 / m2^2) - 3 (excess kurtosis)
        mean_val = col.mean()
        centered = col - mean_val
        var = (centered ** 2).mean()
        if var < 1e-12:
            # Constant feature: high sparsity if all zero, low kurtosis
            kurtosis = 0.0
        else:
            m4 = (centered ** 4).mean()
            kurtosis = float(m4 / (var ** 2)) - 3.0

        # Normalize kurtosis to [0, 1] via sigmoid-like transform
        # Gaussian kurtosis = 0; sparse features have kurtosis >> 0
        normalized_kurtosis = float(1.0 - 1.0 / (1.0 + max(kurtosis, 0.0) / 10.0))

        score = 0.5 * normalized_kurtosis + 0.5 * sparsity
        explainability_scores.append(score)
        kurtosis_values.append(kurtosis)
        sparsity_values.append(sparsity)

    mean_explainability = float(np.mean(explainability_scores))

    return mean_explainability, {
        "n_features_checked": n_check,
        "n_positions": n_positions,
        "mean_kurtosis": float(np.mean(kurtosis_values)),
        "median_kurtosis": float(np.median(kurtosis_values)),
        "mean_sparsity": float(np.mean(sparsity_values)),
        "median_sparsity": float(np.median(sparsity_values)),
        "mean_explainability": mean_explainability,
        "min_explainability": float(np.min(explainability_scores)),
        "max_explainability": float(np.max(explainability_scores)),
    }


@torch.no_grad()
def run_topk_scaling(
    model,
    artifact=None,
    hook_name: str = "blocks.5.hook_resid_pre",
    n_features: int = 50,
    n_tokens: int = 500,
    n_synthetic: int = 5,
) -> list[EvalResult]:
    """Run all three TopK SAE Scaling sub-metrics.

    Args:
        model: HookedTransformer instance.
        artifact: ArtifactAdapter with directions() and activations() methods.
        hook_name: hook point for activation collection.
        n_features: number of features to sample for sparsity/explainability.
        n_tokens: number of tokens for activation collection.
        n_synthetic: number of synthetic directions for feature recovery.

    Returns:
        List of three EvalResult instances (EX7a, EX7b, EX7c).
    """
    if artifact is None:
        log("  WARNING: no artifact adapter provided, skipping topk_scaling")
        return []

    log(f"  Running TopK SAE Scaling metrics at {hook_name}...")

    # Get decoder directions
    try:
        directions = artifact.directions()
    except (NotImplementedError, TypeError):
        log("  WARNING: artifact does not expose directions(), skipping")
        return []

    if directions is None or directions.numel() == 0:
        log("  WARNING: empty directions from artifact, skipping")
        return []

    directions = directions.float()
    d_sae, d_model = directions.shape
    device = directions.device
    log(f"    {d_sae} features, d_model={d_model}")

    results: list[EvalResult] = []

    # ── EX7a: Feature Recovery ────────────────────────────────────────
    log("    [EX7a] Computing feature recovery...")
    recovery_rate, recovery_details = _feature_recovery(directions, n_synthetic, device)
    recovery_passed = recovery_rate > RECOVERY_THRESHOLD
    log(f"    recovery_rate={recovery_rate:.4f} [{'PASS' if recovery_passed else 'FAIL'}]")

    results.append(EvalResult(
        metric_id="EX7a.feature_recovery",
        value=recovery_rate,
        n_samples=n_synthetic,
        instrument_info=INSTRUMENT_INFO_RECOVERY,
        metadata={
            "hook_name": hook_name,
            "d_sae": d_sae,
            "d_model": d_model,
            "passed": recovery_passed,
            "threshold": RECOVERY_THRESHOLD,
            **recovery_details,
        },
    ))

    # ── EX7b: Downstream Sparsity ─────────────────────────────────────
    log("    [EX7b] Computing downstream effect sparsity...")
    try:
        sparsity_value, sparsity_details = _downstream_sparsity(
            directions, model, n_features,
        )
        sparsity_passed = sparsity_value > SPARSITY_THRESHOLD
        log(f"    downstream_sparsity={sparsity_value:.4f} [{'PASS' if sparsity_passed else 'FAIL'}]")

        results.append(EvalResult(
            metric_id="EX7b.downstream_sparsity",
            value=sparsity_value,
            n_samples=sparsity_details["n_features_checked"],
            instrument_info=INSTRUMENT_INFO_SPARSITY,
            metadata={
                "hook_name": hook_name,
                "d_sae": d_sae,
                "d_model": d_model,
                "passed": sparsity_passed,
                "threshold": SPARSITY_THRESHOLD,
                **sparsity_details,
            },
        ))
    except (AttributeError, RuntimeError) as e:
        log(f"    WARNING: downstream sparsity failed ({e}), skipping EX7b")

    # ── EX7c: Activation Explainability ───────────────────────────────
    log("    [EX7c] Computing activation explainability...")
    sample_text = "The quick brown fox jumps over the lazy dog. " * (n_tokens // 10)
    tokens = model.to_tokens(sample_text)
    tokens = tokens[:, :min(tokens.shape[1], n_tokens)]

    try:
        feature_acts = artifact.activations(model, tokens, hook_name)
        feature_acts = feature_acts.reshape(-1, feature_acts.shape[-1]).float()
        n_positions = feature_acts.shape[0]
        log(f"    {n_positions} positions collected")

        explainability_value, explainability_details = _activation_explainability(
            feature_acts, n_features,
        )
        explainability_passed = explainability_value > EXPLAINABILITY_THRESHOLD
        log(f"    explainability={explainability_value:.4f} "
            f"[{'PASS' if explainability_passed else 'FAIL'}]")

        results.append(EvalResult(
            metric_id="EX7c.activation_explainability",
            value=explainability_value,
            n_samples=n_positions,
            instrument_info=INSTRUMENT_INFO_EXPLAINABILITY,
            metadata={
                "hook_name": hook_name,
                "d_sae": d_sae,
                "d_model": d_model,
                "passed": explainability_passed,
                "threshold": EXPLAINABILITY_THRESHOLD,
                **explainability_details,
            },
        ))
    except (NotImplementedError, TypeError, RuntimeError) as e:
        log(f"    WARNING: activation explainability failed ({e}), skipping EX7c")

    return results


def main():
    parser = parse_common_args("EX7: TopK SAE Scaling Metrics")
    parser.add_argument("--hook", default="blocks.5.hook_resid_pre", help="Hook point")
    parser.add_argument("--n-tokens", type=int, default=500, help="Tokens for activations")
    parser.add_argument("--n-features", type=int, default=50, help="Features to sample")
    parser.add_argument("--n-synthetic", type=int, default=5, help="Synthetic directions")
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
    log("EX7: TOPK SAE SCALING METRICS")
    log("=" * 60)

    results = run_topk_scaling(
        model, artifact=artifact, hook_name=args.hook,
        n_features=args.n_features, n_tokens=args.n_tokens,
        n_synthetic=args.n_synthetic,
    )

    out = args.out or "EX7_topk_scaling.json"
    save_results(results, out)
    log(f"\nDone. {len(results)} results.")


if __name__ == "__main__":
    main()
