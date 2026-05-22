"""Latent Self-Consistency (Measurement EX11)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     EX11 — Latent Self-Consistency
Categories:     measurement
Validity layer: Measurement (M4 Sensitivity)
Criteria:       EX11 Encoder-Decoder Mutual Consistency
Establishes:    Whether an artifact's encoder and decoder form a stable
                fixed point — encoding, decoding, and re-encoding gives
                back a structurally coherent latent representation
Requires:       Model, artifact adapter with directions() and activations()
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Motivated by the ICLR 2026 result that perfect reconstruction and correct
dictionary recovery do not guarantee latent variable (concept) recovery
due to encoder-decoder instability. Even an SAE with low reconstruction
error can sit at a saddle point where the encoder and decoder disagree
about the latent basis.

This metric tests stability via iterated encode-decode roundtrips:

1. Encode activations to get feature coefficients z_0 = encode(x).
2. Decode to reconstruct: x_1 = decode(z_0).
3. Re-encode: z_1 = encode(x_1).
4. Repeat for n_roundtrips: x_{k+1} = decode(z_k), z_{k+1} = encode(x_{k+1}).
5. Track cosine_sim(z_0, z_k) for each roundtrip k.

A stable SAE converges quickly (cosine similarity stays high). An SAE
at a saddle point drifts with each roundtrip.

Metrics:
- single_roundtrip_consistency: mean cosine_sim(z_0, z_1) across tokens
- drift_rate: linear slope of cosine_sim(z_0, z_k) vs k
- fixed_point_fraction: fraction of features whose activation changes
  less than 5% after one roundtrip

Pass condition: single_roundtrip_consistency > 0.85

References:
    - Buchanan et al. (ICLR 2026) "Latent variable recovery is not
      guaranteed by reconstruction or dictionary recovery"

Usage:
    mechval.run("latent_self_consistency", artifact=adapter,
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
    name="Latent Self-Consistency",
    paper_ref="Buchanan et al. ICLR 2026",
    paper_cite=(
        "Buchanan et al. 2026, Latent variable recovery is not guaranteed "
        "by reconstruction or dictionary recovery"
    ),
    description=(
        "Tests encoder-decoder mutual consistency via iterated roundtrips: "
        "if the SAE has found a stable feature basis, encoding and decoding "
        "repeatedly should converge rather than drift"
    ),
    category="measurement",
    tier="cogsci",
    origin="established",
)

CONSISTENCY_THRESHOLD = 0.85
FIXED_POINT_TOLERANCE = 0.05


def _batch_cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-row cosine similarity between two (n, d) tensors.

    Returns (n,) tensor. Rows where both vectors are near-zero get
    similarity 1.0 (both effectively represent the same zero state).
    """
    a_norm = a.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    b_norm = b.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    # If both are near-zero, treat as perfectly consistent
    both_zero = (a.norm(dim=-1) < 1e-6) & (b.norm(dim=-1) < 1e-6)

    cos = (a / a_norm * b / b_norm).sum(dim=-1)
    cos[both_zero] = 1.0
    return cos


def _encode(artifact, model, acts: torch.Tensor, hook_name: str) -> torch.Tensor:
    """Encode activations through the artifact.

    Uses artifact.activations() if it accepts raw activations, otherwise
    falls back to manual projection via directions().
    """
    # artifact.activations() expects (model, tokens, hook_name) and internally
    # runs the model. But here we already have the activations and want to
    # re-encode a reconstruction. We need to project through the encoder.
    #
    # For SAEAdapter, the underlying SAE has an encode() method. Try to
    # access it directly; if the adapter doesn't expose it, fall back to
    # projecting via directions (W_dec) pseudo-inverse approximation.
    sae = getattr(artifact, "_sae", None)
    if sae is not None and hasattr(sae, "encode"):
        return sae.encode(acts)

    # For adapters wrapping other artifact types, try a direct encode attr
    inner = getattr(artifact, "_model", None) or getattr(artifact, "_artifact", None)
    if inner is not None and hasattr(inner, "encode"):
        return inner.encode(acts)

    # Fallback: project onto decoder directions (assumes orthogonal-ish decoder)
    directions = artifact.directions()
    directions = directions.to(acts.device).float()
    # (batch, seq, d_model) @ (n_features, d_model).T -> (batch, seq, n_features)
    return acts.float() @ directions.T


def _decode(artifact, z: torch.Tensor) -> torch.Tensor:
    """Decode feature coefficients back to activation space.

    Uses artifact's decoder if available, otherwise directions @ z.
    """
    sae = getattr(artifact, "_sae", None)
    if sae is not None and hasattr(sae, "decode"):
        return sae.decode(z)

    inner = getattr(artifact, "_model", None) or getattr(artifact, "_artifact", None)
    if inner is not None and hasattr(inner, "decode"):
        return inner.decode(z)

    # Fallback: z @ directions (decoder directions are rows of W_dec)
    directions = artifact.directions()
    directions = directions.to(z.device).float()
    return z.float() @ directions


@torch.no_grad()
def run_self_consistency(
    model,
    artifact=None,
    hook_name: str | None = None,
    n_tokens: int = 500,
    n_roundtrips: int = 3,
) -> list[EvalResult]:
    """Test encoder-decoder mutual consistency via iterated roundtrips.

    If an SAE has found a stable feature basis, encoding and decoding
    repeatedly should converge. If it is at a saddle point, the
    representation drifts with each roundtrip.

    Args:
        model: HookedTransformer instance.
        artifact: ArtifactAdapter with directions() and activations().
        hook_name: hook point for activation collection.
        n_tokens: number of tokens to evaluate.
        n_roundtrips: number of encode-decode roundtrips (>= 1).

    Returns:
        List with one EvalResult for EX11.latent_self_consistency.
    """
    if artifact is None:
        log("  WARNING: no artifact adapter provided, skipping self-consistency")
        return []

    effective_hook = (
        hook_name
        or getattr(getattr(artifact, "manifest", None), "hook_point", None)
        or "blocks.5.hook_resid_pre"
    )

    log(f"  Computing latent self-consistency at {effective_hook} "
        f"({n_roundtrips} roundtrips)...")

    # Ensure artifact is loaded (directions() triggers lazy load for most adapters)
    try:
        directions = artifact.directions()
    except (NotImplementedError, TypeError):
        log("  WARNING: artifact does not expose directions(), skipping")
        return []

    if directions is None or directions.numel() == 0:
        log("  WARNING: empty directions from artifact, skipping")
        return []

    n_features = directions.shape[0]
    d_model = directions.shape[1]
    log(f"    {n_features} features, d_model={d_model}")

    # Step 1: get original activations at the hook point
    sample_text = "The quick brown fox jumps over the lazy dog. " * (n_tokens // 10)
    tokens = model.to_tokens(sample_text)
    tokens = tokens[:, :min(tokens.shape[1], n_tokens)]

    # Get feature activations z_0 = encode(x)
    z_0 = artifact.activations(model, tokens, effective_hook)
    # (batch, seq, n_features) -> (n_positions, n_features)
    z_0_flat = z_0.reshape(-1, z_0.shape[-1]).float()
    n_positions = z_0_flat.shape[0]

    log(f"    {n_positions} token positions collected")

    # Step 2: iterated roundtrips
    # z_0 -> x_1 = decode(z_0) -> z_1 = encode(x_1) -> x_2 = decode(z_1) -> ...
    per_roundtrip_cosines = []
    per_roundtrip_feature_deltas = []

    z_prev = z_0_flat
    for k in range(n_roundtrips):
        # Decode: reconstruct activations from feature coefficients
        x_hat = _decode(artifact, z_prev.unsqueeze(0) if z_prev.dim() == 2 else z_prev)
        x_hat_flat = x_hat.reshape(-1, d_model).float()

        # Re-encode: get new feature coefficients
        z_k = _encode(artifact, model, x_hat.reshape(z_0.shape[0], -1, d_model), effective_hook)
        z_k_flat = z_k.reshape(-1, z_k.shape[-1]).float()

        # Cosine similarity between z_0 and z_k (per token position)
        cos_sims = _batch_cosine_similarity(z_0_flat, z_k_flat)
        mean_cos = float(cos_sims.mean())
        per_roundtrip_cosines.append(mean_cos)

        # Per-feature relative change from z_0
        z_0_norms = z_0_flat.abs().clamp(min=1e-8)
        relative_delta = ((z_k_flat - z_0_flat).abs() / z_0_norms).mean(dim=0)
        per_roundtrip_feature_deltas.append(relative_delta)

        log(f"    roundtrip {k+1}: mean_cosine_sim(z_0, z_{k+1}) = {mean_cos:.4f}")
        z_prev = z_k_flat

    # Step 3: compute summary metrics

    # Single roundtrip consistency (the primary metric)
    single_roundtrip_consistency = per_roundtrip_cosines[0]

    # Drift rate: linear regression of cosine_sim vs roundtrip index
    if n_roundtrips >= 2:
        xs = np.arange(1, n_roundtrips + 1, dtype=np.float64)
        ys = np.array(per_roundtrip_cosines, dtype=np.float64)
        drift_rate = float(np.polyfit(xs, ys, 1)[0])
    else:
        drift_rate = 0.0

    # Fixed-point fraction: features whose activation changes < 5% after one roundtrip
    first_delta = per_roundtrip_feature_deltas[0]  # (n_features,)
    fixed_point_fraction = float((first_delta < FIXED_POINT_TOLERANCE).float().mean())

    passed = single_roundtrip_consistency > CONSISTENCY_THRESHOLD

    log(f"    single_roundtrip_consistency = {single_roundtrip_consistency:.4f}")
    log(f"    drift_rate = {drift_rate:.6f}")
    log(f"    fixed_point_fraction = {fixed_point_fraction:.4f}")
    log(f"    [{'PASS' if passed else 'FAIL'}]")

    return [EvalResult(
        metric_id="EX11.latent_self_consistency",
        value=single_roundtrip_consistency,
        n_samples=n_positions,
        instrument_info=INSTRUMENT_INFO,
        metadata={
            "hook_name": effective_hook,
            "n_features": n_features,
            "d_model": d_model,
            "n_positions": n_positions,
            "n_roundtrips": n_roundtrips,
            "single_roundtrip_consistency": single_roundtrip_consistency,
            "drift_rate": drift_rate,
            "fixed_point_fraction": fixed_point_fraction,
            "roundtrip_cosines": per_roundtrip_cosines,
            "passed": passed,
            "threshold": CONSISTENCY_THRESHOLD,
        },
    )]


def main():
    parser = parse_common_args("EX11: Latent Self-Consistency")
    parser.add_argument("--hook", default=None, help="Hook point")
    parser.add_argument("--n-tokens", type=int, default=500, help="Tokens to evaluate")
    parser.add_argument("--n-roundtrips", type=int, default=3,
                        help="Number of encode-decode roundtrips")
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
            hook_point=args.hook or "",
        )

    log("=" * 60)
    log("EX11: LATENT SELF-CONSISTENCY")
    log("=" * 60)

    results = run_self_consistency(
        model, artifact=artifact, hook_name=args.hook,
        n_tokens=args.n_tokens, n_roundtrips=args.n_roundtrips,
    )

    out = args.out or "EX11_self_consistency.json"
    save_results(results, out)
    log(f"\nDone. {len(results)} results.")


if __name__ == "__main__":
    main()
