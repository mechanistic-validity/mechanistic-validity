"""SAEBench Comprehensive SAE Evaluation (Measurement EX9)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     EX9 — SAEBench Multi-Metric Evaluation
Categories:     measurement
Validity layer: External
Criteria:       EX9 Proxy, interpretability, and disentanglement metrics
Establishes:    Whether an artifact's features satisfy basic quality criteria
                across reconstruction, sparsity, interpretability, and
                disentanglement dimensions
Requires:       CPU or GPU, model, artifact adapter
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Implements a subset of the SAEBench evaluation protocol (Karvonen et al.,
ICML 2025). SAEBench evaluates SAEs across 8 metrics in 4 categories and
showed that proxy metric gains (reconstruction loss, L0) do not translate
to practical performance.

Five sub-metrics:

**Proxy metrics** (necessary but not sufficient):
  EX9a. Reconstruction Loss: MSE between original activations and SAE
        reconstruction. Pass: < 0.1.
  EX9b. L0 Sparsity: Mean number of active features per token.
        Pass: < 50.
  EX9c. Explained Variance: Fraction of activation variance explained by
        reconstruction. 1 - var(residual) / var(original). Pass: > 0.85.

**Interpretability** (does the artifact capture concepts?):
  EX9d. Feature Detection: Per-feature AUROC on detecting concept membership
        using built-in concept pairs (sentiment, formal/informal, scientific/
        everyday). Reuses the AxBench concept-pair approach. Pass: > 0.6.

**Disentanglement** (are features independent?):
  EX9e. Feature Disentanglement: Mean pairwise |cosine similarity| of
        sampled decoder direction pairs. Lower overlap means better
        disentanglement. Pass: < 0.1.

Reference:
    Karvonen et al. (2025) "SAEBench: A Comprehensive Benchmark for Sparse
    Autoencoders in Language Models", ICML 2025.

Usage:
    mechval.run("saebench", artifact=adapter, hook_name="blocks.5.hook_resid_pre")
"""

import numpy as np
import torch

from mechval.metrics.common import (
    EvalResult,
    InstrumentInfo,
    load_model,
    log,
    parse_common_args,
    save_results,
)

INSTRUMENT_INFO = InstrumentInfo(
    name="SAEBench Comprehensive SAE Evaluation",
    paper_ref="Karvonen et al. 2025",
    paper_cite="Karvonen et al. 2025, SAEBench: A Comprehensive Benchmark "
               "for Sparse Autoencoders in Language Models, ICML 2025",
    description="Multi-metric SAE evaluation across proxy, interpretability, "
                "and disentanglement dimensions",
    category="measurement",
    tier="external",
    origin="adapted",
    subcategory="saebench",
)

# Pass thresholds
RECONSTRUCTION_LOSS_THRESHOLD = 0.1
L0_THRESHOLD = 50.0
EXPLAINED_VARIANCE_THRESHOLD = 0.85
DETECTION_THRESHOLD = 0.6
DISENTANGLEMENT_THRESHOLD = 0.1


# ---------------------------------------------------------------------------
# AUROC (no sklearn dependency, reused from AxBench pattern)
# ---------------------------------------------------------------------------

def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC via Mann-Whitney U statistic. Direction-agnostic."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    u = 0.0
    for p in pos:
        u += np.sum(p > neg) + 0.5 * np.sum(p == neg)
    auroc = u / (n_pos * n_neg)
    return max(auroc, 1.0 - auroc)


# ---------------------------------------------------------------------------
# Built-in concept pairs (shared with AxBench)
# ---------------------------------------------------------------------------

def _default_concepts() -> list[tuple[list[str], list[str]]]:
    """Small built-in concept set for concept detection evaluation."""
    return [
        # Concept 0: Sentiment (positive vs negative)
        (
            [
                "This movie was absolutely wonderful and I loved every minute of it.",
                "The food at this restaurant is delicious and the service is excellent.",
                "I had an amazing experience and would highly recommend this place.",
                "The weather today is beautiful and perfect for a walk in the park.",
                "She gave a brilliant performance that moved the entire audience.",
                "This is the best book I have read in years, truly outstanding.",
                "The team played magnificently and deserved their victory.",
                "What a fantastic day, everything went perfectly as planned.",
                "The garden looks gorgeous with all the flowers in bloom.",
                "I am thrilled with the results, they exceeded all expectations.",
            ],
            [
                "This movie was terrible and a complete waste of time.",
                "The food at this restaurant is awful and the service is horrible.",
                "I had a dreadful experience and would never come back.",
                "The weather today is miserable with heavy rain and cold wind.",
                "She gave a poor performance that disappointed everyone.",
                "This is the worst book I have ever read, completely boring.",
                "The team played terribly and deserved their humiliating loss.",
                "What a horrible day, everything went wrong from the start.",
                "The garden looks neglected with dead plants everywhere.",
                "I am furious with the results, they are completely unacceptable.",
            ],
        ),
        # Concept 1: Formal vs informal register
        (
            [
                "We respectfully request your presence at the annual conference.",
                "The committee has resolved to implement the proposed amendments.",
                "Pursuant to our agreement, the deliverables shall be submitted.",
                "I am writing to inform you of the changes to our policy.",
                "The board of directors convened to discuss strategic initiatives.",
                "Please find enclosed the documentation for your review.",
                "We acknowledge receipt of your correspondence dated March 15.",
                "The undersigned hereby certifies the accuracy of this report.",
                "In accordance with established protocols, we have initiated.",
                "The organization has undertaken a comprehensive review.",
            ],
            [
                "Hey dude, wanna grab some pizza tonight?",
                "Lol that was so funny I can't even deal right now.",
                "Yo check this out, it's totally insane!",
                "Nah man, I'm just gonna chill at home tonight.",
                "Omg that party was lit, we should do it again!",
                "Bruh you're not gonna believe what happened today.",
                "Yeah whatever, I don't really care about that stuff.",
                "Haha nice one, you totally got me with that joke.",
                "Dude seriously? That's the craziest thing I've heard.",
                "Gonna bounce, catch you later alright?",
            ],
        ),
        # Concept 2: Scientific/technical vs everyday language
        (
            [
                "The mitochondria generate ATP through oxidative phosphorylation.",
                "Quantum entanglement violates Bell's inequality in experiments.",
                "The catalyst lowers the activation energy of the reaction.",
                "Neuroplasticity allows the brain to reorganize synaptic connections.",
                "The algorithm achieves O(n log n) time complexity on average.",
                "Photosynthesis converts carbon dioxide and water into glucose.",
                "The p-value indicates statistical significance below the threshold.",
                "Tectonic plates move due to convection currents in the mantle.",
                "The protein folds into its tertiary structure via hydrophobic.",
                "Electromagnetic radiation propagates as transverse waves.",
            ],
            [
                "I went to the store to buy some milk and bread.",
                "The kids played in the backyard all afternoon.",
                "We had spaghetti for dinner and watched a movie.",
                "She walked the dog around the block before bedtime.",
                "He fixed the leaky faucet in the kitchen sink.",
                "They drove to the beach and built sandcastles.",
                "I called my mom to wish her a happy birthday.",
                "The cat slept on the couch all day long.",
                "We planted tomatoes and peppers in the garden.",
                "She baked cookies for the school bake sale.",
            ],
        ),
    ]


# ---------------------------------------------------------------------------
# Sub-metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def _reconstruction_loss(
    model, artifact, hook_name: str, n_tokens: int,
) -> EvalResult:
    """EX9a: MSE between original activations and SAE reconstruction."""
    texts = _generate_calibration_texts(n_tokens)
    total_mse = 0.0
    n_positions = 0

    dirs = artifact.directions()
    if dirs.ndim == 3:
        dirs = dirs.mean(dim=0)

    for text in texts:
        tokens = model.to_tokens(text)
        _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
        original = cache[hook_name]  # (1, seq, d_model)

        # Encode through artifact, then reconstruct via decoder directions
        feature_acts = artifact.activations(model, tokens, hook_name)  # (1, seq, n_features)
        dirs_dev = dirs.to(feature_acts.device)
        reconstructed = feature_acts @ dirs_dev  # (1, seq, d_model)

        residual = original.to(reconstructed.device) - reconstructed
        total_mse += (residual ** 2).sum().item()
        n_positions += original.shape[0] * original.shape[1] * original.shape[2]

    mse = total_mse / max(n_positions, 1)
    passed = mse < RECONSTRUCTION_LOSS_THRESHOLD

    log(f"    EX9a reconstruction_loss={mse:.6f} [{'PASS' if passed else 'FAIL'}]")

    return EvalResult(
        metric_id="EX9a.reconstruction_loss",
        value=mse,
        n_samples=len(texts),
        instrument_info=INSTRUMENT_INFO,
        metadata={
            "sub_metric": "reconstruction_loss",
            "mse": mse,
            "threshold": RECONSTRUCTION_LOSS_THRESHOLD,
            "passed": passed,
            "hook_name": hook_name,
            "n_texts": len(texts),
        },
    )


@torch.no_grad()
def _l0_sparsity(
    model, artifact, hook_name: str, n_tokens: int,
) -> EvalResult:
    """EX9b: Mean number of active (non-zero) features per token."""
    texts = _generate_calibration_texts(n_tokens)
    total_l0 = 0.0
    n_positions = 0

    for text in texts:
        tokens = model.to_tokens(text)
        feature_acts = artifact.activations(model, tokens, hook_name)  # (1, seq, n_features)
        # Count non-zero features per position, then sum
        nonzero_per_pos = (feature_acts.abs() > 1e-8).float().sum(dim=-1)  # (1, seq)
        total_l0 += nonzero_per_pos.sum().item()
        n_positions += feature_acts.shape[0] * feature_acts.shape[1]

    mean_l0 = total_l0 / max(n_positions, 1)
    passed = mean_l0 < L0_THRESHOLD

    log(f"    EX9b l0_sparsity={mean_l0:.2f} [{'PASS' if passed else 'FAIL'}]")

    return EvalResult(
        metric_id="EX9b.l0_sparsity",
        value=mean_l0,
        n_samples=n_positions,
        instrument_info=INSTRUMENT_INFO,
        metadata={
            "sub_metric": "l0_sparsity",
            "mean_l0": mean_l0,
            "threshold": L0_THRESHOLD,
            "passed": passed,
            "hook_name": hook_name,
        },
    )


@torch.no_grad()
def _explained_variance(
    model, artifact, hook_name: str, n_tokens: int,
) -> EvalResult:
    """EX9c: Fraction of activation variance explained by reconstruction."""
    texts = _generate_calibration_texts(n_tokens)

    dirs = artifact.directions()
    if dirs.ndim == 3:
        dirs = dirs.mean(dim=0)

    all_originals = []
    all_residuals = []

    for text in texts:
        tokens = model.to_tokens(text)
        _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
        original = cache[hook_name]  # (1, seq, d_model)

        feature_acts = artifact.activations(model, tokens, hook_name)
        dirs_dev = dirs.to(feature_acts.device)
        reconstructed = feature_acts @ dirs_dev

        original_flat = original.to(reconstructed.device).reshape(-1)
        residual_flat = (original.to(reconstructed.device) - reconstructed).reshape(-1)

        all_originals.append(original_flat.cpu())
        all_residuals.append(residual_flat.cpu())

    all_originals = torch.cat(all_originals)
    all_residuals = torch.cat(all_residuals)

    var_original = all_originals.var().item()
    var_residual = all_residuals.var().item()

    if var_original < 1e-12:
        explained_var = 0.0
    else:
        explained_var = 1.0 - var_residual / var_original

    passed = explained_var > EXPLAINED_VARIANCE_THRESHOLD

    log(f"    EX9c explained_variance={explained_var:.4f} [{'PASS' if passed else 'FAIL'}]")

    return EvalResult(
        metric_id="EX9c.explained_variance",
        value=explained_var,
        n_samples=len(texts),
        instrument_info=INSTRUMENT_INFO,
        metadata={
            "sub_metric": "explained_variance",
            "explained_variance": explained_var,
            "var_original": var_original,
            "var_residual": var_residual,
            "threshold": EXPLAINED_VARIANCE_THRESHOLD,
            "passed": passed,
            "hook_name": hook_name,
        },
    )


@torch.no_grad()
def _feature_detection(
    model, artifact, hook_name: str, n_features: int,
) -> EvalResult:
    """EX9d: Per-feature AUROC on detecting concept membership."""
    concepts = _default_concepts()
    all_aurocs = []

    for concept_idx, (pos_texts, neg_texts) in enumerate(concepts):
        # Collect activations for positive and negative examples
        pos_acts = []
        for text in pos_texts:
            tokens = model.to_tokens(text)
            acts = artifact.activations(model, tokens, hook_name)
            # Mean-pool across sequence positions
            pos_acts.append(acts[0].mean(dim=0).cpu())
        pos_acts = torch.stack(pos_acts)  # (n_pos, n_features_total)

        neg_acts = []
        for text in neg_texts:
            tokens = model.to_tokens(text)
            acts = artifact.activations(model, tokens, hook_name)
            neg_acts.append(acts[0].mean(dim=0).cpu())
        neg_acts = torch.stack(neg_acts)  # (n_neg, n_features_total)

        n_pos = pos_acts.shape[0]
        n_neg = neg_acts.shape[0]
        n_feat = pos_acts.shape[1]

        all_acts_np = torch.cat([pos_acts, neg_acts], dim=0).numpy()
        labels = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])

        # Evaluate only a subset of features (top by variance ratio)
        n_eval = min(n_features, n_feat)
        # Rank features by abs difference in means (most discriminative)
        mean_diff = np.abs(all_acts_np[:n_pos].mean(axis=0) - all_acts_np[n_pos:].mean(axis=0))
        top_indices = np.argsort(mean_diff)[::-1][:n_eval]

        per_feature_auroc = np.array([
            _auroc(all_acts_np[:, f], labels) for f in top_indices
        ])

        best_auroc = float(per_feature_auroc.max()) if len(per_feature_auroc) > 0 else 0.5
        all_aurocs.append(best_auroc)

        log(f"    concept {concept_idx}: best detection AUROC={best_auroc:.4f}")

    mean_detection = float(np.mean(all_aurocs)) if all_aurocs else 0.5
    passed = mean_detection > DETECTION_THRESHOLD

    log(f"    EX9d feature_detection={mean_detection:.4f} [{'PASS' if passed else 'FAIL'}]")

    return EvalResult(
        metric_id="EX9d.feature_detection",
        value=mean_detection,
        n_samples=sum(len(p) + len(n) for p, n in concepts),
        instrument_info=INSTRUMENT_INFO,
        metadata={
            "sub_metric": "feature_detection",
            "mean_detection_auroc": mean_detection,
            "per_concept_auroc": all_aurocs,
            "n_concepts": len(concepts),
            "n_features_evaluated": n_features,
            "threshold": DETECTION_THRESHOLD,
            "passed": passed,
            "hook_name": hook_name,
        },
    )


@torch.no_grad()
def _feature_disentanglement(artifact, n_features: int) -> EvalResult:
    """EX9e: Mean |cosine similarity| of sampled decoder direction pairs."""
    dirs = artifact.directions()
    if dirs.ndim == 3:
        dirs = dirs.mean(dim=0)
    # dirs: (n_total_features, d_model)

    n_total = dirs.shape[0]
    n_sample = min(n_features, n_total)

    # Sample random pairs from the decoder directions
    indices = torch.randperm(n_total)[:n_sample]
    sampled = dirs[indices]  # (n_sample, d_model)

    # Normalize for cosine similarity
    norms = sampled.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    normed = sampled / norms  # (n_sample, d_model)

    # Pairwise cosine similarity matrix
    cos_sim = normed @ normed.T  # (n_sample, n_sample)

    # Extract upper triangle (exclude diagonal)
    n = cos_sim.shape[0]
    mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
    pairwise_abs_cos = cos_sim[mask].abs()

    if pairwise_abs_cos.numel() == 0:
        mean_overlap = 0.0
    else:
        mean_overlap = float(pairwise_abs_cos.mean().item())

    passed = mean_overlap < DISENTANGLEMENT_THRESHOLD

    log(f"    EX9e feature_disentanglement={mean_overlap:.4f} [{'PASS' if passed else 'FAIL'}]")

    return EvalResult(
        metric_id="EX9e.feature_disentanglement",
        value=mean_overlap,
        n_samples=int(pairwise_abs_cos.numel()),
        instrument_info=INSTRUMENT_INFO,
        metadata={
            "sub_metric": "feature_disentanglement",
            "mean_abs_cosine_similarity": mean_overlap,
            "n_directions_sampled": n_sample,
            "n_pairs": int(pairwise_abs_cos.numel()),
            "threshold": DISENTANGLEMENT_THRESHOLD,
            "passed": passed,
        },
    )


# ---------------------------------------------------------------------------
# Calibration text generation
# ---------------------------------------------------------------------------

def _generate_calibration_texts(n_tokens: int) -> list[str]:
    """Generate diverse calibration texts for proxy metric evaluation.

    Returns enough short texts to cover approximately n_tokens tokens total
    (each text is roughly 15-25 tokens).
    """
    texts = [
        "The president of the United States gave a speech about the economy.",
        "Scientists discovered a new species of deep-sea fish near the coast.",
        "The stock market experienced significant volatility throughout the week.",
        "A major earthquake struck the region causing widespread destruction.",
        "Researchers published findings on the effects of climate change.",
        "The new software update includes several important security patches.",
        "Local authorities announced plans for a new public transit system.",
        "The championship game attracted millions of viewers around the world.",
        "Astronomers observed a distant galaxy using the latest telescope data.",
        "The annual music festival featured performances by international artists.",
        "Engineers developed a prototype for a more efficient solar panel.",
        "The university launched a new program in artificial intelligence research.",
        "Farmers reported record crop yields due to favorable weather conditions.",
        "The documentary film received critical acclaim at the festival premiere.",
        "Volunteers organized a community cleanup event at the local park.",
        "The hospital implemented new procedures to improve patient care quality.",
        "Construction began on the new bridge connecting the two city districts.",
        "The author released a highly anticipated sequel to the bestselling novel.",
        "Marine biologists tracked the migration patterns of humpback whales.",
        "The technology company announced a breakthrough in quantum computing.",
        "Archaeologists uncovered ancient ruins dating back thousands of years.",
        "The city council approved the budget for the upcoming fiscal year.",
        "Professional athletes participated in a charity marathon event.",
        "The orchestra performed a stunning rendition of a classical symphony.",
        "Meteorologists predicted above average temperatures for the summer months.",
    ]
    # Repeat if needed to reach approximately n_tokens total
    avg_tokens_per_text = 20
    n_texts_needed = max(1, n_tokens // avg_tokens_per_text)
    result = []
    for i in range(n_texts_needed):
        result.append(texts[i % len(texts)])
    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_saebench(
    model,
    artifact=None,
    hook_name: str | None = None,
    n_tokens: int = 500,
    n_features: int = 50,
) -> list[EvalResult]:
    """SAEBench-style evaluation across proxy, interpretability, and practical metrics.

    Args:
        model: HookedTransformer instance.
        artifact: ArtifactAdapter with directions() and activations() methods.
        hook_name: Hook point for activations (e.g. "blocks.5.hook_resid_pre").
        n_tokens: Approximate number of tokens for proxy metric calibration texts.
        n_features: Number of features to sample for detection and disentanglement.

    Returns:
        List of EvalResult, one per sub-metric (5 total).
    """
    if artifact is None:
        log("  WARNING: no artifact adapter provided, skipping SAEBench")
        return []

    effective_hook = hook_name or artifact.manifest.hook_point
    if not effective_hook:
        effective_hook = "blocks.5.hook_resid_pre"

    log("=" * 60)
    log("EX9: SAEBENCH COMPREHENSIVE SAE EVALUATION")
    log(f"  hook={effective_hook}, n_tokens={n_tokens}, n_features={n_features}")
    log("=" * 60)

    results = []

    # 1. Proxy: Reconstruction Loss
    log("  [EX9a] Reconstruction Loss...")
    results.append(_reconstruction_loss(model, artifact, effective_hook, n_tokens))

    # 2. Proxy: L0 Sparsity
    log("  [EX9b] L0 Sparsity...")
    results.append(_l0_sparsity(model, artifact, effective_hook, n_tokens))

    # 3. Proxy: Explained Variance
    log("  [EX9c] Explained Variance...")
    results.append(_explained_variance(model, artifact, effective_hook, n_tokens))

    # 4. Interpretability: Feature Detection
    log("  [EX9d] Feature Detection...")
    results.append(_feature_detection(model, artifact, effective_hook, n_features))

    # 5. Disentanglement: Feature Overlap
    log("  [EX9e] Feature Disentanglement...")
    results.append(_feature_disentanglement(artifact, n_features))

    # Summary
    n_passed = sum(1 for r in results if r.metadata.get("passed", False))
    log(f"  SUMMARY: {n_passed}/{len(results)} sub-metrics passed")
    for r in results:
        status = "PASS" if r.metadata.get("passed", False) else "FAIL"
        log(f"    {r.metric_id}: {r.value:.4f} [{status}]")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = parse_common_args("EX9: SAEBench Comprehensive SAE Evaluation")
    parser.add_argument("--hook", default=None,
                        help="Hook point (e.g. blocks.5.hook_resid_pre)")
    parser.add_argument("--artifact-path", default=None,
                        help="SAE release ID for artifact")
    parser.add_argument("--sae-id", default=None,
                        help="SAE ID within release")
    parser.add_argument("--n-tokens", type=int, default=500,
                        help="Approx token count for proxy metrics")
    parser.add_argument("--n-features", type=int, default=50,
                        help="Features to sample for detection/disentanglement")
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

    results = run_saebench(
        model,
        artifact=artifact,
        hook_name=args.hook,
        n_tokens=args.n_tokens,
        n_features=args.n_features,
    )

    out = args.out or "EX9_saebench.json"
    save_results(results, out, args=args)
    log(f"\nDone. {len(results)} sub-metrics evaluated.")


if __name__ == "__main__":
    main()
