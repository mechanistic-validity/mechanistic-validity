"""Rule-Based Feature Descriptions (Measurement EX6)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     EX6 — Rule-Based Feature Descriptions
Categories:     measurement
Validity layer: Measurement (M4 Sensitivity)
Criteria:       EX6 Rule-Based Descriptions
Establishes:    Whether artifact features can be described by formal rules
                (skip-gram, absence, counting) rather than just exemplars
Requires:       Model, artifact adapter
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Implements rule-based feature description following Friedman, Bhaskar
et al. (arXiv Oct 2025, Princeton NLP). Features are tested against
three types of formal rules:

1. **Skip-gram rules**: bigram patterns with a gap. A feature fires
   when a specific token appears at a fixed relative offset from the
   current position (e.g., "[Canadian city] ... speaks -> English").
   Detected via pointwise mutual information (PMI) of token pairs at
   fixed offsets with feature activation.

2. **Absence rules**: features that fire when a specific token is
   ABSENT from the context window. These are invisible to exemplar
   inspection and AutoInterp because they describe what is NOT present.
   Key finding: absence rules appear in >25% of attention features in
   every layer.

3. **Counting rules**: features that activate (or toggle) when a
   token count exceeds a threshold in the context window.

Detection algorithm:
1. Collect activations for each feature across a token corpus.
2. Identify top-activating positions and their context windows.
3. For each feature, test skip-gram, absence, and counting rule
   hypotheses and compute rule coverage (fraction of high activations
   explained by the best-matching rule).
4. Aggregate: mean rule coverage across features, fraction of features
   with absence rules, fraction with counting rules.

Pass condition: mean_rule_coverage > 0.3 (features have identifiable
rule structure).

References:
    - Friedman, Bhaskar et al. (Oct 2025) "Attention Rules"
      github.com/princeton-nlp/AttentionRules
    - Templeton et al. (2024) "Scaling Monosemanticity"

Usage:
    mechval.run("rule_based_descriptions", artifact=adapter,
                hook_name="blocks.5.hook_resid_pre")
"""

from collections import Counter

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
    name="Rule-Based Feature Descriptions",
    paper_ref="Friedman, Bhaskar et al. arXiv Oct 2025",
    paper_cite="Friedman, Bhaskar et al. 2025, Attention Rules",
    description=(
        "Tests whether features can be described by formal rules "
        "(skip-gram, absence, counting) rather than just exemplars"
    ),
    category="measurement",
    tier="cogsci",
    origin="established",
)

RULE_COVERAGE_THRESHOLD = 0.3
CONTEXT_WINDOW = 5
TOP_K_POSITIONS = 50
PMI_THRESHOLD = 1.0
ABSENCE_CORR_THRESHOLD = 0.1
COUNT_CORR_THRESHOLD = 0.1


def _get_top_activating_positions(
    feature_acts: torch.Tensor, top_k: int,
) -> torch.Tensor:
    """Return indices of top-k activating positions for a single feature.

    Args:
        feature_acts: (n_positions,) activation values.
        top_k: number of positions to return.

    Returns:
        (top_k,) indices into feature_acts.
    """
    k = min(top_k, feature_acts.shape[0])
    _, indices = torch.topk(feature_acts.abs(), k)
    return indices


def _test_skipgram_rule(
    tokens_flat: torch.Tensor,
    feature_acts: torch.Tensor,
    top_indices: torch.Tensor,
    seq_len: int,
    window: int,
    vocab_size: int,
) -> float:
    """Test for skip-gram rules via PMI between token pairs and activation.

    For each offset in [-window, +window], compute which token types at
    that offset co-occur with high activation, measuring PMI. The best
    (token, offset) pair's coverage is returned.

    Args:
        tokens_flat: (n_positions,) token ids.
        feature_acts: (n_positions,) activation values.
        top_indices: indices of top-activating positions.
        seq_len: sequence length (to avoid cross-sequence offsets).
        window: context window size.
        vocab_size: vocabulary size.

    Returns:
        Coverage: fraction of top activations explained by the best
        skip-gram rule.
    """
    n_positions = tokens_flat.shape[0]
    top_set = set(top_indices.cpu().tolist())
    n_top = len(top_set)
    if n_top == 0:
        return 0.0

    best_coverage = 0.0

    for offset in range(-window, window + 1):
        if offset == 0:
            continue

        # Count (token_at_offset, is_top_activating) co-occurrences
        token_in_top: Counter[int] = Counter()
        token_total: Counter[int] = Counter()

        for pos in range(n_positions):
            neighbor_pos = pos + offset
            # Avoid crossing sequence boundaries
            if neighbor_pos < 0 or neighbor_pos >= n_positions:
                continue
            if pos % seq_len == 0 and offset < 0:
                continue
            if (pos + 1) % seq_len == 0 and offset > 0:
                continue
            # More precise boundary check
            pos_in_seq = pos % seq_len
            neighbor_in_seq = pos_in_seq + offset
            if neighbor_in_seq < 0 or neighbor_in_seq >= seq_len:
                continue

            tok = int(tokens_flat[neighbor_pos])
            token_total[tok] += 1
            if pos in top_set:
                token_in_top[tok] += 1

        # Find best token at this offset by PMI
        p_top = n_top / n_positions
        for tok, count_in_top in token_in_top.items():
            count_total = token_total[tok]
            if count_total < 3:
                continue
            p_tok = count_total / n_positions
            p_joint = count_in_top / n_positions
            if p_tok * p_top > 0:
                pmi = np.log2(p_joint / (p_tok * p_top) + 1e-30)
                if pmi > PMI_THRESHOLD:
                    coverage = count_in_top / n_top
                    best_coverage = max(best_coverage, coverage)

    return best_coverage


def _test_absence_rule(
    tokens_flat: torch.Tensor,
    feature_acts: torch.Tensor,
    seq_len: int,
    window: int,
    vocab_size: int,
    n_candidate_tokens: int = 100,
) -> float:
    """Test for absence rules: activation correlates with a token NOT being present.

    For the most frequent token types, check if their absence from the
    context window correlates with high feature activation.

    Args:
        tokens_flat: (n_positions,) token ids.
        feature_acts: (n_positions,) activation values.
        seq_len: sequence length.
        window: context window.
        vocab_size: vocabulary size.
        n_candidate_tokens: number of most frequent tokens to test.

    Returns:
        Best absence correlation (positive = absence predicts activation).
    """
    n_positions = tokens_flat.shape[0]
    acts_np = feature_acts.cpu().numpy().astype(np.float64)
    acts_mean = acts_np.mean()
    acts_std = acts_np.std()
    if acts_std < 1e-10:
        return 0.0

    # Find the most frequent tokens as candidates
    token_counts = Counter(tokens_flat.cpu().tolist())
    candidate_tokens = [tok for tok, _ in token_counts.most_common(n_candidate_tokens)]

    best_corr = 0.0

    for candidate_tok in candidate_tokens:
        # For each position, check if candidate_tok is absent from context
        absent = np.ones(n_positions, dtype=np.float64)
        tokens_np = tokens_flat.cpu().numpy()

        for pos in range(n_positions):
            pos_in_seq = pos % seq_len
            seq_start = pos - pos_in_seq
            ctx_start = max(0, pos_in_seq - window)
            ctx_end = min(seq_len, pos_in_seq + window + 1)
            context = tokens_np[seq_start + ctx_start : seq_start + ctx_end]
            if candidate_tok in context:
                absent[pos] = 0.0

        # Correlation between absence and activation
        absent_mean = absent.mean()
        absent_std = absent.std()
        if absent_std < 1e-10:
            continue

        corr = np.dot(absent - absent_mean, acts_np - acts_mean) / (
            n_positions * absent_std * acts_std
        )
        if corr > ABSENCE_CORR_THRESHOLD:
            best_corr = max(best_corr, corr)

    return best_corr


def _test_counting_rule(
    tokens_flat: torch.Tensor,
    feature_acts: torch.Tensor,
    seq_len: int,
    window: int,
    n_candidate_tokens: int = 50,
) -> float:
    """Test for counting rules: activation correlates with count of a token type.

    For the most frequent token types, check if the count of that token
    in the context window correlates with feature activation.

    Args:
        tokens_flat: (n_positions,) token ids.
        feature_acts: (n_positions,) activation values.
        seq_len: sequence length.
        window: context window.
        n_candidate_tokens: number of most frequent tokens to test.

    Returns:
        Best absolute correlation between token count and activation.
    """
    n_positions = tokens_flat.shape[0]
    acts_np = feature_acts.cpu().numpy().astype(np.float64)
    acts_std = acts_np.std()
    if acts_std < 1e-10:
        return 0.0
    acts_mean = acts_np.mean()

    token_counts = Counter(tokens_flat.cpu().tolist())
    candidate_tokens = [tok for tok, _ in token_counts.most_common(n_candidate_tokens)]

    best_corr = 0.0
    tokens_np = tokens_flat.cpu().numpy()

    for candidate_tok in candidate_tokens:
        counts = np.zeros(n_positions, dtype=np.float64)

        for pos in range(n_positions):
            pos_in_seq = pos % seq_len
            seq_start = pos - pos_in_seq
            ctx_start = max(0, pos_in_seq - window)
            ctx_end = min(seq_len, pos_in_seq + window + 1)
            context = tokens_np[seq_start + ctx_start : seq_start + ctx_end]
            counts[pos] = float(np.sum(context == candidate_tok))

        counts_std = counts.std()
        if counts_std < 1e-10:
            continue

        counts_mean = counts.mean()
        corr = abs(
            np.dot(counts - counts_mean, acts_np - acts_mean)
            / (n_positions * counts_std * acts_std)
        )
        if corr > COUNT_CORR_THRESHOLD:
            best_corr = max(best_corr, corr)

    return best_corr


@torch.no_grad()
def run_rule_descriptions(
    model,
    artifact=None,
    hook_name: str = "blocks.5.hook_resid_pre",
    n_features: int = 50,
    n_tokens: int = 500,
) -> list[EvalResult]:
    """Run rule-based feature description analysis.

    Args:
        model: HookedTransformer model.
        artifact: Artifact adapter with directions() and activations().
        hook_name: Hook point for activation extraction.
        n_features: Number of features to analyze.
        n_tokens: Number of tokens in the evaluation corpus.

    Returns:
        List with a single EvalResult containing rule coverage metrics.
    """
    if artifact is None:
        log("  WARNING: no artifact adapter provided, skipping rule descriptions")
        return []

    log(f"  Computing rule-based feature descriptions at {hook_name}...")

    # Step 1: get decoder directions to know feature count
    try:
        directions = artifact.directions()
    except (NotImplementedError, TypeError):
        log("  WARNING: artifact does not expose directions(), skipping")
        return []

    if directions is None or directions.numel() == 0:
        log("  WARNING: empty directions from artifact, skipping")
        return []

    total_features = directions.shape[0]
    n_features = min(n_features, total_features)
    log(f"    {total_features} total features, analyzing {n_features}")

    # Step 2: generate token corpus
    sample_text = "The quick brown fox jumps over the lazy dog. " * (n_tokens // 10)
    tokens = model.to_tokens(sample_text)
    tokens = tokens[:, :min(tokens.shape[1], n_tokens)]
    batch_size, seq_len = tokens.shape
    tokens_flat = tokens.reshape(-1)
    n_positions = tokens_flat.shape[0]
    vocab_size = model.cfg.d_vocab

    log(f"    {n_positions} positions (batch={batch_size}, seq={seq_len})")

    # Step 3: compute activations
    feature_acts = artifact.activations(model, tokens, hook_name)
    # (batch, seq, n_features) -> (n_positions, n_features)
    feature_acts = feature_acts.reshape(-1, feature_acts.shape[-1]).float()

    # Select features to analyze (highest variance for most informative results)
    feature_var = feature_acts.var(dim=0)
    if total_features > n_features:
        _, selected_indices = torch.topk(feature_var, n_features)
        selected_indices = selected_indices.sort().values
    else:
        selected_indices = torch.arange(total_features)

    log(f"    Analyzing {len(selected_indices)} features...")

    # Step 4: test each feature for each rule type
    skipgram_coverages = []
    absence_correlations = []
    counting_correlations = []
    rule_coverages = []
    per_feature_results = []

    for i, feat_idx in enumerate(selected_indices):
        feat_idx_int = int(feat_idx)
        acts = feature_acts[:, feat_idx_int]

        top_indices = _get_top_activating_positions(acts, TOP_K_POSITIONS)

        # Test skip-gram rules
        sg_coverage = _test_skipgram_rule(
            tokens_flat, acts, top_indices, seq_len, CONTEXT_WINDOW, vocab_size,
        )
        skipgram_coverages.append(sg_coverage)

        # Test absence rules
        abs_corr = _test_absence_rule(
            tokens_flat, acts, seq_len, CONTEXT_WINDOW, vocab_size,
        )
        absence_correlations.append(abs_corr)

        # Test counting rules
        cnt_corr = _test_counting_rule(
            tokens_flat, acts, seq_len, CONTEXT_WINDOW,
        )
        counting_correlations.append(cnt_corr)

        # Overall rule coverage: max of the three rule types
        # Skip-gram coverage is already a coverage fraction;
        # absence and counting are correlations, convert to pseudo-coverage
        rule_cov = max(sg_coverage, abs_corr, cnt_corr)
        rule_coverages.append(rule_cov)

        per_feature_results.append({
            "feature_idx": feat_idx_int,
            "skipgram_coverage": float(sg_coverage),
            "absence_correlation": float(abs_corr),
            "counting_correlation": float(cnt_corr),
            "best_rule_coverage": float(rule_cov),
            "best_rule_type": (
                "skipgram" if sg_coverage >= abs_corr and sg_coverage >= cnt_corr
                else "absence" if abs_corr >= cnt_corr
                else "counting"
            ),
        })

        if (i + 1) % 10 == 0:
            log(f"      {i + 1}/{len(selected_indices)} features done")

    # Step 5: aggregate
    mean_rule_coverage = float(np.mean(rule_coverages)) if rule_coverages else 0.0
    mean_skipgram = float(np.mean(skipgram_coverages)) if skipgram_coverages else 0.0
    mean_absence = float(np.mean(absence_correlations)) if absence_correlations else 0.0
    mean_counting = float(np.mean(counting_correlations)) if counting_correlations else 0.0

    frac_with_absence = float(np.mean(
        [1.0 if a > ABSENCE_CORR_THRESHOLD else 0.0 for a in absence_correlations]
    )) if absence_correlations else 0.0
    frac_with_counting = float(np.mean(
        [1.0 if c > COUNT_CORR_THRESHOLD else 0.0 for c in counting_correlations]
    )) if counting_correlations else 0.0
    frac_with_skipgram = float(np.mean(
        [1.0 if s > 0.0 else 0.0 for s in skipgram_coverages]
    )) if skipgram_coverages else 0.0

    passed = mean_rule_coverage > RULE_COVERAGE_THRESHOLD

    # Sort per-feature results by best rule coverage (descending)
    per_feature_results.sort(key=lambda x: x["best_rule_coverage"], reverse=True)

    log(f"    mean_rule_coverage={mean_rule_coverage:.4f}")
    log(f"    mean_skipgram={mean_skipgram:.4f}  mean_absence={mean_absence:.4f}  "
        f"mean_counting={mean_counting:.4f}")
    log(f"    frac_with_absence={frac_with_absence:.4f}  "
        f"frac_with_counting={frac_with_counting:.4f}  "
        f"frac_with_skipgram={frac_with_skipgram:.4f}")
    log(f"    [{'PASS' if passed else 'FAIL'}]")

    return [EvalResult(
        metric_id="EX6.rule_based_descriptions",
        value=mean_rule_coverage,
        n_samples=n_positions,
        instrument_info=INSTRUMENT_INFO,
        metadata={
            "hook_name": hook_name,
            "n_features_analyzed": len(selected_indices),
            "n_features_total": total_features,
            "n_positions": n_positions,
            "mean_rule_coverage": mean_rule_coverage,
            "mean_skipgram_coverage": mean_skipgram,
            "mean_absence_correlation": mean_absence,
            "mean_counting_correlation": mean_counting,
            "frac_with_absence_rules": frac_with_absence,
            "frac_with_counting_rules": frac_with_counting,
            "frac_with_skipgram_rules": frac_with_skipgram,
            "top_features": per_feature_results[:10],
            "passed": passed,
            "threshold": RULE_COVERAGE_THRESHOLD,
            "context_window": CONTEXT_WINDOW,
            "pmi_threshold": PMI_THRESHOLD,
            "absence_corr_threshold": ABSENCE_CORR_THRESHOLD,
            "count_corr_threshold": COUNT_CORR_THRESHOLD,
        },
    )]


def main():
    parser = parse_common_args("EX6: Rule-Based Feature Descriptions")
    parser.add_argument("--hook", default="blocks.5.hook_resid_pre", help="Hook point")
    parser.add_argument("--n-tokens", type=int, default=500, help="Tokens to evaluate")
    parser.add_argument("--n-features", type=int, default=50, help="Features to analyze")
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
    log("EX6: RULE-BASED FEATURE DESCRIPTIONS")
    log("=" * 60)

    results = run_rule_descriptions(
        model, artifact=artifact, hook_name=args.hook,
        n_features=args.n_features, n_tokens=args.n_tokens,
    )

    out = args.out or "EX6_rule_descriptions.json"
    save_results(results, out)
    log(f"\nDone. {len(results)} results.")


if __name__ == "__main__":
    main()
