"""Output-Centric Description Score (Evaluation EX18)
Paper: Gur-Arieh, Mayan, Agassy, Geiger, Geva (2025). ACL 2025. arXiv:2501.08319
=============================================
Instrument:     EX18 --- Output-Centric Description Score
Categories:     evaluation
Validity layer: Internal
Criteria:       I2 Compositional Sufficiency, C5 Convergent Validity
Establishes:    Whether SAE features have consistent input-centric and
                output-centric descriptions. Features with high encoding
                validity but low execution validity are correlational but
                not causal (I2 failure).
Requires:       Model, artifact adapter
=============================================

Implements the encoding/execution validity decomposition from
Gur-Arieh, Mayan, Agassy, Geiger, Geva (ACL 2025, arXiv:2501.08319).

For each SAE feature, computes two descriptions:
  - Input-centric (encoding): top tokens that maximally activate the feature
    (what the feature detects in the input).
  - Output-centric (execution): top tokens promoted/suppressed when
    projecting the feature's decoder direction through the unembedding
    matrix W_U (what the feature causes in the output).

The encoding_execution_agreement measures the overlap between these two
token sets. High agreement means the feature both detects and promotes
the same concept. Low agreement (high encoding, low execution) reveals
correlational features that fire on a concept but do not causally drive
it in the output --- an I2 failure.

The 2x2 decomposition classifies each feature as:
  - HH (high encoding, high execution): well-understood causal feature
  - HL (high encoding, low execution): correlational, I2 failure
  - LH (low encoding, high execution): output-only, encoding gap
  - LL (low encoding, low execution): weak/dead feature

Core logic:
1. Get decoder directions (W_dec) from the artifact.
2. Project each decoder direction through W_U to get output-centric
   logits: vocab_logits = W_dec[i] @ W_U (top-k = promoted tokens).
3. Run model on corpus to collect feature activations at each position.
4. For each feature, rank tokens by mean activation (top-k = input-centric).
5. Compute Jaccard overlap between input-centric and output-centric top-k.
6. Aggregate: encoding_execution_agreement = mean overlap across features.

Pass condition: encoding_execution_agreement > 0.2

Usage:
    uv run python 116_output_centric.py --model gpt2 --device cpu
    uv run python 116_output_centric.py --artifact-path <release> --sae-id <id>
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
    name="Output-Centric Description Score",
    paper_ref="Gur-Arieh, Mayan, Agassy, Geiger, Geva; ACL 2025",
    paper_cite=(
        "Gur-Arieh, Mayan, Agassy, Geiger, Geva 2025, "
        "Encoding and Execution Validity of SAE Feature Descriptions "
        "(ACL 2025, arXiv:2501.08319)"
    ),
    description=(
        "Compares input-centric descriptions (top activating tokens) against "
        "output-centric descriptions (decoder direction projected through W_U) "
        "to detect correlational features that lack causal execution validity"
    ),
    category="evaluation",
    tier="established",
    origin="external",
)

AGREEMENT_THRESHOLD = 0.2
ENCODING_THRESHOLD = 0.3
EXECUTION_THRESHOLD = 0.3


def _get_output_centric_tokens(
    decoder_directions: torch.Tensor,
    W_U: torch.Tensor,
    top_k: int,
) -> list[set[int]]:
    """For each feature, project its decoder direction through W_U.

    Returns a list of sets, one per feature, each containing the top-k
    token indices most promoted by that feature direction.

    Args:
        decoder_directions: (n_features, d_model) decoder weight matrix.
        W_U: (d_model, d_vocab) unembedding matrix.
        top_k: number of top tokens to select per feature.

    Returns:
        List of sets of token indices, one per feature.
    """
    # vocab_logits: (n_features, d_vocab) = decoder_directions @ W_U
    vocab_logits = decoder_directions @ W_U
    # Take top-k by absolute value (promoted or suppressed)
    _, top_indices = vocab_logits.abs().topk(top_k, dim=-1)
    return [set(row.tolist()) for row in top_indices]


def _get_input_centric_tokens(
    feature_acts: torch.Tensor,
    token_ids: torch.Tensor,
    top_k: int,
) -> list[set[int]]:
    """For each feature, find the top-k tokens by mean activation.

    Args:
        feature_acts: (n_positions, n_features) flattened activations.
        token_ids: (n_positions,) token ids at each position.
        top_k: number of top tokens to select per feature.

    Returns:
        List of sets of token indices, one per feature.
    """
    n_features = feature_acts.shape[1]
    # Unique tokens
    unique_tokens = token_ids.unique()
    n_unique = len(unique_tokens)

    # Build per-token mean activation: (n_unique, n_features)
    mean_acts = torch.zeros(n_unique, n_features, device=feature_acts.device)
    for i, tok in enumerate(unique_tokens):
        mask = token_ids == tok
        mean_acts[i] = feature_acts[mask].mean(dim=0)

    # For each feature, find top-k tokens by mean activation magnitude
    k = min(top_k, n_unique)
    _, top_indices = mean_acts.abs().T.topk(k, dim=-1)  # (n_features, k)

    result = []
    for feat_idx in range(n_features):
        token_set = set(unique_tokens[top_indices[feat_idx]].tolist())
        result.append(token_set)
    return result


def _jaccard(a: set, b: set) -> float:
    """Jaccard similarity between two sets."""
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def _classify_feature(
    encoding_score: float,
    execution_score: float,
    enc_threshold: float,
    exec_threshold: float,
) -> str:
    """Classify feature into the 2x2 encoding x execution grid."""
    high_enc = encoding_score >= enc_threshold
    high_exec = execution_score >= exec_threshold
    if high_enc and high_exec:
        return "HH"
    if high_enc and not high_exec:
        return "HL"
    if not high_enc and high_exec:
        return "LH"
    return "LL"


@torch.no_grad()
def run_output_centric(
    model,
    tasks=None,
    n_prompts: int = 50,
    artifact=None,
    hook_name: str | None = None,
    top_k: int = 50,
) -> list[EvalResult]:
    """Compute output-centric description agreement for SAE features.

    For each feature, compares the input-centric description (top activating
    tokens) against the output-centric description (VocabProj of the decoder
    direction through W_U). Measures agreement via Jaccard overlap.

    Args:
        model: HookedTransformer instance.
        tasks: unused (kept for dispatch compatibility).
        n_prompts: number of text segments for activation collection.
        artifact: ArtifactAdapter with directions() and activations().
        hook_name: hook point for activation collection.
        top_k: number of top tokens to compare per feature.

    Returns:
        List with one EvalResult for EX18.output_centric_description.
    """
    if hook_name is None:
        hook_name = "blocks.5.hook_resid_pre"

    if artifact is None:
        log("  WARNING: no artifact adapter provided, skipping output-centric description")
        return []

    log(f"  Computing output-centric description score at {hook_name}...")

    # Step 1: get decoder directions from the artifact
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
    d_model = directions.shape[1]
    log(f"    {n_features} features, d_model={d_model}")

    # Step 2: get W_U (unembedding matrix) from the model
    W_U = model.W_U.detach().float()  # (d_model, d_vocab)
    log(f"    W_U shape: {tuple(W_U.shape)}")

    # Step 3: compute output-centric tokens (VocabProj)
    output_tokens = _get_output_centric_tokens(directions, W_U, top_k)
    log(f"    Computed output-centric top-{top_k} for {n_features} features")

    # Step 4: collect feature activations on a corpus
    sample_text = (
        "The quick brown fox jumps over the lazy dog. "
        "In a shocking turn of events, the president announced a new policy. "
        "The capital of France is Paris. London is the capital of England. "
        "Machine learning models can be trained on large datasets. "
        "The cat sat on the mat and looked out the window. "
    )
    # Repeat to get enough tokens
    n_tokens = 512 * n_prompts // 50  # scale with n_prompts
    n_tokens = max(256, min(n_tokens, 2048))
    corpus = sample_text * (n_tokens // 20 + 1)
    tokens = model.to_tokens(corpus)
    tokens = tokens[:, :min(tokens.shape[1], n_tokens)]

    feature_acts = artifact.activations(model, tokens, hook_name)
    # Flatten: (batch, seq, n_features) -> (n_positions, n_features)
    feature_acts = feature_acts.reshape(-1, feature_acts.shape[-1]).float()
    n_positions = feature_acts.shape[0]

    # Get the token ids for each position
    token_ids = tokens.reshape(-1)[:n_positions]

    log(f"    {n_positions} positions, computing input-centric tokens...")

    # Step 5: compute input-centric tokens
    input_tokens = _get_input_centric_tokens(feature_acts, token_ids, top_k)

    # Step 6: compute per-feature agreement (Jaccard overlap)
    per_feature = []
    overlaps = []
    encoding_scores = []
    execution_scores = []

    for i in range(n_features):
        overlap = _jaccard(input_tokens[i], output_tokens[i])
        overlaps.append(overlap)

        # Encoding score: how peaked the feature's activation distribution is
        # (high = the feature clearly responds to specific tokens)
        feat_acts = feature_acts[:, i]
        act_max = feat_acts.abs().max().item()
        act_mean = feat_acts.abs().mean().item()
        encoding_score = (act_max / (act_mean + 1e-10)) if act_mean > 1e-10 else 0.0
        # Normalize to [0, 1] range: a ratio > 10 is considered high encoding
        encoding_score = min(encoding_score / 10.0, 1.0)
        encoding_scores.append(encoding_score)

        # Execution score: how peaked the output logit distribution is
        # (high = the feature clearly promotes specific tokens)
        vocab_logits = (directions[i] @ W_U).abs()
        exec_max = vocab_logits.max().item()
        exec_mean = vocab_logits.mean().item()
        execution_score = (exec_max / (exec_mean + 1e-10)) if exec_mean > 1e-10 else 0.0
        execution_score = min(execution_score / 10.0, 1.0)
        execution_scores.append(execution_score)

        classification = _classify_feature(
            encoding_score, execution_score, ENCODING_THRESHOLD, EXECUTION_THRESHOLD,
        )

        per_feature.append({
            "feature_idx": i,
            "overlap": overlap,
            "encoding_score": encoding_score,
            "execution_score": execution_score,
            "classification": classification,
        })

    overlaps_arr = np.array(overlaps)
    encoding_arr = np.array(encoding_scores)
    execution_arr = np.array(execution_scores)

    encoding_execution_agreement = float(overlaps_arr.mean())
    median_overlap = float(np.median(overlaps_arr))
    mean_encoding = float(encoding_arr.mean())
    mean_execution = float(execution_arr.mean())

    # 2x2 decomposition counts
    classifications = [f["classification"] for f in per_feature]
    decomposition = {
        "HH": classifications.count("HH"),
        "HL": classifications.count("HL"),
        "LH": classifications.count("LH"),
        "LL": classifications.count("LL"),
    }
    decomposition_fractions = {
        k: v / max(n_features, 1) for k, v in decomposition.items()
    }

    passed = encoding_execution_agreement > AGREEMENT_THRESHOLD

    log(f"    encoding_execution_agreement = {encoding_execution_agreement:.4f}")
    log(f"    median_overlap = {median_overlap:.4f}")
    log(f"    mean_encoding = {mean_encoding:.4f}")
    log(f"    mean_execution = {mean_execution:.4f}")
    log(f"    2x2 decomposition: {decomposition}")
    log(f"    [{'PASS' if passed else 'FAIL'}]")

    # Top features by overlap (best agreement)
    sorted_by_overlap = sorted(per_feature, key=lambda x: x["overlap"], reverse=True)
    top_features = sorted_by_overlap[:10]

    # Worst features (potential I2 failures: high encoding, low execution)
    hl_features = [f for f in per_feature if f["classification"] == "HL"]
    hl_features_sorted = sorted(hl_features, key=lambda x: x["encoding_score"], reverse=True)
    worst_i2_features = hl_features_sorted[:10]

    return [EvalResult(
        metric_id="EX18.output_centric_description",
        value=encoding_execution_agreement,
        n_samples=n_positions,
        instrument_info=INSTRUMENT_INFO,
        metadata={
            "hook_name": hook_name,
            "n_features": n_features,
            "top_k": top_k,
            "encoding_execution_agreement": encoding_execution_agreement,
            "median_overlap": median_overlap,
            "mean_encoding_score": mean_encoding,
            "mean_execution_score": mean_execution,
            "decomposition_2x2": decomposition,
            "decomposition_2x2_fractions": decomposition_fractions,
            "top_agreement_features": top_features,
            "worst_i2_failure_features": worst_i2_features,
            "passed": passed,
            "threshold": AGREEMENT_THRESHOLD,
            "encoding_threshold": ENCODING_THRESHOLD,
            "execution_threshold": EXECUTION_THRESHOLD,
        },
    )]


def main():
    parser = parse_common_args("EX18: Output-Centric Description Score")
    parser.add_argument("--hook", default=None, help="Hook point")
    parser.add_argument("--top-k", type=int, default=50,
                        help="Top-k tokens to compare per feature")
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
            hook_point=args.hook or "blocks.5.hook_resid_pre",
        )

    log("=" * 60)
    log("EX18: OUTPUT-CENTRIC DESCRIPTION SCORE")
    log("=" * 60)

    results = run_output_centric(
        model,
        artifact=artifact,
        hook_name=args.hook,
        top_k=args.top_k,
        n_prompts=args.n_prompts,
    )

    out = args.out or "116_output_centric.json"
    save_results(results, out)
    log(f"\nDone. {len(results)} results.")


if __name__ == "__main__":
    main()
