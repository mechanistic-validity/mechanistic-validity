"""Metric: Neuronpedia Agreement --- cross-reference Neuronpedia auto-interp with weight-based analysis

Paper: Community infrastructure (neuronpedia.org). No single paper.

Neuronpedia provides the largest public database of SAE feature descriptions
and automated interpretability scores. This metric cross-references those
descriptions with weight-based feature semantics to measure external
convergent validity (C5) of feature descriptions across independent methods.

Neuronpedia Agreement (Evaluation EX24)
=============================================
Instrument:     EX24 --- Neuronpedia Agreement
Categories:     evaluation
Validity layer: Construct
Criteria:       C5 Convergent Validity
Establishes:    Whether Neuronpedia's automated feature descriptions agree
                with weight-based top-k token analysis, providing external
                convergent validity for feature semantics
Requires:       CPU or GPU, model, network access to neuronpedia.org API
=============================================

Core logic:
1. For each feature, compute weight-based top-k tokens by projecting
   the decoder direction through the unembedding matrix (W_dec @ W_U).
2. Query Neuronpedia's API for the same feature to get its auto-interp
   description and top activating tokens.
3. Compute token-level overlap (Jaccard) between weight-based top-k
   and Neuronpedia's top activating tokens.
4. Report the mean agreement across sampled features.

Pass condition: neuronpedia_weight_agreement > 0.2

Usage:
    uv run python 126_neuronpedia_agreement.py --model gpt2 --device cpu
    uv run python 126_neuronpedia_agreement.py --neuronpedia-sae-id res-jb --n-features 50
"""

import json
import time
import urllib.error
import urllib.request

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
    name="Neuronpedia Agreement",
    paper_ref="Community infrastructure (neuronpedia.org)",
    paper_cite=(
        "Neuronpedia (neuronpedia.org) — community infrastructure for SAE "
        "feature visualization, automated interpretability, and annotation"
    ),
    description=(
        "Cross-references Neuronpedia's automated feature descriptions and "
        "top activating tokens with weight-based top-k token analysis "
        "(decoder @ unembedding) to measure convergent validity (C5) of "
        "feature semantics across independent methods."
    ),
    category="evaluation",
    tier="community",
    origin="external",
)

AGREEMENT_THRESHOLD = 0.2

NEURONPEDIA_API_BASE = "https://www.neuronpedia.org/api/feature"

# Rate-limit: be polite to the Neuronpedia API
_REQUEST_DELAY_S = 0.5


def _fetch_neuronpedia_feature(
    model_id: str,
    layer: int,
    feature_index: int,
) -> dict | None:
    """Fetch a single feature's data from the Neuronpedia API.

    Returns the parsed JSON response, or None if the request fails.
    The response typically contains 'description', 'activations',
    'top_logits', and other metadata.
    """
    url = f"{NEURONPEDIA_API_BASE}/{model_id}/{layer}/{feature_index}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "mechval/0.1 (https://github.com/mechanistic-validity)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
        log(f"    Neuronpedia API error for feature {feature_index}: {e}")
        return None


def _extract_neuronpedia_tokens(
    feature_data: dict,
    tokenizer,
    top_k: int = 20,
) -> set[int]:
    """Extract top token IDs from Neuronpedia feature data.

    The Neuronpedia API returns:
    - pos_str: list of token strings with highest positive logit contribution
    - pos_values: corresponding logit values
    - activations[*].tokens: list of token strings per activation example
    - activations[*].values: per-token activation values

    We extract token IDs from both logit-space (pos_str) and activation-space
    (top activating tokens) sources to build a combined set for comparison
    against weight-based top-k tokens.
    """
    token_ids: set[int] = set()

    # Source 1: pos_str — tokens this feature most promotes in logit space
    # This is the primary comparison target since weight-based analysis
    # also operates in logit space (decoder @ unembedding).
    pos_str = feature_data.get("pos_str")
    if isinstance(pos_str, list):
        for token_str in pos_str[:top_k]:
            if isinstance(token_str, str) and token_str:
                encoded = tokenizer.encode(token_str, add_special_tokens=False)
                if encoded:
                    token_ids.add(encoded[0])

    # Source 2: activations — top activating contexts
    # Each activation entry has 'tokens' (list of strings) and 'values'
    # (list of floats). We take the token at the max-activation position.
    activations = feature_data.get("activations")
    if isinstance(activations, list):
        for act_entry in activations[:top_k]:
            if not isinstance(act_entry, dict):
                continue
            tokens_list = act_entry.get("tokens")
            values_list = act_entry.get("values")
            if not isinstance(tokens_list, list) or not tokens_list:
                continue
            if isinstance(values_list, list) and len(values_list) == len(tokens_list):
                # Take the token with highest activation value
                max_idx = int(np.argmax(values_list))
                tok_str = tokens_list[max_idx]
            else:
                # Fallback: use the token at maxValueTokenIndex or last position
                max_idx = act_entry.get("maxValueTokenIndex")
                if max_idx is not None and 0 <= max_idx < len(tokens_list):
                    tok_str = tokens_list[max_idx]
                else:
                    continue
            if isinstance(tok_str, str) and tok_str:
                encoded = tokenizer.encode(tok_str, add_special_tokens=False)
                if encoded:
                    token_ids.add(encoded[0])

    return token_ids


def _extract_description(feature_data: dict) -> str:
    """Extract the auto-interp description from Neuronpedia feature data.

    Neuronpedia stores descriptions in an 'explanations' array, each entry
    having a 'description' field. Returns the first available description,
    or empty string if none found.
    """
    explanations = feature_data.get("explanations")
    if isinstance(explanations, list):
        for expl in explanations:
            if isinstance(expl, dict):
                desc = expl.get("description", "")
                if desc:
                    return desc
    return ""


def _compute_weight_topk(
    W_dec: torch.Tensor,
    W_U: torch.Tensor,
    feature_indices: list[int],
    top_k: int = 20,
) -> dict[int, set[int]]:
    """For each feature index, project decoder direction through unembedding.

    Returns a dict mapping feature_index -> set of top-k token IDs.
    """
    result = {}
    W_dec_f = W_dec.float()
    W_U_f = W_U.float()
    for feat_idx in feature_indices:
        if feat_idx >= W_dec_f.shape[0]:
            continue
        logit_vec = W_dec_f[feat_idx] @ W_U_f  # (d_vocab,)
        _, topk_ids = logit_vec.topk(top_k)
        result[feat_idx] = set(topk_ids.tolist())
    return result


def _jaccard(a: set, b: set) -> float:
    """Jaccard similarity between two sets."""
    if not a and not b:
        return 0.0
    union = len(a | b)
    if union == 0:
        return 0.0
    return len(a & b) / union


@torch.no_grad()
def run_neuronpedia_agreement(
    model,
    tasks: list[str] | None = None,
    n_features: int = 100,
    artifact=None,
    hook_name: str | None = None,
    neuronpedia_model_id: str = "gpt2-small",
    neuronpedia_sae_id: str | None = None,
    top_k: int = 20,
) -> list[EvalResult]:
    """Cross-reference Neuronpedia auto-interp with weight-based analysis.

    For sampled features, computes weight-based top-k tokens (decoder @
    unembedding) and fetches Neuronpedia's top tokens for the same feature.
    Reports the fraction of features where these two descriptions agree
    (Jaccard overlap > 0) and the mean Jaccard overlap.

    Args:
        model: HookedTransformer model.
        tasks: Unused (included for registry dispatch compatibility).
        n_features: Number of features to sample and compare.
        artifact: Optional artifact adapter with .directions() method
            providing W_dec. If None, attempts to use model.W_out or
            a default SAE.
        hook_name: Hook point name (used to determine the SAE layer).
        neuronpedia_model_id: Neuronpedia model identifier
            (default: "gpt2-small").
        neuronpedia_sae_id: Neuronpedia SAE identifier
            (e.g., "res-jb"). If None, derived from hook_name.
        top_k: Number of top tokens for comparison per feature.

    Returns:
        List of EvalResult with neuronpedia_weight_agreement scores.
    """
    tokenizer = model.tokenizer

    # Determine layer for Neuronpedia queries
    if hook_name is not None:
        # Extract layer number from hook name like "blocks.5.hook_mlp_out"
        parts = hook_name.split(".")
        layer = int(parts[1]) if len(parts) >= 2 and parts[1].isdigit() else 0
    else:
        layer = model.cfg.n_layers // 2

    # Construct SAE ID for Neuronpedia API path
    if neuronpedia_sae_id is not None:
        sae_layer_id = f"{layer}-{neuronpedia_sae_id}"
    else:
        sae_layer_id = str(layer)

    # Get decoder weight matrix
    if artifact is not None:
        W_dec = artifact.directions()
        if W_dec.ndim == 3:
            W_dec = W_dec.mean(dim=0)
        total_features = W_dec.shape[0]
    else:
        # Without an artifact, we cannot compute weight-based descriptions
        log("  WARNING: no artifact provided; weight-based analysis requires "
            "an artifact adapter with decoder directions")
        return []

    W_U = model.W_U.detach()  # (d_model, d_vocab)

    # Sample features to evaluate
    actual_n = min(n_features, total_features)
    if actual_n < total_features:
        rng = np.random.default_rng(42)
        feature_indices = sorted(rng.choice(total_features, size=actual_n, replace=False).tolist())
    else:
        feature_indices = list(range(total_features))

    log(f"  Neuronpedia agreement: {len(feature_indices)} features, "
        f"model={neuronpedia_model_id}, sae_layer_id={sae_layer_id}")

    # Step 1: Compute weight-based top-k tokens
    weight_topk = _compute_weight_topk(W_dec, W_U, feature_indices, top_k=top_k)
    log(f"  Computed weight-based descriptions for {len(weight_topk)} features")

    # Step 2: Fetch Neuronpedia data and compute per-feature agreement
    agreements = []
    descriptions = []
    n_api_success = 0
    n_api_fail = 0
    per_feature_details = []

    for feat_idx in feature_indices:
        if feat_idx not in weight_topk:
            continue

        # Rate limit
        time.sleep(_REQUEST_DELAY_S)

        feature_data = _fetch_neuronpedia_feature(
            neuronpedia_model_id, sae_layer_id, feat_idx,
        )

        if feature_data is None:
            n_api_fail += 1
            continue

        n_api_success += 1

        # Extract Neuronpedia's tokens
        np_tokens = _extract_neuronpedia_tokens(feature_data, tokenizer, top_k=top_k)
        w_tokens = weight_topk[feat_idx]

        # Extract description from explanations array
        desc = _extract_description(feature_data)

        if not np_tokens:
            # Neuronpedia returned data but no usable tokens
            per_feature_details.append({
                "feature_index": feat_idx,
                "agreement": 0.0,
                "neuronpedia_tokens": 0,
                "weight_tokens": len(w_tokens),
                "description": desc[:200] if desc else "",
            })
            agreements.append(0.0)
            continue

        overlap = _jaccard(w_tokens, np_tokens)
        agreements.append(overlap)
        descriptions.append(desc)

        per_feature_details.append({
            "feature_index": feat_idx,
            "agreement": overlap,
            "neuronpedia_tokens": len(np_tokens),
            "weight_tokens": len(w_tokens),
            "overlap_count": len(w_tokens & np_tokens),
            "description": desc[:200] if desc else "",
        })

    if not agreements:
        log("  No features successfully compared; check API access and feature indices")
        return []

    # Step 3: Aggregate
    mean_agreement = float(np.mean(agreements))
    median_agreement = float(np.median(agreements))
    std_agreement = float(np.std(agreements))
    frac_nonzero = float(np.mean(np.array(agreements) > 0))
    frac_above_threshold = float(np.mean(np.array(agreements) > AGREEMENT_THRESHOLD))
    passed = mean_agreement > AGREEMENT_THRESHOLD

    log(f"  API: {n_api_success} success, {n_api_fail} fail")
    log(f"  Agreement: mean={mean_agreement:.4f}  median={median_agreement:.4f}  "
        f"std={std_agreement:.4f}  frac>0={frac_nonzero:.3f}  "
        f"[{'PASS' if passed else 'FAIL'}]")

    results = [EvalResult(
        metric_id="EX24.neuronpedia_weight_agreement",
        value=mean_agreement,
        n_samples=len(agreements),
        instrument_info=INSTRUMENT_INFO,
        metadata={
            "neuronpedia_weight_agreement": mean_agreement,
            "median_agreement": median_agreement,
            "std_agreement": std_agreement,
            "frac_nonzero_overlap": frac_nonzero,
            "frac_above_threshold": frac_above_threshold,
            "passed": passed,
            "threshold": AGREEMENT_THRESHOLD,
            "n_features_sampled": len(feature_indices),
            "n_features_compared": len(agreements),
            "n_api_success": n_api_success,
            "n_api_fail": n_api_fail,
            "top_k": top_k,
            "neuronpedia_model_id": neuronpedia_model_id,
            "neuronpedia_sae_layer_id": sae_layer_id,
            "layer": layer,
            "hook_name": hook_name or f"blocks.{layer}.hook_mlp_out",
            "per_feature": per_feature_details[:50],  # cap metadata size
        },
    )]

    return results


def main():
    parser = parse_common_args("EX24: Neuronpedia Agreement")
    parser.add_argument("--n-features", type=int, default=100,
                        help="Number of features to sample (default: 100)")
    parser.add_argument("--neuronpedia-model-id", default="gpt2-small",
                        help="Neuronpedia model ID (default: gpt2-small)")
    parser.add_argument("--neuronpedia-sae-id", default=None,
                        help="Neuronpedia SAE ID suffix (e.g., res-jb)")
    parser.add_argument("--hook-name", default=None,
                        help="Hook point name (default: mid-layer MLP out)")
    parser.add_argument("--top-k", type=int, default=20,
                        help="Top-k tokens per feature (default: 20)")
    parser.add_argument("--artifact-type", default="sae",
                        choices=["sae", "transcoder", "crosscoder", "factor_bank"],
                        help="Artifact adapter type")
    parser.add_argument("--artifact-path", default=None,
                        help="Path or release ID for artifact")
    parser.add_argument("--sae-id", default=None,
                        help="SAE ID (for SAELens artifacts)")
    args = parser.parse_args()

    model = load_model(args.model, args.device)

    artifact = _load_artifact(
        args.artifact_type, args.artifact_path,
        args.sae_id, args.hook_name,
    )

    log("=" * 60)
    log("EX24: NEURONPEDIA AGREEMENT")
    log("=" * 60)

    results = run_neuronpedia_agreement(
        model,
        n_features=args.n_features,
        artifact=artifact,
        hook_name=args.hook_name,
        neuronpedia_model_id=args.neuronpedia_model_id,
        neuronpedia_sae_id=args.neuronpedia_sae_id,
        top_k=args.top_k,
    )

    out = args.out or "126_neuronpedia_agreement.json"
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
