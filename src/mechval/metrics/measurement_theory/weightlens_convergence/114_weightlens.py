"""WeightLens Convergence (Measurement M8)
Paper: Golimblevskaia, Jain, Puri, Ibrahim, Samek, Lapuschkin (2026). ICLR 2026. arXiv:2510.14936
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     M08 — WeightLens Convergence
Categories:     measurement
Validity layer: Measurement
Criteria:       C5 Convergent Validity
Establishes:    Whether weight-based and activation-based feature descriptions
                agree — a construct validity check for SAE feature identity
Requires:       One artifact adapter with decoder directions
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Based on Golimblevskaia, Jain, Puri, Ibrahim, Samek, Lapuschkin;
ICLR 2026 (arXiv:2510.14936): SAE features can be described two ways:

  (a) Weight-based: project decoder weight vectors through the unembedding
      matrix (W_dec @ W_U) to get top-k promoted tokens per feature.
  (b) Activation-based: collect top activating tokens from running prompts
      through the model and encoding with the SAE.

Features where these two descriptions diverge have low construct validity —
the feature's structural identity (what it promotes in logit space) does not
match its functional identity (what inputs it fires on).

Method:
    1. Compute weight-based descriptions: for each feature, project its
       decoder direction through the model's unembedding to get a logit
       vector, then take top-k tokens.
    2. Compute activation-based descriptions: run prompts through the model,
       encode at the hook point, and for each feature track which tokens
       produce the highest activations.
    3. Measure agreement: Jaccard overlap of the two top-k token sets,
       averaged over features.

Pass condition: weight_activation_agreement > 0.3

Usage:
    uv run python 114_weightlens.py --artifact-path <release> --sae-id <id>
    uv run python 114_weightlens.py --device cpu
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
    name="WeightLens Convergence (Golimblevskaia et al. 2026)",
    paper_ref="Golimblevskaia et al., ICLR 2026 (arXiv:2510.14936)",
    paper_cite="Golimblevskaia, Jain, Puri, Ibrahim, Samek, Lapuschkin 2026",
    description=(
        "Tests whether weight-based feature descriptions (decoder @ unembedding) "
        "agree with activation-based descriptions (top activating tokens) — "
        "a construct validity check for SAE feature identity"
    ),
    category="measurement",
    tier="measurement_theory",
    origin="established",
)

AGREEMENT_THRESHOLD = 0.3


def compute_weight_topk(
    W_dec: torch.Tensor,
    W_U: torch.Tensor,
    top_k: int = 50,
) -> list[set[int]]:
    """For each feature, project decoder direction through unembedding to get top-k tokens.

    Args:
        W_dec: (n_features, d_model) decoder weight matrix.
        W_U: (d_model, d_vocab) unembedding matrix.
        top_k: number of top tokens per feature.

    Returns:
        List of sets, one per feature, each containing top-k token indices.
    """
    # (n_features, d_vocab) = (n_features, d_model) @ (d_model, d_vocab)
    logit_projections = W_dec.float() @ W_U.float()
    # Take top-k per feature
    _, topk_indices = logit_projections.topk(top_k, dim=-1)
    return [set(row.tolist()) for row in topk_indices]


@torch.no_grad()
def collect_activation_topk(
    model,
    artifact,
    prompts,
    hook_name: str,
    n_features: int,
    top_k: int = 50,
) -> list[set[int]]:
    """For each feature, collect the tokens that produce the highest activations.

    Tracks per-feature top-k activating token IDs across all prompts.

    Args:
        model: HookedTransformer model.
        artifact: Artifact adapter with .activations() method.
        prompts: List of prompt objects.
        hook_name: Hook point for activation collection.
        n_features: Number of features in the artifact.
        top_k: Number of top tokens to track per feature.

    Returns:
        List of sets, one per feature, each containing top-k token indices
        by activation magnitude.
    """
    # Track (activation, token_id) per feature using a simple buffer
    # We keep the top-k across all positions seen.
    feature_top_acts = [[] for _ in range(n_features)]

    for prompt in prompts:
        tokens = model.to_tokens(prompt.text)  # (1, seq_len)
        token_ids = tokens[0].tolist()

        acts = artifact.activations(model, tokens, hook_name)  # (1, seq, n_features)
        acts_2d = acts[0]  # (seq, n_features)

        for pos_idx, tid in enumerate(token_ids):
            for feat_idx in range(n_features):
                val = acts_2d[pos_idx, feat_idx].item()
                if val > 0:
                    feature_top_acts[feat_idx].append((val, tid))

    # Reduce each feature to top-k unique token IDs by max activation
    result = []
    for feat_idx in range(n_features):
        entries = feature_top_acts[feat_idx]
        if not entries:
            result.append(set())
            continue
        # Group by token ID, keep max activation per token
        token_max: dict[int, float] = {}
        for val, tid in entries:
            if tid not in token_max or val > token_max[tid]:
                token_max[tid] = val
        # Sort by activation descending, take top-k
        sorted_tokens = sorted(token_max.items(), key=lambda x: x[1], reverse=True)
        result.append({tid for tid, _ in sorted_tokens[:top_k]})

    return result


def jaccard(a: set, b: set) -> float:
    """Jaccard similarity between two sets."""
    if not a and not b:
        return 0.0
    union = len(a | b)
    if union == 0:
        return 0.0
    return len(a & b) / union


@torch.no_grad()
def run_weightlens_convergence(
    model,
    tasks: list[str] | None = None,
    n_prompts: int = 50,
    artifact=None,
    hook_name: str | None = None,
    top_k: int = 50,
) -> list[EvalResult]:
    """Run WeightLens convergence analysis.

    Compares weight-based feature descriptions (decoder @ unembedding) with
    activation-based descriptions (top activating tokens) and measures
    agreement as mean Jaccard overlap.

    Args:
        model: HookedTransformer model.
        tasks: List of task names to evaluate on. Defaults to CIRCUIT_TASKS.
        n_prompts: Number of prompts per task.
        artifact: Artifact adapter with .directions() and .activations().
        hook_name: Hook point override (defaults to artifact's hook point).
        top_k: Number of top tokens for comparison.

    Returns:
        List of EvalResult with weight_activation_agreement scores.
    """
    if artifact is None:
        log("  WARNING: artifact adapter required, skipping weightlens convergence")
        return []

    if tasks is None:
        tasks = CIRCUIT_TASKS

    tokenizer = model.tokenizer
    results = []

    effective_hook = hook_name or artifact.manifest.hook_point
    if not effective_hook:
        effective_hook = "blocks.0.hook_resid_pre"

    # Step 1: Weight-based descriptions
    W_dec = artifact.directions()
    if W_dec.ndim == 3:
        W_dec = W_dec.mean(dim=0)
    n_features = W_dec.shape[0]

    W_U = model.W_U.detach()  # (d_model, d_vocab)
    weight_topk = compute_weight_topk(W_dec, W_U, top_k=top_k)
    log(f"  Computed weight-based descriptions for {n_features} features")

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

        # Step 2: Activation-based descriptions
        activation_topk = collect_activation_topk(
            model, artifact, prompts, effective_hook, n_features, top_k=top_k,
        )

        # Step 3: Compute Jaccard overlap per feature, then average
        overlaps = []
        active_features = 0
        for feat_idx in range(n_features):
            w_set = weight_topk[feat_idx]
            a_set = activation_topk[feat_idx]
            # Only score features that have activation-based descriptions
            if a_set:
                overlaps.append(jaccard(w_set, a_set))
                active_features += 1

        if not overlaps:
            log(f"  {task}: no active features found, skipping")
            continue

        weight_activation_agreement = float(np.mean(overlaps))
        passed = weight_activation_agreement > AGREEMENT_THRESHOLD

        # Per-feature distribution stats
        overlaps_arr = np.array(overlaps)
        median_overlap = float(np.median(overlaps_arr))
        std_overlap = float(np.std(overlaps_arr))
        frac_above_threshold = float(np.mean(overlaps_arr > AGREEMENT_THRESHOLD))

        log(f"    active_features={active_features}/{n_features}  "
            f"agreement={weight_activation_agreement:.4f}  "
            f"median={median_overlap:.4f}  "
            f"[{'PASS' if passed else 'FAIL'}]")

        results.append(EvalResult(
            metric_id="M8.weightlens_convergence",
            value=weight_activation_agreement,
            n_samples=len(prompts),
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": task,
                "weight_activation_agreement": weight_activation_agreement,
                "median_overlap": median_overlap,
                "std_overlap": std_overlap,
                "frac_above_threshold": frac_above_threshold,
                "passed": passed,
                "threshold": AGREEMENT_THRESHOLD,
                "top_k": top_k,
                "n_features": n_features,
                "active_features": active_features,
                "hook_name": effective_hook,
                "artifact_type": getattr(artifact, "artifact_type", "unknown"),
            },
        ))

    return results


def main():
    parser = parse_common_args("M8: WeightLens Convergence")
    parser.add_argument("--hook", default=None, help="Hook point for artifact activations")
    parser.add_argument("--artifact-type", default="sae",
                        choices=["sae", "transcoder", "crosscoder", "factor_bank"],
                        help="Artifact adapter type")
    parser.add_argument("--artifact-path", default=None,
                        help="Path or release ID for artifact")
    parser.add_argument("--sae-id", default=None,
                        help="SAE ID (for SAELens artifacts)")
    parser.add_argument("--top-k", type=int, default=50,
                        help="Top-k tokens for comparison (default: 50)")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    artifact = _load_artifact(args.artifact_type, args.artifact_path,
                              args.sae_id, args.hook)

    log("=" * 60)
    log("M8: WEIGHTLENS CONVERGENCE")
    log("=" * 60)

    out = args.out or "114_weightlens.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_weightlens_convergence(
            model, [task],
            n_prompts=args.n_prompts,
            artifact=artifact,
            hook_name=args.hook,
            top_k=args.top_k,
        )
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)
            p = "PASS" if r.metadata["passed"] else "FAIL"
            log(f"  {task}: agreement={r.value:.4f}  [{p}]")

    save_results(results, out, args=args)
    log(f"\nDone. {len(results)} tasks evaluated.")


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
