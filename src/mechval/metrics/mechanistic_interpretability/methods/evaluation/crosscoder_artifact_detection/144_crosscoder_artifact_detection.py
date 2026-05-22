"""Crosscoder L1 Artifact Detection (Evaluation EX34)
Paper: Muto et al. (2025). NeurIPS 2025.
=============================================
Instrument:     EX34 --- Crosscoder L1 Artifact Detection
Categories:     evaluation
Validity layer: Measurement / Internal
Criteria:       M6 Construct Coverage, I5 Confound Control
Establishes:    Whether L1-penalized crosscoder training inflates
                apparent feature exclusivity, reclassifying shared
                features as exclusive (the Muto et al. sparsity artifact)
Requires:       CPU or GPU, model
=============================================

Muto et al. (NeurIPS 2025) showed that L1 penalty in crosscoder
training inflates apparent feature exclusivity: features that are
genuinely shared between models are misclassified as exclusive because
L1 pushes decoder norms toward zero in one model. This metric detects
that artifact by comparing two classification methods:

1. Decoder-norm classification: classify features as shared/exclusive
   by the ratio of projection variance across layers (proxy for decoder
   norm ratio under L1 training).
2. Activation-based classification: classify features as shared/exclusive
   by whether they actually activate in both layers on the same inputs.

Features that are "exclusive" by decoder-norm but "shared" by activation
are L1 artifacts. The artifact rate is the fraction of decoder-norm-
exclusive features that are reclassified as shared under activation
analysis.

Core logic:
1. Collect activations at two layers (early = "base", late = "tuned").
2. Extract feature directions from the combined activation space.
3. Classify each feature two ways:
   a. By projection variance ratio (decoder-norm proxy) --- with a
      stringent threshold that mimics L1-induced sparsity.
   b. By activation correlation --- if the feature activates coherently
      in both layers above a noise floor, classify as shared.
4. artifact_rate = (exclusive_by_norm AND shared_by_activation) / exclusive_by_norm.

Pass condition: artifact_rate < 0.15

Usage:
    uv run python 144_crosscoder_artifact_detection.py --model gpt2 --device cpu
    uv run python 144_crosscoder_artifact_detection.py --n-prompts 40 --n-features 30
"""

import numpy as np
import torch
import torch.nn.functional as F

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    InstrumentInfo,
    generate_prompts,
    load_model,
    log,
    parse_common_args,
    save_results,
)

INSTRUMENT_INFO = InstrumentInfo(
    name="Crosscoder L1 Artifact Detection",
    paper_ref="Muto et al. (2025)",
    paper_cite=(
        "Muto et al. 2025, NeurIPS 2025. "
        "L1 penalty in crosscoder training inflates apparent feature "
        "exclusivity, misclassifying shared features as model-exclusive."
    ),
    description=(
        "Detects the sparsity artifact from Muto et al. NeurIPS 2025: "
        "L1 penalty inflates apparent feature exclusivity. Compares "
        "decoder-norm-based classification vs activation-based "
        "classification. Features that switch from exclusive to shared "
        "under reanalysis are artifacts."
    ),
    category="evaluation",
    tier="established",
    origin="external",
)

ARTIFACT_RATE_THRESHOLD = 0.15
NORM_EXCLUSIVITY_RATIO = 5.0  # Stringent threshold mimicking L1 bias
ACTIVATION_CORR_THRESHOLD = 0.2  # Correlation above which feature is "shared"


@torch.no_grad()
def _collect_layer_activations(
    model,
    prompts,
    layer: int,
) -> torch.Tensor:
    """Collect residual stream activations at a single layer.

    Returns tensor of shape (total_tokens, d_model).
    """
    hook_name = f"blocks.{layer}.hook_resid_pre"
    all_acts = []
    for p in prompts:
        tokens = model.to_tokens(p.text)
        captured = {}

        def fwd_hook(value, hook, _c=captured):
            _c["act"] = value.detach()
            return value

        model.run_with_hooks(tokens, fwd_hooks=[(hook_name, fwd_hook)])
        if "act" in captured:
            all_acts.append(captured["act"].squeeze(0))

    if not all_acts:
        return torch.zeros(0, model.cfg.d_model, device=model.cfg.device)
    return torch.cat(all_acts, dim=0)


def _detect_artifacts(
    acts_base: torch.Tensor,
    acts_tuned: torch.Tensor,
    directions: torch.Tensor,
    norm_ratio: float,
    corr_threshold: float,
) -> dict:
    """Classify features by two methods and detect L1 artifacts.

    Method 1 (decoder-norm proxy): classify by projection variance ratio.
    Method 2 (activation-based): classify by activation correlation.

    An artifact is a feature classified as exclusive by method 1 but
    shared by method 2.

    Returns dict with artifact rate and per-feature details.
    """
    n_features = directions.shape[0]
    n_exclusive_by_norm = 0
    n_artifacts = 0
    details = []

    for i in range(n_features):
        direction = directions[i]

        proj_base = acts_base @ direction
        proj_tuned = acts_tuned @ direction

        var_base = proj_base.var().item()
        var_tuned = proj_tuned.var().item()

        # Method 1: decoder-norm proxy classification
        if var_base < 1e-12 and var_tuned < 1e-12:
            norm_label = "dead"
            continue
        elif var_base < 1e-12:
            ratio = float("inf")
            norm_label = "tuned_exclusive"
        elif var_tuned < 1e-12:
            ratio = 0.0
            norm_label = "base_exclusive"
        else:
            ratio = max(var_tuned / var_base, var_base / var_tuned)
            if ratio > norm_ratio:
                norm_label = "tuned_exclusive" if var_tuned > var_base else "base_exclusive"
            else:
                norm_label = "shared"

        # Method 2: activation-based classification
        proj_base_c = proj_base - proj_base.mean()
        proj_tuned_c = proj_tuned - proj_tuned.mean()
        numer = (proj_base_c * proj_tuned_c).sum()
        denom = torch.sqrt((proj_base_c ** 2).sum() * (proj_tuned_c ** 2).sum())
        corr = (numer / denom).item() if denom.item() > 1e-10 else 0.0
        act_label = "shared" if corr > corr_threshold else "exclusive"

        # Detect artifact: exclusive by norm but shared by activation
        is_exclusive_by_norm = norm_label in ("tuned_exclusive", "base_exclusive")
        is_artifact = is_exclusive_by_norm and act_label == "shared"

        if is_exclusive_by_norm:
            n_exclusive_by_norm += 1
        if is_artifact:
            n_artifacts += 1

        details.append({
            "feature_index": i,
            "norm_label": norm_label,
            "act_label": act_label,
            "var_ratio": ratio if ratio != float("inf") else 999.0,
            "activation_corr": corr,
            "is_artifact": is_artifact,
        })

    artifact_rate = n_artifacts / n_exclusive_by_norm if n_exclusive_by_norm > 0 else 0.0
    return {
        "artifact_rate": artifact_rate,
        "n_exclusive_by_norm": n_exclusive_by_norm,
        "n_artifacts": n_artifacts,
        "n_features_evaluated": len(details),
        "details": details,
    }


def run_crosscoder_artifact_detection(
    model,
    tasks: list[str] | None = None,
    n_prompts: int = 40,
    n_features: int = 30,
    base_layer: int | None = None,
    tuned_layer: int | None = None,
    norm_ratio: float = NORM_EXCLUSIVITY_RATIO,
    corr_threshold: float = ACTIVATION_CORR_THRESHOLD,
) -> list[EvalResult]:
    """Detect L1 sparsity artifacts in crosscoder-style feature classification.

    Uses two layers within the same model as proxies for base and
    fine-tuned models, then compares decoder-norm-based vs activation-based
    feature classification.

    Args:
        model: HookedTransformer instance.
        tasks: list of task names (default: CIRCUIT_TASKS).
        n_prompts: number of prompts per task.
        n_features: number of feature directions to test.
        base_layer: layer to use as "base model" proxy (default: layer 1).
        tuned_layer: layer to use as "tuned model" proxy (default: last-1).
        norm_ratio: variance ratio threshold for norm-based exclusive classification.
        corr_threshold: correlation threshold for activation-based shared classification.

    Returns:
        List of EvalResult, one per task plus an aggregate.
    """
    if tasks is None:
        tasks = CIRCUIT_TASKS

    n_layers = model.cfg.n_layers
    if base_layer is None:
        base_layer = min(1, n_layers - 1)
    if tuned_layer is None:
        tuned_layer = max(0, n_layers - 2)

    log(f"  Crosscoder L1 Artifact Detection: n_features={n_features}, "
        f"base_layer={base_layer}, tuned_layer={tuned_layer}, "
        f"norm_ratio={norm_ratio}, corr_threshold={corr_threshold}")

    results = []
    all_artifact_rates = []

    for task in tasks:
        prompts = generate_prompts(task, model.tokenizer, n_prompts=n_prompts)
        if not prompts:
            log(f"    {task}: no prompts, skipping")
            continue

        acts_base = _collect_layer_activations(model, prompts, base_layer)
        acts_tuned = _collect_layer_activations(model, prompts, tuned_layer)

        n_tokens = min(acts_base.shape[0], acts_tuned.shape[0])
        if n_tokens < n_features + 5:
            log(f"    {task}: insufficient tokens ({n_tokens}), skipping")
            continue
        acts_base = acts_base[:n_tokens]
        acts_tuned = acts_tuned[:n_tokens]

        # Extract feature directions from combined space
        d = acts_base.shape[1]
        combined = torch.cat([acts_base, acts_tuned], dim=1)
        centered = combined - combined.mean(dim=0, keepdim=True)

        k = min(n_features, min(centered.shape) - 1)
        if k < 1:
            log(f"    {task}: cannot extract directions, skipping")
            continue

        _, _, Vh = torch.linalg.svd(centered, full_matrices=False)
        directions = F.normalize(Vh[:k, :d], dim=1)

        detection = _detect_artifacts(
            acts_base, acts_tuned, directions, norm_ratio, corr_threshold,
        )

        artifact_rate = detection["artifact_rate"]
        passed = artifact_rate < ARTIFACT_RATE_THRESHOLD
        all_artifact_rates.append(artifact_rate)

        log(f"    {task}: artifact_rate={artifact_rate:.4f} "
            f"({detection['n_artifacts']}/{detection['n_exclusive_by_norm']} exclusive) "
            f"({'PASS' if passed else 'FAIL'})")

        # Summarize label distribution
        label_dist = {}
        for det in detection["details"]:
            key = f"{det['norm_label']}/{det['act_label']}"
            label_dist[key] = label_dist.get(key, 0) + 1

        results.append(EvalResult(
            metric_id="EX34.crosscoder_artifact_detection",
            value=artifact_rate,
            n_samples=detection["n_features_evaluated"],
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": task,
                "artifact_rate": artifact_rate,
                "n_exclusive_by_norm": detection["n_exclusive_by_norm"],
                "n_artifacts": detection["n_artifacts"],
                "n_features_evaluated": detection["n_features_evaluated"],
                "base_layer": base_layer,
                "tuned_layer": tuned_layer,
                "norm_ratio": norm_ratio,
                "corr_threshold": corr_threshold,
                "label_distribution": label_dist,
                "passed": passed,
                "threshold": ARTIFACT_RATE_THRESHOLD,
                "per_feature": detection["details"][:10],
            },
        ))

    # Aggregate
    if all_artifact_rates:
        agg_mean = float(np.mean(all_artifact_rates))
        agg_std = float(np.std(all_artifact_rates))
        agg_passed = agg_mean < ARTIFACT_RATE_THRESHOLD
        log(f"  Aggregate: artifact_rate={agg_mean:.4f} "
            f"+/- {agg_std:.4f} ({'PASS' if agg_passed else 'FAIL'})")

        results.append(EvalResult(
            metric_id="EX34.crosscoder_artifact_detection",
            value=agg_mean,
            n_samples=sum(r.n_samples for r in results),
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": "aggregate",
                "artifact_rate": agg_mean,
                "artifact_rate_std": agg_std,
                "n_tasks_evaluated": len(all_artifact_rates),
                "per_task_artifact_rate": {
                    r.metadata["task"]: r.metadata["artifact_rate"]
                    for r in results if r.metadata.get("task") != "aggregate"
                },
                "base_layer": base_layer,
                "tuned_layer": tuned_layer,
                "passed": agg_passed,
                "threshold": ARTIFACT_RATE_THRESHOLD,
            },
        ))

    return results


def main():
    parser = parse_common_args("EX34: Crosscoder L1 Artifact Detection")
    parser.add_argument("--n-features", type=int, default=30,
                        help="Number of feature directions to test (default: 30)")
    parser.add_argument("--base-layer", type=int, default=None,
                        help="Layer to use as base model proxy (default: 1)")
    parser.add_argument("--tuned-layer", type=int, default=None,
                        help="Layer to use as tuned model proxy (default: last-1)")
    parser.add_argument("--norm-ratio", type=float, default=NORM_EXCLUSIVITY_RATIO,
                        help=f"Variance ratio for norm-based exclusive (default: {NORM_EXCLUSIVITY_RATIO})")
    parser.add_argument("--corr-threshold", type=float, default=ACTIVATION_CORR_THRESHOLD,
                        help=f"Correlation threshold for activation-based shared (default: {ACTIVATION_CORR_THRESHOLD})")
    args = parser.parse_args()

    model = load_model(args.model, args.device)

    log("=" * 60)
    log("EX34: CROSSCODER L1 ARTIFACT DETECTION")
    log("=" * 60)

    tasks = args.tasks or CIRCUIT_TASKS
    results = run_crosscoder_artifact_detection(
        model,
        tasks=tasks,
        n_prompts=args.n_prompts,
        n_features=args.n_features,
        base_layer=args.base_layer,
        tuned_layer=args.tuned_layer,
        norm_ratio=args.norm_ratio,
        corr_threshold=args.corr_threshold,
    )

    out = args.out or "144_crosscoder_artifact_detection.json"
    save_results(results, out)
    log(f"\nDone. {len(results)} results.")


if __name__ == "__main__":
    main()
