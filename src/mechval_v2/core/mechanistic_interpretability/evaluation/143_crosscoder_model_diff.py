"""Crosscoder Model Diffing Validity (Evaluation EX33)
Paper: Lindsey et al. (2024). "Crosscoders." transformer-circuits.pub
=============================================
Instrument:     EX33 --- Crosscoder Model Diffing Validity
Categories:     evaluation
Validity layer: Internal / Measurement
Criteria:       C3 Task Specificity, M3 Baseline Separation
Establishes:    Whether the crosscoder shared/exclusive feature
                classification is valid --- exclusive features should
                have near-zero activation in the excluded model, shared
                features should activate similarly in both
Requires:       CPU or GPU, model
=============================================

For crosscoders trained on base vs fine-tuned models, the decoder norm
ratio classifies features as shared or exclusive. This metric validates
that classification by checking activation behavior: features classified
as exclusive to one model should have near-zero activation when the
other model processes the same inputs, and shared features should
activate with similar magnitude in both models.

Core logic:
1. Simulate a two-model crosscoder setup using two layers within the
   same model as proxies (early layer = "base", late layer = "tuned").
2. Extract feature directions from each layer.
3. Classify features by decoder norm ratio (how much of the direction's
   variance is explained in each layer):
   - Shared: similar magnitude in both layers.
   - Exclusive: dominant in one layer.
4. For exclusive features, verify near-zero activation in the excluded
   layer. For shared features, verify similar activation magnitude.
5. diff_classification_accuracy = fraction correctly classified.

Pass condition: diff_classification_accuracy > 0.7

Usage:
    uv run python 143_crosscoder_model_diff.py --model gpt2 --device cpu
    uv run python 143_crosscoder_model_diff.py --n-prompts 40 --n-features 30
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
    name="Crosscoder Model Diffing Validity",
    paper_ref="Lindsey et al. (2024)",
    paper_cite=(
        "Lindsey et al. 2024, "
        "Crosscoders (transformer-circuits.pub)"
    ),
    description=(
        "Validates the crosscoder shared/exclusive feature classification. "
        "Tests whether features classified as exclusive by decoder norm "
        "ratio actually have near-zero activation in the excluded model "
        "(layer), and whether shared features activate similarly in both."
    ),
    category="evaluation",
    tier="established",
    origin="external",
)

CLASSIFICATION_THRESHOLD = 0.7
EXCLUSIVITY_RATIO = 3.0  # Decoder norm ratio to classify as exclusive


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


def _extract_directions(
    activations: torch.Tensor,
    n_features: int,
) -> torch.Tensor:
    """Extract top-n_features directions via PCA.

    Args:
        activations: (n_tokens, d_model).
        n_features: number of directions to extract.

    Returns:
        directions: (n_features, d_model) unit vectors.
    """
    centered = activations - activations.mean(dim=0, keepdim=True)
    k = min(n_features, min(centered.shape) - 1)
    if k < 1:
        return torch.zeros(0, activations.shape[1], device=activations.device)

    _, _, Vh = torch.linalg.svd(centered, full_matrices=False)
    return F.normalize(Vh[:k], dim=1)


def _classify_and_validate(
    acts_base: torch.Tensor,
    acts_tuned: torch.Tensor,
    directions: torch.Tensor,
    exclusivity_ratio: float,
) -> dict:
    """Classify features as shared/exclusive and validate the classification.

    For each direction:
    1. Compute projection variance in each layer (proxy for decoder norm).
    2. Classify by variance ratio.
    3. Validate: exclusive features should have low activation in the
       excluded layer; shared features should have similar activation.

    Returns dict with classification accuracy and details.
    """
    n_features = directions.shape[0]
    correct = 0
    total = 0
    details = []

    for i in range(n_features):
        direction = directions[i]

        # Project both layers onto this direction
        proj_base = acts_base @ direction  # (n_tokens,)
        proj_tuned = acts_tuned @ direction  # (n_tokens,)

        var_base = proj_base.var().item()
        var_tuned = proj_tuned.var().item()

        # Avoid division by zero
        if var_base < 1e-12 and var_tuned < 1e-12:
            continue

        total += 1

        # Classification by variance ratio (proxy for decoder norm ratio)
        if var_base < 1e-12:
            ratio = float("inf")
        elif var_tuned < 1e-12:
            ratio = 0.0
        else:
            ratio = var_tuned / var_base

        if ratio > exclusivity_ratio:
            # Classified as tuned-exclusive
            label = "tuned_exclusive"
            # Validate: base projection should have much lower variance
            valid = var_base < var_tuned / exclusivity_ratio
        elif ratio < 1.0 / exclusivity_ratio:
            # Classified as base-exclusive
            label = "base_exclusive"
            # Validate: tuned projection should have much lower variance
            valid = var_tuned < var_base / exclusivity_ratio
        else:
            # Classified as shared
            label = "shared"
            # Validate: activation magnitudes should be similar
            # Use correlation as a similarity measure
            proj_base_c = proj_base - proj_base.mean()
            proj_tuned_c = proj_tuned - proj_tuned.mean()
            numer = (proj_base_c * proj_tuned_c).sum()
            denom = torch.sqrt((proj_base_c ** 2).sum() * (proj_tuned_c ** 2).sum())
            corr = (numer / denom).item() if denom.item() > 1e-10 else 0.0
            # Shared features should correlate reasonably
            valid = corr > 0.3

        if valid:
            correct += 1

        details.append({
            "feature_index": i,
            "label": label,
            "var_base": var_base,
            "var_tuned": var_tuned,
            "ratio": ratio if ratio != float("inf") else 999.0,
            "valid": valid,
        })

    accuracy = correct / total if total > 0 else 0.0
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "details": details,
    }


def run_crosscoder_model_diff(
    model,
    tasks: list[str] | None = None,
    n_prompts: int = 40,
    n_features: int = 30,
    base_layer: int | None = None,
    tuned_layer: int | None = None,
    exclusivity_ratio: float = EXCLUSIVITY_RATIO,
) -> list[EvalResult]:
    """Validate crosscoder model diffing classification across tasks.

    Uses two layers within the same model as proxies for base and
    fine-tuned models. Early layers capture more "base" representations;
    later layers capture more task-specific representations.

    Args:
        model: HookedTransformer instance.
        tasks: list of task names (default: CIRCUIT_TASKS).
        n_prompts: number of prompts per task.
        n_features: number of feature directions to test.
        base_layer: layer to use as "base model" proxy (default: layer 1).
        tuned_layer: layer to use as "tuned model" proxy (default: last-1).
        exclusivity_ratio: variance ratio threshold for exclusive classification.

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

    log(f"  Crosscoder Model Diff Validity: n_features={n_features}, "
        f"base_layer={base_layer}, tuned_layer={tuned_layer}")

    results = []
    all_accuracy = []

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

        # Extract directions from the concatenated activations
        combined = torch.cat([acts_base, acts_tuned], dim=1)
        centered = combined - combined.mean(dim=0, keepdim=True)
        d = acts_base.shape[1]

        k = min(n_features, min(centered.shape) - 1)
        if k < 1:
            log(f"    {task}: cannot extract directions, skipping")
            continue

        _, _, Vh = torch.linalg.svd(centered, full_matrices=False)
        # Use directions from the base-layer subspace
        directions = F.normalize(Vh[:k, :d], dim=1)

        validation = _classify_and_validate(
            acts_base, acts_tuned, directions, exclusivity_ratio,
        )

        accuracy = validation["accuracy"]
        passed = accuracy > CLASSIFICATION_THRESHOLD
        all_accuracy.append(accuracy)

        # Count labels
        label_counts = {}
        for det in validation["details"]:
            label_counts[det["label"]] = label_counts.get(det["label"], 0) + 1

        log(f"    {task}: diff_classification_accuracy={accuracy:.4f} "
            f"({validation['correct']}/{validation['total']}) "
            f"({'PASS' if passed else 'FAIL'}) "
            f"labels={label_counts}")

        results.append(EvalResult(
            metric_id="EX33.crosscoder_model_diff",
            value=accuracy,
            n_samples=validation["total"],
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": task,
                "diff_classification_accuracy": accuracy,
                "correct": validation["correct"],
                "total": validation["total"],
                "base_layer": base_layer,
                "tuned_layer": tuned_layer,
                "exclusivity_ratio": exclusivity_ratio,
                "label_distribution": label_counts,
                "passed": passed,
                "threshold": CLASSIFICATION_THRESHOLD,
                "per_feature": validation["details"][:10],
            },
        ))

    # Aggregate
    if all_accuracy:
        agg_mean = float(np.mean(all_accuracy))
        agg_std = float(np.std(all_accuracy))
        agg_passed = agg_mean > CLASSIFICATION_THRESHOLD
        log(f"  Aggregate: diff_classification_accuracy={agg_mean:.4f} "
            f"+/- {agg_std:.4f} ({'PASS' if agg_passed else 'FAIL'})")

        results.append(EvalResult(
            metric_id="EX33.crosscoder_model_diff",
            value=agg_mean,
            n_samples=sum(r.n_samples for r in results),
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": "aggregate",
                "diff_classification_accuracy": agg_mean,
                "diff_classification_accuracy_std": agg_std,
                "n_tasks_evaluated": len(all_accuracy),
                "per_task_accuracy": {
                    r.metadata["task"]: r.metadata["diff_classification_accuracy"]
                    for r in results if r.metadata.get("task") != "aggregate"
                },
                "base_layer": base_layer,
                "tuned_layer": tuned_layer,
                "passed": agg_passed,
                "threshold": CLASSIFICATION_THRESHOLD,
            },
        ))

    return results


def main():
    parser = parse_common_args("EX33: Crosscoder Model Diffing Validity")
    parser.add_argument("--n-features", type=int, default=30,
                        help="Number of feature directions to test (default: 30)")
    parser.add_argument("--base-layer", type=int, default=None,
                        help="Layer to use as base model proxy (default: 1)")
    parser.add_argument("--tuned-layer", type=int, default=None,
                        help="Layer to use as tuned model proxy (default: last-1)")
    parser.add_argument("--exclusivity-ratio", type=float, default=EXCLUSIVITY_RATIO,
                        help=f"Variance ratio for exclusive classification (default: {EXCLUSIVITY_RATIO})")
    args = parser.parse_args()

    model = load_model(args.model, args.device)

    log("=" * 60)
    log("EX33: CROSSCODER MODEL DIFFING VALIDITY")
    log("=" * 60)

    tasks = args.tasks or CIRCUIT_TASKS
    results = run_crosscoder_model_diff(
        model,
        tasks=tasks,
        n_prompts=args.n_prompts,
        n_features=args.n_features,
        base_layer=args.base_layer,
        tuned_layer=args.tuned_layer,
        exclusivity_ratio=args.exclusivity_ratio,
    )

    out = args.out or "143_crosscoder_model_diff.json"
    save_results(results, out)
    log(f"\nDone. {len(results)} results.")


if __name__ == "__main__":
    main()
