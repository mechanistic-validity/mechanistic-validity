"""MIB Causal Variable Localization (Causal C17)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     C17 — MIB Causal Variable Localization
Categories:     causal
Validity layer: Internal
Criteria:       C17 Causal Variable Localization
Establishes:    Whether artifact features localize causal variables
                better than raw neurons (Mueller et al., ICML 2025)
Requires:       CPU or GPU, model, artifact adapter (optional)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Implements the key finding from Mueller et al. (ICML 2025) "Missed
Insights, Burned Budgets" (MIB): SAE features are NOT systematically
better than raw neurons for causal variable localization, and DAS
(supervised) provides an upper bound.

For each task:

1. Neuron baseline: collect activations at the hook point, compute
   per-neuron AUROC at separating positive vs negative prompts
   (by logit-diff sign). neuron_auroc = max over neurons.

2. Feature localization: project activations through artifact feature
   directions, compute per-feature AUROC. feature_auroc = max over
   features.

3. DAS upper bound (supervised): compute mean-difference direction
   between positive and negative activations, project all activations
   onto it, compute AUROC. das_auroc is the supervised ceiling.

4. normalized_score = (feature_auroc - neuron_auroc) / (das_auroc - neuron_auroc)
   0 = no better than neurons, 1 = matches DAS.

AUROC computation uses Mann-Whitney U statistic:
   AUROC = U / (n_pos * n_neg)

Pass threshold: normalized_score > 0.1 (features are at least 10%
of the way from neurons to DAS).

Usage:
    uv run python 103_mib_causal_variable.py --tasks ioi --n-prompts 40
    uv run python 103_mib_causal_variable.py --tasks ioi sva --artifact-path <release>
"""

import numpy as np
import torch

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    generate_prompts,
    get_token_ids,
    load_model,
    log,
    parse_common_args,
    save_incremental,
    save_results,
)


def _mannwhitney_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute AUROC via the Mann-Whitney U statistic.

    Args:
        scores: (n,) array of scalar scores.
        labels: (n,) boolean array, True = positive class.

    Returns:
        AUROC in [0, 1]. Returns 0.5 if either class is empty.
    """
    pos = scores[labels]
    neg = scores[~labels]
    n_pos = len(pos)
    n_neg = len(neg)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    # U = number of (pos, neg) pairs where pos > neg
    # Vectorized: for each pos value, count how many neg values it exceeds
    u = 0.0
    for p in pos:
        u += (p > neg).sum() + 0.5 * (p == neg).sum()
    auroc = u / (n_pos * n_neg)
    # Take max(auroc, 1-auroc) since we don't know the sign convention
    return max(auroc, 1.0 - auroc)


def _best_neuron_auroc(activations: np.ndarray, labels: np.ndarray) -> tuple[float, int]:
    """Find the neuron dimension with highest AUROC.

    Args:
        activations: (n_samples, d_model) array.
        labels: (n_samples,) boolean array.

    Returns:
        (best_auroc, best_neuron_idx)
    """
    n_dims = activations.shape[1]
    best_auroc = 0.5
    best_idx = 0
    for d in range(n_dims):
        auroc = _mannwhitney_auroc(activations[:, d], labels)
        if auroc > best_auroc:
            best_auroc = auroc
            best_idx = d
    return best_auroc, best_idx


def _best_feature_auroc(
    activations: np.ndarray, directions: np.ndarray, labels: np.ndarray,
) -> tuple[float, int]:
    """Find the artifact feature with highest AUROC.

    Projects activations onto each feature direction and computes AUROC.

    Args:
        activations: (n_samples, d_model) array.
        directions: (n_features, d_model) array of feature directions.
        labels: (n_samples,) boolean array.

    Returns:
        (best_auroc, best_feature_idx)
    """
    # Project: (n_samples, d_model) @ (d_model, n_features) -> (n_samples, n_features)
    projections = activations @ directions.T
    n_features = projections.shape[1]
    best_auroc = 0.5
    best_idx = 0
    for f in range(n_features):
        auroc = _mannwhitney_auroc(projections[:, f], labels)
        if auroc > best_auroc:
            best_auroc = auroc
            best_idx = f
    return best_auroc, best_idx


def _das_auroc(activations: np.ndarray, labels: np.ndarray) -> float:
    """Compute DAS (Distributed Alignment Search) upper bound AUROC.

    The DAS direction is the mean-difference direction between positive
    and negative activations (supervised, oracle). Project all activations
    onto this direction and compute AUROC.

    Args:
        activations: (n_samples, d_model) array.
        labels: (n_samples,) boolean array.

    Returns:
        AUROC of the DAS direction.
    """
    pos_acts = activations[labels]
    neg_acts = activations[~labels]
    if len(pos_acts) == 0 or len(neg_acts) == 0:
        return 0.5
    das_dir = pos_acts.mean(axis=0) - neg_acts.mean(axis=0)
    norm = np.linalg.norm(das_dir)
    if norm < 1e-12:
        return 0.5
    das_dir = das_dir / norm
    projections = activations @ das_dir  # (n_samples,)
    return _mannwhitney_auroc(projections, labels)


def _collect_activations_and_labels(
    model, prompts, correct_ids, incorrect_ids, hook_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect last-token activations and positive/negative labels.

    Labels are determined by the sign of the logit diff: positive when
    the model assigns higher logit to the correct answer.

    Args:
        model: HookedTransformer.
        prompts: list of TaskPrompt.
        correct_ids: list of correct token IDs.
        incorrect_ids: list of incorrect token IDs.
        hook_name: hook point for activation collection.

    Returns:
        (activations, labels) where activations is (n, d_model) and
        labels is (n,) boolean array.
    """
    n = min(len(prompts), len(correct_ids), len(incorrect_ids))
    all_acts = []
    all_labels = []
    with torch.no_grad():
        for i in range(n):
            tokens = model.to_tokens(prompts[i].text)
            logits, cache = model.run_with_cache(
                tokens, names_filter=[hook_name],
            )
            act = cache[hook_name][0, -1, :].cpu().numpy()  # (d_model,)
            ld = (logits[0, -1, correct_ids[i]] - logits[0, -1, incorrect_ids[i]]).item()
            all_acts.append(act)
            all_labels.append(ld > 0)
    return np.stack(all_acts), np.array(all_labels)


def run_mib_causal_variable(
    model, tasks: list[str], artifact=None, n_prompts: int = 30,
    hook_name: str | None = None, max_features: int = 200,
) -> list[EvalResult]:
    """Test whether artifact features localize causal variables better than neurons.

    Implements Mueller et al. (ICML 2025) MIB evaluation: compare feature-level
    AUROC against neuron-level AUROC and the DAS supervised upper bound.

    Args:
        model: HookedTransformer model.
        tasks: list of task names to evaluate.
        artifact: ArtifactAdapter with directions() method.
        n_prompts: number of prompts per task.
        hook_name: hook point for activation collection.
        max_features: maximum number of artifact features to evaluate.

    Returns:
        list of EvalResult with metric_id "C17.mib_causal_variable".
    """
    tokenizer = model.tokenizer
    results = []

    effective_hook = hook_name
    if not effective_hook and artifact is not None:
        effective_hook = artifact.manifest.hook_point
    if not effective_hook:
        effective_hook = "blocks.5.hook_resid_pre"

    # Get artifact directions if available
    dirs_np = None
    if artifact is not None:
        dirs = artifact.directions()
        if dirs.ndim == 3:
            dirs = dirs.mean(dim=0)
        dirs_np = dirs.detach().cpu().numpy()  # (n_features, d_model)
        # Normalize each direction
        norms = np.linalg.norm(dirs_np, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        dirs_np = dirs_np / norms
        # Limit number of features for tractability
        if dirs_np.shape[0] > max_features:
            dirs_np = dirs_np[:max_features]

    for task in tasks:
        prompts = generate_prompts(task, tokenizer, n_prompts)
        if not prompts:
            log(f"  {task}: no prompts, skipping")
            continue

        correct_ids, incorrect_ids = get_token_ids(prompts, tokenizer)
        if not correct_ids:
            log(f"  {task}: no valid token ids, skipping")
            continue

        log(f"  {task}: collecting activations at {effective_hook}")
        activations, labels = _collect_activations_and_labels(
            model, prompts, correct_ids, incorrect_ids, effective_hook,
        )

        n_pos = int(labels.sum())
        n_neg = int((~labels).sum())
        if n_pos < 2 or n_neg < 2:
            log(f"  {task}: insufficient contrast (pos={n_pos}, neg={n_neg}), skipping")
            continue

        # 1. Neuron baseline
        neuron_auroc, best_neuron = _best_neuron_auroc(activations, labels)
        log(f"    neuron_auroc={neuron_auroc:.4f} (neuron {best_neuron})")

        # 2. Feature localization (if artifact available)
        if dirs_np is not None:
            feature_auroc, best_feature = _best_feature_auroc(
                activations, dirs_np, labels,
            )
            log(f"    feature_auroc={feature_auroc:.4f} (feature {best_feature})")
        else:
            # Without an artifact, use neuron baseline as the feature result
            # (normalized_score will be 0.0)
            feature_auroc = neuron_auroc
            best_feature = -1
            log(f"    no artifact provided, feature_auroc defaults to neuron_auroc")

        # 3. DAS upper bound
        das_auroc_val = _das_auroc(activations, labels)
        log(f"    das_auroc={das_auroc_val:.4f}")

        # 4. Compute normalized score
        localization_advantage = feature_auroc - neuron_auroc
        das_gap = das_auroc_val - neuron_auroc
        if abs(das_gap) < 1e-8:
            # DAS and neurons are equivalent; avoid division by zero
            normalized_score = 0.0
        else:
            normalized_score = localization_advantage / das_gap

        passed = bool(normalized_score > 0.1)
        log(f"    normalized_score={normalized_score:.4f}  "
            f"[{'PASS' if passed else 'FAIL'}]")

        results.append(EvalResult(
            metric_id="C17.mib_causal_variable",
            value=float(normalized_score),
            n_samples=len(prompts),
            metadata={
                "task": task,
                "feature_auroc": float(feature_auroc),
                "neuron_auroc": float(neuron_auroc),
                "das_auroc": float(das_auroc_val),
                "localization_advantage": float(localization_advantage),
                "normalized_score": float(normalized_score),
                "best_neuron_idx": int(best_neuron),
                "best_feature_idx": int(best_feature),
                "n_positive": n_pos,
                "n_negative": n_neg,
                "n_features_evaluated": int(dirs_np.shape[0]) if dirs_np is not None else 0,
                "passed": passed,
                "threshold": 0.1,
                "hook_name": effective_hook,
                "has_artifact": artifact is not None,
            },
        ))

    return results


def main():
    parser = parse_common_args("C17: MIB Causal Variable Localization")
    parser.add_argument("--hook", default=None,
                        help="Hook point for activation collection")
    parser.add_argument("--max-features", type=int, default=200,
                        help="Max artifact features to evaluate")
    parser.add_argument("--artifact-path", default=None,
                        help="SAE release ID")
    parser.add_argument("--sae-id", default=None,
                        help="SAE ID within release")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
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
    log("C17: MIB CAUSAL VARIABLE LOCALIZATION")
    log("=" * 60)

    out = args.out or "103_mib_causal_variable.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_mib_causal_variable(
            model, [task], artifact=artifact,
            n_prompts=args.n_prompts, hook_name=args.hook,
            max_features=args.max_features,
        )
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)
            p = "PASS" if r.metadata["passed"] else "FAIL"
            log(f"  {task}: normalized_score={r.value:.4f}  [{p}]")

    save_results(results, out)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
