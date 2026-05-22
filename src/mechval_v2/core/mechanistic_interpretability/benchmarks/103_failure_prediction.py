"""Failure Prediction (Behavioral B22)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     B22 -- Failure Prediction
Categories:     behavioral
Validity layer: External
Criteria:       Feature-level failure discrimination via AUROC
Establishes:    Whether artifact features can predict model failures,
                not just successes
Requires:       CPU or GPU, model, artifact adapter
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Inspired by the "Why It Failed" benchmark (Mathew et al., AAAI 2026),
which found that linear probes on learned features achieve near-chance
performance at predicting model failures. This metric tests whether
individual artifact features can distinguish prompts where the model
succeeds (correct_token logit > incorrect_token logit) from those where
it fails.

For each task:
  1. Run the model on all prompts and classify each as success/failure.
  2. Collect per-feature activations from the artifact at the last token.
  3. Compute per-feature AUROC at separating success vs failure cases.
  4. Compute the same AUROC on raw activation dimensions (neuron baseline).
  5. Report best feature AUROC, best neuron AUROC, and the advantage.
  6. Compute a combined AUROC using the mean of top-k features as a score.

Pass condition: best_feature_auroc > 0.6

Usage:
    # Programmatic (from metric_registry dispatch):
    run_failure_prediction(model, artifact=artifact,
        hook_name="blocks.5.hook_resid_pre", tasks=["ioi"])

    # CLI:
    uv run python 103_failure_prediction.py --model gpt2 \\
        --hook blocks.5.hook_resid_pre --tasks ioi

Reference:
    Mathew et al., "Why It Failed: Probing Failure Modes of Learned
    Representations", AAAI 2026.
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


# ---------------------------------------------------------------------------
# AUROC via Mann-Whitney U (no sklearn dependency)
# ---------------------------------------------------------------------------

def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute AUROC from scores and binary labels without sklearn.

    Uses the Mann-Whitney U statistic formulation:
        AUROC = U / (n_pos * n_neg)
    """
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Vectorized: count how many (pos, neg) pairs the positive scores higher
    u = 0.0
    for p in pos:
        u += np.sum(p > neg) + 0.5 * np.sum(p == neg)

    return float(u / (n_pos * n_neg))


# ---------------------------------------------------------------------------
# Classify model success/failure per prompt
# ---------------------------------------------------------------------------

@torch.no_grad()
def _classify_success_failure(
    model, prompts, correct_ids: list[int], incorrect_ids: list[int],
) -> np.ndarray:
    """Return binary labels: 1 = success, 0 = failure.

    Success means the model assigns higher logit to the correct token
    than the incorrect token at the last sequence position.
    """
    n_valid = min(len(prompts), len(correct_ids))
    labels = np.zeros(n_valid, dtype=np.int32)

    for i in range(n_valid):
        tokens = model.to_tokens(prompts[i].text)
        logits = model(tokens)
        last_logits = logits[0, -1]
        if last_logits[correct_ids[i]] > last_logits[incorrect_ids[i]]:
            labels[i] = 1

    return labels


# ---------------------------------------------------------------------------
# Collect activations
# ---------------------------------------------------------------------------

@torch.no_grad()
def _collect_feature_activations(
    artifact, model, prompts, hook_name: str, n_valid: int,
) -> np.ndarray | None:
    """Collect last-token feature activations for each prompt.

    Returns shape (n_valid, n_features) or None if artifact is unavailable.
    """
    all_acts = []
    for i in range(n_valid):
        tokens = model.to_tokens(prompts[i].text)
        acts = artifact.activations(model, tokens, hook_name)
        # acts shape: (batch, seq, n_features) -- take last token
        last_act = acts[0, -1].detach().cpu().numpy()
        all_acts.append(last_act)

    return np.stack(all_acts, axis=0)


@torch.no_grad()
def _collect_neuron_activations(
    model, prompts, hook_name: str, n_valid: int,
) -> np.ndarray:
    """Collect last-token raw activations (neuron baseline) for each prompt.

    Returns shape (n_valid, d_model).
    """
    all_acts = []
    for i in range(n_valid):
        tokens = model.to_tokens(prompts[i].text)
        _, cache = model.run_with_cache(
            tokens, names_filter=[hook_name],
        )
        last_act = cache[hook_name][0, -1].detach().cpu().numpy()
        all_acts.append(last_act)

    return np.stack(all_acts, axis=0)


# ---------------------------------------------------------------------------
# Per-dimension AUROC scan
# ---------------------------------------------------------------------------

def _best_auroc_over_dims(
    activations: np.ndarray, labels: np.ndarray,
) -> tuple[float, int]:
    """Compute AUROC for each dimension and return the best.

    Returns (best_auroc, best_dim_index).
    For each dimension, we take max(auroc, 1-auroc) since the sign of
    the feature's relationship to success/failure is unknown.
    """
    n_dims = activations.shape[1]
    best_auroc = 0.5
    best_dim = 0

    for d in range(n_dims):
        auc = _auroc(activations[:, d], labels)
        # The feature might be anti-correlated -- take the better direction
        auc = max(auc, 1.0 - auc)
        if auc > best_auroc:
            best_auroc = auc
            best_dim = d

    return best_auroc, best_dim


def _topk_combined_auroc(
    activations: np.ndarray, labels: np.ndarray, k: int = 5,
) -> float:
    """Compute AUROC using the mean of top-k feature activations as a score.

    Selects the k features with the highest individual AUROC, averages
    their activations (flipping sign for anti-correlated features), and
    computes AUROC on the combined score.
    """
    n_dims = activations.shape[1]
    k = min(k, n_dims)

    # Score each dimension
    dim_aurocs = []
    for d in range(n_dims):
        auc = _auroc(activations[:, d], labels)
        dim_aurocs.append((max(auc, 1.0 - auc), auc, d))

    dim_aurocs.sort(key=lambda x: x[0], reverse=True)
    top_dims = dim_aurocs[:k]

    # Combine: average activations, flipping sign for anti-correlated dims
    combined = np.zeros(activations.shape[0], dtype=np.float64)
    for _, raw_auc, d in top_dims:
        vals = activations[:, d].astype(np.float64)
        # If raw AUROC < 0.5, the feature is anti-correlated: flip sign
        if raw_auc < 0.5:
            vals = -vals
        combined += vals

    combined /= k
    return _auroc(combined, labels)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_failure_prediction(
    model,
    tasks: list[str] | None = None,
    artifact=None,
    hook_name: str | None = None,
    n_prompts: int = 50,
    top_k: int = 5,
) -> list[EvalResult]:
    """Test whether artifact features predict model failures.

    For each task, classifies model predictions as success/failure, then
    measures how well individual artifact features (and raw neuron
    activations) discriminate between the two classes using AUROC.

    Args:
        model: HookedTransformer instance.
        tasks: List of task names to evaluate.
        artifact: ArtifactAdapter with activations() method.
        hook_name: Hook point for activation collection.
        n_prompts: Number of prompts per task.
        top_k: Number of top features to combine for the combined AUROC.

    Returns:
        List of EvalResult, one per task.
    """
    if tasks is None:
        tasks = CIRCUIT_TASKS

    if artifact is None:
        log("  WARNING: no artifact adapter provided, skipping failure prediction")
        return []

    effective_hook = hook_name or getattr(
        getattr(artifact, "manifest", None), "hook_point", None,
    )
    if not effective_hook:
        effective_hook = "blocks.5.hook_resid_pre"

    tokenizer = model.tokenizer

    log(f"  Failure prediction: {len(tasks)} tasks, hook={effective_hook}, "
        f"n_prompts={n_prompts}, top_k={top_k}")

    results = []

    for task in tasks:
        prompts = generate_prompts(task, tokenizer, n_prompts)
        if not prompts:
            log(f"  {task}: no prompts, skipping")
            continue

        correct_ids, incorrect_ids = get_token_ids(prompts, tokenizer)
        if not correct_ids:
            log(f"  {task}: no valid token pairs, skipping")
            continue

        n_valid = min(len(prompts), len(correct_ids))

        # 1. Classify each prompt as success (1) or failure (0)
        labels = _classify_success_failure(
            model, prompts, correct_ids, incorrect_ids,
        )
        n_success = int(np.sum(labels == 1))
        n_failure = int(np.sum(labels == 0))

        log(f"  {task}: {n_valid} prompts, {n_success} success, {n_failure} failure")

        if n_success == 0 or n_failure == 0:
            log(f"    all-same class, AUROC undefined, skipping")
            results.append(EvalResult(
                metric_id="B22.failure_prediction",
                value=0.5,
                n_samples=n_valid,
                metadata={
                    "task": task,
                    "n_success": n_success,
                    "n_failure": n_failure,
                    "best_feature_auroc": 0.5,
                    "best_neuron_auroc": 0.5,
                    "prediction_advantage": 0.0,
                    "top_k_combined_auroc": 0.5,
                    "passed": False,
                    "skipped_reason": "all prompts in same class",
                },
            ))
            continue

        # 2. Feature activations from artifact
        feature_acts = _collect_feature_activations(
            artifact, model, prompts, effective_hook, n_valid,
        )

        # 3. Neuron baseline activations
        neuron_acts = _collect_neuron_activations(
            model, prompts, effective_hook, n_valid,
        )

        # 4. Per-feature AUROC
        best_feature_auroc, best_feature_idx = _best_auroc_over_dims(
            feature_acts, labels,
        )

        # 5. Per-neuron AUROC (baseline)
        best_neuron_auroc, best_neuron_idx = _best_auroc_over_dims(
            neuron_acts, labels,
        )

        # 6. Advantage
        advantage = best_feature_auroc - best_neuron_auroc

        # 7. Combined top-k feature AUROC
        combined_auroc = _topk_combined_auroc(feature_acts, labels, k=top_k)

        passed = bool(best_feature_auroc > 0.6)

        log(f"    best_feature_auroc = {best_feature_auroc:.3f} (dim {best_feature_idx})")
        log(f"    best_neuron_auroc  = {best_neuron_auroc:.3f} (dim {best_neuron_idx})")
        log(f"    advantage          = {advantage:+.3f}")
        log(f"    top_{top_k}_combined   = {combined_auroc:.3f}")
        log(f"    [{'PASS' if passed else 'FAIL'}]")

        results.append(EvalResult(
            metric_id="B22.failure_prediction",
            value=best_feature_auroc,
            n_samples=n_valid,
            metadata={
                "task": task,
                "n_success": n_success,
                "n_failure": n_failure,
                "best_feature_auroc": best_feature_auroc,
                "best_feature_index": best_feature_idx,
                "best_neuron_auroc": best_neuron_auroc,
                "best_neuron_index": best_neuron_idx,
                "prediction_advantage": advantage,
                "top_k_combined_auroc": combined_auroc,
                "top_k": top_k,
                "hook_name": effective_hook,
                "passed": passed,
                "threshold": 0.6,
            },
        ))

    # Log aggregate summary
    if results:
        mean_feat = np.mean([r.metadata["best_feature_auroc"] for r in results])
        mean_neur = np.mean([r.metadata["best_neuron_auroc"] for r in results])
        mean_adv = np.mean([r.metadata["prediction_advantage"] for r in results])
        n_passed = sum(1 for r in results if r.metadata["passed"])
        log(f"  SUMMARY: mean_feature_auroc={mean_feat:.3f}, "
            f"mean_neuron_auroc={mean_neur:.3f}, "
            f"mean_advantage={mean_adv:+.3f}")
        log(f"  passed: {n_passed}/{len(results)}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = parse_common_args("B22: Failure Prediction")
    parser.add_argument("--hook", default=None,
                        help="Hook point (e.g. blocks.5.hook_resid_pre)")
    parser.add_argument("--artifact-path", default=None,
                        help="SAE release ID for artifact")
    parser.add_argument("--sae-id", default=None,
                        help="SAE ID within release")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of top features to combine (default: 5)")
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

    tasks = args.tasks or CIRCUIT_TASKS

    log("=" * 60)
    log("B22: FAILURE PREDICTION")
    log("=" * 60)

    out = args.out or "103_failure_prediction.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_failure_prediction(
            model, [task],
            artifact=artifact,
            hook_name=args.hook,
            n_prompts=args.n_prompts,
            top_k=args.top_k,
        )
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)
            p = "PASS" if r.metadata["passed"] else "FAIL"
            log(f"  {task}: best_feature_auroc={r.value:.3f}  "
                f"advantage={r.metadata['prediction_advantage']:+.3f}  [{p}]")

    save_results(results, out)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
