"""Representation Engineering (Causal C16)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     C16 — Representation Engineering
Categories:     causal
Validity layer: External (E1 Intervention Reach)
Criteria:       C16 Concept Dimensionality
Establishes:    Whether task-relevant concepts occupy low-dimensional subspaces;
                extends CAA with PCA to discover multi-component concept directions
Requires:       CPU or GPU, model, artifact adapter (optional)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Implements RepE (Zou et al., 2023) as a validation metric.  RepE generalizes
Contrastive Activation Addition (CAA) by using PCA on the difference of
positive/negative activation distributions to discover principal concept
directions.

For each task:

1. Collect residual-stream activations for positive and negative prompts.
2. Compute the mean-difference vector (same as CAA).
3. Apply PCA to the centered contrast matrix to extract top-k directions.
4. Steer with cumulative PCA components (1, then 1+2, ...) and measure
   logit-diff shift at each level.
5. Report concept_dimensionality (number of PCA components needed to reach
   90% of the full steering effect), per-component steerability, and
   explained variance of each component.
6. Optionally compare discovered directions against artifact adapter
   directions via cosine similarity for convergent validity.

Pass condition: concept_dimensionality <= 5 (concepts should be
low-dimensional in the residual stream).

Usage:
    uv run python 100_representation_engineering.py --tasks ioi --n-prompts 40
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


def _collect_activations(model, prompts, hook_name: str, max_prompts: int) -> torch.Tensor:
    """Collect last-token residual stream activations at the given hook point.

    Returns tensor of shape (n_prompts, d_model).
    """
    acts = []
    with torch.no_grad():
        for p in prompts[:max_prompts]:
            tokens = model.to_tokens(p.text)
            _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
            act = cache[hook_name][0, -1, :]  # (d_model,)
            acts.append(act.cpu())
    return torch.stack(acts)  # (n, d_model)


def _pca_directions(contrast_matrix: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute PCA on the contrast matrix (positive - negative activations).

    Returns:
        directions: (n_components, d_model) principal directions
        explained_variance_ratio: (n_components,) fraction of variance per component
    """
    centered = contrast_matrix - contrast_matrix.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    total_var = (s ** 2).sum()
    k = min(n_components, len(s))
    directions = vt[:k]  # (k, d_model)
    explained = (s[:k] ** 2) / (total_var + 1e-12)
    return directions, explained


def _add_steering_hook(hook_name: str, direction: torch.Tensor, coeff: float):
    """Create a hook that adds coeff * direction to the residual stream."""
    def hook_fn(act, hook):
        act[:, :, :] = act + coeff * direction.to(act.device)
        return act
    return (hook_name, hook_fn)


def _measure_steering_effect(
    model, prompts, correct_ids, incorrect_ids,
    hook_name: str, direction: torch.Tensor,
    baseline_ld: float, coeff: float = 2.0,
) -> float:
    """Measure the logit-diff shift when steering with the given direction.

    Returns the absolute shift relative to the baseline.
    """
    n = min(len(prompts), len(correct_ids), len(incorrect_ids))
    shifted_lds = []
    with torch.no_grad():
        for i in range(n):
            tokens = model.to_tokens(prompts[i].text)
            hook = _add_steering_hook(hook_name, direction, coeff)
            logits = model.run_with_hooks(tokens, fwd_hooks=[hook])
            ld = (logits[0, -1, correct_ids[i]] - logits[0, -1, incorrect_ids[i]]).item()
            shifted_lds.append(ld)
    mean_shifted = float(np.mean(shifted_lds))
    return abs(mean_shifted - baseline_ld)


def _compute_baseline_ld(model, prompts, correct_ids, incorrect_ids) -> float:
    """Compute mean baseline logit diff across prompts."""
    n = min(len(prompts), len(correct_ids), len(incorrect_ids))
    lds = []
    with torch.no_grad():
        for i in range(n):
            tokens = model.to_tokens(prompts[i].text)
            logits = model(tokens)
            ld = (logits[0, -1, correct_ids[i]] - logits[0, -1, incorrect_ids[i]]).item()
            lds.append(ld)
    return float(np.mean(lds)) if lds else 0.0


def run_representation_engineering(
    model, tasks: list[str], artifact=None, n_prompts: int = 40,
    hook_name: str | None = None, n_components: int = 10,
) -> list[EvalResult]:
    """Run RepE evaluation: PCA-based concept direction discovery and steering.

    Args:
        model: HookedTransformer model.
        tasks: List of task names to evaluate.
        artifact: Optional artifact adapter for convergent validity comparison.
        n_prompts: Number of prompts per task.
        hook_name: Hook point for activation collection and steering.
        n_components: Maximum number of PCA components to extract.

    Returns:
        List of EvalResult with metric_id "C16.representation_engineering".
    """
    tokenizer = model.tokenizer
    results = []

    effective_hook = hook_name
    if not effective_hook and artifact is not None:
        effective_hook = artifact.manifest.hook_point
    if not effective_hook:
        effective_hook = "blocks.5.hook_resid_pre"

    for task in tasks:
        prompts = generate_prompts(task, tokenizer, n_prompts)
        if not prompts:
            log(f"  {task}: no prompts, skipping")
            continue

        correct_ids, incorrect_ids = get_token_ids(prompts, tokenizer)
        if not correct_ids:
            log(f"  {task}: no valid token ids, skipping")
            continue

        # Split prompts into positive (correct answer expected) and negative
        # (incorrect answer expected).  We use the same prompts but collect
        # activations once for the "clean" run.  The contrast is formed by
        # the sign of the logit diff: positive prompts are those where the
        # model already gets it right (logit_diff > 0), negative otherwise.
        log(f"  {task}: collecting activations at {effective_hook}")
        all_acts = _collect_activations(model, prompts, effective_hook, len(prompts))
        baseline_ld = _compute_baseline_ld(model, prompts, correct_ids, incorrect_ids)

        # Compute per-prompt logit diffs to split into positive/negative
        per_prompt_lds = []
        with torch.no_grad():
            for i in range(min(len(prompts), len(correct_ids))):
                tokens = model.to_tokens(prompts[i].text)
                logits = model(tokens)
                ld = (logits[0, -1, correct_ids[i]] - logits[0, -1, incorrect_ids[i]]).item()
                per_prompt_lds.append(ld)

        median_ld = float(np.median(per_prompt_lds))
        pos_mask = np.array(per_prompt_lds) >= median_ld
        neg_mask = ~pos_mask

        n_pos = int(pos_mask.sum())
        n_neg = int(neg_mask.sum())
        if n_pos < 2 or n_neg < 2:
            log(f"  {task}: insufficient contrast pairs (pos={n_pos}, neg={n_neg}), skipping")
            continue

        pos_acts = all_acts[pos_mask].numpy()  # (n_pos, d_model)
        neg_acts = all_acts[neg_mask].numpy()  # (n_neg, d_model)

        # Mean difference vector (CAA-style)
        mean_diff = pos_acts.mean(axis=0) - neg_acts.mean(axis=0)  # (d_model,)
        mean_diff_norm = mean_diff / (np.linalg.norm(mean_diff) + 1e-12)

        # Contrast matrix for PCA: each row is a paired difference.
        # When counts differ, use the minimum and pair in order.
        n_pairs = min(n_pos, n_neg)
        contrast_matrix = pos_acts[:n_pairs] - neg_acts[:n_pairs]  # (n_pairs, d_model)

        k = min(n_components, n_pairs, contrast_matrix.shape[1])
        pca_dirs, explained_var = _pca_directions(contrast_matrix, k)

        log(f"  {task}: {k} PCA components, explained variance: "
            f"{', '.join(f'{v:.3f}' for v in explained_var[:5])}")

        # Measure steering with cumulative PCA directions
        steer_coeff = 2.0
        cumulative_effects = []
        per_component_effects = []

        for n_dims in range(1, k + 1):
            # Cumulative direction: sum of first n_dims PCA components
            cumulative_dir = torch.tensor(pca_dirs[:n_dims].sum(axis=0), dtype=torch.float32)
            cumulative_dir = cumulative_dir / (cumulative_dir.norm() + 1e-12)

            effect = _measure_steering_effect(
                model, prompts, correct_ids, incorrect_ids,
                effective_hook, cumulative_dir, baseline_ld, coeff=steer_coeff,
            )
            cumulative_effects.append(float(effect))

        # Per-component individual effects
        for i in range(k):
            single_dir = torch.tensor(pca_dirs[i], dtype=torch.float32)
            single_dir = single_dir / (single_dir.norm() + 1e-12)
            effect = _measure_steering_effect(
                model, prompts, correct_ids, incorrect_ids,
                effective_hook, single_dir, baseline_ld, coeff=steer_coeff,
            )
            per_component_effects.append(float(effect))

        # Also test the mean-diff vector for comparison
        mean_diff_dir = torch.tensor(mean_diff_norm, dtype=torch.float32)
        mean_diff_effect = _measure_steering_effect(
            model, prompts, correct_ids, incorrect_ids,
            effective_hook, mean_diff_dir, baseline_ld, coeff=steer_coeff,
        )

        # Concept dimensionality: number of PCA components to reach 90% of
        # the maximum cumulative steering effect
        max_cumulative = max(cumulative_effects) if cumulative_effects else 1e-12
        threshold = 0.9 * max_cumulative
        concept_dimensionality = k  # default: all components needed
        for i, eff in enumerate(cumulative_effects):
            if eff >= threshold:
                concept_dimensionality = i + 1
                break

        # Steerability: max cumulative effect relative to baseline
        steerability = max_cumulative / (abs(baseline_ld) + 1e-8)

        # Convergent validity: compare PCA direction 1 with artifact directions
        artifact_cosine_sim = None
        if artifact is not None:
            dirs = artifact.directions()
            if dirs.ndim == 3:
                dirs = dirs.mean(dim=0)
            dirs_np = dirs.detach().cpu().numpy()
            pc1 = pca_dirs[0]
            # Max absolute cosine similarity against any artifact direction
            dots = np.abs(dirs_np @ pc1) / (np.linalg.norm(dirs_np, axis=1) * np.linalg.norm(pc1) + 1e-12)
            artifact_cosine_sim = float(np.max(dots))
            log(f"    artifact convergent validity (max |cos|): {artifact_cosine_sim:.4f}")

        passed = bool(concept_dimensionality <= 5)
        log(f"    concept_dimensionality={concept_dimensionality}, "
            f"steerability={steerability:.4f}, "
            f"mean_diff_effect={mean_diff_effect:.4f}")
        log(f"    [{('PASS' if passed else 'FAIL')}]")

        results.append(EvalResult(
            metric_id="C16.representation_engineering",
            value=float(concept_dimensionality),
            n_samples=len(prompts),
            metadata={
                "task": task,
                "concept_dimensionality": concept_dimensionality,
                "steerability": float(steerability),
                "mean_diff_effect": float(mean_diff_effect),
                "cumulative_effects": cumulative_effects,
                "per_component_steerability": per_component_effects,
                "subspace_explained_variance": explained_var.tolist(),
                "n_pca_components": k,
                "n_positive": n_pos,
                "n_negative": n_neg,
                "baseline_ld": float(baseline_ld),
                "max_cumulative_effect": float(max_cumulative),
                "artifact_cosine_similarity": artifact_cosine_sim,
                "passed": passed,
                "threshold_dimensionality": 5,
                "hook_name": effective_hook,
                "steer_coeff": steer_coeff,
            },
        ))

    return results


def main():
    parser = parse_common_args("C16: Representation Engineering (RepE)")
    parser.add_argument("--hook", default=None, help="Hook point for activations and steering")
    parser.add_argument("--n-components", type=int, default=10, help="Max PCA components")
    parser.add_argument("--artifact-path", default=None, help="SAE release ID")
    parser.add_argument("--sae-id", default=None, help="SAE ID within release")
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
    log("C16: REPRESENTATION ENGINEERING (RepE)")
    log("=" * 60)

    out = args.out or "100_representation_engineering.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_representation_engineering(
            model, [task], artifact=artifact,
            n_prompts=args.n_prompts, hook_name=args.hook,
            n_components=args.n_components,
        )
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)

    save_results(results, out)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
