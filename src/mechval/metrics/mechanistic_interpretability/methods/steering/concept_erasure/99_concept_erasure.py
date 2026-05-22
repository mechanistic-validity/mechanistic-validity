"""LEACE Concept Erasure Dissociation (Causal C15)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     C15 — Concept Erasure Dissociation
Categories:     causal
Validity layer: Internal
Criteria:       C15 Concept Erasure Dissociation
Establishes:    Whether erasing a claimed feature direction via LEACE
                (Belrose et al., NeurIPS 2023) causes the predicted
                behavioral change to disappear
Requires:       CPU or GPU, model, artifact adapter
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Implements the LEACE (Least-Squares Concept Erasure) dissociation test.
Given an artifact adapter's top-k feature directions as a concept subspace:

1. Collect residual-stream activations at the specified hook point.
2. Compute the LEACE erasure projector for the concept subspace.
3. Re-run the model with erased activations via hooks.
4. Measure behavioral change: KL divergence and logit-diff delta.

LEACE core math (closed-form, no iterative optimization):
    Given concept directions D of shape (k, d_model):
    P = D^T (D D^T)^{-1} D           (projection onto concept subspace)
    X_erased = X - X @ P              (project out the concept subspace)

Pass condition: dissociation_strength > 0.3

Usage:
    uv run python 99_concept_erasure.py --tasks ioi --n-prompts 40
    uv run python 99_concept_erasure.py --tasks ioi sva --artifact-path <release>
"""

import numpy as np
import torch
import torch.nn.functional as F

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


def leace_projector(directions: torch.Tensor) -> torch.Tensor:
    """Compute the LEACE erasure projection matrix.

    Given concept directions D of shape (k, d_model), returns the
    orthogonal projector onto the concept subspace:
        P = D^T (D D^T)^{-1} D

    Erasing is then: X_erased = X - X @ P = X @ (I - P).

    Uses SVD for numerical stability when directions may be
    near-collinear.

    Args:
        directions: (k, d_model) tensor of concept directions.

    Returns:
        P: (d_model, d_model) projection matrix.
    """
    # Normalize rows for numerical stability
    D = directions.float()
    # SVD-based orthogonal projector: robust to near-collinear directions
    U, S, Vh = torch.linalg.svd(D, full_matrices=False)
    # Keep components with non-negligible singular values
    threshold = S.max() * 1e-5
    mask = S > threshold
    Vh_kept = Vh[mask]  # (r, d_model) where r <= k
    # P = Vh_kept^T @ Vh_kept projects onto the column space of D^T
    P = Vh_kept.T @ Vh_kept  # (d_model, d_model)
    return P


def _make_erasure_hook(hook_name: str, projector: torch.Tensor):
    """Create a forward hook that erases the concept subspace from activations."""
    def hook_fn(act, hook):
        P = projector.to(act.device, act.dtype)
        # act shape: (batch, seq, d_model)
        # Erase: X_erased = X - X @ P = X @ (I - P)
        erased = act - act @ P
        return erased
    return (hook_name, hook_fn)


def compute_erasure_scores(
    model, directions: torch.Tensor, prompts, correct_ids, incorrect_ids,
    hook_name: str, top_k: int = 10,
) -> dict:
    """Compute behavioral change from LEACE concept erasure.

    Args:
        model: HookedTransformer model.
        directions: (n_features, d_model) feature directions from artifact.
        prompts: list of prompt objects.
        correct_ids: list of correct token IDs.
        incorrect_ids: list of incorrect token IDs.
        hook_name: hook point for intervention.
        top_k: number of top feature directions to use as concept subspace.

    Returns:
        dict with erasure_kl, behavioral_change, dissociation_strength, etc.
    """
    n = min(len(prompts), len(correct_ids), len(incorrect_ids))
    if n == 0:
        return {"erasure_kl": 0.0, "behavioral_change": 0.0, "dissociation_strength": 0.0}

    # Select top-k directions as the concept subspace
    k = min(top_k, directions.shape[0])
    concept_dirs = directions[:k]  # (k, d_model)

    # Compute the LEACE projector
    P = leace_projector(concept_dirs)

    # Collect clean baselines and erased outputs
    clean_logit_diffs = []
    erased_logit_diffs = []
    kl_divs = []

    hook = _make_erasure_hook(hook_name, P)

    with torch.no_grad():
        for i in range(n):
            tokens = model.to_tokens(prompts[i].text)

            # Clean forward pass
            clean_logits = model(tokens)
            clean_ld = (clean_logits[0, -1, correct_ids[i]] - clean_logits[0, -1, incorrect_ids[i]]).item()
            clean_logit_diffs.append(clean_ld)

            # Erased forward pass
            erased_logits = model.run_with_hooks(tokens, fwd_hooks=[hook])
            erased_ld = (erased_logits[0, -1, correct_ids[i]] - erased_logits[0, -1, incorrect_ids[i]]).item()
            erased_logit_diffs.append(erased_ld)

            # KL divergence: KL(clean || erased) over the full vocabulary
            clean_lp = F.log_softmax(clean_logits[0, -1], dim=-1)
            erased_lp = F.log_softmax(erased_logits[0, -1], dim=-1)
            kl = F.kl_div(erased_lp, clean_lp.exp(), reduction="sum", log_target=False).item()
            kl_divs.append(kl)

    clean_ld_arr = np.array(clean_logit_diffs)
    erased_ld_arr = np.array(erased_logit_diffs)

    mean_clean_ld = float(clean_ld_arr.mean())
    mean_erased_ld = float(erased_ld_arr.mean())
    mean_kl = float(np.mean(kl_divs))

    # Behavioral change: absolute change in mean logit diff
    behavioral_change = abs(mean_clean_ld - mean_erased_ld)

    # Dissociation strength: normalized behavioral change
    # How much of the clean logit diff was destroyed by erasure
    denominator = abs(mean_clean_ld) + 1e-8
    dissociation_strength = behavioral_change / denominator

    return {
        "erasure_kl": mean_kl,
        "behavioral_change": behavioral_change,
        "dissociation_strength": float(dissociation_strength),
        "mean_clean_ld": mean_clean_ld,
        "mean_erased_ld": mean_erased_ld,
        "n_concept_directions": k,
        "per_prompt_kl": [float(x) for x in kl_divs[:20]],
        "per_prompt_clean_ld": [float(x) for x in clean_logit_diffs[:20]],
        "per_prompt_erased_ld": [float(x) for x in erased_logit_diffs[:20]],
    }


def run_concept_erasure(
    model, tasks: list[str], artifact=None, n_prompts: int = 40,
    hook_name: str | None = None, top_k: int = 10,
) -> list[EvalResult]:
    """Run LEACE concept erasure dissociation test.

    Args:
        model: HookedTransformer model.
        tasks: list of task names to evaluate.
        artifact: ArtifactAdapter with directions() method.
        n_prompts: number of prompts per task.
        hook_name: hook point for intervention (default: from artifact or blocks.5.hook_resid_pre).
        top_k: number of top feature directions to use as concept subspace.

    Returns:
        list of EvalResult objects.
    """
    if artifact is None:
        log("  WARNING: no artifact adapter provided, skipping concept erasure")
        return []

    tokenizer = model.tokenizer
    results = []

    effective_hook = hook_name or artifact.manifest.hook_point
    if not effective_hook:
        effective_hook = "blocks.5.hook_resid_pre"

    dirs = artifact.directions()
    if dirs.ndim == 3:
        dirs = dirs.mean(dim=0)

    for task in tasks:
        prompts = generate_prompts(task, tokenizer, n_prompts)
        if not prompts:
            log(f"  {task}: no prompts, skipping")
            continue

        correct_ids, incorrect_ids = get_token_ids(prompts, tokenizer)
        if not correct_ids:
            continue

        log(f"  {task}: erasing top-{top_k} directions at {effective_hook}, {len(prompts)} prompts")

        scores = compute_erasure_scores(
            model, dirs, prompts, correct_ids, incorrect_ids,
            effective_hook, top_k=top_k,
        )

        passed = bool(scores["dissociation_strength"] > 0.3)

        log(f"    erasure_kl={scores['erasure_kl']:.4f}")
        log(f"    behavioral_change={scores['behavioral_change']:.4f}")
        log(f"    dissociation_strength={scores['dissociation_strength']:.4f}")
        log(f"    [{('PASS' if passed else 'FAIL')}]")

        results.append(EvalResult(
            metric_id="C15.concept_erasure_dissociation",
            value=float(scores["dissociation_strength"]),
            n_samples=len(prompts),
            metadata={
                "task": task,
                "erasure_kl": scores["erasure_kl"],
                "behavioral_change": scores["behavioral_change"],
                "dissociation_strength": scores["dissociation_strength"],
                "mean_clean_ld": scores["mean_clean_ld"],
                "mean_erased_ld": scores["mean_erased_ld"],
                "n_concept_directions": scores["n_concept_directions"],
                "passed": passed,
                "threshold": 0.3,
                "hook_name": effective_hook,
                "top_k": top_k,
                "per_prompt_kl": scores["per_prompt_kl"],
                "per_prompt_clean_ld": scores["per_prompt_clean_ld"],
                "per_prompt_erased_ld": scores["per_prompt_erased_ld"],
            },
        ))

    return results


def main():
    parser = parse_common_args("C15: LEACE Concept Erasure Dissociation")
    parser.add_argument("--hook", default=None, help="Hook point for erasure")
    parser.add_argument("--top-k", type=int, default=10, help="Number of top directions to erase")
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
    log("C15: LEACE CONCEPT ERASURE DISSOCIATION")
    log("=" * 60)

    out = args.out or "99_concept_erasure.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_concept_erasure(
            model, [task], artifact=artifact,
            n_prompts=args.n_prompts, hook_name=args.hook,
            top_k=args.top_k,
        )
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)
            p = "PASS" if r.metadata["passed"] else "FAIL"
            log(f"  {task}: dissociation={r.value:.4f}  [{p}]")

    save_results(results, out)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
