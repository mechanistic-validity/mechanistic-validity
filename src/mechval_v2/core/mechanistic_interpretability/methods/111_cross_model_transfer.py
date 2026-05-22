"""Cross-Model Steering Transfer (External E6)
Paper: Oozeer et al. (2025). ICML 2025.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     EX15 — Cross-Model Steering Transfer
Categories:     steering, external validity
Validity layer: External
Criteria:       E6 Cross-Architecture Generalization
Establishes:    Whether steering vectors can be transferred across model
                families via learned linear mappings, indicating the concept
                is genuinely represented in both models
Requires:       GPU (two models), or CPU for small models
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Implements the cross-model steering transfer approach from Oozeer et al.
(ICML 2025). The key finding is that steering vectors can be transferred
across model families (Llama -> Qwen -> Gemma) via learned autoencoder
mappings, and smaller models can align larger ones via steering transfer.

For a source model A and target model B:

1. Collect activations from both models on the same prompts at
   corresponding layers.
2. Learn a linear mapping (autoencoder) between activation spaces:
   M: R^{d_A} -> R^{d_B}.
3. Extract a steering vector v_A from model A (mean difference between
   positive/negative concept prompts).
4. Transfer the steering vector to model B: v_B = M @ v_A.
5. Measure the behavioral effect of v_B on model B and compare to the
   effect of the natively-extracted v_B* on model B.
6. Report transfer_fidelity = correlation between steering effects
   (logit diff shifts) from transferred v_B and native v_B*.

Pass condition: transfer_fidelity > 0.3

Transfer success indicates the concept is genuinely represented in both
models' activation spaces, providing external validity evidence.

Usage:
    uv run python 111_cross_model_transfer.py --source-model gpt2 \\
        --target-model gpt2 --tasks ioi --n-prompts 30

    uv run python 111_cross_model_transfer.py --source-model gpt2 \\
        --target-model gpt2-medium --layer-source 5 --layer-target 12

Reference:
    Oozeer et al., "Cross-Model Steering Transfer via Learned Activation
    Mappings", ICML 2025.
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
# Core: collect activations at a hook point
# ---------------------------------------------------------------------------

@torch.no_grad()
def _collect_activations(
    model, prompts, hook_name: str,
) -> torch.Tensor:
    """Collect last-token activations at a hook point for each prompt.

    Returns tensor of shape (n_prompts, d_model).
    """
    acts = []
    for prompt in prompts:
        tokens = model.to_tokens(prompt.text)
        cache_dict = {}

        def cache_hook(act, hook, _name=hook_name):
            cache_dict[_name] = act.detach()
            return act

        model.run_with_hooks(tokens, fwd_hooks=[(hook_name, cache_hook)])
        act = cache_dict.get(hook_name)
        if act is not None:
            acts.append(act[0, -1])  # last token, (d_model,)

    if not acts:
        return torch.zeros(0)
    return torch.stack(acts)  # (n_prompts, d_model)


# ---------------------------------------------------------------------------
# Core: learn linear mapping between activation spaces
# ---------------------------------------------------------------------------

def _learn_linear_mapping(
    source_acts: torch.Tensor,
    target_acts: torch.Tensor,
) -> torch.Tensor:
    """Learn a linear mapping M such that target_acts ~ source_acts @ M.T.

    Uses least-squares: M = (X^T X)^{-1} X^T Y, where X = source_acts,
    Y = target_acts. Returns M of shape (d_target, d_source).
    """
    X = source_acts.float()  # (n, d_source)
    Y = target_acts.float()  # (n, d_target)

    # Solve Y = X @ M.T  =>  M.T = (X^T X)^{-1} X^T Y
    XtX = X.T @ X  # (d_source, d_source)
    # Regularize for numerical stability
    XtX += 1e-4 * torch.eye(XtX.shape[0], device=XtX.device)
    XtY = X.T @ Y  # (d_source, d_target)
    MT = torch.linalg.solve(XtX, XtY)  # (d_source, d_target)
    return MT.T  # (d_target, d_source)


# ---------------------------------------------------------------------------
# Core: extract steering vector via mean difference
# ---------------------------------------------------------------------------

@torch.no_grad()
def _extract_steering_vector(
    model, positive_prompts, negative_prompts, hook_name: str,
) -> torch.Tensor:
    """Extract a steering vector as the mean activation difference.

    v = mean(act(positive)) - mean(act(negative)), normalized to unit norm.
    Returns a (d_model,) tensor.
    """
    pos_acts = _collect_activations(model, positive_prompts, hook_name)
    neg_acts = _collect_activations(model, negative_prompts, hook_name)

    if pos_acts.numel() == 0 or neg_acts.numel() == 0:
        return torch.zeros(0)

    v = pos_acts.mean(dim=0) - neg_acts.mean(dim=0)
    norm = v.norm()
    if norm > 1e-12:
        v = v / norm
    return v


# ---------------------------------------------------------------------------
# Core: measure steering effect on logit diff
# ---------------------------------------------------------------------------

@torch.no_grad()
def _measure_steering_effects(
    model, prompts, correct_ids: list[int], incorrect_ids: list[int],
    steering_vec: torch.Tensor, hook_name: str,
    coefficients: list[float],
) -> list[float]:
    """Measure per-coefficient logit diff shift when adding steering_vec.

    Returns list of mean logit diff shifts (one per coefficient).
    """
    n = min(len(prompts), len(correct_ids), len(incorrect_ids))
    if n == 0 or steering_vec.numel() == 0:
        return [0.0] * len(coefficients)

    # Baseline logit diffs
    baseline_lds = []
    for i in range(n):
        tokens = model.to_tokens(prompts[i].text)
        logits = model(tokens)
        ld = (logits[0, -1, correct_ids[i]] - logits[0, -1, incorrect_ids[i]]).item()
        baseline_lds.append(ld)

    shifts = []
    for coeff in coefficients:
        coeff_lds = []
        for i in range(n):
            tokens = model.to_tokens(prompts[i].text)

            def steering_hook(act, hook, _v=steering_vec, _c=coeff):
                act[:, :, :] = act + _c * _v.to(act.device)
                return act

            logits = model.run_with_hooks(
                tokens, fwd_hooks=[(hook_name, steering_hook)],
            )
            ld = (logits[0, -1, correct_ids[i]] - logits[0, -1, incorrect_ids[i]]).item()
            coeff_lds.append(ld)

        mean_shift = float(np.mean(coeff_lds) - np.mean(baseline_lds))
        shifts.append(mean_shift)

    return shifts


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_cross_model_transfer(
    model=None,
    tasks: list[str] | None = None,
    source_model=None,
    target_model=None,
    source_hook: str | None = None,
    target_hook: str | None = None,
    n_prompts: int = 30,
    n_mapping_prompts: int = 50,
    coefficients: list[float] | None = None,
) -> list[EvalResult]:
    """Run cross-model steering transfer evaluation.

    If source_model/target_model are not provided, falls back to using
    ``model`` as both source and target (same-model transfer, useful
    for testing the pipeline or evaluating within-model consistency).

    Args:
        model: Default model (used as both source and target if specific
            models are not provided).
        tasks: List of task names to evaluate.
        source_model: Source HookedTransformer for extracting the steering vector.
        target_model: Target HookedTransformer for applying the transferred vector.
        source_hook: Hook point on the source model (e.g. "blocks.5.hook_resid_pre").
        target_hook: Hook point on the target model.
        n_prompts: Number of prompts for steering effect measurement.
        n_mapping_prompts: Number of prompts for learning the linear mapping.
        coefficients: Steering coefficients to sweep.

    Returns:
        List of EvalResult with transfer_fidelity as the key metric.
    """
    if tasks is None:
        tasks = CIRCUIT_TASKS
    if coefficients is None:
        coefficients = [0.5, 1.0, 2.0, 4.0]

    if source_model is None:
        source_model = model
    if target_model is None:
        target_model = model

    if source_model is None or target_model is None:
        log("  ERROR: no models provided")
        return []

    if source_hook is None:
        n_source = source_model.cfg.n_layers
        source_hook = f"blocks.{n_source // 2}.hook_resid_pre"
    if target_hook is None:
        n_target = target_model.cfg.n_layers
        target_hook = f"blocks.{n_target // 2}.hook_resid_pre"

    source_tokenizer = source_model.tokenizer
    target_tokenizer = target_model.tokenizer

    log(f"  Cross-model transfer: {len(tasks)} tasks")
    log(f"    source hook={source_hook}, target hook={target_hook}")
    log(f"    n_prompts={n_prompts}, n_mapping_prompts={n_mapping_prompts}")

    results = []

    for task in tasks:
        # Generate prompts using both tokenizers (need prompts valid for both)
        source_prompts = generate_prompts(task, source_tokenizer, max(n_prompts, n_mapping_prompts))
        target_prompts = generate_prompts(task, target_tokenizer, max(n_prompts, n_mapping_prompts))
        if not source_prompts or not target_prompts:
            log(f"  {task}: no prompts, skipping")
            continue

        # Use same count for mapping
        n_map = min(n_mapping_prompts, len(source_prompts), len(target_prompts))
        map_source_prompts = source_prompts[:n_map]
        map_target_prompts = target_prompts[:n_map]

        # Evaluation prompts (separate from mapping prompts where possible)
        eval_source_prompts = source_prompts[n_map:n_map + n_prompts] or source_prompts[:n_prompts]
        eval_target_prompts = target_prompts[n_map:n_map + n_prompts] or target_prompts[:n_prompts]

        # If we don't have enough for a split, reuse
        if len(eval_source_prompts) == 0:
            eval_source_prompts = source_prompts[:n_prompts]
        if len(eval_target_prompts) == 0:
            eval_target_prompts = target_prompts[:n_prompts]

        source_correct_ids, source_incorrect_ids = get_token_ids(eval_source_prompts, source_tokenizer)
        target_correct_ids, target_incorrect_ids = get_token_ids(eval_target_prompts, target_tokenizer)

        if not source_correct_ids or not target_correct_ids:
            log(f"  {task}: no valid token pairs, skipping")
            continue

        log(f"  {task}: {n_map} mapping prompts, {len(eval_source_prompts)}/{len(eval_target_prompts)} eval prompts")

        # Step 1: Collect paired activations for learning the mapping
        log(f"    collecting activations for linear mapping...")
        source_acts = _collect_activations(source_model, map_source_prompts, source_hook)
        target_acts = _collect_activations(target_model, map_target_prompts, target_hook)

        if source_acts.numel() == 0 or target_acts.numel() == 0:
            log(f"    failed to collect activations, skipping")
            continue

        # Step 2: Learn the linear mapping
        M = _learn_linear_mapping(source_acts, target_acts)

        # Measure mapping quality (reconstruction R^2 on mapping data)
        reconstructed = (source_acts.float() @ M.T.float())
        ss_res = ((target_acts.float() - reconstructed) ** 2).sum().item()
        ss_tot = ((target_acts.float() - target_acts.float().mean(dim=0)) ** 2).sum().item()
        mapping_r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        log(f"    linear mapping R^2 = {mapping_r2:.4f}")

        # Step 3: Extract steering vectors
        # Split eval prompts into positive/negative for concept extraction
        # Use correct-answer prompts as positive, incorrect as negative
        half = len(eval_source_prompts) // 2
        pos_source = eval_source_prompts[:half]
        neg_source = eval_source_prompts[half:]
        pos_target = eval_target_prompts[:half]
        neg_target = eval_target_prompts[half:]

        if len(pos_source) == 0 or len(neg_source) == 0:
            log(f"    not enough prompts for steering vector extraction, skipping")
            continue

        log(f"    extracting steering vectors...")
        v_source = _extract_steering_vector(source_model, pos_source, neg_source, source_hook)
        v_target_native = _extract_steering_vector(target_model, pos_target, neg_target, target_hook)

        if v_source.numel() == 0 or v_target_native.numel() == 0:
            log(f"    steering vector extraction failed, skipping")
            continue

        # Step 4: Transfer steering vector
        v_target_transferred = M.float() @ v_source.float()
        norm = v_target_transferred.norm()
        if norm > 1e-12:
            v_target_transferred = v_target_transferred / norm

        # Cosine similarity between transferred and native vectors
        cos_sim = float(torch.dot(v_target_transferred, v_target_native.float()).item())
        log(f"    cosine(transferred, native) = {cos_sim:.4f}")

        # Step 5: Measure steering effects on target model
        log(f"    measuring transferred steering effects...")
        transferred_effects = _measure_steering_effects(
            target_model, eval_target_prompts,
            target_correct_ids, target_incorrect_ids,
            v_target_transferred, target_hook, coefficients,
        )

        log(f"    measuring native steering effects...")
        native_effects = _measure_steering_effects(
            target_model, eval_target_prompts,
            target_correct_ids, target_incorrect_ids,
            v_target_native, target_hook, coefficients,
        )

        # Step 6: Compute transfer fidelity
        if len(coefficients) >= 2:
            t_arr = np.array(transferred_effects)
            n_arr = np.array(native_effects)
            if np.std(t_arr) > 1e-12 and np.std(n_arr) > 1e-12:
                transfer_fidelity = float(np.corrcoef(t_arr, n_arr)[0, 1])
                if np.isnan(transfer_fidelity):
                    transfer_fidelity = 0.0
            else:
                transfer_fidelity = 0.0
        else:
            transfer_fidelity = 0.0

        passed = bool(transfer_fidelity > 0.3)

        log(f"    transfer_fidelity = {transfer_fidelity:.4f}")
        log(f"    [{'PASS' if passed else 'FAIL'}]")

        results.append(EvalResult(
            metric_id="EX15.cross_model_transfer",
            value=float(transfer_fidelity),
            n_samples=min(len(eval_source_prompts), len(eval_target_prompts)),
            metadata={
                "task": task,
                "transfer_fidelity": float(transfer_fidelity),
                "cosine_similarity": cos_sim,
                "mapping_r2": float(mapping_r2),
                "passed": passed,
                "threshold": 0.3,
                "source_hook": source_hook,
                "target_hook": target_hook,
                "n_mapping_prompts": n_map,
                "coefficients": coefficients,
                "transferred_effects": transferred_effects,
                "native_effects": native_effects,
            },
        ))

    if results:
        mean_fidelity = np.mean([r.value for r in results])
        mean_cos = np.mean([r.metadata["cosine_similarity"] for r in results])
        n_passed = sum(1 for r in results if r.metadata["passed"])
        log(f"  SUMMARY: mean_transfer_fidelity={mean_fidelity:.4f}, "
            f"mean_cosine_sim={mean_cos:.4f}, "
            f"passed={n_passed}/{len(results)}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = parse_common_args("EX15: Cross-Model Steering Transfer")
    parser.add_argument("--source-model", default=None,
                        help="Source model name (default: same as --model)")
    parser.add_argument("--target-model", default=None,
                        help="Target model name (default: same as --model)")
    parser.add_argument("--layer-source", type=int, default=None,
                        help="Source layer index for hook point")
    parser.add_argument("--layer-target", type=int, default=None,
                        help="Target layer index for hook point")
    parser.add_argument("--n-mapping-prompts", type=int, default=50,
                        help="Number of prompts for learning the linear mapping")
    parser.add_argument("--coefficients", type=float, nargs="+",
                        default=None,
                        help="Steering coefficients (default: 0.5 1.0 2.0 4.0)")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS

    # Load models
    log("=" * 60)
    log("EX15: CROSS-MODEL STEERING TRANSFER")
    log("=" * 60)

    model = load_model(args.model, args.device)

    source_model = model
    target_model = model
    if args.source_model:
        log(f"  Loading source model: {args.source_model}")
        source_model = load_model(args.source_model, args.device)
    if args.target_model:
        log(f"  Loading target model: {args.target_model}")
        target_model = load_model(args.target_model, args.device)

    source_hook = None
    target_hook = None
    if args.layer_source is not None:
        source_hook = f"blocks.{args.layer_source}.hook_resid_pre"
    if args.layer_target is not None:
        target_hook = f"blocks.{args.layer_target}.hook_resid_pre"

    out = args.out or "111_cross_model_transfer.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_cross_model_transfer(
            model=model,
            tasks=[task],
            source_model=source_model,
            target_model=target_model,
            source_hook=source_hook,
            target_hook=target_hook,
            n_prompts=args.n_prompts,
            n_mapping_prompts=args.n_mapping_prompts,
            coefficients=args.coefficients,
        )
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)
            p = "PASS" if r.metadata["passed"] else "FAIL"
            log(f"  {task}: transfer_fidelity={r.value:.4f}  [{p}]")

    save_results(results, out)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
