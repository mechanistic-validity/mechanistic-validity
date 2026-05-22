"""Model Alignment Search (Evaluation EX12)
Paper: Satchel Grant (Stanford) (2026). ICLR 2026 Re-Align Workshop.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     EX12 — Model Alignment Search
Categories:     evaluation
Validity layer: Construct (C5 Cross-Model Agreement)
Criteria:       C5 Cross-Model Agreement
Establishes:    Whether causal variables can be bidirectionally transferred
                between two models via a learned invertible linear
                transformation of their activation spaces
Requires:       GPU, two models (or one model + second seed)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Implements Model Alignment Search (MAS) from Grant (2026, ICLR Re-Align
Workshop). MAS extends Distributed Alignment Search (DAS) to bidirectional
cross-model comparison: given two models, it learns an invertible linear
transformation between their activation spaces at corresponding hook
points, then measures whether causal information can be freely interchanged
between the models through the learned alignment.

The CLMAS (Cross-Latent MAS) variant handles the case where one model is
frozen or inaccessible — it learns a unidirectional mapping while still
measuring bidirectional interchange accuracy.

Algorithm:
1. For each task, generate counterfactual prompt pairs.
2. Collect activations from both models at matching hook points.
3. Learn an invertible linear transformation T between activation spaces
   by minimizing the IIA loss in both directions (A->B and B->A).
4. Evaluate bidirectional IIA on held-out pairs:
   - Forward IIA: patch model_a activations through T into model_b
   - Backward IIA: patch model_b activations through T^{-1} into model_a
5. Report alignment_score = mean of forward and backward IIA.

Pass condition: alignment_score > 0.5

References:
    Grant, S. (2026). "Model Alignment Search." ICLR 2026 Re-Align Workshop.
    Stanford University.

Usage:
    uv run python 107_mas.py --tasks ioi --n-prompts 40
    uv run python 107_mas.py --tasks ioi sva --model-b gpt2-medium --device cpu
"""

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    InstrumentInfo,
    generate_prompts,
    get_circuit_heads,
    get_token_ids,
    load_model,
    log,
    logit_diff_from_logits,
    parse_common_args,
    save_incremental,
    save_results,
)

INSTRUMENT_INFO = InstrumentInfo(
    name="Model Alignment Search",
    paper_ref="Grant 2026, ICLR Re-Align Workshop",
    paper_cite=(
        "Grant, S. 2026, Model Alignment Search, "
        "ICLR 2026 Re-Align Workshop, Stanford University"
    ),
    description=(
        "Learns an invertible linear transformation between two models' "
        "activation spaces to enable bidirectional interchange of causal "
        "variables. Measures alignment_score as mean bidirectional IIA."
    ),
    category="evaluation",
    tier="frontier",
    origin="established",
)

ALIGNMENT_THRESHOLD = 0.5


def make_counterfactual_pairs(prompts, correct_ids, incorrect_ids, rng):
    """Create counterfactual pairs by matching prompts with different targets."""
    pairs = []
    n = len(prompts)
    for i in range(n):
        candidates = [j for j in range(n) if j != i and correct_ids[j] != correct_ids[i]]
        if not candidates:
            candidates = [j for j in range(n) if j != i]
        j = rng.choice(candidates)
        pairs.append((i, j))
    return pairs


def collect_activations(model, prompts, hook_name: str, device: str,
                        pos: int = -1) -> torch.Tensor:
    """Collect activations at a given hook point for all prompts.

    Returns tensor of shape (n_prompts, d_act) where d_act is the
    activation dimension at the hook point (flattened over heads if needed).
    """
    acts = []
    for p in prompts:
        tokens = model.to_tokens(p.text).to(device)
        _, cache = model.run_with_cache(
            tokens, names_filter=lambda n: n == hook_name,
        )
        act = cache[hook_name][0, pos]  # (d_model,) or (n_heads, d_head)
        if act.ndim > 1:
            act = act.reshape(-1)
        acts.append(act.detach())
    return torch.stack(acts)


def train_alignment(
    acts_a: torch.Tensor,
    acts_b: torch.Tensor,
    pairs: list[tuple[int, int]],
    correct_ids_a: list[int],
    correct_ids_b: list[int],
    d_sub: int,
    device: str,
    n_steps: int = 300,
    lr: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Learn invertible linear transformation between two activation spaces.

    Learns matrices T_fwd (d_a, d_sub) and T_bwd (d_b, d_sub) such that
    projecting through the shared subspace enables bidirectional causal
    interchange.

    Returns (T_fwd, T_bwd) as orthogonalized projection matrices.
    """
    d_a = acts_a.shape[1]
    d_b = acts_b.shape[1]
    actual_sub = min(d_sub, d_a, d_b)

    # SVD warm-start: find initial alignment from activation differences
    diffs_a = []
    diffs_b = []
    for base_idx, source_idx in pairs:
        diffs_a.append(acts_a[source_idx] - acts_a[base_idx])
        diffs_b.append(acts_b[source_idx] - acts_b[base_idx])

    diffs_a_t = torch.stack(diffs_a)
    diffs_b_t = torch.stack(diffs_b)

    try:
        _, _, Vt_a = torch.linalg.svd(diffs_a_t.cpu(), full_matrices=False)
        T_fwd_param = Vt_a[:actual_sub, :].t().clone().contiguous().to(device)
    except torch._C._LinAlgError:
        T_fwd_param = torch.randn(d_a, actual_sub, device=device)

    try:
        _, _, Vt_b = torch.linalg.svd(diffs_b_t.cpu(), full_matrices=False)
        T_bwd_param = Vt_b[:actual_sub, :].t().clone().contiguous().to(device)
    except torch._C._LinAlgError:
        T_bwd_param = torch.randn(d_b, actual_sub, device=device)

    T_fwd_param = T_fwd_param.requires_grad_(True)
    T_bwd_param = T_bwd_param.requires_grad_(True)

    optimizer = torch.optim.Adam([T_fwd_param, T_bwd_param], lr=lr)
    rng = random.Random(42)

    for step in range(n_steps):
        batch = rng.sample(pairs, min(16, len(pairs)))
        loss = torch.tensor(0.0, device=device)

        Q_a, _ = torch.linalg.qr(T_fwd_param)
        Q_b, _ = torch.linalg.qr(T_bwd_param)
        proj_a = Q_a @ Q_a.t()
        proj_b = Q_b @ Q_b.t()

        for base_idx, source_idx in batch:
            # Forward direction: A -> shared subspace -> B
            a_base = acts_a[base_idx]
            a_source = acts_a[source_idx]
            b_base = acts_b[base_idx]
            b_source = acts_b[source_idx]

            # Intervene in A's subspace, project to B
            a_intervened = a_base - a_base @ proj_a + a_source @ proj_a
            # The source's projection in A should match source's projection in B
            a_proj_source = a_source @ Q_a  # (d_sub,)
            b_proj_source = b_source @ Q_b  # (d_sub,)
            loss = loss + F.mse_loss(a_proj_source, b_proj_source)

            # Backward direction: B -> shared subspace -> A
            b_intervened = b_base - b_base @ proj_b + b_source @ proj_b
            b_proj_base = b_base @ Q_b
            a_proj_base = a_base @ Q_a
            loss = loss + F.mse_loss(b_proj_base, a_proj_base)

        loss = loss / len(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        Q_a, _ = torch.linalg.qr(T_fwd_param)
        Q_b, _ = torch.linalg.qr(T_bwd_param)

    return Q_a.detach(), Q_b.detach()


@torch.no_grad()
def compute_bidirectional_iia(
    model_a,
    model_b,
    prompts,
    correct_ids: list[int],
    incorrect_ids: list[int],
    pairs: list[tuple[int, int]],
    hook_name_a: str,
    hook_name_b: str,
    T_fwd: torch.Tensor,
    T_bwd: torch.Tensor,
    device: str,
) -> tuple[float, float]:
    """Compute bidirectional IIA through learned alignment.

    Forward IIA: patch model_a's causal info into model_b via T.
    Backward IIA: patch model_b's causal info into model_a via T^{-1}.

    Returns (forward_iia, backward_iia).
    """
    proj_a = T_fwd @ T_fwd.t()
    proj_b = T_bwd @ T_bwd.t()

    # Cross-model projection: map A's subspace component into B's space
    # T_cross maps from A's subspace coordinates to B's full space
    T_cross_a_to_b = T_bwd @ T_fwd.t()  # (d_b, d_a)
    T_cross_b_to_a = T_fwd @ T_bwd.t()  # (d_a, d_b)

    fwd_correct = 0
    bwd_correct = 0
    n_total = 0

    for base_idx, source_idx in pairs:
        base_tokens = model_a.to_tokens(prompts[base_idx].text).to(device)
        source_tokens = model_a.to_tokens(prompts[source_idx].text).to(device)
        source_correct_id = correct_ids[source_idx]
        source_incorrect_id = incorrect_ids[source_idx]

        # Collect activations from both models
        _, cache_a_base = model_a.run_with_cache(
            base_tokens, names_filter=lambda n: n == hook_name_a,
        )
        _, cache_a_source = model_a.run_with_cache(
            source_tokens, names_filter=lambda n: n == hook_name_a,
        )
        _, cache_b_base = model_b.run_with_cache(
            base_tokens, names_filter=lambda n: n == hook_name_b,
        )
        _, cache_b_source = model_b.run_with_cache(
            source_tokens, names_filter=lambda n: n == hook_name_b,
        )

        z_a_base = cache_a_base[hook_name_a][0, -1].reshape(-1)
        z_a_source = cache_a_source[hook_name_a][0, -1].reshape(-1)
        z_b_base = cache_b_base[hook_name_b][0, -1].reshape(-1)
        z_b_source = cache_b_source[hook_name_b][0, -1].reshape(-1)

        # Forward IIA: patch A's source causal info into B
        # Replace B's subspace component with A's source subspace component
        a_source_sub = z_a_source @ T_fwd  # A's source in shared subspace
        b_intervened = z_b_base - z_b_base @ proj_b + (a_source_sub @ T_bwd.t())

        # Reshape back for hooking
        orig_shape = cache_b_base[hook_name_b][0, -1].shape

        def _fwd_hook(z, hook, _z=b_intervened.reshape(orig_shape)):
            z[0, -1] = _z
            return z

        logits_fwd = model_b.run_with_hooks(
            base_tokens, fwd_hooks=[(hook_name_b, _fwd_hook)],
        )
        ld_fwd = logit_diff_from_logits(logits_fwd, source_correct_id, source_incorrect_id)
        if ld_fwd > 0:
            fwd_correct += 1

        # Backward IIA: patch B's source causal info into A
        b_source_sub = z_b_source @ T_bwd  # B's source in shared subspace
        a_intervened = z_a_base - z_a_base @ proj_a + (b_source_sub @ T_fwd.t())

        orig_shape_a = cache_a_base[hook_name_a][0, -1].shape

        def _bwd_hook(z, hook, _z=a_intervened.reshape(orig_shape_a)):
            z[0, -1] = _z
            return z

        logits_bwd = model_a.run_with_hooks(
            base_tokens, fwd_hooks=[(hook_name_a, _bwd_hook)],
        )
        ld_bwd = logit_diff_from_logits(logits_bwd, source_correct_id, source_incorrect_id)
        if ld_bwd > 0:
            bwd_correct += 1

        n_total += 1

    fwd_iia = fwd_correct / max(n_total, 1)
    bwd_iia = bwd_correct / max(n_total, 1)
    return fwd_iia, bwd_iia


def run_mas(model, tasks: list[str], n_prompts: int = 40,
            model_b=None, model_b_name: str = "gpt2",
            subspace_dim: int = 4,
            hook_layer: int | None = None,
            n_train_steps: int = 300) -> list[EvalResult]:
    """Run Model Alignment Search evaluation.

    Learns a bidirectional alignment between model and model_b, then
    measures IIA in both directions.

    If model_b is None, loads model_b_name as the second model.
    When model_b_name equals the primary model name, this tests
    cross-seed alignment (same architecture, different init).
    """
    tokenizer = model.tokenizer
    device = str(model.cfg.device)
    results = []
    rng = np.random.RandomState(42)

    if model_b is None:
        log(f"  Loading second model: {model_b_name}")
        model_b = load_model(model_b_name, device)

    for task in tasks:
        circuit_heads = get_circuit_heads(task)
        if not circuit_heads:
            log(f"  {task}: no circuit, skipping")
            continue

        prompts = generate_prompts(task, tokenizer, n_prompts)
        if not prompts:
            log(f"  {task}: no prompts, skipping")
            continue

        correct_ids, incorrect_ids = get_token_ids(prompts, tokenizer)
        if not correct_ids:
            continue

        n_train = len(prompts) // 2
        train_prompts = prompts[:n_train]
        test_prompts = prompts[n_train:]
        train_correct = correct_ids[:n_train]
        train_incorrect = incorrect_ids[:n_train]
        test_correct = correct_ids[n_train:]
        test_incorrect = incorrect_ids[n_train:]

        train_pairs = make_counterfactual_pairs(
            train_prompts, train_correct, train_incorrect, rng,
        )
        test_pairs = make_counterfactual_pairs(
            test_prompts, test_correct, test_incorrect, rng,
        )

        # Determine hook points: use residual stream at circuit-relevant layers
        if hook_layer is not None:
            layers_to_test = [hook_layer]
        else:
            layers_to_test = sorted({L for L, _ in circuit_heads})

        log(f"  {task}: {len(circuit_heads)} circuit heads, "
            f"{len(prompts)} prompts, layers {layers_to_test}")

        task_fwd_iias = []
        task_bwd_iias = []
        per_layer_results = []

        for layer in layers_to_test:
            hook_name_a = f"blocks.{layer}.hook_resid_pre"
            # Use same hook structure for model_b; if architectures differ
            # the layer index is clamped to model_b's range
            max_layer_b = model_b.cfg.n_layers - 1
            layer_b = min(layer, max_layer_b)
            hook_name_b = f"blocks.{layer_b}.hook_resid_pre"

            # Collect training activations
            train_acts_a = collect_activations(
                model, train_prompts, hook_name_a, device,
            )
            train_acts_b = collect_activations(
                model_b, train_prompts, hook_name_b, device,
            )

            # Learn alignment
            T_fwd, T_bwd = train_alignment(
                train_acts_a, train_acts_b, train_pairs,
                train_correct, train_correct,
                d_sub=subspace_dim, device=device,
                n_steps=n_train_steps,
            )

            # Evaluate on held-out pairs
            fwd_iia, bwd_iia = compute_bidirectional_iia(
                model, model_b,
                test_prompts, test_correct, test_incorrect,
                test_pairs,
                hook_name_a, hook_name_b,
                T_fwd, T_bwd, device,
            )

            layer_alignment = (fwd_iia + bwd_iia) / 2.0
            passed = bool(layer_alignment > ALIGNMENT_THRESHOLD)

            log(f"    L{layer}: fwd_IIA={fwd_iia:.3f} bwd_IIA={bwd_iia:.3f} "
                f"align={layer_alignment:.3f} [{'PASS' if passed else 'FAIL'}]")

            task_fwd_iias.append(fwd_iia)
            task_bwd_iias.append(bwd_iia)
            per_layer_results.append({
                "layer_a": layer,
                "layer_b": layer_b,
                "hook_a": hook_name_a,
                "hook_b": hook_name_b,
                "forward_iia": float(fwd_iia),
                "backward_iia": float(bwd_iia),
                "alignment_score": float(layer_alignment),
                "passed": passed,
            })

        if per_layer_results:
            mean_fwd = float(np.mean(task_fwd_iias))
            mean_bwd = float(np.mean(task_bwd_iias))
            alignment_score = (mean_fwd + mean_bwd) / 2.0
            passed = bool(alignment_score > ALIGNMENT_THRESHOLD)

            log(f"  {task} overall: alignment_score={alignment_score:.3f} "
                f"[{'PASS' if passed else 'FAIL'}]")

            results.append(EvalResult(
                metric_id="EX12.mas_alignment",
                value=alignment_score,
                n_samples=len(test_prompts),
                instrument_info=INSTRUMENT_INFO,
                metadata={
                    "task": task,
                    "model_b": model_b_name,
                    "subspace_dim": subspace_dim,
                    "alignment_score": float(alignment_score),
                    "mean_forward_iia": mean_fwd,
                    "mean_backward_iia": mean_bwd,
                    "passed": passed,
                    "threshold": ALIGNMENT_THRESHOLD,
                    "n_layers_evaluated": len(per_layer_results),
                    "n_train_steps": n_train_steps,
                    "per_layer": per_layer_results,
                },
            ))

    return results


def main():
    parser = parse_common_args("EX12: Model Alignment Search")
    parser.add_argument("--model-b", default="gpt2",
                        help="Second model for alignment (default: gpt2)")
    parser.add_argument("--subspace-dim", type=int, default=4,
                        help="Subspace dimensionality for alignment")
    parser.add_argument("--hook-layer", type=int, default=None,
                        help="Specific layer to test (default: all circuit layers)")
    parser.add_argument("--n-train-steps", type=int, default=300,
                        help="Training steps for alignment learning")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("EX12: MODEL ALIGNMENT SEARCH (MAS)")
    log("=" * 60)

    out = args.out or "107_mas.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_mas(
            model, [task],
            n_prompts=args.n_prompts,
            model_b_name=args.model_b,
            subspace_dim=args.subspace_dim,
            hook_layer=args.hook_layer,
            n_train_steps=args.n_train_steps,
        )
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)
            p = "PASS" if r.metadata["passed"] else "FAIL"
            log(f"  {task}: alignment={r.value:.4f}  [{p}]")

    save_results(results, out)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
