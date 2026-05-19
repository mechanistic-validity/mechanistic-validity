"""Cross-Task Representation Overlap
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     E10 — Cross-Task Overlap
Categories:     representational
Validity layer: External
Criteria:       E5 Cross-task
Establishes:    Task representations diverge at circuit-active layers
Requires:       GPU, model
Doc:            /instruments_v2/representational/e10-cross-task-overlap
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For pairs of tasks, measures how much their residual stream
representations overlap at each layer.

Two metrics per layer:
  (1) Subspace overlap via principal angles between the top-k PCA
      subspaces of each task's activations.
  (2) CKA (Centered Kernel Alignment) between activation matrices.

Expected pattern: high overlap at early layers (shared input
representations) decreasing at later layers (task-specific computation
emerges).

Usage:
    uv run python 63_cross_task_overlap.py --tasks ioi sva greater_than --device cpu
    uv run python 63_cross_task_overlap.py --device cuda --n-prompts 40
"""
from itertools import combinations

import numpy as np
import torch

from mechanistic_validity.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    generate_prompts,
    get_circuit_heads,
    load_model,
    log,
    parse_common_args,
    save_results,
)


def subspace_overlap(X: np.ndarray, Y: np.ndarray, k: int = 10) -> float:
    """Compute subspace overlap via mean squared cosine of principal angles.

    X, Y: (n_samples, d). Returns mean cos^2(theta_i) for i=1..k.
    Value in [0, 1]: 1 = identical subspaces, 0 = orthogonal.
    """
    # Center
    X_c = X - X.mean(axis=0, keepdims=True)
    Y_c = Y - Y.mean(axis=0, keepdims=True)

    # Top-k right singular vectors (principal directions)
    _, _, Vx = np.linalg.svd(X_c, full_matrices=False)
    _, _, Vy = np.linalg.svd(Y_c, full_matrices=False)

    k = min(k, Vx.shape[0], Vy.shape[0])
    Vx_k = Vx[:k].T  # (d, k)
    Vy_k = Vy[:k].T  # (d, k)

    # Principal angles via SVD of Vx_k.T @ Vy_k
    M = Vx_k.T @ Vy_k  # (k, k)
    singular_values = np.linalg.svd(M, compute_uv=False)
    # Clamp to [0, 1] for numerical stability
    cos_angles = np.clip(singular_values, 0.0, 1.0)

    return float(np.mean(cos_angles ** 2))


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute linear CKA (Centered Kernel Alignment).

    X: (n, p), Y: (n, q). Returns CKA in [0, 1].
    """
    n = X.shape[0]
    if n < 3:
        return 0.0

    # Center
    X_c = X - X.mean(axis=0, keepdims=True)
    Y_c = Y - Y.mean(axis=0, keepdims=True)

    # Linear kernel: K = X @ X.T, L = Y @ Y.T
    # CKA = ||Y.T @ X||_F^2 / (||X.T @ X||_F * ||Y.T @ Y||_F)
    XTX = X_c.T @ X_c
    YTY = Y_c.T @ Y_c
    YTX = Y_c.T @ X_c

    num = np.sum(YTX ** 2)
    denom = np.sqrt(np.sum(XTX ** 2) * np.sum(YTY ** 2))

    if denom < 1e-12:
        return 0.0
    return float(num / denom)


@torch.no_grad()
def collect_residual_activations(model, prompts, layer: int) -> np.ndarray:
    """Collect last-token residual stream. Returns (n_prompts, d_model)."""
    activations = []
    hook_name = f"blocks.{layer}.hook_resid_post"
    for p in prompts:
        tokens = model.to_tokens(p.text)
        _, cache = model.run_with_cache(tokens, names_filter=lambda n: n == hook_name)
        act = cache[hook_name][0, -1].cpu().float().numpy()
        activations.append(act)
    return np.stack(activations, axis=0)


@torch.no_grad()
def main():
    parser = parse_common_args("E63: Cross-Task Representation Overlap")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS[:4]  # Limit default to avoid combinatorial explosion
    model = load_model(args.model, args.device)
    tokenizer = model.tokenizer
    n_layers = model.cfg.n_layers

    log("=" * 60)
    log("E63: CROSS-TASK REPRESENTATION OVERLAP")
    log("=" * 60)
    log(f"Tasks: {tasks} ({len(list(combinations(tasks, 2)))} pairs)")

    # Pre-collect activations for all tasks at all layers
    task_activations = {}
    for task in tasks:
        prompts = generate_prompts(task, tokenizer, args.n_prompts)
        if not prompts:
            log(f"  {task}: no prompts, skipping")
            continue
        log(f"  Collecting activations for {task} ({len(prompts)} prompts)...")
        layer_acts = {}
        for layer in range(n_layers):
            layer_acts[layer] = collect_residual_activations(model, prompts, layer)
        task_activations[task] = layer_acts

    results = []
    valid_tasks = [t for t in tasks if t in task_activations]

    for task_a, task_b in combinations(valid_tasks, 2):
        log(f"  Comparing {task_a} vs {task_b}...")
        acts_a = task_activations[task_a]
        acts_b = task_activations[task_b]

        # Use min samples for CKA (requires same n)
        n_shared = min(acts_a[0].shape[0], acts_b[0].shape[0])

        per_layer_overlap = []
        per_layer_cka = []

        for layer in range(n_layers):
            Xa = acts_a[layer][:n_shared]
            Xb = acts_b[layer][:n_shared]

            overlap = subspace_overlap(Xa, Xb, k=10)
            cka = linear_cka(Xa, Xb)

            per_layer_overlap.append(overlap)
            per_layer_cka.append(cka)

        # Measure the decline from early to late layers
        early_cka = float(np.mean(per_layer_cka[:n_layers // 3]))
        late_cka = float(np.mean(per_layer_cka[2 * n_layers // 3:]))
        cka_decline = early_cka - late_cka

        early_overlap = float(np.mean(per_layer_overlap[:n_layers // 3]))
        late_overlap = float(np.mean(per_layer_overlap[2 * n_layers // 3:]))

        log(f"    CKA: early={early_cka:.3f}, late={late_cka:.3f}, decline={cka_decline:.3f}")
        log(f"    Overlap: early={early_overlap:.3f}, late={late_overlap:.3f}")

        pair_name = f"{task_a}_vs_{task_b}"
        results.append(EvalResult(
            metric_id="E63.cka_decline",
            value=cka_decline,
            n_samples=n_shared,
            metadata={
                "task_pair": pair_name,
                "early_cka": early_cka,
                "late_cka": late_cka,
                "per_layer_cka": per_layer_cka,
            },
        ))
        results.append(EvalResult(
            metric_id="E63.subspace_overlap",
            value=float(np.mean(per_layer_overlap)),
            n_samples=n_shared,
            metadata={
                "task_pair": pair_name,
                "early_overlap": early_overlap,
                "late_overlap": late_overlap,
                "per_layer_overlap": per_layer_overlap,
            },
        ))

    out = args.out or "63_cross_task_overlap.json"
    save_results(results, out)
    log(f"\nDone. {len(results)} metrics across {len(list(combinations(valid_tasks, 2)))} pairs.")


if __name__ == "__main__":
    main()
