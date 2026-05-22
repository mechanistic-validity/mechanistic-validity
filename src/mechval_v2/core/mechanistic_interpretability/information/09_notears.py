"""NOTEARS Structure Learning (Activation DAG)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     C09 — NOTEARS DAG Discovery
Categories:     information, causal
Validity layer: Internal
Criteria:       I4 Consistency
Establishes:    Recovers causal DAG structure over circuit heads via continuous optimization
Requires:       GPU, model
Doc:            /instruments_v2/information/c09-notears-dag
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Learns a DAG over component activations using continuous optimization,
recovering which heads causally precede which others without prior
assumptions. Compares to known circuit structure via Structural Hamming
Distance (SHD).

Optional: pip install causal-learn (for PC/GES alternatives).
Falls back to L-BFGS-B NOTEARS implementation.

Usage:
    uv run python 09_notears.py --tasks ioi sva --n-prompts 80
"""

import numpy as np
import torch
from scipy.linalg import expm
from scipy.optimize import minimize

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    generate_prompts,
    get_circuit_heads,
    get_token_ids,
    load_model,
    log,
    logit_diff_from_logits,
    parse_common_args,
    save_results,
)

try:
    from causallearn.search.ConstraintBased.PC import pc as run_pc
    HAS_CAUSAL_LEARN = True
except ImportError:
    HAS_CAUSAL_LEARN = False


def notears_linear(X: np.ndarray, lambda1: float = 0.01,
                   max_iter: int = 100, h_tol: float = 1e-8,
                   w_threshold: float = 0.1) -> np.ndarray:
    """NOTEARS: continuous optimization for DAG structure learning.

    Zheng et al. NeurIPS 2018. Solves:
        min ||X - X @ W||^2 / n + lambda1 * |W|
        s.t. h(W) = tr(e^{W ◦ W}) - d = 0 (acyclicity)
    """
    n, d = X.shape

    def _h(W):
        return np.trace(expm(W * W)) - d

    def _h_grad(W):
        return expm(W * W).T * W * 2

    W_est = np.zeros((d, d))
    rho, alpha, h_prev = 1.0, 0.0, np.inf

    for _ in range(max_iter):
        def _func_grad(w_flat):
            W = w_flat.reshape(d, d)
            R = X - X @ W
            loss = 0.5 / n * (R ** 2).sum()
            h_val = _h(W)
            obj = loss + 0.5 * rho * h_val ** 2 + alpha * h_val + lambda1 * np.abs(W).sum()
            G_loss = -X.T @ R / n
            G_h = _h_grad(W)
            G = G_loss + (rho * h_val + alpha) * G_h + lambda1 * np.sign(W)
            return obj, G.ravel()

        sol = minimize(_func_grad, W_est.ravel(), jac=True,
                       method="L-BFGS-B", options={"maxiter": 100})
        W_est = sol.x.reshape(d, d)

        h_new = _h(W_est)
        if h_new < h_tol:
            break
        if h_new > 0.25 * h_prev:
            rho *= 10
        alpha += rho * h_new
        h_prev = h_new

    W_est[np.abs(W_est) < w_threshold] = 0
    return W_est


def compute_shd(W_true: np.ndarray, W_est: np.ndarray) -> int:
    """Structural Hamming Distance between two adjacency matrices."""
    true_edges = set(zip(*np.nonzero(W_true)))
    est_edges = set(zip(*np.nonzero(W_est)))
    return len(true_edges.symmetric_difference(est_edges))


@torch.no_grad()
def collect_activation_matrix(model, prompts, correct_ids, incorrect_ids,
                               head_subset: list[tuple[int, int]] | None = None):
    """Collect DLA matrix for structure learning.

    Each head's feature is its direct logit attribution (z @ W_O @ diff_dir),
    preserving sign and task-relevant direction. Last column is logit diff.
    Returns (n_prompts, n_heads+1) matrix.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    n_valid = min(len(prompts), len(correct_ids))

    if head_subset is not None:
        head_list = sorted(head_subset)
    else:
        head_list = [(L, H) for L in range(n_layers) for H in range(n_heads)]

    n_components = len(head_list)
    A = np.zeros((n_valid, n_components + 1))

    for i in range(n_valid):
        tokens = model.to_tokens(prompts[i].text)
        logits, cache = model.run_with_cache(tokens, names_filter=lambda n: "hook_z" in n)
        A[i, -1] = logit_diff_from_logits(logits, correct_ids[i], incorrect_ids[i])

        diff_dir = model.W_U[:, correct_ids[i]] - model.W_U[:, incorrect_ids[i]]
        for j, (L, H) in enumerate(head_list):
            z = cache[f"blocks.{L}.attn.hook_z"][0, -1, H]
            A[i, j] = (z @ model.W_O[L, H] @ diff_dir).item()

    return A, head_list


def run_notears(model, tasks: list[str], n_prompts: int = 80) -> list[EvalResult]:
    tokenizer = model.tokenizer
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    results = []
    rng = np.random.RandomState(42)

    for task in tasks:
        circuit_heads = get_circuit_heads(task)
        if not circuit_heads:
            log(f"  {task}: no circuit, skipping")
            continue

        prompts = generate_prompts(task, tokenizer, n_prompts)
        if not prompts:
            continue

        correct_ids, incorrect_ids = get_token_ids(prompts, tokenizer)
        if not correct_ids:
            continue

        top_heads = sorted(circuit_heads) + [(L, H) for L in range(n_layers)
                                              for H in range(n_heads)
                                              if (L, H) not in circuit_heads][:20]
        top_heads = top_heads[:min(30, len(top_heads))]

        log(f"  {task} ({len(circuit_heads)} circuit heads, {len(top_heads)} total, "
            f"{len(prompts)} prompts)...")

        A, head_list = collect_activation_matrix(model, prompts, correct_ids,
                                                  incorrect_ids, top_heads)

        A_norm = (A - A.mean(axis=0)) / (A.std(axis=0) + 1e-8)

        W = notears_linear(A_norm, lambda1=0.01, max_iter=50)

        output_col = len(head_list)
        causal_parents_idx = np.nonzero(W[:, output_col])[0]
        discovered_heads = {head_list[j] for j in causal_parents_idx if j < len(head_list)}

        tp = len(circuit_heads & discovered_heads)
        fp = len(discovered_heads - circuit_heads)
        fn = len(circuit_heads - discovered_heads)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        perm_A = A_norm.copy()
        rng.shuffle(perm_A)
        W_perm = notears_linear(perm_A, lambda1=0.01, max_iter=50)
        perm_parents = np.nonzero(W_perm[:, output_col])[0]
        baseline_n_parents = len(perm_parents)

        log(f"    F1={f1:.3f} (P={precision:.3f}, R={recall:.3f}) "
            f"parents={len(discovered_heads)} baseline_parents={baseline_n_parents}")

        results.append(EvalResult(
            metric_id="C9.notears",
            value=f1,
            baseline_random=float(baseline_n_parents),
            n_samples=len(prompts),
            metadata={
                "task": task,
                "precision": precision,
                "recall": recall,
                "discovered_heads": [f"L{L}H{H}" for L, H in sorted(discovered_heads)],
                "n_parents": len(discovered_heads),
                "n_circuit_heads": len(circuit_heads),
                "n_total_heads_tested": len(head_list),
            },
        ))

    return results


def main():
    parser = parse_common_args("C9: NOTEARS Structure Learning")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("C9: NOTEARS STRUCTURE LEARNING")
    log("=" * 60)

    results = run_notears(model, tasks, args.n_prompts)

    out = args.out or "09_notears.json"
    save_results(results, out)

    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
