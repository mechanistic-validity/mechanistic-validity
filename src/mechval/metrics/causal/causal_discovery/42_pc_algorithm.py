"""PC Algorithm for Causal Discovery (Spirtes, Glymour & Scheines 2000)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     A13 — Causal Discovery
Categories:     causal, information
Validity layer: Internal
Criteria:       I4 Consistency
Establishes:    PC algorithm recovers circuit structure from observational CI tests (no interventions)
Requires:       GPU, model
Doc:            /instruments_v2/causal/a13-causal-discovery
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The PC algorithm discovers causal DAG structure from observational data
using conditional independence (CI) tests. Unlike NOTEARS (which uses
continuous optimization), PC uses statistical CI tests and produces a
CPDAG (completed partially directed acyclic graph).

Applied to circuits: run the model on many prompts, collect per-head
DLA (Direct Logit Attribution) or activation magnitudes, then run
PC to discover which heads are conditionally independent given others.
Compare the discovered graph to the known circuit structure.

Key insight: PC discovers structure WITHOUT interventions. If PC's
graph matches the intervention-based circuit, the causal structure
is identifiable from observations alone — a much cheaper path to
circuit discovery. If they disagree, the disagreement reveals where
confounding or faithfulness violations hide.

Outputs per task:
  - Discovered adjacency matrix (from PC)
  - Comparison to known circuit edges (precision, recall, SHD)
  - Conditional independence test results for circuit vs non-circuit pairs

Usage:
    uv run python 42_pc_algorithm.py --tasks ioi sva --n-prompts 100
    uv run python 42_pc_algorithm.py --device cuda --alpha 0.05
"""

import numpy as np
import torch
from scipy import stats

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    generate_prompts,
    get_circuit_heads,
    get_all_edges,
    get_circuit,
    get_token_ids,
    load_model,
    log,
    parse_common_args,
    save_results,
)


@torch.no_grad()
def collect_head_activations(
    model, prompts, correct_ids, incorrect_ids,
) -> np.ndarray:
    """Collect per-head DLA (contribution to logit diff) across prompts.

    Returns array of shape (n_prompts, n_layers * n_heads).
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    n_prompts = min(len(prompts), len(correct_ids))

    activations = np.zeros((n_prompts, n_layers * n_heads))

    for i in range(n_prompts):
        tokens = model.to_tokens(prompts[i].text)
        _, cache = model.run_with_cache(tokens, names_filter=lambda n: "hook_result" in n)

        correct_dir = model.W_U[:, correct_ids[i]] - model.W_U[:, incorrect_ids[i]]

        for L in range(n_layers):
            result = cache[f"blocks.{L}.attn.hook_result"]
            for H in range(n_heads):
                head_out = result[0, -1, H, :]
                dla = torch.dot(head_out, correct_dir).item()
                activations[i, L * n_heads + H] = dla

    return activations


def partial_correlation(X: np.ndarray, i: int, j: int, cond_set: list[int]) -> tuple[float, float]:
    """Compute partial correlation between X[:,i] and X[:,j] given X[:,cond_set].

    Returns (partial_corr, p_value).
    """
    n = X.shape[0]
    if len(cond_set) == 0:
        r, p = stats.pearsonr(X[:, i], X[:, j])
        return r, p

    if len(cond_set) >= n - 3:
        return 0.0, 1.0

    Z = X[:, cond_set]
    residuals_i = X[:, i] - Z @ np.linalg.lstsq(Z, X[:, i], rcond=None)[0]
    residuals_j = X[:, j] - Z @ np.linalg.lstsq(Z, X[:, j], rcond=None)[0]

    if np.std(residuals_i) < 1e-10 or np.std(residuals_j) < 1e-10:
        return 0.0, 1.0

    r, p = stats.pearsonr(residuals_i, residuals_j)
    return r, p


def pc_algorithm(
    X: np.ndarray,
    alpha: float = 0.05,
    max_cond_size: int = 3,
    node_subset: list[int] | None = None,
) -> np.ndarray:
    """Run the PC algorithm on data matrix X.

    Args:
        X: (n_samples, n_variables) data matrix
        alpha: significance level for CI tests
        max_cond_size: maximum conditioning set size
        node_subset: if provided, only discover structure among these nodes

    Returns:
        adjacency: (n_nodes, n_nodes) binary adjacency matrix (undirected skeleton)
    """
    if node_subset is not None:
        X_sub = X[:, node_subset]
        n_nodes = len(node_subset)
    else:
        X_sub = X
        n_nodes = X.shape[1]

    adj = np.ones((n_nodes, n_nodes), dtype=int)
    np.fill_diagonal(adj, 0)

    sep_sets: dict[tuple[int, int], list[int]] = {}

    for cond_size in range(max_cond_size + 1):
        edges_to_test = [(i, j) for i in range(n_nodes) for j in range(i+1, n_nodes) if adj[i, j] == 1]

        for i, j in edges_to_test:
            if adj[i, j] == 0:
                continue

            neighbors_i = [k for k in range(n_nodes) if k != j and adj[i, k] == 1]

            if len(neighbors_i) < cond_size:
                continue

            from itertools import combinations
            found_independence = False

            combos = list(combinations(neighbors_i, cond_size))
            if len(combos) > 100:
                combos = combos[:100]

            for cond_set in combos:
                _, p_value = partial_correlation(X_sub, i, j, list(cond_set))

                if p_value > alpha:
                    adj[i, j] = 0
                    adj[j, i] = 0
                    sep_sets[(i, j)] = list(cond_set)
                    sep_sets[(j, i)] = list(cond_set)
                    found_independence = True
                    break

            if found_independence:
                break

    return adj


def compare_to_known_circuit(
    discovered_adj: np.ndarray,
    node_indices: list[int],
    known_edges: set[tuple[int, int, int, int]],
    n_heads: int,
) -> dict:
    """Compare PC-discovered skeleton to known circuit edges."""
    discovered_edges = set()
    n_nodes = len(node_indices)

    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            if discovered_adj[i, j] == 1:
                li, hi = divmod(node_indices[i], n_heads)
                lj, hj = divmod(node_indices[j], n_heads)
                if li < lj:
                    discovered_edges.add((li, hi, lj, hj))
                else:
                    discovered_edges.add((lj, hj, li, hi))

    true_positives = discovered_edges & known_edges
    false_positives = discovered_edges - known_edges
    false_negatives = known_edges - discovered_edges

    precision = len(true_positives) / max(len(discovered_edges), 1)
    recall = len(true_positives) / max(len(known_edges), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    shd = len(false_positives) + len(false_negatives)

    return {
        "n_discovered_edges": len(discovered_edges),
        "n_known_edges": len(known_edges),
        "true_positives": len(true_positives),
        "false_positives": len(false_positives),
        "false_negatives": len(false_negatives),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "structural_hamming_distance": shd,
    }


@torch.no_grad()
def main():
    parser = parse_common_args("A13 — PC Algorithm for Causal Discovery")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Significance level for CI tests")
    parser.add_argument("--max-cond-size", type=int, default=2,
                        help="Max conditioning set size")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)
    n_heads = model.cfg.n_heads
    n_layers = model.cfg.n_layers

    all_results = {}

    for task in tasks:
        log(f"\n{'='*60}")
        log(f"Task: {task}")
        circuit_heads = get_circuit_heads(task)
        if not circuit_heads:
            log(f"  No circuit for {task}, skipping")
            continue

        prompts = generate_prompts(task, model.tokenizer, args.n_prompts)
        if not prompts:
            log(f"  No prompts for {task}, skipping")
            continue

        correct_ids, incorrect_ids = get_token_ids(prompts, model.tokenizer)

        log(f"  Collecting head DLA activations ({len(prompts)} prompts)...")
        X = collect_head_activations(model, prompts, correct_ids, incorrect_ids)
        log(f"  Activation matrix shape: {X.shape}")

        circuit_indices = sorted([L * n_heads + H for L, H in circuit_heads])
        expanded_indices = list(circuit_indices)
        for idx in circuit_indices:
            L, H = divmod(idx, n_heads)
            for dL in [-1, 0, 1]:
                for dH in [-1, 0, 1]:
                    neighbor = (L + dL) * n_heads + (H + dH)
                    if 0 <= neighbor < n_layers * n_heads and neighbor not in expanded_indices:
                        expanded_indices.append(neighbor)
        expanded_indices = sorted(set(expanded_indices))[:40]

        log(f"  Running PC algorithm on {len(expanded_indices)} nodes "
            f"(alpha={args.alpha}, max_cond={args.max_cond_size})...")

        adj = pc_algorithm(
            X, alpha=args.alpha,
            max_cond_size=args.max_cond_size,
            node_subset=expanded_indices,
        )

        n_edges_discovered = int(adj.sum()) // 2
        log(f"  Discovered skeleton: {n_edges_discovered} edges")

        try:
            circuit = get_circuit(task)
            known_edges = get_all_edges(circuit)
        except Exception:
            known_edges = set()

        if known_edges:
            comparison = compare_to_known_circuit(adj, expanded_indices, known_edges, n_heads)
            log(f"  vs known circuit: precision={comparison['precision']:.3f}, "
                f"recall={comparison['recall']:.3f}, F1={comparison['f1']:.3f}, "
                f"SHD={comparison['structural_hamming_distance']}")
        else:
            comparison = None

        all_results[task] = {
            "n_circuit_heads": len(circuit_heads),
            "n_nodes_tested": len(expanded_indices),
            "n_edges_discovered": n_edges_discovered,
            "alpha": args.alpha,
            "max_cond_size": args.max_cond_size,
            "comparison_to_known": comparison,
            "adjacency_nonzero": [[int(expanded_indices[i]), int(expanded_indices[j])]
                                  for i in range(len(expanded_indices))
                                  for j in range(i+1, len(expanded_indices))
                                  if adj[i, j] == 1],
        }

    save_results(all_results, "a13_pc_algorithm.json")
    log("\nDone.")


if __name__ == "__main__":
    main()
