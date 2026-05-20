"""Network Motifs in Circuit Graphs
Tests whether the circuit contains enriched network motifs (feed-forward
loops, fan-out, fan-in) compared to random graphs with the same degree
sequence. Motif enrichment indicates non-random functional organization.

Pass: at least one motif Z-score > 2.0
Ref: Milo et al. 2002, Science 298:824-827

Usage:
    uv run python CB1_network_motifs.py --tasks ioi --n-prompts 40
"""

from collections import Counter

import numpy as np
import torch

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    InstrumentInfo,
    generate_prompts,
    get_circuit_heads,
    get_circuit_info,
    get_token_ids,
    load_model,
    log,
    logit_diff_from_logits,
    parse_common_args,
    save_incremental,
    save_results,
)

INSTRUMENT_INFO = InstrumentInfo(
    name="Network Motifs",
    paper_ref="Milo et al. 2002, Science 298:824-827",
    paper_cite="Milo et al. 2002, Network Motifs: Simple Building Blocks of Complex Networks",
    description="Tests whether circuit contains enriched 3-node subgraph motifs compared to random graphs",
    category="cross_discipline",
    tier="cross_discipline",
    origin="established",
)

Z_THRESHOLD = 2.0
N_RANDOM_GRAPHS = 200


def _classify_triad(adj: np.ndarray, i: int, j: int, k: int) -> str:
    """Classify a 3-node subgraph pattern by its edge configuration.

    Returns a canonical string like 'FFL' (feed-forward loop),
    'fan_out', 'fan_in', 'chain', 'mutual', or 'other'.
    """
    edges = (
        adj[i, j], adj[j, i],
        adj[i, k], adj[k, i],
        adj[j, k], adj[k, j],
    )
    n_edges = sum(edges)
    if n_edges == 0:
        return "empty"

    # Feed-forward loop: A->B, B->C, A->C
    if adj[i, j] and adj[j, k] and adj[i, k]:
        return "feed_forward_loop"
    if adj[i, k] and adj[k, j] and adj[i, j]:
        return "feed_forward_loop"
    if adj[j, i] and adj[i, k] and adj[j, k]:
        return "feed_forward_loop"
    if adj[j, k] and adj[k, i] and adj[j, i]:
        return "feed_forward_loop"
    if adj[k, i] and adj[i, j] and adj[k, j]:
        return "feed_forward_loop"
    if adj[k, j] and adj[j, i] and adj[k, i]:
        return "feed_forward_loop"

    # Fan-out: one node sends to both others
    for src in [i, j, k]:
        others = [x for x in [i, j, k] if x != src]
        if adj[src, others[0]] and adj[src, others[1]] and n_edges == 2:
            return "fan_out"

    # Fan-in: one node receives from both others
    for tgt in [i, j, k]:
        others = [x for x in [i, j, k] if x != tgt]
        if adj[others[0], tgt] and adj[others[1], tgt] and n_edges == 2:
            return "fan_in"

    # Chain: A->B->C (no other edges)
    if n_edges == 2:
        return "chain"

    # Mutual: bidirectional edge exists
    if any(adj[a, b] and adj[b, a] for a, b in [(i, j), (i, k), (j, k)]):
        return "mutual"

    return "other"


def _count_motifs(adj: np.ndarray) -> Counter:
    """Count all 3-node subgraph motifs in a directed adjacency matrix."""
    n = adj.shape[0]
    counts: Counter = Counter()
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                motif = _classify_triad(adj, i, j, k)
                if motif != "empty":
                    counts[motif] += 1
    return counts


def _random_rewire(adj: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Degree-preserving random rewiring of a directed graph."""
    edges = list(zip(*np.where(adj > 0)))
    if len(edges) < 2:
        return adj.copy()

    new_adj = adj.copy()
    n_swaps = len(edges) * 2

    for _ in range(n_swaps):
        idx1, idx2 = rng.choice(len(edges), 2, replace=False)
        u1, v1 = edges[idx1]
        u2, v2 = edges[idx2]
        # Swap targets: u1->v2, u2->v1
        if u1 == v2 or u2 == v1:
            continue
        if new_adj[u1, v2] or new_adj[u2, v1]:
            continue
        new_adj[u1, v1] = 0
        new_adj[u2, v2] = 0
        new_adj[u1, v2] = 1
        new_adj[u2, v1] = 1
        edges[idx1] = (u1, v2)
        edges[idx2] = (u2, v1)

    return new_adj


@torch.no_grad()
def run_network_motifs(model, tasks: list[str],
                       n_prompts: int = 40) -> list[EvalResult]:
    """Enumerate 3-node motifs in circuit graph and compare to random baseline."""
    results = []

    for task in tasks:
        circuit, circuit_heads, circuit_edges = get_circuit_info(task)
        if not circuit_heads or not circuit_edges:
            # Fall back: build edges from layer ordering
            circuit_heads = get_circuit_heads(task)
            if len(circuit_heads) < 3:
                log(f"  {task}: need >= 3 circuit heads, skipping")
                continue
            # Create edges: head A -> head B if A's layer < B's layer
            head_list = sorted(circuit_heads)
            circuit_edges = set()
            for i, (la, ha) in enumerate(head_list):
                for lb, hb in head_list[i + 1:]:
                    if la < lb:
                        circuit_edges.add((la, ha, lb, hb))

        if len(circuit_heads) < 3:
            log(f"  {task}: need >= 3 circuit heads, skipping")
            continue

        head_list = sorted(circuit_heads)
        head_idx = {h: i for i, h in enumerate(head_list)}
        n = len(head_list)

        log(f"  {task}: {n} heads, {len(circuit_edges)} edges")

        # Build adjacency matrix
        adj = np.zeros((n, n), dtype=int)
        for la, ha, lb, hb in circuit_edges:
            src = head_idx.get((la, ha))
            dst = head_idx.get((lb, hb))
            if src is not None and dst is not None:
                adj[src, dst] = 1

        # Count motifs in actual circuit
        actual_counts = _count_motifs(adj)
        if not actual_counts:
            log(f"    no motifs found, skipping")
            continue

        # Count motifs in random rewired graphs
        rng = np.random.default_rng(42)
        all_motif_types = set(actual_counts.keys())
        random_counts: dict[str, list[int]] = {m: [] for m in all_motif_types}

        for _ in range(N_RANDOM_GRAPHS):
            rand_adj = _random_rewire(adj, rng)
            rc = _count_motifs(rand_adj)
            for m in all_motif_types:
                random_counts[m].append(rc.get(m, 0))

        # Compute Z-scores
        motif_details = []
        z_scores = []
        for motif_type in sorted(all_motif_types):
            actual = actual_counts[motif_type]
            rand_vals = np.array(random_counts[motif_type])
            rand_mean = float(np.mean(rand_vals))
            rand_std = float(np.std(rand_vals))
            z = (actual - rand_mean) / max(rand_std, 1e-8)
            z_scores.append(z)

            motif_details.append({
                "motif": motif_type,
                "count": int(actual),
                "random_mean": rand_mean,
                "random_std": rand_std,
                "z_score": float(z),
            })

        max_z = float(np.max(z_scores))
        passed = max_z > Z_THRESHOLD

        log(f"    motifs: {dict(actual_counts)}")
        log(f"    max_z={max_z:.2f}  [{'PASS' if passed else 'FAIL'}]")

        results.append(EvalResult(
            metric_id="CB1.network_motifs",
            value=max_z,
            n_samples=n,
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": task,
                "n_heads": n,
                "n_edges": len(circuit_edges),
                "max_z_score": max_z,
                "n_random_graphs": N_RANDOM_GRAPHS,
                "passed": passed,
                "threshold": Z_THRESHOLD,
                "motif_details": sorted(motif_details,
                                        key=lambda d: d["z_score"],
                                        reverse=True),
            },
        ))

    return results


def main():
    parser = parse_common_args("CB1: Network Motifs")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("CB1: NETWORK MOTIFS")
    log("=" * 60)

    out = args.out or "CB1_network_motifs.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_network_motifs(model, [task], args.n_prompts)
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)

    save_results(results, out, args=args)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
