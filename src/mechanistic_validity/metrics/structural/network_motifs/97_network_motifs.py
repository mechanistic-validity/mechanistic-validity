"""Network Motif Enrichment (Graph Structure G5)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     G05 — Graph Structure
Categories:     structural
Validity layer: Internal
Criteria:       G5 Network Motif Enrichment
Establishes:    Whether the circuit's directed graph contains over-represented
                subgraph patterns (motifs) compared to random graphs
Requires:       CPU only, no model
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Inspired by Alon (2002) network motif analysis from computational biology.
For each task circuit, we:

1. Build a directed graph from circuit edges (head-to-head connections)
2. Enumerate 3-node directed subgraph patterns (triads):
   - Feed-forward chain: A->B->C
   - Fan-in: A->C, B->C (two nodes feed into one)
   - Fan-out: A->B, A->C (one node feeds two)
3. Enumerate 4-node bi-fan pattern: A->C, A->D, B->C, B->D
4. Generate N=1000 random graphs (Erdos-Renyi) with the same node/edge count
5. Compute z-score: (circuit_count - mean_random) / std_random
6. A motif is "enriched" if z-score > 2.0 (p < 0.05)

Pass condition: at least one motif is significantly enriched.

Usage:
    uv run python 97_network_motifs.py --tasks ioi --n-random 1000
    uv run python 97_network_motifs.py --tasks ioi sva greater_than
"""

import numpy as np

from mechanistic_validity.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    get_circuit_info,
    log,
    parse_common_args,
    save_incremental,
    save_results,
)

Z_THRESHOLD = 2.0
N_RANDOM_DEFAULT = 1000


def _edges_to_adjacency(nodes: list, edges: set[tuple]) -> dict[tuple, set[tuple]]:
    """Build adjacency set: adj[a] = set of nodes a points to."""
    adj: dict[tuple, set[tuple]] = {n: set() for n in nodes}
    for a, b in edges:
        if a in adj:
            adj[a].add(b)
    return adj


def count_triads(nodes: list, adj: dict[tuple, set[tuple]]) -> dict[str, int]:
    """Count 3-node directed motif patterns.

    Patterns counted:
    - feed_forward_chain: A->B->C (and A does NOT directly connect to C)
    - fan_in: A->C and B->C (convergence onto C, where A!=B)
    - fan_out: A->B and A->C (divergence from A, where B!=C)
    """
    chain_count = 0
    fan_in_count = 0
    fan_out_count = 0

    # Fan-out: for each node A, count pairs of distinct targets
    for a in nodes:
        targets = adj[a]
        n_targets = len(targets)
        if n_targets >= 2:
            fan_out_count += n_targets * (n_targets - 1) // 2

    # Fan-in: for each node C, count pairs of distinct sources
    # Build reverse adjacency
    rev: dict[tuple, set[tuple]] = {n: set() for n in nodes}
    for a in nodes:
        for b in adj[a]:
            rev[b].add(a)

    for c in nodes:
        sources = rev[c]
        n_sources = len(sources)
        if n_sources >= 2:
            fan_in_count += n_sources * (n_sources - 1) // 2

    # Feed-forward chain: A->B->C where A does not directly edge to C
    for b in nodes:
        sources_of_b = rev[b]
        targets_of_b = adj[b]
        for a in sources_of_b:
            for c in targets_of_b:
                if a != c and c not in adj[a]:
                    chain_count += 1

    return {
        "feed_forward_chain": chain_count,
        "fan_in": fan_in_count,
        "fan_out": fan_out_count,
    }


def count_bifan(nodes: list, adj: dict[tuple, set[tuple]]) -> int:
    """Count 4-node bi-fan motifs: A->C, A->D, B->C, B->D.

    For every pair of sources (A, B) that share at least 2 common targets,
    count C(shared, 2) bi-fan instances.
    """
    count = 0
    for i in range(len(nodes)):
        a = nodes[i]
        targets_a = adj[a]
        if len(targets_a) < 2:
            continue
        for j in range(i + 1, len(nodes)):
            b = nodes[j]
            shared = targets_a & adj[b]
            n_shared = len(shared)
            if n_shared >= 2:
                count += n_shared * (n_shared - 1) // 2
    return count


def count_all_motifs(nodes: list, edges: set[tuple]) -> dict[str, int]:
    """Count all motif patterns in a directed graph."""
    adj = _edges_to_adjacency(nodes, edges)
    counts = count_triads(nodes, adj)
    counts["bi_fan"] = count_bifan(nodes, adj)
    return counts


def generate_random_graph_edges(nodes: list, n_edges: int, rng: np.random.Generator) -> set[tuple]:
    """Generate an Erdos-Renyi random directed graph with the given number of edges.

    Edges are sampled uniformly from all possible directed pairs (no self-loops).
    """
    n = len(nodes)
    if n < 2:
        return set()
    all_possible = [(nodes[i], nodes[j]) for i in range(n) for j in range(n) if i != j]
    max_edges = len(all_possible)
    n_edges = min(n_edges, max_edges)
    if n_edges == 0:
        return set()
    chosen_indices = rng.choice(max_edges, size=n_edges, replace=False)
    return {all_possible[idx] for idx in chosen_indices}


def run_network_motifs(tasks: list[str], n_random: int = N_RANDOM_DEFAULT) -> list[EvalResult]:
    rng = np.random.default_rng(42)
    results = []

    for task in tasks:
        circuit, all_heads, all_edges = get_circuit_info(task)
        if circuit is None:
            log(f"  {task}: no circuit info, skipping")
            continue

        # Build directed graph from circuit edges
        # Nodes are (layer, head) tuples; edges are directed connections
        nodes = sorted(all_heads)
        directed_edges = {((ul, uh), (dl, dh)) for ul, uh, dl, dh in all_edges}

        if len(nodes) < 3:
            log(f"  {task}: only {len(nodes)} nodes, need >=3 for triad analysis, skipping")
            continue

        n_edges = len(directed_edges)
        log(f"  {task}: {len(nodes)} nodes, {n_edges} edges")

        # Count motifs in the actual circuit
        circuit_counts = count_all_motifs(nodes, directed_edges)
        log(f"    circuit motifs: {circuit_counts}")

        # Count motifs in random graphs
        motif_names = list(circuit_counts.keys())
        random_counts = {name: [] for name in motif_names}

        for _ in range(n_random):
            rand_edges = generate_random_graph_edges(nodes, n_edges, rng)
            rand_motifs = count_all_motifs(nodes, rand_edges)
            for name in motif_names:
                random_counts[name].append(rand_motifs[name])

        # Compute z-scores
        motif_results = {}
        any_enriched = False
        for name in motif_names:
            observed = circuit_counts[name]
            rand_arr = np.array(random_counts[name], dtype=float)
            mean_rand = float(rand_arr.mean())
            std_rand = float(rand_arr.std())

            if std_rand < 1e-10:
                # All random graphs have the same count; cap at 100 to avoid inf
                z_score = 0.0 if abs(observed - mean_rand) < 1e-10 else 100.0
            else:
                z_score = (observed - mean_rand) / std_rand

            enriched = z_score > Z_THRESHOLD
            if enriched:
                any_enriched = True

            motif_results[name] = {
                "observed": observed,
                "mean_random": round(mean_rand, 2),
                "std_random": round(std_rand, 2),
                "z_score": round(z_score, 2),
                "enriched": enriched,
            }

            status = "ENRICHED" if enriched else "not enriched"
            log(f"    {name}: observed={observed} random={mean_rand:.1f}+/-{std_rand:.1f} "
                f"z={z_score:.2f} [{status}]")

        passed = any_enriched
        # Best z-score as the summary value
        best_z = max(m["z_score"] for m in motif_results.values())

        log(f"    result: best_z={best_z:.2f} [{'PASS' if passed else 'FAIL'}]")

        results.append(EvalResult(
            metric_id="G5.network_motif_enrichment",
            value=best_z,
            n_samples=n_random,
            metadata={
                "task": task,
                "n_nodes": len(nodes),
                "n_edges": n_edges,
                "n_random_graphs": n_random,
                "z_threshold": Z_THRESHOLD,
                "motifs": motif_results,
                "passed": passed,
            },
        ))

    return results


def main():
    parser = parse_common_args("G5: Network Motif Enrichment")
    parser.add_argument("--n-random", type=int, default=N_RANDOM_DEFAULT,
                        help="Number of random graphs for null distribution (default: 1000)")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS

    log("=" * 60)
    log("G5: NETWORK MOTIF ENRICHMENT")
    log("=" * 60)

    out = args.out or "97_network_motifs.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_network_motifs([task], n_random=args.n_random)
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)
            p = "PASS" if r.metadata["passed"] else "FAIL"
            log(f"  {task}: best_z={r.value:.2f} [{p}]")

    save_results(results, out)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
