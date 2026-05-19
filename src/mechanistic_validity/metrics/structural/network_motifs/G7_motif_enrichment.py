"""Paper-Faithful Network Motif Enrichment (Alon 2007)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     G7 — Network Motif Enrichment
Categories:     structural
Validity layer: Internal
Criteria:       G7 Motif Enrichment (paper-faithful)
Establishes:    Whether the circuit graph has over-represented subgraph motifs
                relative to degree-preserving random rewirings
Requires:       CPU only, no model
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Follows Alon (2007) methodology faithfully:

1. Enumerate all 13 possible 3-node directed subgraph patterns (triads)
   using the triad census convention from graph theory.
2. Enumerate the 4-node bi-fan pattern.
3. Use degree-preserving edge swaps (not Erdos-Renyi) for the null model.
4. Compute z-score profile as the motif signature.
5. Apply Bonferroni correction for multiple testing.

Pass condition: at least one motif enriched at p < 0.05 after correction.

Usage:
    uv run python G7_motif_enrichment.py --tasks ioi --n-random 1000
    uv run python G7_motif_enrichment.py --tasks ioi sva greater_than
"""

import numpy as np
from scipy import stats

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

# All 13 three-node directed subgraph isomorphism classes (triad census).
# Each is identified by its adjacency pattern among nodes (0, 1, 2).
# Represented as frozensets of directed edges.
TRIAD_PATTERNS = {
    "003": frozenset(),
    "012": frozenset({(0, 1)}),
    "102": frozenset({(0, 1), (1, 0)}),
    "021D": frozenset({(0, 1), (0, 2)}),
    "021U": frozenset({(1, 0), (2, 0)}),
    "021C": frozenset({(0, 1), (2, 0)}),
    "111D": frozenset({(0, 1), (1, 0), (0, 2)}),
    "111U": frozenset({(0, 1), (1, 0), (2, 0)}),
    "030T": frozenset({(0, 1), (0, 2), (1, 2)}),
    "030C": frozenset({(0, 1), (1, 2), (2, 0)}),
    "201": frozenset({(0, 1), (1, 0), (0, 2), (2, 0)}),
    "120D": frozenset({(0, 1), (1, 0), (0, 2), (1, 2)}),
    "120U": frozenset({(0, 1), (1, 0), (2, 0), (2, 1)}),
    "120C": frozenset({(0, 1), (1, 0), (0, 2), (2, 1)}),
    "210": frozenset({(0, 1), (1, 0), (0, 2), (2, 0), (1, 2)}),
    "300": frozenset({(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)}),
}


def _edges_to_adj_matrix(nodes: list, edges: set[tuple]) -> np.ndarray:
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    adj = np.zeros((n, n), dtype=np.int8)
    for a, b in edges:
        if a in idx and b in idx:
            adj[idx[a], idx[b]] = 1
    return adj


def _triad_class(adj: np.ndarray, i: int, j: int, k: int) -> frozenset:
    """Determine the triad class for nodes at indices i, j, k."""
    local_edges = set()
    triple = (i, j, k)
    for li, gi in enumerate(triple):
        for lj, gj in enumerate(triple):
            if li != lj and adj[gi, gj]:
                local_edges.add((li, lj))
    return frozenset(local_edges)


def triad_census(nodes: list, edges: set[tuple]) -> dict[str, int]:
    """Count all 16 triad isomorphism classes (13 non-trivial + 003)."""
    adj = _edges_to_adj_matrix(nodes, edges)
    n = len(nodes)
    counts = {name: 0 for name in TRIAD_PATTERNS}

    pattern_to_name = {pattern: name for name, pattern in TRIAD_PATTERNS.items()}

    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                local = _triad_class(adj, i, j, k)
                # Try all 6 permutations to find the canonical form
                perms = [
                    (0, 1, 2), (0, 2, 1), (1, 0, 2),
                    (1, 2, 0), (2, 0, 1), (2, 1, 0),
                ]
                matched = False
                for perm in perms:
                    remapped = frozenset(
                        (perm[a], perm[b]) for a, b in local
                    )
                    if remapped in pattern_to_name:
                        counts[pattern_to_name[remapped]] += 1
                        matched = True
                        break
                if not matched:
                    counts["003"] += 1

    return counts


def count_bifan(nodes: list, edges: set[tuple]) -> int:
    """Count 4-node bi-fan motifs: A->C, A->D, B->C, B->D."""
    adj_sets: dict = {n: set() for n in nodes}
    for a, b in edges:
        if a in adj_sets:
            adj_sets[a].add(b)

    count = 0
    for i in range(len(nodes)):
        a = nodes[i]
        targets_a = adj_sets[a]
        if len(targets_a) < 2:
            continue
        for j in range(i + 1, len(nodes)):
            b = nodes[j]
            shared = targets_a & adj_sets[b]
            n_shared = len(shared)
            if n_shared >= 2:
                count += n_shared * (n_shared - 1) // 2
    return count


def degree_preserving_rewire(nodes: list, edges: set[tuple],
                             n_swaps: int, rng: np.random.Generator) -> set[tuple]:
    """Rewire edges via degree-preserving random swaps.

    Each swap picks two edges (a->b, c->d) and swaps targets to get
    (a->d, c->b), provided no self-loops or multi-edges are created.
    """
    edge_list = list(edges)
    edge_set = set(edges)
    n = len(edge_list)
    if n < 2:
        return set(edge_list)

    for _ in range(n_swaps):
        i = rng.integers(n)
        j = rng.integers(n)
        if i == j:
            continue

        a, b = edge_list[i]
        c, d = edge_list[j]

        if a == d or c == b:
            continue
        if (a, d) in edge_set or (c, b) in edge_set:
            continue

        edge_set.discard((a, b))
        edge_set.discard((c, d))
        edge_set.add((a, d))
        edge_set.add((c, b))

        edge_list[i] = (a, d)
        edge_list[j] = (c, b)

    return edge_set


def run_motif_enrichment(tasks: list[str], n_random: int = N_RANDOM_DEFAULT) -> list[EvalResult]:
    rng = np.random.default_rng()
    results = []

    for task in tasks:
        circuit, all_heads, all_edges = get_circuit_info(task)
        if circuit is None:
            log(f"  {task}: no circuit info, skipping")
            continue

        nodes = sorted(all_heads)
        directed_edges = {((ul, uh), (dl, dh)) for ul, uh, dl, dh in all_edges}

        if len(nodes) < 3:
            log(f"  {task}: only {len(nodes)} nodes, need >=3, skipping")
            continue

        n_edges = len(directed_edges)
        n_swaps = max(n_edges * 10, 100)
        log(f"  {task}: {len(nodes)} nodes, {n_edges} edges, {n_swaps} swaps per rewiring")

        circuit_triads = triad_census(nodes, directed_edges)
        circuit_bifan = count_bifan(nodes, directed_edges)

        # Only analyze non-empty triad classes (skip 003)
        motif_names = [name for name in TRIAD_PATTERNS if name != "003"]
        motif_names.append("bi_fan")

        random_counts = {name: [] for name in motif_names}

        for _ in range(n_random):
            rewired = degree_preserving_rewire(nodes, directed_edges, n_swaps, rng)
            rand_triads = triad_census(nodes, rewired)
            for name in motif_names:
                if name == "bi_fan":
                    random_counts[name].append(count_bifan(nodes, rewired))
                else:
                    random_counts[name].append(rand_triads[name])

        n_tests = len(motif_names)
        corrected_threshold = Z_THRESHOLD
        bonferroni_p = 0.05 / n_tests

        motif_results = {}
        any_enriched = False

        for name in motif_names:
            if name == "bi_fan":
                observed = circuit_bifan
            else:
                observed = circuit_triads[name]

            rand_arr = np.array(random_counts[name], dtype=float)
            mean_rand = float(rand_arr.mean())
            std_rand = float(rand_arr.std())

            if std_rand < 1e-10:
                z_score = 0.0 if abs(observed - mean_rand) < 1e-10 else 100.0
            else:
                z_score = (observed - mean_rand) / std_rand

            # Two-sided p-value from z-score, then check against Bonferroni threshold
            p_value = 2.0 * (1.0 - stats.norm.cdf(abs(z_score)))
            enriched = (z_score > 0) and (p_value < bonferroni_p)
            if enriched:
                any_enriched = True

            motif_results[name] = {
                "observed": observed,
                "mean_random": round(mean_rand, 2),
                "std_random": round(std_rand, 2),
                "z_score": round(z_score, 2),
                "p_value": round(p_value, 6),
                "enriched": enriched,
            }

        z_profile = {name: motif_results[name]["z_score"] for name in motif_names}
        best_z = max(m["z_score"] for m in motif_results.values())
        passed = any_enriched

        log(f"    best_z={best_z:.2f} [{'PASS' if passed else 'FAIL'}]")

        results.append(EvalResult(
            metric_id="G7.motif_enrichment",
            value=best_z,
            n_samples=n_random,
            metadata={
                "task": task,
                "n_nodes": len(nodes),
                "n_edges": n_edges,
                "n_random_graphs": n_random,
                "n_swaps_per_rewiring": n_swaps,
                "bonferroni_p_threshold": round(bonferroni_p, 6),
                "z_profile": z_profile,
                "motifs": motif_results,
                "passed": passed,
            },
        ))

    return results


def main():
    parser = parse_common_args("G7: Motif Enrichment (Alon 2007)")
    parser.add_argument("--n-random", type=int, default=N_RANDOM_DEFAULT,
                        help="Number of rewired graphs for null distribution (default: 1000)")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS

    log("=" * 60)
    log("G7: MOTIF ENRICHMENT (ALON 2007)")
    log("=" * 60)

    out = args.out or "G7_motif_enrichment.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_motif_enrichment([task], n_random=args.n_random)
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)
            p = "PASS" if r.metadata["passed"] else "FAIL"
            log(f"  {task}: best_z={r.value:.2f} [{p}]")

    save_results(results, out, args=args)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
