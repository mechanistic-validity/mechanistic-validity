"""Circuit Metric Distance (Metric #38)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     B06 — Template Distance
Categories:     structural
Validity layer: Construct
Criteria:       C3 Task specificity
Establishes:    Different tasks produce structurally distinct circuits
Requires:       CPU, data-only
Doc:            /instruments_v2/structural/b06-template-distance
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For each pair of tasks, compute the graph-edit distance between their
circuits. Uses Jaccard distance on the head sets as the primary metric.
Also computes: number of shared heads, number of unique heads per
circuit, and normalized overlap. Reports a task x task distance matrix.

Usage:
    uv run python 26_cmd.py --tasks ioi sva induction
    uv run python 26_cmd.py --device cpu
"""

import numpy as np

from mechanistic_validity.instruments.common import (
    CIRCUIT_TASKS,
    EvalResult,
    get_circuit_heads,
    log,
    parse_common_args,
    save_results,
)


def jaccard_distance(set_a: set, set_b: set) -> float:
    """Jaccard distance = 1 - |A & B| / |A | B|."""
    union = set_a | set_b
    if not union:
        return 0.0
    intersection = set_a & set_b
    return 1.0 - len(intersection) / len(union)


def run_cmd(tasks: list[str]) -> list[EvalResult]:
    results = []

    # Gather circuit heads for all tasks
    task_heads = {}
    for task in tasks:
        heads = get_circuit_heads(task)
        if heads:
            task_heads[task] = heads

    valid_tasks = sorted(task_heads.keys())
    if len(valid_tasks) < 2:
        log("  Need at least 2 tasks with circuits for CMD analysis")
        return results

    log(f"  Computing pairwise distances for {len(valid_tasks)} tasks...")

    # Build distance matrix
    n = len(valid_tasks)
    distance_matrix = np.zeros((n, n))
    overlap_matrix = np.zeros((n, n))
    pairwise_details = {}

    for i in range(n):
        for j in range(i + 1, n):
            task_a = valid_tasks[i]
            task_b = valid_tasks[j]
            heads_a = task_heads[task_a]
            heads_b = task_heads[task_b]

            jd = jaccard_distance(heads_a, heads_b)
            shared = heads_a & heads_b
            only_a = heads_a - heads_b
            only_b = heads_b - heads_a
            union_size = len(heads_a | heads_b)
            normalized_overlap = len(shared) / union_size if union_size > 0 else 0.0

            distance_matrix[i, j] = jd
            distance_matrix[j, i] = jd
            overlap_matrix[i, j] = normalized_overlap
            overlap_matrix[j, i] = normalized_overlap

            pair_key = f"{task_a}_vs_{task_b}"
            pairwise_details[pair_key] = {
                "jaccard_distance": jd,
                "n_shared": len(shared),
                "n_only_a": len(only_a),
                "n_only_b": len(only_b),
                "n_union": union_size,
                "normalized_overlap": normalized_overlap,
                "shared_heads": sorted(shared),
            }

            log(f"    {task_a} vs {task_b}: "
                f"jaccard={jd:.3f}, shared={len(shared)}, "
                f"only_{task_a}={len(only_a)}, only_{task_b}={len(only_b)}")

    # Aggregate statistics
    upper_tri = distance_matrix[np.triu_indices(n, k=1)]
    mean_distance = float(np.mean(upper_tri))
    std_distance = float(np.std(upper_tri))
    min_distance = float(np.min(upper_tri))
    max_distance = float(np.max(upper_tri))

    log(f"\n  Mean Jaccard distance: {mean_distance:.4f} +/- {std_distance:.4f}")
    log(f"  Range: [{min_distance:.4f}, {max_distance:.4f}]")

    results.append(EvalResult(
        metric_id="C26.cmd",
        value=mean_distance,
        n_samples=len(upper_tri),
        metadata={
            "tasks": valid_tasks,
            "distance_matrix": distance_matrix.tolist(),
            "overlap_matrix": overlap_matrix.tolist(),
            "pairwise_details": pairwise_details,
            "mean_jaccard_distance": mean_distance,
            "std_jaccard_distance": std_distance,
            "min_jaccard_distance": min_distance,
            "max_jaccard_distance": max_distance,
            "circuit_sizes": {t: len(h) for t, h in task_heads.items()},
        },
    ))

    return results


def main():
    parser = parse_common_args("C26: Circuit Metric Distance")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS

    log("=" * 60)
    log("C26: CIRCUIT METRIC DISTANCE")
    log("=" * 60)

    results = run_cmd(tasks)

    out = args.out or "26_cmd.json"
    save_results(results, out)

    log(f"\nDone. {len(results)} results.")
    for r in results:
        log(f"  Mean Jaccard distance: {r.value:.4f}")


if __name__ == "__main__":
    main()
