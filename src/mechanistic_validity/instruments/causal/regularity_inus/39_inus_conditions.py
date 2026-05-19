"""INUS Condition Analysis (Mackie 1965)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     A10 — Regularity/INUS
Categories:     causal
Validity layer: Internal
Criteria:       I1 Necessity
Establishes:    Circuit heads are INUS conditions (non-redundant parts of sufficient subcircuits)
Requires:       GPU, model
Doc:            /instruments_v2/causal/a10-regularity-inus
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Identifies whether circuit heads are INUS conditions for the target behavior:
  Insufficient but Non-redundant parts of an
  Unnecessary but Sufficient condition.

Concretely: a head H is INUS if:
  1. H is part of at least one minimal sufficient subcircuit (Non-redundant)
  2. That subcircuit is not the only sufficient one (Unnecessary)
  3. H alone is not sufficient (Insufficient)

This maps Mackie's regularity theory of causation onto circuits. The key
insight: most circuit heads are INUS conditions, not causes in isolation.
Heads that are individually sufficient are rare (full necessity). Heads
that appear in ALL minimal sufficient sets are stronger causes than those
appearing in only one.

Outputs per task:
  - Minimal sufficient subcircuits (greedy search)
  - Per-head INUS classification
  - Redundancy index: fraction of sufficient sets containing each head

Usage:
    uv run python 39_inus_conditions.py --tasks ioi sva --n-prompts 40
    uv run python 39_inus_conditions.py --device cuda --threshold 0.7
"""
from itertools import combinations

import torch

from mechanistic_validity.instruments.common import (
    CIRCUIT_TASKS,
    EvalResult,
    calibrate_mean_z,
    compute_faithfulness,
    generate_prompts,
    get_circuit_heads,
    get_token_ids,
    load_model,
    log,
    parse_common_args,
    save_results,
)


def find_minimal_sufficient_sets(
    model, prompts, correct_ids, incorrect_ids,
    circuit_heads: set[tuple[int, int]], mean_z: torch.Tensor,
    threshold: float = 0.7, max_set_size: int | None = None,
) -> list[set[tuple[int, int]]]:
    """Find minimal sufficient subcircuits via greedy forward search.

    A set S is sufficient if faithfulness(S) >= threshold.
    A set S is minimal if no proper subset is also sufficient.
    """
    heads_list = sorted(circuit_heads)
    n = len(heads_list)
    if max_set_size is None:
        max_set_size = min(n, 8)

    sufficient_sets: list[set[tuple[int, int]]] = []

    full_faith = compute_faithfulness(
        model, prompts, correct_ids, incorrect_ids, circuit_heads, mean_z
    )
    if full_faith < threshold:
        log(f"  Full circuit faithfulness {full_faith:.3f} < threshold {threshold}, skipping")
        return []

    for size in range(1, max_set_size + 1):
        if size > 5:
            n_combos = 1
            for i in range(size):
                n_combos = n_combos * (n - i) // (i + 1)
            if n_combos > 500:
                log(f"  Skipping size {size} ({n_combos} combos > 500)")
                break

        for combo in combinations(range(n), size):
            subset = {heads_list[i] for i in combo}

            is_superset_of_existing = any(
                existing <= subset for existing in sufficient_sets
            )
            if is_superset_of_existing:
                continue

            faith = compute_faithfulness(
                model, prompts, correct_ids, incorrect_ids, subset, mean_z
            )
            if faith >= threshold:
                sufficient_sets.append(subset)

    minimal_sets = []
    for s in sufficient_sets:
        is_minimal = not any(
            other < s for other in sufficient_sets if other != s
        )
        if is_minimal:
            minimal_sets.append(s)

    return minimal_sets


def classify_inus(
    circuit_heads: set[tuple[int, int]],
    minimal_sufficient_sets: list[set[tuple[int, int]]],
) -> dict[tuple[int, int], dict]:
    """Classify each head's INUS status."""
    n_sets = len(minimal_sufficient_sets)
    results = {}

    for head in sorted(circuit_heads):
        sets_containing = [s for s in minimal_sufficient_sets if head in s]
        n_containing = len(sets_containing)

        if n_sets == 0:
            status = "undetermined"
        elif n_containing == n_sets:
            status = "necessary"
        elif n_containing > 0 and n_sets > 1:
            status = "inus"
        elif n_containing > 0 and n_sets == 1:
            status = "non-redundant_necessary"
        else:
            status = "redundant"

        results[head] = {
            "status": status,
            "in_n_sets": n_containing,
            "total_sets": n_sets,
            "redundancy_index": n_containing / max(n_sets, 1),
        }

    return results


@torch.no_grad()
def main():
    parser = parse_common_args("A10 — INUS Condition Analysis")
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="Sufficiency threshold for faithfulness")
    parser.add_argument("--max-set-size", type=int, default=None,
                        help="Max subcircuit size to search (default: min(n_heads, 8))")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

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
        mean_z = calibrate_mean_z(model, prompts)

        log(f"  Circuit size: {len(circuit_heads)} heads")
        log(f"  Finding minimal sufficient subcircuits (threshold={args.threshold})...")

        minimal_sets = find_minimal_sufficient_sets(
            model, prompts, correct_ids, incorrect_ids,
            circuit_heads, mean_z,
            threshold=args.threshold,
            max_set_size=args.max_set_size,
        )

        log(f"  Found {len(minimal_sets)} minimal sufficient sets")
        for i, s in enumerate(minimal_sets):
            log(f"    Set {i+1}: {sorted(s)}")

        classifications = classify_inus(circuit_heads, minimal_sets)

        status_counts = {}
        for head, info in classifications.items():
            status = info["status"]
            status_counts[status] = status_counts.get(status, 0) + 1
            log(f"    {head}: {status} (in {info['in_n_sets']}/{info['total_sets']} sets)")

        log(f"  Status distribution: {status_counts}")

        all_results[task] = {
            "n_circuit_heads": len(circuit_heads),
            "n_minimal_sufficient_sets": len(minimal_sets),
            "minimal_sufficient_sets": [sorted(list(s)) for s in minimal_sets],
            "classifications": {
                f"L{h[0]}H{h[1]}": info for h, info in classifications.items()
            },
            "status_distribution": status_counts,
            "threshold": args.threshold,
        }

    save_results(all_results, "a10_inus_conditions.json")
    log("\nDone.")


if __name__ == "__main__":
    main()
