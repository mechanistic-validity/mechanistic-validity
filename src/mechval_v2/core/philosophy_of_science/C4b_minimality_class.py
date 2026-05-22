"""Minimality Classification (Hadad, Katz & Bassan, ICLR 2026)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     C4b — Minimality Class
Categories:     causal
Validity layer: Internal
Criteria:       C4b Minimality Classification
Establishes:    Which minimality class a circuit belongs to
Requires:       CPU or GPU, model
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Classifies circuits into one of four minimality classes from
Hadad, Katz & Bassan (ICLR 2026):

  - Subset-minimal: no proper sub-circuit of size k-1 is faithful
  - Locally minimal: removing any single edge breaks faithfulness
  - Quasi-minimal: at least one edge whose removal breaks faithfulness
  - Not minimal: every edge can be removed without breaking faithfulness

A sub-circuit is "faithful" if faithfulness > 0.5.

Value encoding:
  1.0  = subset_minimal
  0.75 = locally_minimal
  0.5  = quasi_minimal
  0.0  = not_minimal

Pass condition: classified as at least locally_minimal.

Usage:
    uv run python C4b_minimality_class.py --tasks ioi --n-prompts 10
    uv run python C4b_minimality_class.py --tasks ioi sva --device cpu
"""

import itertools
import math

import numpy as np
import torch

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    InstrumentInfo,
    calibrate_mean_z,
    compute_faithfulness,
    generate_prompts,
    get_circuit_info,
    get_token_ids,
    load_model,
    log,
    parse_common_args,
    save_incremental,
    save_results,
)

INSTRUMENT_INFO = InstrumentInfo(
    name="Minimality Classification",
    paper_ref="https://openreview.net/forum?id=ICLR2026_C4b",
    paper_cite="Hadad, Katz & Bassan 2026 (ICLR)",
    description="Four-class minimality hierarchy for mechanistic circuits",
    category="causal",
    tier="core",
    origin="established",
)

FAITHFULNESS_THRESHOLD = 0.5
MAX_SUBSET_SAMPLES = 20

CLASS_VALUES = {
    "subset_minimal": 1.0,
    "locally_minimal": 0.75,
    "quasi_minimal": 0.5,
    "not_minimal": 0.0,
}


def _edges_to_head_set(all_edges: set[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    return sorted(all_edges)


@torch.no_grad()
def _faithfulness_without_edges(
    model, prompts, correct_ids, incorrect_ids,
    all_heads: set[tuple[int, int]], all_edges: list[tuple[int, int, int, int]],
    removed_edges: set[int], mean_z: torch.Tensor,
) -> float:
    """Compute faithfulness using only heads that remain connected after edge removal.

    When edges are removed, heads that lose ALL their connections (both
    incoming and outgoing within the circuit) are also removed.
    """
    remaining_edges = {all_edges[i] for i in range(len(all_edges)) if i not in removed_edges}

    connected_heads = set()
    for ul, uh, dl, dh in remaining_edges:
        connected_heads.add((ul, uh))
        connected_heads.add((dl, dh))

    # Heads with no edges at all still count if they were in the original circuit
    # and have no edges to begin with — but for minimality we care about the
    # sub-circuit defined by remaining edges, so use connected heads only.
    if not connected_heads:
        return 0.0

    return compute_faithfulness(
        model, prompts, correct_ids, incorrect_ids, connected_heads, mean_z
    )


@torch.no_grad()
def classify_minimality(
    model, task: str, n_prompts: int = 10,
) -> tuple[str, dict]:
    """Classify the circuit's minimality class.

    Returns (class_name, details_dict).
    """
    circuit, all_heads, all_edges_set = get_circuit_info(task)
    if circuit is None or not all_heads:
        return "not_minimal", {"reason": "no_circuit"}

    prompts = generate_prompts(task, model.tokenizer, n_prompts)
    if not prompts:
        return "not_minimal", {"reason": "no_prompts"}

    correct_ids, incorrect_ids = get_token_ids(prompts, model.tokenizer)
    if not correct_ids:
        return "not_minimal", {"reason": "no_token_ids"}

    mean_z = calibrate_mean_z(model, prompts, n_calibration=min(100, len(prompts)))
    edge_list = _edges_to_head_set(all_edges_set)
    n_edges = len(edge_list)

    full_faith = compute_faithfulness(
        model, prompts, correct_ids, incorrect_ids, all_heads, mean_z
    )

    if full_faith < FAITHFULNESS_THRESHOLD:
        return "not_minimal", {
            "reason": "full_circuit_not_faithful",
            "full_faithfulness": full_faith,
        }

    # Test local minimality: ablate each edge individually
    edge_necessary = []
    for i in range(n_edges):
        faith = _faithfulness_without_edges(
            model, prompts, correct_ids, incorrect_ids,
            all_heads, edge_list, {i}, mean_z,
        )
        edge_necessary.append(faith < FAITHFULNESS_THRESHOLD)

    all_necessary = all(edge_necessary)
    any_necessary = any(edge_necessary)

    details = {
        "full_faithfulness": full_faith,
        "n_edges": n_edges,
        "edges_necessary": sum(edge_necessary),
        "edges_total": n_edges,
        "per_edge": {
            f"({e[0]},{e[1]})->({e[2]},{e[3]})": edge_necessary[i]
            for i, e in enumerate(edge_list)
        },
    }

    if all_necessary:
        # Every single edge removal breaks faithfulness — candidate for
        # locally_minimal. Check subset minimality: no sub-circuit of
        # size k-1 edges is faithful.
        if n_edges <= 1:
            details["subset_check"] = "trivial"
            return "subset_minimal", details

        n_subsets = math.comb(n_edges, n_edges - 1)
        rng = np.random.default_rng()

        if n_subsets <= MAX_SUBSET_SAMPLES:
            combos = list(itertools.combinations(range(n_edges), n_edges - 1))
        else:
            combos = []
            seen = set()
            while len(combos) < MAX_SUBSET_SAMPLES:
                removed = int(rng.integers(n_edges))
                if removed not in seen:
                    seen.add(removed)
                    combo = tuple(j for j in range(n_edges) if j != removed)
                    combos.append(combo)

        any_faithful_subset = False
        for combo in combos:
            removed = set(range(n_edges)) - set(combo)
            faith = _faithfulness_without_edges(
                model, prompts, correct_ids, incorrect_ids,
                all_heads, edge_list, removed, mean_z,
            )
            if faith >= FAITHFULNESS_THRESHOLD:
                any_faithful_subset = True
                break

        if not any_faithful_subset:
            details["subset_check"] = "passed"
            return "subset_minimal", details
        else:
            details["subset_check"] = "failed"
            return "locally_minimal", details

    elif any_necessary:
        return "quasi_minimal", details
    else:
        return "not_minimal", details


@torch.no_grad()
def run_minimality_class(model, tasks: list[str], n_prompts: int = 10) -> list[EvalResult]:
    results = []

    for task in tasks:
        log(f"  {task}: classifying minimality...")
        cls, details = classify_minimality(model, task, n_prompts)
        value = CLASS_VALUES[cls]
        passed = cls in ("subset_minimal", "locally_minimal")

        log(f"    class={cls} value={value:.2f} [{'PASS' if passed else 'FAIL'}]")

        results.append(EvalResult(
            metric_id="C4b.minimality_class",
            value=value,
            n_samples=n_prompts,
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": task,
                "minimality_class": cls,
                "passed": passed,
                **details,
            },
        ))

    return results


def main():
    parser = parse_common_args("C4b: Minimality Classification")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("C4b: MINIMALITY CLASSIFICATION")
    log("=" * 60)

    out = args.out or "C4b_minimality_class.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_minimality_class(model, [task], args.n_prompts)
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)
            p = "PASS" if r.metadata["passed"] else "FAIL"
            log(f"  {task}: {r.metadata['minimality_class']} [{p}]")

    save_results(results, out, args=args)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
