"""Niche Partitioning of Circuit Heads
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Metric:         EC1 — Niche Partitioning
Categories:     cross_discipline, ecology
Evidence family: cross_discipline

Tests whether circuit heads specialize on non-overlapping subsets of
prompts, analogous to niche partitioning in ecological communities.
Low pairwise Jaccard overlap indicates functional specialization.

Pass condition: mean Jaccard < 0.5

Usage:
    uv run python EC1_niche_partitioning.py --tasks ioi sva --n-prompts 40
"""

import numpy as np
import torch

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    InstrumentInfo,
    generate_prompts,
    get_circuit_heads,
    load_model,
    log,
    parse_common_args,
    save_incremental,
    save_results,
)

INSTRUMENT_INFO = InstrumentInfo(
    name="Niche Partitioning",
    paper_ref="MacArthur 1958, Ecology 39:599-619",
    paper_cite="MacArthur 1958, Population Ecology of Some Warblers of Northeastern Coniferous Forests",
    description="Tests whether circuit heads specialize on non-overlapping prompt subsets (Jaccard overlap)",
    category="cross_discipline",
    tier="cross_discipline",
    origin="established",
)

JACCARD_THRESHOLD = 0.5
TOP_K_FRACTION = 0.5


@torch.no_grad()
def run_niche_partitioning(model, tasks: list[str],
                           n_prompts: int = 40) -> list[EvalResult]:
    """Measure pairwise Jaccard overlap of top-activated prompts across circuit heads."""
    tokenizer = model.tokenizer
    results = []

    for task in tasks:
        circuit_heads = get_circuit_heads(task)
        if len(circuit_heads) < 2:
            log(f"  {task}: need >= 2 circuit heads, skipping")
            continue

        prompts = generate_prompts(task, tokenizer, n_prompts)
        if not prompts:
            continue

        n_valid = len(prompts)
        head_list = sorted(circuit_heads)

        log(f"  {task}: {len(head_list)} heads, {n_valid} prompts")

        # Collect activation magnitudes per head per prompt
        # activations[h_idx, prompt_idx] = magnitude
        activations = np.zeros((len(head_list), n_valid))

        for p_idx in range(n_valid):
            tokens = model.to_tokens(prompts[p_idx].text)
            _, cache = model.run_with_cache(
                tokens, names_filter=lambda n: "hook_z" in n)
            for h_idx, (L, H) in enumerate(head_list):
                z = cache[f"blocks.{L}.attn.hook_z"]
                activations[h_idx, p_idx] = z[0, -1, H].norm().item()

        # For each head, find top-k activated prompts
        top_k = max(1, int(n_valid * TOP_K_FRACTION))
        top_sets = []
        for h_idx in range(len(head_list)):
            top_indices = set(np.argsort(activations[h_idx])[-top_k:])
            top_sets.append(top_indices)

        # Pairwise Jaccard
        jaccard_values = []
        pair_details = []
        for i in range(len(head_list)):
            for j in range(i + 1, len(head_list)):
                intersection = len(top_sets[i] & top_sets[j])
                union = len(top_sets[i] | top_sets[j])
                jaccard = intersection / union if union > 0 else 0.0
                jaccard_values.append(jaccard)
                pair_details.append({
                    "head_a": f"L{head_list[i][0]}H{head_list[i][1]}",
                    "head_b": f"L{head_list[j][0]}H{head_list[j][1]}",
                    "jaccard": float(jaccard),
                })

        mean_jaccard = float(np.mean(jaccard_values))
        passed = mean_jaccard < JACCARD_THRESHOLD

        log(f"    mean_jaccard={mean_jaccard:.3f}  [{'PASS' if passed else 'FAIL'}]")

        results.append(EvalResult(
            metric_id="EC1.niche_partitioning",
            value=mean_jaccard,
            n_samples=len(prompts),
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": task,
                "n_heads": len(head_list),
                "mean_jaccard": mean_jaccard,
                "min_jaccard": float(np.min(jaccard_values)),
                "max_jaccard": float(np.max(jaccard_values)),
                "top_k": top_k,
                "n_pairs": len(jaccard_values),
                "passed": passed,
                "threshold": JACCARD_THRESHOLD,
                "pair_details": sorted(pair_details, key=lambda d: d["jaccard"])[:10],
            },
        ))

    return results


def main():
    parser = parse_common_args("EC1: Niche Partitioning")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("EC1: NICHE PARTITIONING")
    log("=" * 60)

    out = args.out or "EC1_niche_partitioning.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_niche_partitioning(model, [task], args.n_prompts)
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)

    save_results(results, out, args=args)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
