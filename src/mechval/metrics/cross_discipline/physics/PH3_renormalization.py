"""Renormalization Group: Scale Invariance Under Coarse-Graining
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Metric:         PH3 — Renormalization Group
Categories:     cross_discipline, physics
Tier:           cross_discipline
Origin:         established

Tests whether a circuit's behavior is preserved when heads are
coarse-grained (merged by averaging activation pairs within each
layer). A circuit whose logit diff survives progressive merging has
scale-invariant structure analogous to RG fixed points.

Pass: faithfulness > 0.7 after merging 50% of circuit heads.
Ref: Wilson 1971, Renormalization Group and Critical Phenomena,
     Physical Review B 4:3174.

Usage:
    uv run python PH3_renormalization.py --tasks ioi sva --n-prompts 40
"""

import numpy as np
import torch

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    InstrumentInfo,
    generate_prompts,
    get_circuit_heads,
    get_token_ids,
    load_model,
    log,
    logit_diff_from_logits,
    parse_common_args,
    save_incremental,
    save_results,
)

INSTRUMENT_INFO = InstrumentInfo(
    name="Renormalization Group",
    paper_ref="Wilson 1971",
    paper_cite="Wilson 1971, Renormalization Group and Critical Phenomena, Physical Review B 4:3174",
    description="Tests circuit robustness under coarse-graining by progressively merging head activations",
    category="cross_discipline",
    tier="cross_discipline",
    origin="established",
)

FAITHFULNESS_THRESHOLD = 0.7


@torch.no_grad()
def run_renormalization(model, tasks: list[str],
                        n_prompts: int = 40) -> list[EvalResult]:
    tokenizer = model.tokenizer
    results = []

    for task in tasks:
        circuit_heads = get_circuit_heads(task)
        if not circuit_heads:
            log(f"  {task}: no circuit heads, skipping")
            continue

        prompts = generate_prompts(task, tokenizer, n_prompts)
        if not prompts:
            continue

        correct_ids, incorrect_ids = get_token_ids(prompts, tokenizer)
        if not correct_ids:
            continue

        log(f"  {task}: {len(circuit_heads)} heads, {len(prompts)} prompts")

        heads_by_layer: dict[int, list[int]] = {}
        for L, H in circuit_heads:
            heads_by_layer.setdefault(L, []).append(H)

        # Determine merge levels: each level merges pairs within each layer
        # Level 0 = original, level k merges 2^k heads into 1
        max_heads_in_layer = max(len(hs) for hs in heads_by_layer.values())
        max_levels = 0
        tmp = max_heads_in_layer
        while tmp > 1:
            max_levels += 1
            tmp = (tmp + 1) // 2
        # Include at least 1 merge level even if all layers have single heads
        max_levels = max(max_levels, 1)

        n_valid = min(len(prompts), len(correct_ids))

        # Baseline: full-model logit diffs
        baseline_lds = []
        for idx in range(n_valid):
            tokens = model.to_tokens(prompts[idx].text)
            logits = model(tokens)
            ld = logit_diff_from_logits(logits, correct_ids[idx], incorrect_ids[idx])
            baseline_lds.append(ld)
        baseline_mean = float(np.mean(baseline_lds))

        coarse_grain_curve = []

        for level in range(max_levels + 1):
            # Build merge plan: for each layer, group heads into clusters of 2^level
            # and average their activations, writing result to the first head, zeroing rest
            cluster_size = 2 ** level
            merge_plan: dict[int, list[tuple[list[int], int]]] = {}
            total_original = sum(len(hs) for hs in heads_by_layer.values())
            total_after = 0

            for L, hs in heads_by_layer.items():
                sorted_hs = sorted(hs)
                clusters = []
                for i in range(0, len(sorted_hs), cluster_size):
                    cluster = sorted_hs[i:i + cluster_size]
                    # The representative head is the first in each cluster
                    clusters.append((cluster, cluster[0]))
                    total_after += 1
                merge_plan[L] = clusters

            fraction_merged = 1.0 - total_after / max(total_original, 1)

            # Build hooks that implement the merge
            hooks = []
            for L, clusters in merge_plan.items():
                def _hook(z, hook, _clusters=clusters):
                    for cluster, rep in _clusters:
                        if len(cluster) == 1:
                            continue
                        # Average all heads in cluster, write to representative, zero others
                        avg = torch.mean(
                            torch.stack([z[0, :, h, :] for h in cluster]), dim=0)
                        z[0, :, rep, :] = avg
                        for h in cluster:
                            if h != rep:
                                z[0, :, h, :] = 0.0
                    return z
                hooks.append((f"blocks.{L}.attn.hook_z", _hook))

            merged_lds = []
            for idx in range(n_valid):
                tokens = model.to_tokens(prompts[idx].text)
                logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
                ld = logit_diff_from_logits(logits, correct_ids[idx], incorrect_ids[idx])
                merged_lds.append(ld)

            merged_mean = float(np.mean(merged_lds))
            faithfulness = merged_mean / baseline_mean if abs(baseline_mean) > 1e-8 else 0.0

            coarse_grain_curve.append({
                "level": level,
                "cluster_size": cluster_size,
                "n_effective_heads": total_after,
                "fraction_merged": float(fraction_merged),
                "mean_ld": merged_mean,
                "faithfulness": float(faithfulness),
            })

            log(f"    level={level} cluster_size={cluster_size} "
                f"heads={total_after}/{total_original} "
                f"faith={faithfulness:.4f}")

        # Find the faithfulness at ~50% merge or the closest level
        faith_at_half = None
        for entry in coarse_grain_curve:
            if entry["fraction_merged"] >= 0.5 - 1e-6:
                faith_at_half = entry["faithfulness"]
                break
        # If no level reached 50%, use the last level
        if faith_at_half is None:
            faith_at_half = coarse_grain_curve[-1]["faithfulness"]

        passed = faith_at_half > FAITHFULNESS_THRESHOLD

        log(f"    faithfulness at ~50% merge: {faith_at_half:.4f}")
        log(f"    [{'PASS' if passed else 'FAIL'}]")

        results.append(EvalResult(
            metric_id="PH3.renormalization",
            value=float(faith_at_half),
            n_samples=len(prompts),
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": task,
                "n_heads": len(circuit_heads),
                "baseline_mean_ld": baseline_mean,
                "coarse_grain_curve": coarse_grain_curve,
                "faithfulness_at_half_merge": float(faith_at_half),
                "threshold": FAITHFULNESS_THRESHOLD,
                "passed": passed,
            },
        ))

    return results


def main():
    parser = parse_common_args("PH3: Renormalization Group")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("PH3: RENORMALIZATION GROUP")
    log("=" * 60)

    out = args.out or "PH3_renormalization.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_renormalization(model, [task], args.n_prompts)
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)

    save_results(results, out, args=args)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
