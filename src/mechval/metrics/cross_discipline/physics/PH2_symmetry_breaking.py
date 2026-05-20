"""Symmetry Breaking Among Circuit Heads
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Metric:         PH2 — Symmetry Breaking
Categories:     cross_discipline, physics
Tier:           cross_discipline
Origin:         established

Tests whether nominally equivalent heads (same layer) develop
functional specialization. Computes pairwise cosine similarity of
hook_z activation vectors across prompts for heads within the same
layer. Low similarity = symmetry breaking (heads specialized despite
architectural equivalence).

Pass: mean within-layer cosine similarity < 0.5.
Ref: Anderson 1972, More is Different, Science 177:393-396.

Usage:
    uv run python PH2_symmetry_breaking.py --tasks ioi sva --n-prompts 40
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
    parse_common_args,
    save_incremental,
    save_results,
)

INSTRUMENT_INFO = InstrumentInfo(
    name="Symmetry Breaking",
    paper_ref="Anderson 1972",
    paper_cite="Anderson 1972, More is Different, Science 177:393-396",
    description="Measures functional specialization among architecturally equivalent heads via activation similarity",
    category="cross_discipline",
    tier="cross_discipline",
    origin="established",
)

SIMILARITY_THRESHOLD = 0.5


@torch.no_grad()
def run_symmetry_breaking(model, tasks: list[str],
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

        # Group circuit heads by layer
        heads_by_layer: dict[int, list[int]] = {}
        for L, H in circuit_heads:
            heads_by_layer.setdefault(L, []).append(H)

        # Only consider layers with 2+ circuit heads
        multi_head_layers = {L: hs for L, hs in heads_by_layer.items() if len(hs) >= 2}

        if not multi_head_layers:
            log(f"    No layers with 2+ circuit heads, skipping")
            continue

        # Collect activation vectors: {(layer, head): [n_prompts, d_head]}
        n_valid = min(len(prompts), len(correct_ids))
        head_activations: dict[tuple[int, int], list[torch.Tensor]] = {}
        for L, hs in multi_head_layers.items():
            for H in hs:
                head_activations[(L, H)] = []

        for idx in range(n_valid):
            tokens = model.to_tokens(prompts[idx].text)
            layers_needed = set(multi_head_layers.keys())
            _, cache = model.run_with_cache(
                tokens,
                names_filter=lambda n: any(f"blocks.{L}.attn.hook_z" in n for L in layers_needed),
            )
            for L, hs in multi_head_layers.items():
                z = cache[f"blocks.{L}.attn.hook_z"]  # (1, seq, n_heads, d_head)
                for H in hs:
                    head_activations[(L, H)].append(z[0, -1, H, :].cpu())

        # Compute within-layer pairwise cosine similarities
        per_layer_sims = {}
        all_sims = []

        for L, hs in multi_head_layers.items():
            # Stack activations: (n_prompts, d_head) per head
            stacked = {}
            for H in hs:
                acts = torch.stack(head_activations[(L, H)])  # (n_valid, d_head)
                # Flatten to single vector per head for comparison
                stacked[H] = acts.reshape(-1).float()

            pair_sims = []
            for i in range(len(hs)):
                for j in range(i + 1, len(hs)):
                    v_i = stacked[hs[i]]
                    v_j = stacked[hs[j]]
                    cos_sim = float(torch.nn.functional.cosine_similarity(
                        v_i.unsqueeze(0), v_j.unsqueeze(0)).item())
                    pair_sims.append(cos_sim)
                    all_sims.append(cos_sim)

            if pair_sims:
                per_layer_sims[L] = {
                    "heads": hs,
                    "mean_similarity": float(np.mean(pair_sims)),
                    "pairwise_sims": pair_sims,
                }

        mean_sim = float(np.mean(all_sims)) if all_sims else 1.0
        passed = mean_sim < SIMILARITY_THRESHOLD

        log(f"    mean within-layer similarity={mean_sim:.4f}")
        log(f"    layers analyzed: {sorted(multi_head_layers.keys())}")
        log(f"    [{'PASS' if passed else 'FAIL'}]")

        results.append(EvalResult(
            metric_id="PH2.symmetry_breaking",
            value=mean_sim,
            n_samples=len(prompts),
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": task,
                "n_heads": len(circuit_heads),
                "mean_within_layer_similarity": mean_sim,
                "per_layer_details": {str(k): v for k, v in per_layer_sims.items()},
                "n_layers_analyzed": len(multi_head_layers),
                "threshold": SIMILARITY_THRESHOLD,
                "passed": passed,
            },
        ))

    return results


def main():
    parser = parse_common_args("PH2: Symmetry Breaking")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("PH2: SYMMETRY BREAKING")
    log("=" * 60)

    out = args.out or "PH2_symmetry_breaking.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_symmetry_breaking(model, [task], args.n_prompts)
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)

    save_results(results, out, args=args)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
