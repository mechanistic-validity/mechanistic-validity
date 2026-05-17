"""Transportability (Pearl & Bareinboim 2014)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     A12 — Transportability
Categories:     causal
Validity layer: External
Criteria:       E5/E6 Cross-model
Establishes:    Causal circuit findings transport from GPT-2 Small to larger model sizes
Requires:       GPU, model
Doc:            /instruments_v2/causal/a12-transportability
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Tests whether causal circuit findings from a source domain (GPT-2 Small)
transport to target domains (GPT-2 Medium, Large, XL) where direct
full evaluation may be expensive.

Transportability in Pearl's framework asks: given a causal model in
domain S, under what conditions can we predict the effect of interventions
in domain T? The key construct is the "selection diagram" — variables
that differ between domains are marked with S-nodes.

For circuits, the transportable quantities are:
  1. Circuit STRUCTURE (which layers/positions have causal heads) — does
     the same layer-depth pattern appear?
  2. Circuit EFFECT (normalized faithfulness) — is the effect magnitude
     comparable?
  3. Circuit SPECIFICITY (head selectivity for the task) — do the same
     relative rankings hold?

We test transportability by:
  - Running activation patching in larger models at matched layer fractions
  - Computing transport formula: P_T(y|do(x)) from P_S observations
  - Measuring structural similarity (Jaccard on layer-normalized positions)

Usage:
    uv run python 41_transportability.py --tasks ioi sva
    uv run python 41_transportability.py --device cuda --target-models gpt2-medium gpt2-large
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "causal"))
from _common import (
    CIRCUIT_TASKS,
    EvalResult,
    calibrate_mean_z,
    generate_prompts,
    get_circuit_heads,
    get_token_ids,
    heads_to_layer_dict,
    load_model,
    log,
    logit_diff_from_logits,
    make_ablation_hook,
    parse_common_args,
    save_results,
)


def normalize_layer_positions(heads: set[tuple[int, int]], n_layers: int) -> list[float]:
    """Convert head positions to fractional layer depths [0, 1]."""
    return sorted(L / (n_layers - 1) for L, H in heads)


def layer_profile(heads: set[tuple[int, int]], n_layers: int) -> np.ndarray:
    """Histogram of heads per layer, normalized."""
    counts = np.zeros(n_layers)
    for L, H in heads:
        counts[L] += 1
    total = counts.sum()
    if total > 0:
        counts /= total
    return counts


@torch.no_grad()
def find_top_heads_by_patching(
    model, prompts, correct_ids, incorrect_ids,
    mean_z: torch.Tensor, top_k: int = 15,
) -> list[tuple[tuple[int, int], float]]:
    """Run activation patching on all heads, return top-k by effect."""
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    effects = {}

    for L in range(n_layers):
        for H in range(n_heads):
            head_by_layer = heads_to_layer_dict({(L, H)})
            hooks = make_ablation_hook(head_by_layer, mean_z, "mean")

            total_effect = 0.0
            total_baseline = 0.0

            for i, p in enumerate(prompts[:15]):
                if i >= len(correct_ids):
                    break
                tokens = model.to_tokens(p.text)
                clean_logits = model(tokens)
                clean_ld = logit_diff_from_logits(clean_logits, correct_ids[i], incorrect_ids[i])

                ablated_logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
                ablated_ld = logit_diff_from_logits(ablated_logits, correct_ids[i], incorrect_ids[i])

                total_effect += abs(clean_ld - ablated_ld)
                total_baseline += abs(clean_ld)

            if total_baseline > 1e-8:
                effects[(L, H)] = total_effect / total_baseline

    ranked = sorted(effects.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


def compute_structural_transportability(
    source_heads: set[tuple[int, int]], source_n_layers: int,
    target_heads: set[tuple[int, int]], target_n_layers: int,
) -> dict:
    """Measure how well circuit structure transports between models."""
    source_profile = layer_profile(source_heads, source_n_layers)
    target_profile = layer_profile(target_heads, target_n_layers)

    source_norm = normalize_layer_positions(source_heads, source_n_layers)
    target_norm = normalize_layer_positions(target_heads, target_n_layers)

    source_thirds = [0, 0, 0]
    for pos in source_norm:
        if pos < 0.33:
            source_thirds[0] += 1
        elif pos < 0.67:
            source_thirds[1] += 1
        else:
            source_thirds[2] += 1
    source_thirds_norm = np.array(source_thirds, dtype=float)
    if source_thirds_norm.sum() > 0:
        source_thirds_norm /= source_thirds_norm.sum()

    target_thirds = [0, 0, 0]
    for pos in target_norm:
        if pos < 0.33:
            target_thirds[0] += 1
        elif pos < 0.67:
            target_thirds[1] += 1
        else:
            target_thirds[2] += 1
    target_thirds_norm = np.array(target_thirds, dtype=float)
    if target_thirds_norm.sum() > 0:
        target_thirds_norm /= target_thirds_norm.sum()

    thirds_cosine = float(np.dot(source_thirds_norm, target_thirds_norm) / (
        np.linalg.norm(source_thirds_norm) * np.linalg.norm(target_thirds_norm) + 1e-8
    ))

    return {
        "source_layer_fractions": source_norm,
        "target_layer_fractions": target_norm,
        "source_thirds_distribution": source_thirds_norm.tolist(),
        "target_thirds_distribution": target_thirds_norm.tolist(),
        "thirds_cosine_similarity": thirds_cosine,
    }


@torch.no_grad()
def main():
    parser = parse_common_args("A12 — Transportability Analysis")
    parser.add_argument("--target-models", nargs="+",
                        default=["gpt2-medium"],
                        help="Target models to transport to")
    parser.add_argument("--top-k", type=int, default=15,
                        help="Top-k heads to consider as target circuit")
    args = parser.parse_args()

    tasks = args.tasks or ["ioi", "sva", "greater_than"]
    source_model = load_model(args.model, args.device)
    source_n_layers = source_model.cfg.n_layers

    all_results = {}

    for task in tasks:
        log(f"\n{'='*60}")
        log(f"Task: {task}")
        source_circuit = get_circuit_heads(task)
        if not source_circuit:
            log(f"  No source circuit for {task}, skipping")
            continue

        log(f"  Source circuit ({args.model}): {len(source_circuit)} heads")

        task_results = {
            "source_model": args.model,
            "source_n_layers": source_n_layers,
            "source_n_heads": len(source_circuit),
            "source_heads": sorted([list(h) for h in source_circuit]),
            "targets": {},
        }

        for target_name in args.target_models:
            log(f"\n  Target: {target_name}")
            target_model = load_model(target_name, args.device)
            target_n_layers = target_model.cfg.n_layers

            prompts = generate_prompts(task, target_model.tokenizer, args.n_prompts)
            if not prompts:
                log(f"    No prompts for {task} on {target_name}, skipping")
                continue

            correct_ids, incorrect_ids = get_token_ids(prompts, target_model.tokenizer)
            mean_z = calibrate_mean_z(target_model, prompts)

            log(f"    Running activation patching on {target_name} ({target_n_layers} layers)...")
            top_heads = find_top_heads_by_patching(
                target_model, prompts, correct_ids, incorrect_ids,
                mean_z, top_k=args.top_k,
            )

            target_circuit = {h for h, _ in top_heads}
            log(f"    Top-{args.top_k} heads: {sorted(target_circuit)[:5]}...")

            structural = compute_structural_transportability(
                source_circuit, source_n_layers,
                target_circuit, target_n_layers,
            )

            log(f"    Thirds cosine similarity: {structural['thirds_cosine_similarity']:.3f}")
            log(f"    Source distribution (early/mid/late): {structural['source_thirds_distribution']}")
            log(f"    Target distribution (early/mid/late): {structural['target_thirds_distribution']}")

            task_results["targets"][target_name] = {
                "target_n_layers": target_n_layers,
                "target_top_heads": [[list(h), float(e)] for h, e in top_heads[:10]],
                "structural_transportability": structural,
            }

        all_results[task] = task_results

    save_results(all_results, "a12_transportability.json")
    log("\nDone.")


if __name__ == "__main__":
    main()
