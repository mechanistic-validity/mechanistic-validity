"""Incremental Validity
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     F02 — Convergent Validity
Categories:     measurement
Validity layer: Measurement
Criteria:       C5 Convergent validity
Establishes:    Weight-classifier circuits outperform naive top-k activation baselines
Requires:       CPU, data-only
Doc:            /instruments_v2/measurement/f02-convergent-validity
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Does our weight-classifier circuit add value BEYOND the simplest baseline
(top-k heads by activation magnitude)? For each task:
  (1) compute faithfulness of our circuit,
  (2) compute faithfulness of top-k heads by mean activation magnitude
      (where k = |our circuit|),
  (3) report delta = our_faithfulness - topk_faithfulness.

Positive delta = our method finds better circuits than the naive baseline.

Usage:
    uv run python 36_incremental_validity.py --tasks ioi sva --n-prompts 40
    uv run python 36_incremental_validity.py --device cuda
"""

import numpy as np
import torch

from mechanistic_validity.metrics.common import (
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


@torch.no_grad()
def compute_head_activation_magnitudes(model, prompts) -> np.ndarray:
    """Compute mean activation magnitude per head across prompts.

    Returns array of shape (n_layers, n_heads) with mean |z| at last position.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    magnitudes = np.zeros((n_layers, n_heads))
    count = 0

    for p in prompts:
        tokens = model.to_tokens(p.text)
        _, cache = model.run_with_cache(tokens, names_filter=lambda n: "hook_z" in n)
        for layer in range(n_layers):
            z = cache[f"blocks.{layer}.attn.hook_z"]
            # Mean absolute activation at last position
            for head in range(n_heads):
                magnitudes[layer, head] += z[0, -1, head].abs().mean().item()
        count += 1

    if count > 0:
        magnitudes /= count
    return magnitudes


def get_topk_heads_by_magnitude(magnitudes: np.ndarray, k: int) -> set[tuple[int, int]]:
    """Return the k heads with highest mean activation magnitude."""
    n_layers, n_heads = magnitudes.shape
    flat = []
    for layer in range(n_layers):
        for head in range(n_heads):
            flat.append((magnitudes[layer, head], layer, head))
    flat.sort(reverse=True)
    return {(layer, head) for _, layer, head in flat[:k]}


@torch.no_grad()
def run_incremental_validity(model, tasks: list[str],
                             n_prompts: int = 40) -> list[EvalResult]:
    tokenizer = model.tokenizer
    results = []

    for task in tasks:
        circuit_heads = get_circuit_heads(task)
        if not circuit_heads:
            log(f"  {task}: no circuit, skipping")
            continue

        prompts = generate_prompts(task, tokenizer, n_prompts)
        if not prompts:
            log(f"  {task}: no prompts, skipping")
            continue

        correct_ids, incorrect_ids = get_token_ids(prompts, tokenizer)
        if not correct_ids:
            continue

        k = len(circuit_heads)
        log(f"  {task} (k={k} heads, {len(prompts)} prompts)...")

        # Calibrate
        mean_z = calibrate_mean_z(model, prompts, n_calibration=min(100, len(prompts)))

        # Our circuit faithfulness
        our_faith = compute_faithfulness(
            model, prompts, correct_ids, incorrect_ids, circuit_heads, mean_z)

        # Top-k by activation magnitude
        magnitudes = compute_head_activation_magnitudes(model, prompts)
        topk_heads = get_topk_heads_by_magnitude(magnitudes, k)
        topk_faith = compute_faithfulness(
            model, prompts, correct_ids, incorrect_ids, topk_heads, mean_z)

        delta = our_faith - topk_faith
        overlap = len(circuit_heads & topk_heads)
        jaccard = overlap / len(circuit_heads | topk_heads) if (circuit_heads | topk_heads) else 0.0

        log(f"    our_faith={our_faith:.3f}  topk_faith={topk_faith:.3f}  "
            f"delta={delta:+.3f}  overlap={overlap}/{k}")

        # Format heads for metadata
        our_heads_str = sorted(f"L{L}H{H}" for L, H in circuit_heads)
        topk_heads_str = sorted(f"L{L}H{H}" for L, H in topk_heads)
        overlap_heads_str = sorted(f"L{L}H{H}" for L, H in (circuit_heads & topk_heads))

        results.append(EvalResult(
            metric_id="C36.incremental_validity",
            value=delta,
            n_samples=len(correct_ids),
            metadata={
                "task": task,
                "our_faithfulness": our_faith,
                "topk_faithfulness": topk_faith,
                "delta": delta,
                "k": k,
                "our_heads": our_heads_str,
                "topk_heads": topk_heads_str,
                "overlap_heads": overlap_heads_str,
                "overlap_count": overlap,
                "jaccard": jaccard,
            },
        ))

    return results


def main():
    parser = parse_common_args("C36: Incremental Validity")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("C36: INCREMENTAL VALIDITY (Metric #82)")
    log("=" * 60)

    results = run_incremental_validity(model, tasks, args.n_prompts)

    out = args.out or "36_incremental_validity.json"
    save_results(results, out)

    log(f"\nDone. {len(results)} tasks evaluated.")
    for r in results:
        t = r.metadata["task"]
        log(f"  {t}: delta={r.value:+.3f} (ours={r.metadata['our_faithfulness']:.3f} "
            f"vs topk={r.metadata['topk_faithfulness']:.3f})")


if __name__ == "__main__":
    main()
