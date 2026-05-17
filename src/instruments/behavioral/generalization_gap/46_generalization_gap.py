"""Generalization Gap
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     D09 — Generalization Gap
Categories:     behavioral
Validity layer: External
Criteria:       E1 Replication
Establishes:    Circuit faithfulness generalizes beyond training distribution templates
Requires:       GPU, model
Doc:            /instruments_v2/behavioral/d09-generalization-gap
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Compares circuit faithfulness on "in-distribution" prompts (standard
templates) vs "out-of-distribution" prompts (perturbed variants with
longer sentences, unusual names, or different syntax). The gap between
in-dist and OOD faithfulness reveals how template-dependent the circuit is.

A small gap means the circuit captures genuine computational structure;
a large gap means it may be an artifact of the specific prompt format.

Framework reference: Behavioral Pillar D07 -- template robustness test
for circuit generalization beyond training distribution.

Usage:
    uv run python 46_generalization_gap.py --tasks ioi sva
    uv run python 46_generalization_gap.py --device cuda --n-prompts 40
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "causal"))
from _common import (
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


# ---------------------------------------------------------------------------
# OOD prompt perturbation
# ---------------------------------------------------------------------------

PADDING_PREFIXES = [
    "Well, you know, ",
    "As it turns out, ",
    "Interestingly enough, ",
    "If you think about it, ",
    "To be perfectly honest, ",
]

PADDING_SUFFIXES = [
    " That was what happened.",
    " And that is the whole story.",
    " Everyone agreed on this.",
    " Nobody could deny it.",
    " It was quite clear.",
]


def perturb_prompts(prompts, seed=123):
    """Create OOD variants by adding padding text around the core prompt."""
    rng = np.random.RandomState(seed)
    perturbed = []
    for p in prompts:
        variant = type(p).__new__(type(p))
        # Copy all attributes
        for attr in vars(p):
            setattr(variant, attr, getattr(p, attr))
        # Perturb the text with prefix/suffix padding
        prefix = rng.choice(PADDING_PREFIXES)
        suffix = rng.choice(PADDING_SUFFIXES)
        choice = rng.randint(3)
        if choice == 0:
            variant.text = prefix + p.text
        elif choice == 1:
            variant.text = p.text + suffix
        else:
            variant.text = prefix + p.text + suffix
        perturbed.append(variant)
    return perturbed


@torch.no_grad()
def run_generalization_gap(model, tasks, n_prompts):
    tokenizer = model.tokenizer
    results = []

    for task in tasks:
        circuit_heads = get_circuit_heads(task)
        if not circuit_heads:
            log(f"  {task}: no circuit, skipping")
            continue

        prompts_id = generate_prompts(task, tokenizer, n_prompts)
        if not prompts_id:
            continue

        correct_ids_id, incorrect_ids_id = get_token_ids(prompts_id, tokenizer)
        if not correct_ids_id:
            continue

        # Generate OOD prompts via perturbation
        prompts_ood = perturb_prompts(prompts_id)
        correct_ids_ood, incorrect_ids_ood = get_token_ids(prompts_ood, tokenizer)
        if not correct_ids_ood:
            continue

        log(f"  {task} ({len(circuit_heads)} heads)...")

        # Calibrate mean_z on in-distribution prompts
        mean_z = calibrate_mean_z(model, prompts_id, n_calibration=min(100, len(prompts_id)))

        # Compute faithfulness on in-distribution
        faith_id = compute_faithfulness(
            model, prompts_id, correct_ids_id, incorrect_ids_id, circuit_heads, mean_z,
        )

        # Compute faithfulness on OOD
        faith_ood = compute_faithfulness(
            model, prompts_ood, correct_ids_ood, incorrect_ids_ood, circuit_heads, mean_z,
        )

        gap = faith_id - faith_ood
        relative_gap = gap / abs(faith_id) if abs(faith_id) > 1e-8 else 0.0

        log(f"    faith_ID={faith_id:.4f}, faith_OOD={faith_ood:.4f}, "
            f"gap={gap:.4f}, relative_gap={relative_gap:.3f}")

        results.append(EvalResult(
            metric_id="D07.generalization_gap",
            value=gap,
            n_samples=len(prompts_id),
            metadata={
                "task": task,
                "faithfulness_in_dist": faith_id,
                "faithfulness_ood": faith_ood,
                "absolute_gap": gap,
                "relative_gap": relative_gap,
                "n_circuit_heads": len(circuit_heads),
                "n_prompts_id": len(prompts_id),
                "n_prompts_ood": len(prompts_ood),
                "perturbation_type": "prefix_suffix_padding",
            },
        ))

    return results


def main():
    parser = parse_common_args("D07: Generalization Gap")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("D07: GENERALIZATION GAP")
    log("=" * 60)

    results = run_generalization_gap(model, tasks, args.n_prompts)

    out = args.out or "46_generalization_gap.json"
    save_results(results, out)

    log(f"\nDone. {len(results)} tasks evaluated.")
    for r in results:
        t = r.metadata["task"]
        log(f"  {t}: gap={r.value:.4f} (ID={r.metadata['faithfulness_in_dist']:.3f}, "
            f"OOD={r.metadata['faithfulness_ood']:.3f})")


if __name__ == "__main__":
    main()
