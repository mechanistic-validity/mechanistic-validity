"""Boundary Sweep (Behavioral)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     RI3b — Boundary Sweep
Categories:     behavioral
Validity layer: Internal
Criteria:       Construct boundary identification
Establishes:    Where a circuit's activation boundary lies across
                different prompt categories
Requires:       CPU or GPU, model
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Defines 6 prompt categories with synthetic prompts and measures
circuit head activation strength (mean absolute attention score at
circuit heads on the last token position) for each.

Pass condition: at least one category pair has ratio > 2.0

Usage:
    uv run python RI3b_boundary_sweep.py --tasks ioi --n-prompts 5
    uv run python RI3b_boundary_sweep.py --device cpu
"""

import numpy as np
import torch

from mechanistic_validity.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    get_circuit_heads,
    heads_to_layer_dict,
    load_model,
    log,
    parse_common_args,
    save_incremental,
    save_results,
)

PROMPT_CATEGORIES: dict[str, list[str]] = {
    "epistemic": [
        "I think the answer is",
        "She believes the result is",
    ],
    "evidential": [
        "Apparently the answer is",
        "According to reports the result is",
    ],
    "temporal": [
        "Yesterday the answer was",
        "In 2019 the result was",
    ],
    "spatial": [
        "In Paris the answer is",
        "At the lab the result is",
    ],
    "causal": [
        "Because of this the answer is",
        "As a result the answer is",
    ],
    "neutral": [
        "The answer is",
        "The result is",
    ],
}


@torch.no_grad()
def measure_circuit_activation(
    model,
    text: str,
    circuit_heads: set[tuple[int, int]],
) -> float:
    """Mean absolute attention score at circuit heads on the last token."""
    tokens = model.to_tokens(text)
    heads_by_layer = heads_to_layer_dict(circuit_heads)
    hook_names = [f"blocks.{L}.attn.hook_pattern" for L in heads_by_layer]
    _, cache = model.run_with_cache(
        tokens, names_filter=lambda n: n in hook_names,
    )

    activations = []
    for layer, head_list in heads_by_layer.items():
        pattern = cache[f"blocks.{layer}.attn.hook_pattern"]
        for h in head_list:
            last_token_attn = pattern[0, h, -1, :]
            activations.append(last_token_attn.abs().mean().item())

    if not activations:
        return 0.0
    return float(np.mean(activations))


@torch.no_grad()
def run_boundary_sweep(
    model,
    task: str,
    n_prompts_per_category: int = 5,
) -> EvalResult:
    circuit_heads = get_circuit_heads(task)
    if not circuit_heads:
        return EvalResult(
            metric_id="RI3b.boundary_sweep",
            value=0.0,
            n_samples=0,
            metadata={"task": task, "error": "no circuit"},
        )

    per_category: dict[str, float] = {}

    for category, templates in PROMPT_CATEGORIES.items():
        cat_activations = []
        for template in templates:
            for _ in range(n_prompts_per_category):
                act = measure_circuit_activation(model, template, circuit_heads)
                cat_activations.append(act)
        per_category[category] = float(np.mean(cat_activations))
        log(f"  {category}: activation={per_category[category]:.6f}")

    strengths = list(per_category.values())
    max_strength = max(strengths)
    min_strength = min(strengths)

    if abs(min_strength) < 1e-10:
        activation_ratio = float("inf")
    else:
        activation_ratio = max_strength / min_strength

    category_ranking = sorted(per_category.keys(), key=lambda c: per_category[c], reverse=True)

    passed = bool(activation_ratio > 2.0)

    log(f"  ratio={activation_ratio:.3f}  [{'PASS' if passed else 'FAIL'}]")

    return EvalResult(
        metric_id="RI3b.boundary_sweep",
        value=activation_ratio,
        n_samples=n_prompts_per_category * sum(len(v) for v in PROMPT_CATEGORIES.values()),
        metadata={
            "task": task,
            "per_category": per_category,
            "category_ranking": category_ranking,
            "max_strength": max_strength,
            "min_strength": min_strength,
            "activation_ratio": activation_ratio,
            "n_circuit_heads": len(circuit_heads),
            "circuit_heads": sorted(circuit_heads),
            "n_prompts_per_category": n_prompts_per_category,
            "passed": passed,
            "threshold_ratio": 2.0,
        },
    )


def main():
    parser = parse_common_args("RI3b: Boundary Sweep")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS[:3]
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("RI3b: BOUNDARY SWEEP")
    log("=" * 60)

    out = args.out or "RI3b_boundary_sweep.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        log(f"Task: {task}")
        r = run_boundary_sweep(model, task, n_prompts_per_category=args.n_prompts)
        results.append(r)
        save_incremental(r, jsonl_out)

    save_results(results, out, args=args)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
