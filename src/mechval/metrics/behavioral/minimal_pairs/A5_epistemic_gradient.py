"""Epistemic Gradient (Behavioral)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     A5 — Epistemic Gradient
Categories:     behavioral
Validity layer: Internal
Criteria:       Minimal-pair gradient test
Establishes:    Whether circuit responds monotonically to graded
                manipulation of a target construct
Requires:       CPU or GPU, model
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Creates minimal pairs that grade from weak to strong epistemic framing:
  Level 0 (neutral):  "The door is open"
  Level 1 (weak):     "Maybe the door is open"
  Level 2 (medium):   "I think the door is open"
  Level 3 (strong):   "I firmly believe the door is open"

For each level, measures circuit head activation strength via hook_z
at circuit head positions (mean absolute activation at last token).

Pass condition: monotonicity >= 0.75

Usage:
    uv run python A5_epistemic_gradient.py --tasks ioi --n-prompts 5
    uv run python A5_epistemic_gradient.py --device cpu
"""

import numpy as np
import torch

from mechval.metrics.common import (
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

GRADIENT_TEMPLATES = [
    {
        0: "The door is open",
        1: "Maybe the door is open",
        2: "I think the door is open",
        3: "I firmly believe the door is open",
    },
    {
        0: "The sky is blue",
        1: "Perhaps the sky is blue",
        2: "I suspect the sky is blue",
        3: "I am certain the sky is blue",
    },
    {
        0: "The cat sat on the mat",
        1: "Possibly the cat sat on the mat",
        2: "I believe the cat sat on the mat",
        3: "I strongly believe the cat sat on the mat",
    },
    {
        0: "It will rain tomorrow",
        1: "It might rain tomorrow",
        2: "I expect it will rain tomorrow",
        3: "I am convinced it will rain tomorrow",
    },
    {
        0: "The answer is correct",
        1: "Maybe the answer is correct",
        2: "I think the answer is correct",
        3: "I firmly believe the answer is correct",
    },
]


def compute_monotonicity(values: list[float]) -> float:
    """Fraction of adjacent pairs in expected (non-decreasing) order."""
    if len(values) < 2:
        return 1.0
    n_ordered = sum(1 for a, b in zip(values[:-1], values[1:]) if b >= a)
    return n_ordered / (len(values) - 1)


@torch.no_grad()
def measure_hook_z_activation(
    model,
    text: str,
    circuit_heads: set[tuple[int, int]],
) -> float:
    """Mean absolute hook_z activation at circuit heads on the last token."""
    tokens = model.to_tokens(text)
    heads_by_layer = heads_to_layer_dict(circuit_heads)
    hook_names = [f"blocks.{L}.attn.hook_z" for L in heads_by_layer]
    _, cache = model.run_with_cache(
        tokens, names_filter=lambda n: n in hook_names,
    )

    activations = []
    for layer, head_list in heads_by_layer.items():
        z = cache[f"blocks.{layer}.attn.hook_z"]
        for h in head_list:
            last_z = z[0, -1, h, :]
            activations.append(last_z.abs().mean().item())

    if not activations:
        return 0.0
    return float(np.mean(activations))


@torch.no_grad()
def run_epistemic_gradient(
    model,
    task: str,
    n_templates: int = 5,
) -> EvalResult:
    circuit_heads = get_circuit_heads(task)
    if not circuit_heads:
        return EvalResult(
            metric_id="A5.epistemic_gradient",
            value=0.0,
            n_samples=0,
            metadata={"task": task, "error": "no circuit"},
        )

    templates_to_use = GRADIENT_TEMPLATES[:n_templates]
    levels = sorted(templates_to_use[0].keys())
    per_level_activations: dict[int, list[float]] = {lv: [] for lv in levels}

    for template in templates_to_use:
        for level in levels:
            text = template[level]
            act = measure_hook_z_activation(model, text, circuit_heads)
            per_level_activations[level].append(act)

    per_level_means = {lv: float(np.mean(acts)) for lv, acts in per_level_activations.items()}
    mean_values = [per_level_means[lv] for lv in levels]

    monotonicity = compute_monotonicity(mean_values)

    # Gradient slope: linear regression slope across levels
    x = np.array(levels, dtype=float)
    y = np.array(mean_values, dtype=float)
    if len(x) > 1 and np.std(x) > 0:
        gradient_slope = float(np.polyfit(x, y, 1)[0])
    else:
        gradient_slope = 0.0

    passed = bool(monotonicity >= 0.75)

    log(f"  monotonicity={monotonicity:.3f}  slope={gradient_slope:.6f}  "
        f"[{'PASS' if passed else 'FAIL'}]")

    templates_used = [
        {lv: t[lv] for lv in levels} for t in templates_to_use
    ]

    return EvalResult(
        metric_id="A5.epistemic_gradient",
        value=monotonicity,
        n_samples=len(templates_to_use) * len(levels),
        metadata={
            "task": task,
            "per_level_means": per_level_means,
            "templates_used": templates_used,
            "gradient_slope": gradient_slope,
            "monotonicity": monotonicity,
            "n_circuit_heads": len(circuit_heads),
            "circuit_heads": sorted(circuit_heads),
            "passed": passed,
            "threshold_monotonicity": 0.75,
        },
    )


def main():
    parser = parse_common_args("A5: Epistemic Gradient")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS[:3]
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("A5: EPISTEMIC GRADIENT")
    log("=" * 60)

    out = args.out or "A5_epistemic_gradient.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        log(f"Task: {task}")
        r = run_epistemic_gradient(model, task, n_templates=args.n_prompts)
        results.append(r)
        save_incremental(r, jsonl_out)

    save_results(results, out, args=args)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
