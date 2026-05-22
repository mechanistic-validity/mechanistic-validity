"""Metric: Surprise Reduction --- information-theoretic circuit coverage

Paper: ARC (2026). "Formal verification, heuristic explanations and
surprise accounting." alignment.org. Related: Hilton et al. (2026).
"Mechanistic estimation for wide random MLPs." AlignmentForum, May 2026.

Measures how much knowing the circuit reduces uncertainty about model
outputs.  Given a set of prompts, compute the entropy of the output
distribution with and without the circuit's contribution.  The ratio
of entropy increase upon circuit ablation to the ablated entropy gives
the fraction of the model's certainty that the circuit accounts for.

Surprise Reduction (Evaluation EX34)
=============================================
Instrument:     EX34 --- Surprise Reduction
Categories:     evaluation
Validity layer: Measurement
Criteria:       M4 Construct Coverage
Establishes:    How much of the model's output certainty is accounted
                for by the circuit
Requires:       CPU or GPU, model
=============================================

Core logic:
1. Run model on N prompts, collect logit distributions.
2. Compute mean entropy of logit distributions = H_full.
3. Ablate all circuit heads (mean ablation), compute logit entropy
   = H_ablated.
4. surprise_reduction = (H_ablated - H_full) / H_ablated.
5. Higher values mean the circuit explains more of the model's
   certainty.

Pass condition: surprise_reduction > 0.05

Usage:
    uv run python 129_surprise_reduction.py --model gpt2 --device cpu
    uv run python 129_surprise_reduction.py --tasks ioi --n-prompts 40
"""

import numpy as np
import torch
import torch.nn.functional as F

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    InstrumentInfo,
    calibrate_mean_z,
    generate_prompts,
    get_circuit_heads,
    get_token_ids,
    heads_to_layer_dict,
    load_model,
    log,
    make_ablation_hook,
    parse_common_args,
    save_results,
)

INSTRUMENT_INFO = InstrumentInfo(
    name="Surprise Reduction",
    paper_ref="ARC 2026; Hilton et al. AlignmentForum May 2026",
    paper_cite=(
        "ARC 2026, Formal verification, heuristic explanations and "
        "surprise accounting (alignment.org); "
        "Hilton et al. 2026, Mechanistic estimation for wide random "
        "MLPs (AlignmentForum)"
    ),
    description=(
        "Measures the fraction of model output certainty accounted for "
        "by the circuit. Computes entropy of output distributions with "
        "and without circuit heads, reporting the normalized entropy "
        "increase upon ablation."
    ),
    category="evaluation",
    tier="emerging",
    origin="external",
)

SURPRISE_THRESHOLD = 0.05


def _logit_entropy(logits: torch.Tensor) -> float:
    """Compute entropy of the softmax distribution from logits at the last position."""
    probs = F.softmax(logits[0, -1], dim=-1)
    # Clamp to avoid log(0)
    log_probs = torch.log(probs.clamp(min=1e-10))
    return -(probs * log_probs).sum().item()


@torch.no_grad()
def run_surprise_reduction(
    model,
    tasks: list[str] | None = None,
    n_prompts: int = 40,
) -> list[EvalResult]:
    """Measure surprise reduction for each task's circuit.

    For each task, computes output entropy with the full model vs with
    circuit heads ablated, then reports the fractional entropy increase.

    Args:
        model: HookedTransformer instance.
        tasks: list of task names (default: all circuit tasks).
        n_prompts: prompts per task.

    Returns:
        List of EvalResult, one per task plus aggregate.
    """
    if tasks is None:
        tasks = CIRCUIT_TASKS

    results = []
    all_reductions = []

    for task in tasks:
        circuit_heads = get_circuit_heads(task)
        if not circuit_heads:
            continue

        prompts = generate_prompts(task, model.tokenizer, n_prompts=n_prompts)
        if not prompts:
            continue

        correct_ids, incorrect_ids = get_token_ids(prompts, model.tokenizer)
        if not correct_ids:
            continue

        mean_z = calibrate_mean_z(model, prompts)

        # Build ablation hooks for circuit heads
        circuit_by_layer = heads_to_layer_dict(circuit_heads)
        hooks = make_ablation_hook(circuit_by_layer, mean_z, "mean")

        entropies_full = []
        entropies_ablated = []

        for i, p in enumerate(prompts):
            if i >= len(correct_ids):
                break
            tokens = model.to_tokens(p.text)

            # Full model entropy
            full_logits = model(tokens)
            h_full = _logit_entropy(full_logits)
            entropies_full.append(h_full)

            # Ablated model entropy
            abl_logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
            h_abl = _logit_entropy(abl_logits)
            entropies_ablated.append(h_abl)

        mean_h_full = float(np.mean(entropies_full))
        mean_h_ablated = float(np.mean(entropies_ablated))

        if mean_h_ablated < 1e-8:
            reduction = 0.0
        else:
            reduction = (mean_h_ablated - mean_h_full) / mean_h_ablated

        all_reductions.append(reduction)
        passed = reduction > SURPRISE_THRESHOLD

        log(f"  {task}: H_full={mean_h_full:.4f} H_ablated={mean_h_ablated:.4f} "
            f"reduction={reduction:.4f} ({'PASS' if passed else 'FAIL'})")

        results.append(EvalResult(
            metric_id="EX34.surprise_reduction",
            value=reduction,
            n_samples=len(entropies_full),
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": task,
                "surprise_reduction": reduction,
                "mean_entropy_full": mean_h_full,
                "mean_entropy_ablated": mean_h_ablated,
                "entropy_increase": mean_h_ablated - mean_h_full,
                "n_circuit_heads": len(circuit_heads),
                "passed": passed,
                "threshold": SURPRISE_THRESHOLD,
            },
        ))

    # Aggregate
    if all_reductions:
        agg = float(np.mean(all_reductions))
        agg_passed = agg > SURPRISE_THRESHOLD
        log(f"  Aggregate: surprise_reduction={agg:.4f} "
            f"({'PASS' if agg_passed else 'FAIL'})")
        results.append(EvalResult(
            metric_id="EX34.surprise_reduction",
            value=agg,
            n_samples=sum(r.n_samples for r in results),
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": "aggregate",
                "mean_surprise_reduction": agg,
                "n_tasks": len(all_reductions),
                "passed": agg_passed,
                "threshold": SURPRISE_THRESHOLD,
            },
        ))

    return results


def main():
    parser = parse_common_args("EX34: Surprise Reduction")
    args = parser.parse_args()

    model = load_model(args.model, args.device)

    log("=" * 60)
    log("EX34: SURPRISE REDUCTION")
    log("=" * 60)

    results = run_surprise_reduction(
        model,
        tasks=args.tasks or CIRCUIT_TASKS,
        n_prompts=args.n_prompts,
    )

    out = args.out or "129_surprise_reduction.json"
    save_results(results, out, args=args)
    log(f"\nDone. {len(results)} results.")


if __name__ == "__main__":
    main()
