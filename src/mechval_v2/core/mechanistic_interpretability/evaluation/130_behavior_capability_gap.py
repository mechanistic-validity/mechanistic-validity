"""Metric: Behavior vs Capability Gap --- does the circuit explain behavior or just capability?

Paper: Steinhardt (2026). "The Case for Evaluating Model Behaviors."
AlignmentForum, May 20 2026.
normalscience.org/article/the-case-for-evaluating-model-behaviors

Tests whether a circuit explains model *behavior* (what it does on
all inputs) vs just *capability* (what it can do on easy inputs).
A circuit that achieves high faithfulness only on prompts where the
model already gets the answer right is measuring capability, not
behavior.  By comparing faithfulness on correct-output vs error
prompts, we detect this gap.

Behavior vs Capability Gap (Evaluation EX35)
=============================================
Instrument:     EX35 --- Behavior vs Capability Gap
Categories:     evaluation
Validity layer: External
Criteria:       E3 Generalizability
Establishes:    Whether the circuit explains general model behavior
                (including errors) or only capability (correct outputs)
Requires:       CPU or GPU, model
=============================================

Core logic:
1. Run model on all prompts, split into correct (model gets right
   answer) and error (model gets wrong answer) sets.
2. For each set, compute circuit faithfulness (logit-diff recovery
   after ablating non-circuit heads).
3. behavior_gap = faithfulness_correct - faithfulness_error.
4. Lower gap = circuit explains behavior generally; high gap = circuit
   only explains capability.

Pass condition: behavior_gap < 0.3

Usage:
    uv run python 130_behavior_capability_gap.py --model gpt2 --device cpu
    uv run python 130_behavior_capability_gap.py --tasks ioi --n-prompts 40
"""

import numpy as np
import torch

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
    logit_diff_from_logits,
    make_ablation_hook,
    parse_common_args,
    save_results,
)

INSTRUMENT_INFO = InstrumentInfo(
    name="Behavior vs Capability Gap",
    paper_ref="Steinhardt, AlignmentForum May 2026",
    paper_cite=(
        "Steinhardt 2026, "
        "The Case for Evaluating Model Behaviors "
        "(AlignmentForum / normalscience.org)"
    ),
    description=(
        "Tests whether a circuit explains general model behavior "
        "(including error cases) or only capability (correct outputs). "
        "Compares circuit faithfulness on correct-output vs error "
        "prompts. A low gap indicates the circuit captures behavior, "
        "not just capability."
    ),
    category="evaluation",
    tier="emerging",
    origin="external",
)

GAP_THRESHOLD = 0.3


@torch.no_grad()
def _compute_faithfulness_on_subset(
    model,
    circuit_heads: set[tuple[int, int]],
    prompts,
    correct_ids: list[int],
    incorrect_ids: list[int],
    mean_z: torch.Tensor,
) -> float:
    """Faithfulness on a subset of prompts: logit_diff(circuit_only) / logit_diff(full)."""
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    non_circuit = {(L, H) for L in range(n_layers) for H in range(n_heads)} - circuit_heads
    non_circuit_by_layer = heads_to_layer_dict(non_circuit)
    hooks = make_ablation_hook(non_circuit_by_layer, mean_z, "mean")

    faith_num, faith_den = 0.0, 0.0
    for i, p in enumerate(prompts):
        if i >= len(correct_ids):
            break
        tokens = model.to_tokens(p.text)
        clean_logits = model(tokens)
        clean_ld = logit_diff_from_logits(clean_logits, correct_ids[i], incorrect_ids[i])
        ablated_logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
        ablated_ld = logit_diff_from_logits(ablated_logits, correct_ids[i], incorrect_ids[i])
        faith_num += ablated_ld
        faith_den += clean_ld

    if abs(faith_den) < 1e-8:
        return 0.0
    return faith_num / faith_den


def run_behavior_capability_gap(
    model,
    tasks: list[str] | None = None,
    n_prompts: int = 40,
) -> list[EvalResult]:
    """Measure the behavior vs capability gap for each task's circuit.

    Splits prompts into those the model gets right (correct) and wrong
    (error), then computes circuit faithfulness on each subset.

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
    all_gaps = []

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

        # Split prompts by whether model gets them right
        correct_prompts, correct_c, correct_i = [], [], []
        error_prompts, error_c, error_i = [], [], []

        for i, p in enumerate(prompts):
            if i >= len(correct_ids):
                break
            tokens = model.to_tokens(p.text)
            logits = model(tokens)
            ld = logit_diff_from_logits(logits, correct_ids[i], incorrect_ids[i])

            if ld > 0:
                correct_prompts.append(p)
                correct_c.append(correct_ids[i])
                correct_i.append(incorrect_ids[i])
            else:
                error_prompts.append(p)
                error_c.append(correct_ids[i])
                error_i.append(incorrect_ids[i])

        n_correct = len(correct_prompts)
        n_error = len(error_prompts)

        # Compute faithfulness on each subset
        if n_correct >= 2:
            faith_correct = _compute_faithfulness_on_subset(
                model, circuit_heads, correct_prompts,
                correct_c, correct_i, mean_z,
            )
        else:
            faith_correct = float("nan")

        if n_error >= 2:
            faith_error = _compute_faithfulness_on_subset(
                model, circuit_heads, error_prompts,
                error_c, error_i, mean_z,
            )
        else:
            faith_error = float("nan")

        # Compute gap (only if both subsets have enough data)
        if n_correct >= 2 and n_error >= 2:
            gap = faith_correct - faith_error
            all_gaps.append(gap)
            passed = abs(gap) < GAP_THRESHOLD
        else:
            gap = float("nan")
            passed = False

        log(f"  {task}: faith_correct={faith_correct:.4f} "
            f"faith_error={faith_error:.4f} gap={gap:.4f} "
            f"(n_correct={n_correct}, n_error={n_error}) "
            f"({'PASS' if passed else 'FAIL'})")

        results.append(EvalResult(
            metric_id="EX35.behavior_capability_gap",
            value=gap if not np.isnan(gap) else 0.0,
            n_samples=len(prompts),
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": task,
                "behavior_gap": gap if not np.isnan(gap) else None,
                "faithfulness_correct": faith_correct if not np.isnan(faith_correct) else None,
                "faithfulness_error": faith_error if not np.isnan(faith_error) else None,
                "n_correct": n_correct,
                "n_error": n_error,
                "n_circuit_heads": len(circuit_heads),
                "passed": passed,
                "threshold": GAP_THRESHOLD,
            },
        ))

    # Aggregate
    if all_gaps:
        agg_gap = float(np.mean(all_gaps))
        agg_passed = abs(agg_gap) < GAP_THRESHOLD
        log(f"  Aggregate: behavior_gap={agg_gap:.4f} "
            f"({'PASS' if agg_passed else 'FAIL'})")
        results.append(EvalResult(
            metric_id="EX35.behavior_capability_gap",
            value=agg_gap,
            n_samples=sum(r.n_samples for r in results),
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": "aggregate",
                "mean_behavior_gap": agg_gap,
                "n_tasks": len(all_gaps),
                "passed": agg_passed,
                "threshold": GAP_THRESHOLD,
            },
        ))

    return results


def main():
    parser = parse_common_args("EX35: Behavior vs Capability Gap")
    args = parser.parse_args()

    model = load_model(args.model, args.device)

    log("=" * 60)
    log("EX35: BEHAVIOR VS CAPABILITY GAP")
    log("=" * 60)

    results = run_behavior_capability_gap(
        model,
        tasks=args.tasks or CIRCUIT_TASKS,
        n_prompts=args.n_prompts,
    )

    out = args.out or "130_behavior_capability_gap.json"
    save_results(results, out, args=args)
    log(f"\nDone. {len(results)} results.")


if __name__ == "__main__":
    main()
