"""Cross-Task Generalization (Behavioral)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     E6b — Cross-Task Generalization
Categories:     behavioral
Validity layer: External
Criteria:       Task-transfer test (ModCirc, He et al., ICML 2025)
Establishes:    Whether a circuit identified on one task also activates
                on related tasks
Requires:       CPU or GPU, model
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For a given task's circuit heads:
1. Generate prompts from N different tasks (including the source task)
2. For each task's prompts, compute mean logit attribution of the circuit
   heads (mean absolute logit diff contribution when ablating circuit
   vs non-circuit)
3. Transfer score = mean attribution on held-out tasks / attribution on
   source task
4. Selectivity = source task attribution / mean of ALL tasks attribution

Pass condition: circuit activates > 2x baseline on >= 2 held-out tasks

Usage:
    uv run python E6b_cross_task_generalization.py --tasks ioi --n-prompts 10
    uv run python E6b_cross_task_generalization.py --device cpu
"""

import numpy as np
import torch

from mechval.metrics.common import (
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
    save_incremental,
    save_results,
)

DEFAULT_TEST_TASKS = ["ioi", "greater_than", "induction", "sva", "gendered_pronoun"]


@torch.no_grad()
def compute_attribution(
    model,
    prompts,
    correct_ids,
    incorrect_ids,
    circuit_heads: set[tuple[int, int]],
    mean_z: torch.Tensor,
) -> float:
    """Mean absolute logit-diff contribution of circuit heads.

    Measures |logit_diff(clean) - logit_diff(circuit_ablated)|.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    circuit_by_layer = heads_to_layer_dict(circuit_heads)
    hooks = make_ablation_hook(circuit_by_layer, mean_z, "mean")

    attributions = []
    for i, p in enumerate(prompts):
        if i >= len(correct_ids):
            break
        tokens = model.to_tokens(p.text)
        clean_logits = model(tokens)
        clean_ld = logit_diff_from_logits(clean_logits, correct_ids[i], incorrect_ids[i])
        ablated_logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
        ablated_ld = logit_diff_from_logits(ablated_logits, correct_ids[i], incorrect_ids[i])
        attributions.append(abs(clean_ld - ablated_ld))

    if not attributions:
        return 0.0
    return float(np.mean(attributions))


@torch.no_grad()
def run_cross_task_generalization(
    model,
    source_task: str,
    test_tasks: list[str] | None = None,
    n_prompts: int = 10,
) -> EvalResult:
    tokenizer = model.tokenizer

    circuit_heads = get_circuit_heads(source_task)
    if not circuit_heads:
        return EvalResult(
            metric_id="E6b.cross_task_generalization",
            value=0.0,
            n_samples=0,
            metadata={"source_task": source_task, "error": "no circuit"},
        )

    if test_tasks is None:
        test_tasks = [t for t in DEFAULT_TEST_TASKS if t in CIRCUIT_TASKS]

    if source_task not in test_tasks:
        test_tasks = [source_task] + test_tasks

    # Calibrate mean_z on source task prompts
    source_prompts = generate_prompts(source_task, tokenizer, n_prompts)
    mean_z = calibrate_mean_z(model, source_prompts, n_calibration=min(100, len(source_prompts)))

    per_task_attribution: dict[str, float] = {}

    for task in test_tasks:
        prompts = generate_prompts(task, tokenizer, n_prompts)
        if not prompts:
            log(f"  {task}: no prompts, skipping")
            continue
        correct_ids, incorrect_ids = get_token_ids(prompts, tokenizer)
        if not correct_ids:
            continue

        attr = compute_attribution(
            model, prompts, correct_ids, incorrect_ids, circuit_heads, mean_z,
        )
        per_task_attribution[task] = attr
        log(f"  {task}: attribution={attr:.4f}")

    source_attr = per_task_attribution.get(source_task, 0.0)
    held_out_attrs = {t: v for t, v in per_task_attribution.items() if t != source_task}

    if abs(source_attr) < 1e-8 or not held_out_attrs:
        transfer_score = 0.0
    else:
        transfer_score = float(np.mean(list(held_out_attrs.values()))) / source_attr

    all_attrs = list(per_task_attribution.values())
    mean_all = float(np.mean(all_attrs)) if all_attrs else 0.0
    selectivity = source_attr / mean_all if abs(mean_all) > 1e-8 else float("inf")

    # Pass: circuit activates > 2x baseline on >= 2 held-out tasks
    # "baseline" = mean of all held-out attributions
    if held_out_attrs:
        baseline = float(np.mean(list(held_out_attrs.values())))
    else:
        baseline = 0.0
    n_above_2x = sum(1 for v in held_out_attrs.values() if v > 2.0 * baseline) if abs(baseline) > 1e-8 else 0
    passed = n_above_2x >= 2

    log(f"  transfer_score={transfer_score:.3f}  selectivity={selectivity:.3f}  "
        f"[{'PASS' if passed else 'FAIL'}]")

    return EvalResult(
        metric_id="E6b.cross_task_generalization",
        value=transfer_score,
        n_samples=n_prompts,
        metadata={
            "source_task": source_task,
            "per_task_attribution": per_task_attribution,
            "selectivity": selectivity,
            "transfer_score": transfer_score,
            "n_circuit_heads": len(circuit_heads),
            "circuit_heads": sorted(circuit_heads),
            "test_tasks": test_tasks,
            "n_above_2x_baseline": n_above_2x,
            "passed": passed,
        },
    )


def main():
    parser = parse_common_args("E6b: Cross-Task Generalization")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS[:3]
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("E6b: CROSS-TASK GENERALIZATION")
    log("=" * 60)

    out = args.out or "E6b_cross_task_generalization.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for source_task in tasks:
        log(f"Source task: {source_task}")
        r = run_cross_task_generalization(model, source_task, n_prompts=args.n_prompts)
        results.append(r)
        save_incremental(r, jsonl_out)

    save_results(results, out, args=args)
    log(f"\nDone. {len(results)} source tasks evaluated.")


if __name__ == "__main__":
    main()
