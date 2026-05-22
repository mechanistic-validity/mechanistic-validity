"""Ablation Method Invariance (E1b)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     E1b — Ablation Method Invariance
Categories:     measurement
Validity layer: Measurement
Criteria:       E1b Method Invariance
Establishes:    Whether faithfulness scores are robust across ablation methods
Requires:       CPU or GPU, model
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Based on "Towards Best Practices of Activation Patching" (ICLR 2024)
and "Faithfulness Metrics are Not Robust" (Miller et al. 2024).

Runs faithfulness under three ablation methods (zero, mean, noise)
and measures consistency. The value is 1.0 - max_divergence: higher
means the circuit's faithfulness score is more robust to ablation choice.

Pass condition: max_divergence < 0.2 (value > 0.8).

Usage:
    uv run python E1b_method_invariance.py --tasks ioi --n-prompts 10
    uv run python E1b_method_invariance.py --tasks ioi sva --device cpu
"""

import torch

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    calibrate_mean_z,
    generate_prompts,
    get_circuit_info,
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

ABLATION_METHODS = ["zero", "mean", "noise"]
DIVERGENCE_THRESHOLD = 0.20


@torch.no_grad()
def _faithfulness_with_method(model, prompts, correct_ids, incorrect_ids,
                              non_circuit_by_layer, mean_z, method):
    n_prompts = min(len(prompts), len(correct_ids))
    faith_num = 0.0
    faith_den = 0.0

    hooks = make_ablation_hook(non_circuit_by_layer, mean_z, method)

    for i in range(n_prompts):
        tokens = model.to_tokens(prompts[i].text)
        clean_logits = model(tokens)
        clean_ld = logit_diff_from_logits(clean_logits, correct_ids[i], incorrect_ids[i])
        ablated_logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
        ablated_ld = logit_diff_from_logits(ablated_logits, correct_ids[i], incorrect_ids[i])
        faith_num += ablated_ld
        faith_den += clean_ld

    if abs(faith_den) < 1e-8:
        return 0.0
    return faith_num / faith_den


@torch.no_grad()
def run_ablation_method_invariance(model, tasks: list[str], n_prompts: int = 10) -> list[EvalResult]:
    tokenizer = model.tokenizer
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    results = []

    for task in tasks:
        circuit, all_heads, all_edges = get_circuit_info(task)
        if circuit is None or not all_heads:
            log(f"  {task}: no circuit, skipping")
            continue

        prompts = generate_prompts(task, tokenizer, n_prompts)
        if not prompts:
            log(f"  {task}: no prompts, skipping")
            continue

        correct_ids, incorrect_ids = get_token_ids(prompts, tokenizer)
        if not correct_ids:
            log(f"  {task}: no valid token ids, skipping")
            continue

        log(f"  {task}: {len(all_heads)} circuit heads, {len(prompts)} prompts")

        mean_z = calibrate_mean_z(model, prompts, n_calibration=min(100, len(prompts)))
        non_circuit = {(L, H) for L in range(n_layers) for H in range(n_heads)} - all_heads
        non_circuit_by_layer = heads_to_layer_dict(non_circuit)

        scores = {}
        for method in ABLATION_METHODS:
            faith = _faithfulness_with_method(
                model, prompts, correct_ids, incorrect_ids,
                non_circuit_by_layer, mean_z, method,
            )
            scores[method] = faith
            log(f"    {method}: faithfulness={faith:.4f}")

        # Pairwise divergences
        pairwise = {}
        for i, m1 in enumerate(ABLATION_METHODS):
            for m2 in ABLATION_METHODS[i + 1:]:
                key = f"{m1}_vs_{m2}"
                pairwise[key] = abs(scores[m1] - scores[m2])

        values = list(scores.values())
        max_divergence = max(values) - min(values)
        value = 1.0 - max_divergence
        passed = max_divergence < DIVERGENCE_THRESHOLD

        log(f"    max_divergence={max_divergence:.4f} value={value:.4f} [{'PASS' if passed else 'FAIL'}]")

        results.append(EvalResult(
            metric_id="E1b.method_invariance",
            value=value,
            n_samples=len(prompts),
            metadata={
                "task": task,
                "n_circuit_heads": len(all_heads),
                "per_method_faithfulness": scores,
                "pairwise_divergences": pairwise,
                "max_divergence": max_divergence,
                "passed": passed,
                "threshold": DIVERGENCE_THRESHOLD,
            },
        ))

    return results


def main():
    parser = parse_common_args("E1b: Ablation Method Invariance")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("E1b: ABLATION METHOD INVARIANCE")
    log("=" * 60)

    out = args.out or "E1b_method_invariance.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_ablation_method_invariance(model, [task], args.n_prompts)
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)
            p = "PASS" if r.metadata["passed"] else "FAIL"
            log(f"  {task}: value={r.value:.4f} [{p}]")

    save_results(results, out, args=args)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
