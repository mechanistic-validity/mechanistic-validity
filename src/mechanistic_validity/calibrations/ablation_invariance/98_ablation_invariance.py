"""Ablation Method Invariance (Measurement Robustness)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     M98 — Ablation Method Invariance
Categories:     measurement
Validity layer: Measurement
Criteria:       Method Robustness
Establishes:    Whether circuit faithfulness scores are consistent across ablation methods
Requires:       CPU or GPU, model
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Inspired by Miller et al. 2024 ("Transformer Circuit Faithfulness Metrics
are not Robust"), this instrument tests whether a circuit's faithfulness
score is stable across three ablation methods:

  1. Zero ablation: replace non-circuit head outputs with zeros
  2. Mean ablation: replace with mean activation across a calibration set
  3. Resample ablation: replace with activations from a different prompt

For each task, we compute faithfulness (logit-diff recovery) under all
three methods and report the maximum divergence between any pair.

Pass condition: divergence < 0.20 (scores agree within 20pp).

Usage:
    uv run python 98_ablation_invariance.py --tasks ioi --n-prompts 40
    uv run python 98_ablation_invariance.py --tasks ioi sva --device cpu
"""

import numpy as np
import torch

from mechanistic_validity.metrics.common import (
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

ABLATION_METHODS = ["zero", "mean", "resample"]
DIVERGENCE_THRESHOLD = 0.20


@torch.no_grad()
def _faithfulness_with_ablation(model, prompts, correct_ids, incorrect_ids,
                                circuit_heads, non_circuit_by_layer,
                                ablation_type, mean_z):
    """Compute faithfulness under a single ablation method.

    Faithfulness = sum(ablated_ld) / sum(clean_ld), where non-circuit
    heads are ablated and circuit heads are left intact.
    """
    n_prompts = min(len(prompts), len(correct_ids))
    faith_num = 0.0
    faith_den = 0.0

    for i in range(n_prompts):
        tokens = model.to_tokens(prompts[i].text)

        # Clean forward pass
        clean_logits = model(tokens)
        clean_ld = logit_diff_from_logits(clean_logits, correct_ids[i], incorrect_ids[i])

        if ablation_type == "resample":
            # Pick a different prompt to source replacement activations
            j = (i + 1) % n_prompts
            resample_tokens = model.to_tokens(prompts[j].text)
            _, resample_cache = model.run_with_cache(
                resample_tokens, names_filter=lambda n: "hook_z" in n)

            hooks = []
            for layer, head_list in non_circuit_by_layer.items():
                resample_z = resample_cache[f"blocks.{layer}.attn.hook_z"]

                def _hook(z, hook, _heads=head_list, _rz=resample_z):
                    for H in _heads:
                        # Match sequence length: use the shorter length
                        seq_len = min(z.shape[1], _rz.shape[1])
                        z[0, :seq_len, H, :] = _rz[0, :seq_len, H, :].to(z.device)
                    return z
                hooks.append((f"blocks.{layer}.attn.hook_z", _hook))
        else:
            hooks = make_ablation_hook(non_circuit_by_layer, mean_z, ablation_type)

        ablated_logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
        ablated_ld = logit_diff_from_logits(ablated_logits, correct_ids[i], incorrect_ids[i])

        faith_num += ablated_ld
        faith_den += clean_ld

    if abs(faith_den) < 1e-8:
        return 0.0
    return faith_num / faith_den


@torch.no_grad()
def run_ablation_invariance(model, tasks, n_prompts=40):
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

        # Calibrate mean activations for mean ablation
        mean_z = calibrate_mean_z(model, prompts, n_calibration=min(100, len(prompts)))

        # Identify non-circuit heads
        non_circuit = {(L, H) for L in range(n_layers) for H in range(n_heads)} - all_heads
        non_circuit_by_layer = heads_to_layer_dict(non_circuit)

        # Compute faithfulness under each method
        scores = {}
        for method in ABLATION_METHODS:
            faith = _faithfulness_with_ablation(
                model, prompts, correct_ids, incorrect_ids,
                all_heads, non_circuit_by_layer, method, mean_z)
            scores[method] = faith
            log(f"    {method}: faithfulness={faith:.4f}")

        # Compute max divergence between any pair
        values = list(scores.values())
        divergence = max(values) - min(values)
        passed = divergence < DIVERGENCE_THRESHOLD

        log(f"    divergence={divergence:.4f}  [{'PASS' if passed else 'FAIL'}]")

        results.append(EvalResult(
            metric_id="M98.ablation_invariance",
            value=divergence,
            n_samples=len(prompts),
            metadata={
                "task": task,
                "n_circuit_heads": len(all_heads),
                "scores": scores,
                "divergence": divergence,
                "passed": passed,
                "threshold": DIVERGENCE_THRESHOLD,
            },
        ))

    return results


def main():
    parser = parse_common_args("M98: Ablation Method Invariance")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("M98: ABLATION METHOD INVARIANCE")
    log("=" * 60)

    out = args.out or "98_ablation_invariance.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_ablation_invariance(model, [task], args.n_prompts)
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)
            p = "PASS" if r.metadata["passed"] else "FAIL"
            log(f"  {task}: divergence={r.value:.4f}  [{p}]")

    save_results(results, out)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
