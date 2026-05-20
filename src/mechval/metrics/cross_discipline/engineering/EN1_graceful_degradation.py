"""Graceful Degradation Analysis
Tests whether circuit performance degrades smoothly as components are
removed, or collapses catastrophically. Heads are removed one at a time
in order of decreasing activation norm. The area under the normalized
degradation curve (AUC) measures resilience.

Pass: degradation_auc > 0.5 (better than catastrophic collapse)
Ref: Avizienis et al. 2004, IEEE TDSC 1:11-33

Usage:
    uv run python EN1_graceful_degradation.py --tasks ioi --n-prompts 40
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
    save_incremental,
    save_results,
)

INSTRUMENT_INFO = InstrumentInfo(
    name="Graceful Degradation",
    paper_ref="Avizienis et al. 2004, IEEE TDSC 1:11-33",
    paper_cite="Avizienis et al. 2004, Basic Concepts and Taxonomy of Dependable Computing",
    description="Tests whether circuit performance degrades smoothly or catastrophically as heads are removed",
    category="cross_discipline",
    tier="cross_discipline",
    origin="established",
)

AUC_THRESHOLD = 0.5


@torch.no_grad()
def run_graceful_degradation(model, tasks: list[str],
                             n_prompts: int = 40) -> list[EvalResult]:
    tokenizer = model.tokenizer
    results = []

    for task in tasks:
        circuit_heads = get_circuit_heads(task)
        if not circuit_heads:
            log(f"  {task}: no circuit heads, skipping")
            continue

        prompts = generate_prompts(task, tokenizer, n_prompts)
        if not prompts:
            continue

        correct_ids, incorrect_ids = get_token_ids(prompts, tokenizer)
        if not correct_ids:
            continue

        log(f"  {task}: {len(circuit_heads)} heads, {len(prompts)} prompts")

        mean_z = calibrate_mean_z(model, prompts, n_calibration=min(100, len(prompts)))

        # Compute baseline (clean) logit diff
        clean_lds = []
        for i, p in enumerate(prompts):
            if i >= len(correct_ids):
                break
            tokens = model.to_tokens(p.text)
            logits = model(tokens)
            clean_lds.append(logit_diff_from_logits(logits, correct_ids[i], incorrect_ids[i]))
        baseline_ld = float(np.mean(clean_lds))

        if abs(baseline_ld) < 1e-8:
            log(f"    baseline LD ~ 0, skipping")
            continue

        # Rank heads by mean activation norm (descending = most important first)
        head_norms = {}
        for L, H in circuit_heads:
            norms = []
            for i, p in enumerate(prompts):
                if i >= len(correct_ids):
                    break
                tokens = model.to_tokens(p.text)
                _, cache = model.run_with_cache(tokens, names_filter=lambda n: "hook_z" in n)
                z = cache[f"blocks.{L}.attn.hook_z"][0, -1, H, :]
                norms.append(z.norm().item())
            head_norms[(L, H)] = float(np.mean(norms))

        sorted_heads = sorted(circuit_heads, key=lambda h: head_norms[h], reverse=True)

        # Remove heads one at a time, measure LD at each step
        degradation_curve = [1.0]  # Normalized: full performance = 1.0
        removed = set()

        for head in sorted_heads:
            removed.add(head)
            ablation_hooks = make_ablation_hook(
                heads_to_layer_dict(removed), mean_z, "mean")

            step_lds = []
            for i, p in enumerate(prompts):
                if i >= len(correct_ids):
                    break
                tokens = model.to_tokens(p.text)
                logits = model.run_with_hooks(tokens, fwd_hooks=ablation_hooks)
                step_lds.append(logit_diff_from_logits(logits, correct_ids[i], incorrect_ids[i]))

            normalized_ld = float(np.mean(step_lds)) / baseline_ld
            degradation_curve.append(max(normalized_ld, 0.0))

        # Compute AUC using trapezoidal rule
        # x-axis: fraction of heads removed (0 to 1)
        n_steps = len(degradation_curve)
        xs = np.linspace(0, 1, n_steps)
        degradation_auc = float(np.trapz(degradation_curve, xs))

        # Ideal linear degradation AUC = 0.5
        passed = degradation_auc > AUC_THRESHOLD

        log(f"    degradation_auc={degradation_auc:.3f}  curve={[f'{v:.2f}' for v in degradation_curve]}")
        log(f"    [{'PASS' if passed else 'FAIL'}]")

        results.append(EvalResult(
            metric_id="EN1.graceful_degradation",
            value=degradation_auc,
            n_samples=len(prompts),
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": task,
                "n_heads": len(circuit_heads),
                "degradation_auc": degradation_auc,
                "degradation_curve": degradation_curve,
                "head_removal_order": [(L, H) for L, H in sorted_heads],
                "head_norms": {f"L{L}H{H}": v for (L, H), v in head_norms.items()},
                "baseline_ld": baseline_ld,
                "passed": passed,
                "auc_threshold": AUC_THRESHOLD,
            },
        ))

    return results


def main():
    parser = parse_common_args("EN1: Graceful Degradation")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("EN1: GRACEFUL DEGRADATION")
    log("=" * 60)

    out = args.out or "EN1_graceful_degradation.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_graceful_degradation(model, [task], args.n_prompts)
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)

    save_results(results, out, args=args)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
