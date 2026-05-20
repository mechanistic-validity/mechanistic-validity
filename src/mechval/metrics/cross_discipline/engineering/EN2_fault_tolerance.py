"""Fault Tolerance: Single-Point-of-Failure Analysis
Counts how many individual circuit heads, if removed alone, cause >50%
performance drop. A robust circuit has no single points of failure.

Pass: spof_count == 0 (no single point of failure)
Ref: Laprie 1985, Dependable Computing and Fault Tolerance, Springer LNCS

Usage:
    uv run python EN2_fault_tolerance.py --tasks ioi --n-prompts 40
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
    name="Fault Tolerance (SPOF Count)",
    paper_ref="Laprie 1985, Springer LNCS",
    paper_cite="Laprie 1985, Dependable Computing and Fault Tolerance",
    description="Counts single-point-of-failure heads whose individual removal causes >50% performance drop",
    category="cross_discipline",
    tier="cross_discipline",
    origin="established",
)

SPOF_DROP_THRESHOLD = 0.5


@torch.no_grad()
def run_fault_tolerance(model, tasks: list[str],
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

        # Ablate each head individually and measure drop
        per_head_drops = {}
        spof_heads = []

        for L, H in sorted(circuit_heads):
            ablation_hooks = make_ablation_hook(
                heads_to_layer_dict({(L, H)}), mean_z, "mean")

            ablated_lds = []
            for i, p in enumerate(prompts):
                if i >= len(correct_ids):
                    break
                tokens = model.to_tokens(p.text)
                logits = model.run_with_hooks(tokens, fwd_hooks=ablation_hooks)
                ablated_lds.append(logit_diff_from_logits(logits, correct_ids[i], incorrect_ids[i]))

            ablated_mean = float(np.mean(ablated_lds))
            relative_drop = 1.0 - (ablated_mean / baseline_ld)
            per_head_drops[(L, H)] = relative_drop

            if relative_drop > SPOF_DROP_THRESHOLD:
                spof_heads.append((L, H))
                log(f"    SPOF: L{L}H{H} drop={relative_drop:.3f}")

        spof_count = len(spof_heads)
        max_drop = max(per_head_drops.values()) if per_head_drops else 0.0
        mean_drop = float(np.mean(list(per_head_drops.values()))) if per_head_drops else 0.0
        passed = spof_count == 0

        log(f"    spof_count={spof_count}  max_drop={max_drop:.3f}  mean_drop={mean_drop:.3f}")
        log(f"    [{'PASS' if passed else 'FAIL'}]")

        results.append(EvalResult(
            metric_id="EN2.fault_tolerance",
            value=float(spof_count),
            n_samples=len(prompts),
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": task,
                "n_heads": len(circuit_heads),
                "spof_count": spof_count,
                "spof_heads": [(L, H) for L, H in spof_heads],
                "per_head_drops": {f"L{L}H{H}": v for (L, H), v in per_head_drops.items()},
                "max_drop": max_drop,
                "mean_drop": mean_drop,
                "baseline_ld": baseline_ld,
                "passed": passed,
                "spof_drop_threshold": SPOF_DROP_THRESHOLD,
            },
        ))

    return results


def main():
    parser = parse_common_args("EN2: Fault Tolerance (SPOF Analysis)")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("EN2: FAULT TOLERANCE (SPOF ANALYSIS)")
    log("=" * 60)

    out = args.out or "EN2_fault_tolerance.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_fault_tolerance(model, [task], args.n_prompts)
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)

    save_results(results, out, args=args)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
