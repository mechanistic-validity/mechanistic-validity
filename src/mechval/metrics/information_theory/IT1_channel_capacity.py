"""Channel Capacity Utilization of Circuit Heads
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Metric:         IT1 — Channel Capacity
Categories:     extended, information_theory
Tier:           extended
Origin:         established

Tests how much task-relevant information each circuit head transmits.
Computes mutual information between discretized head activations and
the correct output token. MI / log2(n_bins) = utilization ratio.

Pass: mean utilization > 0.3
Ref: Shannon 1948 "A Mathematical Theory of Communication",
     Bell System Technical Journal 27:379-423

Usage:
    uv run python IT1_channel_capacity.py --tasks ioi sva --n-prompts 40
"""

import numpy as np
import torch

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    InstrumentInfo,
    generate_prompts,
    get_circuit_heads,
    get_token_ids,
    load_model,
    log,
    logit_diff_from_logits,
    parse_common_args,
    save_incremental,
    save_results,
)

INSTRUMENT_INFO = InstrumentInfo(
    name="Channel Capacity",
    paper_ref="Shannon 1948",
    paper_cite="Shannon 1948, A Mathematical Theory of Communication, Bell System Technical Journal 27:379-423",
    description="Tests how much task-relevant information each circuit head transmits via MI utilization ratio",
    category="extended",
    tier="extended",
    origin="established",
)

N_BINS = 10
UTILIZATION_THRESHOLD = 0.3


def _mutual_information(x_bins: np.ndarray, y_labels: np.ndarray) -> float:
    """Compute MI(X; Y) where X is discretized and Y is categorical."""
    n = len(x_bins)
    if n == 0:
        return 0.0

    # Joint and marginal counts
    x_vals = np.unique(x_bins)
    y_vals = np.unique(y_labels)
    mi = 0.0
    for xv in x_vals:
        for yv in y_vals:
            p_xy = np.sum((x_bins == xv) & (y_labels == yv)) / n
            p_x = np.sum(x_bins == xv) / n
            p_y = np.sum(y_labels == yv) / n
            if p_xy > 0 and p_x > 0 and p_y > 0:
                mi += p_xy * np.log2(p_xy / (p_x * p_y))
    return mi


@torch.no_grad()
def run_channel_capacity(model, tasks: list[str],
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

        n_valid = min(len(prompts), len(correct_ids))
        head_list = sorted(circuit_heads)

        log(f"  {task}: {len(head_list)} heads, {n_valid} prompts")

        # Collect activation norms and correct token IDs per prompt
        activations = {h: np.zeros(n_valid) for h in head_list}
        labels = np.array(correct_ids[:n_valid])

        for p_idx in range(n_valid):
            tokens = model.to_tokens(prompts[p_idx].text)
            _, cache = model.run_with_cache(
                tokens, names_filter=lambda n: "hook_z" in n)
            for L, H in head_list:
                z = cache[f"blocks.{L}.attn.hook_z"]
                activations[(L, H)][p_idx] = z[0, -1, H].norm().item()

        # Compute MI utilization per head
        head_utilizations = {}
        max_mi = np.log2(N_BINS)

        for L, H in head_list:
            acts = activations[(L, H)]
            # Discretize into bins via equal-width binning
            a_min, a_max = acts.min(), acts.max()
            if a_max - a_min < 1e-10:
                bins = np.zeros(n_valid, dtype=int)
            else:
                bins = np.clip(
                    ((acts - a_min) / (a_max - a_min) * N_BINS).astype(int),
                    0, N_BINS - 1,
                )

            mi = _mutual_information(bins, labels)
            utilization = mi / max_mi if max_mi > 0 else 0.0
            head_utilizations[f"L{L}H{H}"] = {
                "mi": float(mi),
                "utilization": float(utilization),
            }

        mean_util = float(np.mean([v["utilization"] for v in head_utilizations.values()]))
        passed = mean_util > UTILIZATION_THRESHOLD

        log(f"    mean_utilization={mean_util:.4f}  [{'PASS' if passed else 'FAIL'}]")

        results.append(EvalResult(
            metric_id="IT1.channel_capacity",
            value=mean_util,
            n_samples=len(prompts),
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": task,
                "n_heads": len(head_list),
                "mean_utilization": mean_util,
                "n_bins": N_BINS,
                "max_possible_mi": float(max_mi),
                "per_head": head_utilizations,
                "threshold": UTILIZATION_THRESHOLD,
                "passed": passed,
            },
        ))

    return results


def main():
    parser = parse_common_args("IT1: Channel Capacity")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("IT1: CHANNEL CAPACITY")
    log("=" * 60)

    out = args.out or "IT1_channel_capacity.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_channel_capacity(model, [task], args.n_prompts)
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)

    save_results(results, out, args=args)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
