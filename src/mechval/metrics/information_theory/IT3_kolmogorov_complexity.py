"""Kolmogorov Complexity Proxy via Weight Compression
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Metric:         IT3 — Kolmogorov Complexity Proxy
Categories:     extended, information_theory
Tier:           extended
Origin:         established

Tests compressibility of circuit weight matrices as a proxy for
algorithmic complexity. Flattens W_Q, W_K, W_V, W_O per head to
float16 bytes and measures gzip compression ratio. High ratio =
structured weights; low ratio = random-looking.

Pass: mean compression ratio > 1.5
Ref: Kolmogorov 1965 "Three Approaches to the Definition of the
     Concept of Information", Problems Inform. Transmission 1:1-7

Usage:
    uv run python IT3_kolmogorov_complexity.py --tasks ioi sva --n-prompts 40
"""

import gzip

import numpy as np
import torch

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    InstrumentInfo,
    get_circuit_heads,
    load_model,
    log,
    parse_common_args,
    save_incremental,
    save_results,
)

INSTRUMENT_INFO = InstrumentInfo(
    name="Kolmogorov Complexity Proxy",
    paper_ref="Kolmogorov 1965",
    paper_cite="Kolmogorov 1965, Three Approaches to the Definition of the Concept of Information, Problems Inform. Transmission 1:1-7",
    description="Tests compressibility of circuit weight matrices as a proxy for algorithmic complexity",
    category="extended",
    tier="extended",
    origin="established",
)

COMPRESSION_THRESHOLD = 1.5
WEIGHT_NAMES = ["W_Q", "W_K", "W_V", "W_O"]


def _compression_ratio(tensor: torch.Tensor) -> float:
    """Compute gzip compression ratio for a tensor converted to float16 bytes."""
    raw = tensor.detach().cpu().to(torch.float16).numpy().tobytes()
    compressed = gzip.compress(raw)
    if len(compressed) == 0:
        return 1.0
    return len(raw) / len(compressed)


@torch.no_grad()
def run_kolmogorov_complexity(model, tasks: list[str],
                              n_prompts: int = 40) -> list[EvalResult]:
    results = []

    for task in tasks:
        circuit_heads = get_circuit_heads(task)
        if not circuit_heads:
            log(f"  {task}: no circuit heads, skipping")
            continue

        head_list = sorted(circuit_heads)
        log(f"  {task}: {len(head_list)} heads (weight-space, no forward pass)")

        head_ratios = {}

        for L, H in head_list:
            block = model.blocks[L].attn
            ratios_for_head = {}

            for wname in WEIGHT_NAMES:
                w = getattr(block, wname)  # (n_heads, d_model, d_head) or transposed
                # Extract the slice for this head
                head_w = w[H]
                ratio = _compression_ratio(head_w)
                ratios_for_head[wname] = float(ratio)

            mean_ratio = float(np.mean(list(ratios_for_head.values())))
            head_ratios[f"L{L}H{H}"] = {
                "per_weight": ratios_for_head,
                "mean_ratio": mean_ratio,
            }

        mean_compression = float(np.mean([v["mean_ratio"] for v in head_ratios.values()]))
        passed = mean_compression > COMPRESSION_THRESHOLD

        log(f"    mean_compression_ratio={mean_compression:.3f}  [{'PASS' if passed else 'FAIL'}]")

        results.append(EvalResult(
            metric_id="IT3.kolmogorov_complexity",
            value=mean_compression,
            n_samples=0,
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": task,
                "n_heads": len(head_list),
                "mean_compression_ratio": mean_compression,
                "per_head": head_ratios,
                "threshold": COMPRESSION_THRESHOLD,
                "passed": passed,
            },
        ))

    return results


def main():
    parser = parse_common_args("IT3: Kolmogorov Complexity Proxy")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("IT3: KOLMOGOROV COMPLEXITY PROXY")
    log("=" * 60)

    out = args.out or "IT3_kolmogorov_complexity.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_kolmogorov_complexity(model, [task], args.n_prompts)
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)

    save_results(results, out, args=args)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
