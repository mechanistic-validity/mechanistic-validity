"""MIB Faithfulness Curve
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     MIB — Faithfulness Curve
Categories:     measurement
Validity layer: Measurement
Criteria:       Multi-threshold faithfulness for CPR/CMD computation
Establishes:    Circuit quality via area under the faithfulness curve
                across edge-count thresholds
Requires:       CPU or GPU, model
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Based on Mueller et al. 2025 ("MIB", ICML 2025).

For each threshold t in MIB_THRESHOLDS:
  1. n = max(1, int(t * total_edges))
  2. Keep top-n edges (by layer order)
  3. Convert kept edges to heads, compute faithfulness
  4. Record faithfulness at threshold t

CPR = area under the faithfulness curve (trapezoidal rule)
CMD = area between curve and y=1 line

Pass condition: CPR > 0.5

Usage:
    uv run python MIB_faithfulness_curve.py --tasks ioi --n-prompts 40
"""

import numpy as np
import torch

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    calibrate_mean_z,
    compute_faithfulness,
    generate_prompts,
    get_circuit_info,
    get_token_ids,
    load_model,
    log,
    parse_common_args,
    save_incremental,
    save_results,
)

MIB_THRESHOLDS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]


def edges_to_heads(edges: set[tuple[int, int, int, int]]) -> set[tuple[int, int]]:
    heads = set()
    for sl, sh, rl, rh in edges:
        heads.add((sl, sh))
        heads.add((rl, rh))
    return heads


def compute_cpr(thresholds: list[float], faithfulness_values: list[float]) -> float:
    """Area under the faithfulness curve via trapezoidal rule."""
    return float(np.trapz(faithfulness_values, thresholds))


def compute_cmd(thresholds: list[float], faithfulness_values: list[float]) -> float:
    """Area between faithfulness curve and y=1 (perfect) line."""
    perfect = np.ones(len(faithfulness_values))
    deficit = np.array(perfect) - np.array(faithfulness_values)
    return float(np.trapz(deficit, thresholds))


@torch.no_grad()
def run_mib_faithfulness(model, tasks: list[str],
                         n_prompts: int = 10) -> list[EvalResult]:
    tokenizer = model.tokenizer
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
            continue

        total_edges = len(all_edges)
        if total_edges == 0:
            log(f"  {task}: no edges, skipping")
            continue

        log(f"  {task}: {len(all_heads)} heads, {total_edges} edges, {len(prompts)} prompts")

        mean_z = calibrate_mean_z(model, prompts, n_calibration=min(100, len(prompts)))

        # Sort edges by sender layer then receiver layer (layer-order heuristic)
        sorted_edges = sorted(all_edges, key=lambda e: (e[0], e[2], e[1], e[3]))

        thresholds_used = []
        faithfulness_values = []
        per_threshold = {}

        for t in MIB_THRESHOLDS:
            n_keep = max(1, int(t * total_edges))
            kept_edges = set(sorted_edges[:n_keep])
            kept_heads = edges_to_heads(kept_edges)

            if not kept_heads:
                kept_heads = {sorted_edges[0][:2]}

            faith = compute_faithfulness(
                model, prompts, correct_ids, incorrect_ids, kept_heads, mean_z)

            thresholds_used.append(t)
            faithfulness_values.append(faith)
            per_threshold[str(t)] = faith

            log(f"    t={t:.3f}  n_edges={n_keep}  n_heads={len(kept_heads)}  "
                f"faith={faith:.3f}")

        cpr = compute_cpr(thresholds_used, faithfulness_values)
        cmd = compute_cmd(thresholds_used, faithfulness_values)
        passed = cpr > 0.5

        faith_curve = {t: f for t, f in zip(thresholds_used, faithfulness_values)}

        log(f"    CPR={cpr:.4f}  CMD={cmd:.4f}  [{'PASS' if passed else 'FAIL'}]")

        results.append(EvalResult(
            metric_id="MIB.faithfulness_curve",
            value=cpr,
            n_samples=len(prompts),
            faithfulness_curve=faith_curve,
            cpr=cpr,
            cmd=cmd,
            metadata={
                "task": task,
                "n_heads": len(all_heads),
                "n_edges": total_edges,
                "thresholds": MIB_THRESHOLDS,
                "per_threshold_faithfulness": per_threshold,
                "passed": passed,
            },
        ))

    return results


def main():
    parser = parse_common_args("MIB: Faithfulness Curve")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("MIB: FAITHFULNESS CURVE")
    log("=" * 60)

    out = args.out or "MIB_faithfulness_curve.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_mib_faithfulness(model, [task], args.n_prompts)
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)
            p = "PASS" if r.metadata["passed"] else "FAIL"
            log(f"  {task}: CPR={r.value:.4f}  [{p}]")

    save_results(results, out, args=args)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
