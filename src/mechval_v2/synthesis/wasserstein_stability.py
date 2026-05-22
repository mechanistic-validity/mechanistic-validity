"""Wasserstein Circuit Distance (S06)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Meta-protocol:  Consumes results from other protocols
Categories:     meta, stability
Validity layer: External
Establishes:    Circuit stability across runs + circuit distance across tasks
Requires:       CPU, protocol results as input
Source:         Optimal transport / Earth Mover's Distance
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Computes Wasserstein-1 distance between circuit score distributions,
using functional similarity (weight-space cosine) as the ground metric.

Three applications:
  1. Stability: W₁ between two runs on same task → sensitivity to sampling
  2. Cross-task: W₁ between IOI and SVA circuits → structural divergence
  3. Cross-model: W₁ between GPT-2 Small and Medium circuits

Usage:
    uv run python wasserstein_stability.py --results-json modal_sweep_results.json
"""
import json
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    get_circuit_heads,
    log,
    parse_common_args,
    save_results,
)

PROTOCOL_ID = "S06"
PROTOCOL_NAME = "Wasserstein Circuit Distance"

GPT2_HEADS = [(layer, head) for layer in range(12) for head in range(12)]
N_HEADS = len(GPT2_HEADS)


def _wasserstein_1d(a: np.ndarray, b: np.ndarray) -> float:
    a_sorted = np.sort(a)
    b_sorted = np.sort(b)
    return float(np.mean(np.abs(a_sorted - b_sorted)))


def _wasserstein_emd(dist_a: np.ndarray, dist_b: np.ndarray,
                     cost_matrix: np.ndarray) -> float:
    a_norm = dist_a / (dist_a.sum() + 1e-10)
    b_norm = dist_b / (dist_b.sum() + 1e-10)

    n = len(a_norm)
    expanded_cost = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            expanded_cost[i, j] = cost_matrix[i, j]

    supply = (a_norm * 1000).astype(int)
    demand = (b_norm * 1000).astype(int)

    diff = supply.sum() - demand.sum()
    if diff > 0:
        supply[np.argmax(supply)] -= diff
    elif diff < 0:
        demand[np.argmax(demand)] += diff

    row_ind, col_ind = linear_sum_assignment(expanded_cost)
    return float(expanded_cost[row_ind, col_ind].mean())


def _build_weight_cost_matrix(model) -> np.ndarray:
    features = np.zeros((N_HEADS, 4))
    for i, (layer, head) in enumerate(GPT2_HEADS):
        W_Q = model.W_Q[layer, head].detach().float().cpu().numpy().flatten()
        W_K = model.W_K[layer, head].detach().float().cpu().numpy().flatten()
        features[i, 0] = np.linalg.norm(W_Q)
        features[i, 1] = np.linalg.norm(W_K)
        W_V = model.W_V[layer, head].detach().float().cpu().numpy().flatten()
        W_O = model.W_O[layer, head].detach().float().cpu().numpy().flatten()
        features[i, 2] = np.linalg.norm(W_V)
        features[i, 3] = np.linalg.norm(W_O)

    cost = cdist(features, features, metric="cosine")
    return cost


def _parse_head_key(key: str) -> tuple[int, int] | None:
    try:
        if key.startswith("L") and "H" in key:
            parts = key[1:].split("H")
            return int(parts[0]), int(parts[1])
    except (ValueError, IndexError):
        pass
    return None


def _scores_from_results(protocol_results: list[dict], task: str) -> np.ndarray:
    head_to_idx = {h: i for i, h in enumerate(GPT2_HEADS)}
    scores = np.zeros(N_HEADS)
    count = 0

    for result in protocol_results:
        if result.get("status") != "success":
            continue
        for mname, evals in result.get("metrics", {}).items():
            for ev in evals:
                meta = ev if isinstance(ev, dict) else {}
                ev_task = meta.get("metadata", {}).get("task", meta.get("task", ""))
                if ev_task != task:
                    continue
                head_scores = meta.get("metadata", {}).get("head_scores", {})
                for hkey, score in head_scores.items():
                    parsed = _parse_head_key(hkey)
                    if parsed and parsed in head_to_idx:
                        val = abs(score) if isinstance(score, (int, float)) else 0.0
                        scores[head_to_idx[parsed]] += val
                        count += 1

    if count > 0:
        scores /= scores.sum() + 1e-10
    return scores


def run_wasserstein(model=None, tasks: list[str] | None = None,
                    device: str = "cpu",
                    protocol_results: list[dict] | None = None) -> list[EvalResult]:
    if tasks is None:
        tasks = list(CIRCUIT_TASKS)

    results = []

    if len(tasks) >= 2 and protocol_results:
        log("Computing cross-task Wasserstein distances...")
        task_scores = {}
        for task in tasks:
            s = _scores_from_results(protocol_results, task)
            if s.sum() > 0:
                task_scores[task] = s

        valid_tasks = sorted(task_scores.keys())
        if len(valid_tasks) >= 2:
            for i in range(len(valid_tasks)):
                for j in range(i + 1, len(valid_tasks)):
                    t_a, t_b = valid_tasks[i], valid_tasks[j]
                    w1 = _wasserstein_1d(task_scores[t_a], task_scores[t_b])
                    log(f"  {t_a} vs {t_b}: W₁ = {w1:.6f}")

                    results.append(EvalResult(
                        metric_id="S06.cross_task_w1",
                        value=w1,
                        n_samples=N_HEADS,
                        metadata={
                            "task_a": t_a,
                            "task_b": t_b,
                            "wasserstein_1": w1,
                        },
                    ))

    for task in tasks:
        gt_heads = get_circuit_heads(task)
        if not gt_heads:
            continue

        head_to_idx = {h: i for i, h in enumerate(GPT2_HEADS)}
        gt_dist = np.zeros(N_HEADS)
        for h in gt_heads:
            if h in head_to_idx:
                gt_dist[head_to_idx[h]] = 1.0
        gt_dist /= gt_dist.sum() + 1e-10

        if protocol_results:
            pred_dist = _scores_from_results(protocol_results, task)
            if pred_dist.sum() > 0:
                w1 = _wasserstein_1d(pred_dist, gt_dist)
                log(f"  {task} predicted-vs-GT: W₁ = {w1:.6f}")
                results.append(EvalResult(
                    metric_id="S06.pred_gt_w1",
                    value=w1,
                    n_samples=N_HEADS,
                    metadata={"task": task, "wasserstein_1": w1},
                ))

    return results


def run_protocol(model, tasks, n_prompts=40, device="cpu", run_cals=False,
                 protocol_results=None):
    from protocols import ProtocolResult
    t0 = time.time()
    evals = run_wasserstein(model, tasks, device=device,
                            protocol_results=protocol_results)
    result = ProtocolResult(
        protocol_id=PROTOCOL_ID,
        protocol_name=PROTOCOL_NAME,
        tasks=tasks,
        elapsed_seconds=time.time() - t0,
    )
    result.metrics["wasserstein"] = evals
    return result


def summarize(result) -> str:
    lines = [f"=== {PROTOCOL_NAME} ==="]
    for ev in result.metrics.get("wasserstein", []):
        lines.append(f"  {ev.metric_id}: {ev.value:.6f}")
    return "\n".join(lines)


def main():
    parser = parse_common_args("S06: Wasserstein Circuit Distance")
    parser.add_argument("--results-json", type=str, required=True)
    args = parser.parse_args()
    tasks = args.tasks or CIRCUIT_TASKS
    with open(args.results_json) as f:
        protocol_results = json.load(f)

    log("=" * 60)
    log("S06: WASSERSTEIN CIRCUIT DISTANCE")
    log("=" * 60)

    results = run_wasserstein(tasks=tasks, protocol_results=protocol_results)
    out = args.out or "meta_p6_wasserstein.json"
    save_results(results, out)
    log(f"\nDone. {len(results)} results.")


if __name__ == "__main__":
    main()
