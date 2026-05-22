"""ModCirc Vocabulary (S09)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Meta-protocol:  Consumes results from other protocols
Categories:     meta, parcellation
Validity layer: External
Establishes:    Reusable circuit subgraphs shared across tasks
Requires:       CPU, protocol results as input
Source:         He et al. ICML 2025 (ModCirc)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Discovers a "modular circuit vocabulary" — a fixed set of task-agnostic
computational subgraphs reused across different tasks.

Criteria (from the paper):
  1. Consistency: same subgraph = same function across tasks
  2. Locality: subgraph is spatially localized (few layers)
  3. Composability: different tasks = different combinations of vocabulary

Reports: vocabulary size, coverage per task, consistency score.

Usage:
    uv run python modcirc_vocabulary.py --results-json modal_sweep_results.json
"""
import json
import time
from collections import defaultdict

import numpy as np

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    get_circuit_heads,
    get_circuit_info,
    log,
    parse_common_args,
    save_results,
)

PROTOCOL_ID = "S09"
PROTOCOL_NAME = "ModCirc Vocabulary"

GPT2_HEADS = [(layer, head) for layer in range(12) for head in range(12)]
N_HEADS = len(GPT2_HEADS)


def _find_shared_subgraphs(tasks: list[str],
                           min_tasks: int = 2) -> list[dict]:
    task_circuits = {}
    task_edges = {}
    for task in tasks:
        heads = get_circuit_heads(task)
        _, _, edges = get_circuit_info(task)
        if heads:
            task_circuits[task] = set(heads)
            task_edges[task] = edges or set()

    if len(task_circuits) < 2:
        return []

    head_tasks = defaultdict(set)
    for task, heads in task_circuits.items():
        for h in heads:
            head_tasks[h].add(task)

    shared_heads = {h: ts for h, ts in head_tasks.items() if len(ts) >= min_tasks}
    if not shared_heads:
        return []

    clusters = []
    visited = set()
    for head in sorted(shared_heads.keys()):
        if head in visited:
            continue
        cluster = {head}
        tasks_in_common = shared_heads[head].copy()
        queue = [head]
        while queue:
            current = queue.pop(0)
            layer, h_idx = current
            for other in sorted(shared_heads.keys()):
                if other in cluster or other in visited:
                    continue
                o_layer, _ = other
                if abs(o_layer - layer) <= 2 and shared_heads[other] & tasks_in_common:
                    cluster.add(other)
                    tasks_in_common &= shared_heads[other]
                    queue.append(other)
        if len(cluster) >= 2 and len(tasks_in_common) >= min_tasks:
            visited.update(cluster)
            clusters.append({
                "heads": sorted(cluster),
                "tasks": sorted(tasks_in_common),
                "size": len(cluster),
                "n_tasks": len(tasks_in_common),
                "layers": sorted(set(h[0] for h in cluster)),
                "layer_span": max(h[0] for h in cluster) - min(h[0] for h in cluster) + 1,
            })

    return sorted(clusters, key=lambda c: c["n_tasks"] * c["size"], reverse=True)


def _compute_coverage(vocabulary: list[dict], task: str) -> float:
    heads = get_circuit_heads(task)
    if not heads:
        return 0.0
    covered = set()
    for module in vocabulary:
        if task in module["tasks"]:
            for h in module["heads"]:
                covered.add(h)
    return len(covered & set(heads)) / len(heads)


def _compute_consistency(vocabulary: list[dict],
                         protocol_results: list[dict] | None) -> float:
    if not protocol_results or not vocabulary:
        return 0.0

    consistencies = []
    for module in vocabulary:
        task_scores = defaultdict(list)
        for task in module["tasks"]:
            for result in protocol_results:
                if result.get("status") != "success":
                    continue
                for mname, evals in result.get("metrics", {}).items():
                    for ev in evals:
                        meta = ev if isinstance(ev, dict) else {}
                        ev_task = meta.get("metadata", {}).get("task", meta.get("task", ""))
                        if ev_task != task:
                            continue
                        value = meta.get("value", ev.get("value", 0))
                        if isinstance(value, (int, float)):
                            task_scores[task].append(value)

        if len(task_scores) >= 2:
            means = [np.mean(v) for v in task_scores.values() if v]
            if means:
                cv = np.std(means) / (np.mean(means) + 1e-10)
                consistencies.append(1.0 - min(cv, 1.0))

    return float(np.mean(consistencies)) if consistencies else 0.0


def run_modcirc(model=None, tasks: list[str] | None = None,
                device: str = "cpu",
                protocol_results: list[dict] | None = None,
                min_tasks: int = 2) -> list[EvalResult]:
    if tasks is None:
        tasks = list(CIRCUIT_TASKS)

    log("Discovering shared circuit subgraphs...")
    vocabulary = _find_shared_subgraphs(tasks, min_tasks=min_tasks)
    log(f"  Found {len(vocabulary)} vocabulary modules")

    results = []

    for i, module in enumerate(vocabulary):
        heads_str = [f"L{h[0]}H{h[1]}" for h in module["heads"]]
        log(f"  Module {i}: {heads_str} — shared by {module['tasks']} "
            f"(span={module['layer_span']} layers)")

    total_vocab_heads = set()
    for m in vocabulary:
        total_vocab_heads.update(m["heads"])

    task_coverage = {}
    for task in tasks:
        cov = _compute_coverage(vocabulary, task)
        task_coverage[task] = cov
        if cov > 0:
            log(f"  {task} coverage: {cov:.3f}")

    mean_coverage = np.mean(list(task_coverage.values())) if task_coverage else 0.0

    consistency = _compute_consistency(vocabulary, protocol_results)

    locality = 0.0
    if vocabulary:
        locality = 1.0 - np.mean([m["layer_span"] / 12 for m in vocabulary])

    results.append(EvalResult(
        metric_id="S09.vocab_size",
        value=float(len(vocabulary)),
        n_samples=len(tasks),
        metadata={
            "n_modules": len(vocabulary),
            "n_unique_heads": len(total_vocab_heads),
            "modules": [
                {
                    **m,
                    "heads": [f"L{h[0]}H{h[1]}" for h in m["heads"]],
                }
                for m in vocabulary
            ],
        },
    ))

    results.append(EvalResult(
        metric_id="S09.mean_coverage",
        value=mean_coverage,
        n_samples=len(tasks),
        metadata={"task_coverage": task_coverage},
    ))

    results.append(EvalResult(
        metric_id="S09.consistency",
        value=consistency,
        n_samples=len(vocabulary),
        metadata={},
    ))

    results.append(EvalResult(
        metric_id="S09.locality",
        value=locality,
        n_samples=len(vocabulary),
        metadata={
            "mean_layer_span": float(np.mean([m["layer_span"] for m in vocabulary])) if vocabulary else 0.0,
        },
    ))

    return results


def run_protocol(model, tasks, n_prompts=40, device="cpu", run_cals=False,
                 protocol_results=None):
    from protocols import ProtocolResult
    t0 = time.time()
    evals = run_modcirc(model, tasks, device=device,
                        protocol_results=protocol_results)
    result = ProtocolResult(
        protocol_id=PROTOCOL_ID,
        protocol_name=PROTOCOL_NAME,
        tasks=tasks,
        elapsed_seconds=time.time() - t0,
    )
    result.metrics["modcirc"] = evals
    return result


def summarize(result) -> str:
    lines = [f"=== {PROTOCOL_NAME} ==="]
    for ev in result.metrics.get("modcirc", []):
        lines.append(f"  {ev.metric_id}: {ev.value:.4f}")
    return "\n".join(lines)


def main():
    parser = parse_common_args("S09: ModCirc Vocabulary")
    parser.add_argument("--results-json", type=str, default=None)
    parser.add_argument("--min-tasks", type=int, default=2)
    args = parser.parse_args()
    tasks = args.tasks or CIRCUIT_TASKS

    protocol_results = None
    if args.results_json:
        with open(args.results_json) as f:
            protocol_results = json.load(f)

    log("=" * 60)
    log("S09: MODCIRC VOCABULARY")
    log("=" * 60)

    results = run_modcirc(tasks=tasks, protocol_results=protocol_results,
                          min_tasks=args.min_tasks)
    out = args.out or "meta_p9_modcirc.json"
    save_results(results, out)
    log(f"\nDone. {len(results)} results.")


if __name__ == "__main__":
    main()
