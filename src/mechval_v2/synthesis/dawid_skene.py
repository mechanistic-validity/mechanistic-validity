"""Dawid-Skene Protocol Consensus (S02)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Meta-protocol:  Consumes results from other protocols
Categories:     meta, ensemble
Validity layer: Construct + Internal
Establishes:    Consensus circuit membership + per-protocol reliability
Requires:       CPU, protocol results as input
Source:         Dawid & Skene 1979 (crowdsourcing EM)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Treats each protocol as an "annotator" that labels heads as circuit
members or not. EM algorithm jointly estimates:
  1. True binary label per head (in-circuit vs not)
  2. Per-protocol confusion matrix (sensitivity + specificity)

This answers: "which heads are circuit members when we account for
the fact that some protocols are more reliable than others?"

Usage:
    uv run python dawid_skene.py --results-json modal_sweep_results.json --task ioi
"""
import json
import time

import numpy as np

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    get_circuit_heads,
    log,
    parse_common_args,
    save_results,
)

PROTOCOL_ID = "S02"
PROTOCOL_NAME = "Dawid-Skene Protocol Consensus"

GPT2_HEADS = [(layer, head) for layer in range(12) for head in range(12)]
N_HEADS = len(GPT2_HEADS)


def _extract_head_labels(protocol_results: list[dict], task: str,
                         threshold: float = 0.5) -> dict[str, np.ndarray]:
    head_to_idx = {h: i for i, h in enumerate(GPT2_HEADS)}
    annotations = {}

    for result in protocol_results:
        if result.get("status") != "success":
            continue
        proto_id = result.get("protocol_id", "unknown")

        labels = np.full(N_HEADS, -1, dtype=int)  # -1 = no annotation
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
                        labels[head_to_idx[parsed]] = 1 if val >= threshold else 0

        annotated = (labels >= 0).sum()
        if annotated > 0:
            annotations[proto_id] = labels

    return annotations


def _parse_head_key(key: str) -> tuple[int, int] | None:
    try:
        if key.startswith("L") and "H" in key:
            parts = key[1:].split("H")
            return int(parts[0]), int(parts[1])
    except (ValueError, IndexError):
        pass
    return None


def dawid_skene_em(annotations: dict[str, np.ndarray],
                   n_items: int, n_classes: int = 2,
                   max_iter: int = 50, tol: float = 1e-4) -> tuple[np.ndarray, dict]:
    annotators = list(annotations.keys())
    n_annotators = len(annotators)

    prevalence = np.array([0.7, 0.3])
    true_labels = np.random.rand(n_items, n_classes)
    true_labels /= true_labels.sum(axis=1, keepdims=True)

    confusion = {}
    for a in annotators:
        cm = np.eye(n_classes) * 0.7 + 0.15
        cm /= cm.sum(axis=1, keepdims=True)
        confusion[a] = cm

    for iteration in range(max_iter):
        old_labels = true_labels.copy()

        true_labels = np.tile(np.log(prevalence + 1e-10), (n_items, 1))
        for a_idx, a in enumerate(annotators):
            labels = annotations[a]
            for i in range(n_items):
                if labels[i] < 0:
                    continue
                for c in range(n_classes):
                    true_labels[i, c] += np.log(confusion[a][c, labels[i]] + 1e-10)

        true_labels -= true_labels.max(axis=1, keepdims=True)
        true_labels = np.exp(true_labels)
        true_labels /= true_labels.sum(axis=1, keepdims=True)

        prevalence = true_labels.mean(axis=0)

        for a in annotators:
            cm = np.zeros((n_classes, n_classes))
            labels = annotations[a]
            for i in range(n_items):
                if labels[i] < 0:
                    continue
                for c in range(n_classes):
                    cm[c, labels[i]] += true_labels[i, c]
            row_sums = cm.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            confusion[a] = cm / row_sums

        delta = np.abs(true_labels - old_labels).max()
        if delta < tol:
            log(f"  EM converged at iteration {iteration + 1} (delta={delta:.6f})")
            break

    reliability = {}
    for a in annotators:
        cm = confusion[a]
        sensitivity = cm[1, 1] if n_classes > 1 else 1.0
        specificity = cm[0, 0] if n_classes > 1 else 1.0
        reliability[a] = {
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "accuracy": float(np.diag(cm).sum() / n_classes),
        }

    return true_labels, reliability


def run_dawid_skene(model=None, tasks: list[str] | None = None,
                    device: str = "cpu",
                    protocol_results: list[dict] | None = None,
                    threshold: float = 0.5) -> list[EvalResult]:
    if tasks is None:
        tasks = list(CIRCUIT_TASKS)
    if protocol_results is None:
        log("  No protocol results provided, skipping")
        return []

    results = []
    for task in tasks:
        annotations = _extract_head_labels(protocol_results, task, threshold)
        if len(annotations) < 2:
            log(f"  {task}: only {len(annotations)} protocols annotated, skipping")
            continue

        log(f"  {task}: {len(annotations)} protocol annotations, running EM...")
        posteriors, reliability = dawid_skene_em(annotations, N_HEADS)

        consensus_circuit = [
            GPT2_HEADS[i] for i in range(N_HEADS) if posteriors[i, 1] > 0.5
        ]

        gt_heads = get_circuit_heads(task)
        if gt_heads:
            consensus_set = set(consensus_circuit)
            gt_set = set(gt_heads)
            intersection = consensus_set & gt_set
            union = consensus_set | gt_set
            jaccard = len(intersection) / len(union) if union else 0.0
            precision = len(intersection) / len(consensus_set) if consensus_set else 0.0
            recall = len(intersection) / len(gt_set) if gt_set else 0.0
        else:
            jaccard = precision = recall = 0.0

        sorted_reliability = sorted(reliability.items(),
                                     key=lambda x: x[1]["accuracy"], reverse=True)
        log(f"    Consensus: {len(consensus_circuit)} heads")
        log(f"    GT overlap: Jaccard={jaccard:.3f}, P={precision:.3f}, R={recall:.3f}")
        log(f"    Most reliable: {sorted_reliability[0][0]} "
            f"(acc={sorted_reliability[0][1]['accuracy']:.3f})")
        log(f"    Least reliable: {sorted_reliability[-1][0]} "
            f"(acc={sorted_reliability[-1][1]['accuracy']:.3f})")

        results.append(EvalResult(
            metric_id="S02.consensus_jaccard",
            value=jaccard,
            n_samples=len(annotations),
            metadata={
                "task": task,
                "n_protocols": len(annotations),
                "n_consensus_heads": len(consensus_circuit),
                "consensus_heads": [f"L{l}H{h}" for l, h in consensus_circuit],
                "jaccard_with_gt": jaccard,
                "precision": precision,
                "recall": recall,
                "protocol_reliability": {k: v for k, v in sorted_reliability},
            },
        ))

    return results


def run_protocol(model, tasks, n_prompts=40, device="cpu", run_cals=False,
                 protocol_results=None):
    from protocols import ProtocolResult
    t0 = time.time()
    evals = run_dawid_skene(model, tasks, device=device,
                            protocol_results=protocol_results)
    result = ProtocolResult(
        protocol_id=PROTOCOL_ID,
        protocol_name=PROTOCOL_NAME,
        tasks=tasks,
        elapsed_seconds=time.time() - t0,
    )
    result.metrics["consensus"] = evals
    return result


def summarize(result) -> str:
    lines = [f"=== {PROTOCOL_NAME} ==="]
    for ev in result.metrics.get("consensus", []):
        lines.append(f"  {ev.metric_id}: {ev.value:.4f}")
    return "\n".join(lines)


def main():
    parser = parse_common_args("S02: Dawid-Skene Consensus")
    parser.add_argument("--results-json", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    tasks = args.tasks or CIRCUIT_TASKS
    with open(args.results_json) as f:
        protocol_results = json.load(f)

    log("=" * 60)
    log("S02: DAWID-SKENE PROTOCOL CONSENSUS")
    log("=" * 60)

    results = run_dawid_skene(tasks=tasks, protocol_results=protocol_results,
                              threshold=args.threshold)
    out = args.out or "meta_p2_dawid_skene.json"
    save_results(results, out)
    log(f"\nDone. {len(results)} results.")


if __name__ == "__main__":
    main()
