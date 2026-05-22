"""Parallel Ensemble (S04)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Meta-protocol:  Consumes results from other protocols
Categories:     meta, ensemble, PARCEL Layer 2a
Validity layer: Internal
Establishes:    Weighted average + minimum ensemble circuit from all method scores
Requires:       CPU, protocol results as input
Source:         Mondorf et al. BlackboxNLP 2025 (MIB Shared Task Ensemble)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Three fusion rules applied to rank-normalized component scores:
  1. Equal-weight average
  2. Method-type-weighted average (causal 2x, statistical 1x, spectral 1.5x)
  3. Minimum (conservative — only keeps components ALL methods agree on)

Reports circuit precision/recall vs ground truth for each fusion rule.

Usage:
    uv run python parallel_ensemble.py --results-json modal_sweep_results.json --task ioi
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

PROTOCOL_ID = "S04"
PROTOCOL_NAME = "Parallel Ensemble (PARCEL L2a)"

GPT2_HEADS = [(layer, head) for layer in range(12) for head in range(12)]
N_HEADS = len(GPT2_HEADS)

METHOD_TYPE_WEIGHTS = {
    "causal": 2.0,
    "statistical": 1.0,
    "spectral": 1.5,
    "structural": 1.0,
    "behavioral": 1.0,
    "representational": 1.0,
}

PROTOCOL_TYPES = {
    "A01": "causal", "A02": "causal", "A03": "causal", "A04": "causal",
    "A05": "causal", "A06": "causal", "A07": "causal", "A08": "causal",
    "A09": "causal", "A10": "causal", "A11": "causal", "A12": "causal",
    "A13": "causal",
    "B01": "structural", "B02": "structural", "B03": "structural", "B04": "structural",
    "D01": "behavioral", "D02": "behavioral", "D03": "behavioral",
    "C01": "causal", "C02": "causal", "C03": "causal",
    "E01": "representational",
    "WC_M1": "causal", "WC_M2": "causal", "WC_M3": "statistical",
    "WC_M4": "structural", "WC_M5": "spectral", "WC_M6": "spectral",
    "WC_M7": "spectral", "WC_M8": "spectral", "WC_M9": "statistical",
    "WC_M10": "spectral", "WC_M11": "statistical", "WC_M12": "spectral",
    "WC_M13": "statistical",
    "MB_KH": "causal", "MB_RE": "causal", "MB_MR": "causal",
    "MB_SS": "causal", "MB_GI": "causal", "MB_IP": "causal",
    "MB_DR": "causal", "MB_TE": "causal", "MB_MRX": "causal",
    "MB_MD": "causal", "MB_SA": "causal", "MB_CO": "causal",
    "MB_EA": "causal", "MB_QE": "causal", "MB_CD": "causal",
    "MB_SB": "causal",
}


def _parse_head_key(key: str) -> tuple[int, int] | None:
    try:
        if key.startswith("L") and "H" in key:
            parts = key[1:].split("H")
            return int(parts[0]), int(parts[1])
    except (ValueError, IndexError):
        pass
    return None


def _extract_score_vectors(protocol_results: list[dict],
                           task: str) -> dict[str, np.ndarray]:
    head_to_idx = {h: i for i, h in enumerate(GPT2_HEADS)}
    score_vectors = {}

    for result in protocol_results:
        if result.get("status") != "success":
            continue
        proto_id = result.get("protocol_id", "unknown")
        scores = np.zeros(N_HEADS)
        found = False

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
                        scores[head_to_idx[parsed]] = max(scores[head_to_idx[parsed]], val)
                        found = True

        if found:
            score_vectors[proto_id] = scores

    return score_vectors


def _rank_normalize(scores: np.ndarray) -> np.ndarray:
    n = len(scores)
    ranks = np.zeros(n)
    order = np.argsort(-scores)
    for rank, idx in enumerate(order):
        ranks[idx] = (rank + 1) / n
    return 1.0 - ranks  # higher = more important


def _circuit_metrics(predicted_set: set, gt_set: set) -> dict:
    if not gt_set:
        return {"jaccard": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    intersection = predicted_set & gt_set
    union = predicted_set | gt_set
    jaccard = len(intersection) / len(union) if union else 0.0
    precision = len(intersection) / len(predicted_set) if predicted_set else 0.0
    recall = len(intersection) / len(gt_set) if gt_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"jaccard": jaccard, "precision": precision, "recall": recall, "f1": f1}


def run_parallel_ensemble(model=None, tasks: list[str] | None = None,
                          device: str = "cpu",
                          protocol_results: list[dict] | None = None,
                          top_k: int = 20) -> list[EvalResult]:
    if tasks is None:
        tasks = list(CIRCUIT_TASKS)
    if protocol_results is None:
        return []

    results = []
    for task in tasks:
        score_vectors = _extract_score_vectors(protocol_results, task)
        if len(score_vectors) < 2:
            log(f"  {task}: only {len(score_vectors)} protocols, skipping")
            continue

        normalized = {pid: _rank_normalize(sv) for pid, sv in score_vectors.items()}
        log(f"  {task}: {len(normalized)} protocols, fusing...")

        matrix = np.stack(list(normalized.values()))  # (M, K)

        equal_avg = matrix.mean(axis=0)

        weights = []
        for pid in normalized:
            ptype = PROTOCOL_TYPES.get(pid, "causal")
            weights.append(METHOD_TYPE_WEIGHTS.get(ptype, 1.0))
        weights = np.array(weights) / sum(weights)
        weighted_avg = (matrix * weights[:, None]).sum(axis=0)

        minimum = matrix.min(axis=0)

        gt_heads = get_circuit_heads(task)
        gt_set = set(gt_heads) if gt_heads else set()

        for rule_name, fused_scores in [("equal_avg", equal_avg),
                                         ("weighted_avg", weighted_avg),
                                         ("minimum", minimum)]:
            top_indices = np.argsort(-fused_scores)[:top_k]
            predicted = {GPT2_HEADS[i] for i in top_indices}
            metrics = _circuit_metrics(predicted, gt_set)

            top_heads = [(GPT2_HEADS[i], float(fused_scores[i])) for i in top_indices[:10]]
            log(f"    {rule_name}: F1={metrics['f1']:.3f} P={metrics['precision']:.3f} "
                f"R={metrics['recall']:.3f} J={metrics['jaccard']:.3f}")

            results.append(EvalResult(
                metric_id=f"S04.{rule_name}_f1",
                value=metrics["f1"],
                n_samples=len(normalized),
                metadata={
                    "task": task,
                    "fusion_rule": rule_name,
                    "n_protocols": len(normalized),
                    "top_k": top_k,
                    **metrics,
                    "top_10_heads": [
                        {"head": f"L{h[0]}H{h[1]}", "score": s}
                        for h, s in top_heads
                    ],
                    "protocol_ids": list(normalized.keys()),
                },
            ))

        corr_matrix = np.corrcoef(matrix)
        proto_ids = list(normalized.keys())
        log(f"    Method correlation (Spearman) — {len(proto_ids)} methods")

        results.append(EvalResult(
            metric_id="S04.method_correlation",
            value=float(np.mean(corr_matrix[np.triu_indices(len(corr_matrix), k=1)])),
            n_samples=len(proto_ids) * (len(proto_ids) - 1) // 2,
            metadata={
                "task": task,
                "protocol_ids": proto_ids,
                "correlation_matrix": corr_matrix.tolist(),
            },
        ))

    return results


def run_protocol(model, tasks, n_prompts=40, device="cpu", run_cals=False,
                 protocol_results=None):
    from protocols import ProtocolResult
    t0 = time.time()
    evals = run_parallel_ensemble(model, tasks, device=device,
                                  protocol_results=protocol_results)
    result = ProtocolResult(
        protocol_id=PROTOCOL_ID,
        protocol_name=PROTOCOL_NAME,
        tasks=tasks,
        elapsed_seconds=time.time() - t0,
    )
    result.metrics["ensemble"] = evals
    return result


def summarize(result) -> str:
    lines = [f"=== {PROTOCOL_NAME} ==="]
    for ev in result.metrics.get("ensemble", []):
        lines.append(f"  {ev.metric_id}: {ev.value:.4f}")
    return "\n".join(lines)


def main():
    parser = parse_common_args("S04: Parallel Ensemble")
    parser.add_argument("--results-json", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()
    tasks = args.tasks or CIRCUIT_TASKS
    with open(args.results_json) as f:
        protocol_results = json.load(f)

    log("=" * 60)
    log("S04: PARALLEL ENSEMBLE (PARCEL L2a)")
    log("=" * 60)

    results = run_parallel_ensemble(tasks=tasks, protocol_results=protocol_results,
                                     top_k=args.top_k)
    out = args.out or "meta_p4_parallel_ensemble.json"
    save_results(results, out)
    log(f"\nDone. {len(results)} results.")


if __name__ == "__main__":
    main()
