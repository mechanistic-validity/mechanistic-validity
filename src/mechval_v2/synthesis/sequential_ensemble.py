"""Sequential Ensemble (S05)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Meta-protocol:  Consumes results from other protocols
Categories:     meta, ensemble, PARCEL Layer 2b
Validity layer: Internal
Establishes:    Filter-then-refine circuit discovery pipeline
Requires:       CPU for filtering, GPU for refinement phase
Source:         Mondorf et al. BlackboxNLP 2025 (sequential ensembling)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Phase 1: Run all cheap methods (EAP, Kyle lambda, Hawkes, cosine,
         graph topology — forward-pass only, no interventions) → rank
         components → keep top-P% candidates.
Phase 2: Run expensive methods (IIA, full patching) ONLY on filtered
         candidate set. Dramatically reduces forward passes.
Phase 3: Re-rank filtered candidates using expensive method scores.

Reports filtering compression ratio, candidate overlap with GT, and
final circuit quality.

Usage:
    uv run python sequential_ensemble.py --results-json modal_sweep_results.json --task ioi --filter-pct 0.2
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

PROTOCOL_ID = "S05"
PROTOCOL_NAME = "Sequential Ensemble (PARCEL L2b)"

GPT2_HEADS = [(layer, head) for layer in range(12) for head in range(12)]
N_HEADS = len(GPT2_HEADS)

CHEAP_PROTOCOLS = {
    "WC_M3", "WC_M9", "WC_M11", "WC_M13",  # statistical wildcards
    "WC_M5", "WC_M6", "WC_M8", "WC_M10",    # spectral wildcards
    "B01", "B02", "B03", "B04",               # structural
}

EXPENSIVE_PROTOCOLS = {
    "A01", "A02", "A03", "A04", "A05", "A06",  # causal
    "MB_KH", "MB_RE", "MB_TE",                  # biology causal
}


def _parse_head_key(key: str) -> tuple[int, int] | None:
    try:
        if key.startswith("L") and "H" in key:
            parts = key[1:].split("H")
            return int(parts[0]), int(parts[1])
    except (ValueError, IndexError):
        pass
    return None


def _extract_scores(protocol_results: list[dict], task: str,
                    protocol_filter: set | None = None) -> dict[str, np.ndarray]:
    head_to_idx = {h: i for i, h in enumerate(GPT2_HEADS)}
    vectors = {}

    for result in protocol_results:
        if result.get("status") != "success":
            continue
        proto_id = result.get("protocol_id", "unknown")
        if protocol_filter and proto_id not in protocol_filter:
            continue

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
            vectors[proto_id] = scores

    return vectors


def run_sequential_ensemble(model=None, tasks: list[str] | None = None,
                            device: str = "cpu",
                            protocol_results: list[dict] | None = None,
                            filter_pct: float = 0.2) -> list[EvalResult]:
    if tasks is None:
        tasks = list(CIRCUIT_TASKS)
    if protocol_results is None:
        return []

    results = []
    for task in tasks:
        cheap_scores = _extract_scores(protocol_results, task, CHEAP_PROTOCOLS)
        expensive_scores = _extract_scores(protocol_results, task, EXPENSIVE_PROTOCOLS)

        if not cheap_scores:
            log(f"  {task}: no cheap protocol scores, skipping")
            continue

        log(f"  {task}: {len(cheap_scores)} cheap, {len(expensive_scores)} expensive protocols")

        cheap_matrix = np.stack(list(cheap_scores.values()))
        cheap_avg = cheap_matrix.mean(axis=0)

        n_keep = max(1, int(N_HEADS * filter_pct))
        candidate_indices = set(np.argsort(-cheap_avg)[:n_keep])
        candidate_heads = {GPT2_HEADS[i] for i in candidate_indices}

        gt_heads = get_circuit_heads(task)
        gt_set = set(gt_heads) if gt_heads else set()

        if gt_set:
            filter_recall = len(candidate_heads & gt_set) / len(gt_set)
        else:
            filter_recall = 0.0

        log(f"    Phase 1: filtered to {n_keep}/{N_HEADS} candidates "
            f"(recall of GT: {filter_recall:.3f})")

        if expensive_scores:
            exp_matrix = np.stack(list(expensive_scores.values()))
            exp_avg = exp_matrix.mean(axis=0)

            refined_scores = np.full(N_HEADS, -np.inf)
            for idx in candidate_indices:
                refined_scores[idx] = exp_avg[idx]

            final_top = set(np.argsort(-refined_scores)[:len(gt_set) if gt_set else n_keep])
            final_heads = {GPT2_HEADS[i] for i in final_top}

            if gt_set:
                intersection = final_heads & gt_set
                union = final_heads | gt_set
                jaccard = len(intersection) / len(union) if union else 0.0
                precision = len(intersection) / len(final_heads) if final_heads else 0.0
                recall = len(intersection) / len(gt_set)
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            else:
                jaccard = precision = recall = f1 = 0.0

            log(f"    Phase 2+3: refined to {len(final_heads)} heads "
                f"(F1={f1:.3f} P={precision:.3f} R={recall:.3f})")

            results.append(EvalResult(
                metric_id="S05.sequential_f1",
                value=f1,
                n_samples=len(cheap_scores) + len(expensive_scores),
                metadata={
                    "task": task,
                    "filter_pct": filter_pct,
                    "n_candidates": n_keep,
                    "filter_recall": filter_recall,
                    "n_cheap": len(cheap_scores),
                    "n_expensive": len(expensive_scores),
                    "jaccard": jaccard,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "final_heads": sorted(f"L{h[0]}H{h[1]}" for h in final_heads),
                    "compression_ratio": N_HEADS / n_keep,
                },
            ))
        else:
            log(f"    No expensive protocols available for refinement")

        results.append(EvalResult(
            metric_id="S05.filter_recall",
            value=filter_recall,
            n_samples=n_keep,
            metadata={
                "task": task,
                "filter_pct": filter_pct,
                "n_candidates": n_keep,
                "n_cheap_protocols": len(cheap_scores),
                "cheap_protocols": list(cheap_scores.keys()),
            },
        ))

    return results


def run_protocol(model, tasks, n_prompts=40, device="cpu", run_cals=False,
                 protocol_results=None):
    from protocols import ProtocolResult
    t0 = time.time()
    evals = run_sequential_ensemble(model, tasks, device=device,
                                    protocol_results=protocol_results)
    result = ProtocolResult(
        protocol_id=PROTOCOL_ID,
        protocol_name=PROTOCOL_NAME,
        tasks=tasks,
        elapsed_seconds=time.time() - t0,
    )
    result.metrics["sequential"] = evals
    return result


def summarize(result) -> str:
    lines = [f"=== {PROTOCOL_NAME} ==="]
    for ev in result.metrics.get("sequential", []):
        lines.append(f"  {ev.metric_id}: {ev.value:.4f}")
    return "\n".join(lines)


def main():
    parser = parse_common_args("S05: Sequential Ensemble")
    parser.add_argument("--results-json", type=str, required=True)
    parser.add_argument("--filter-pct", type=float, default=0.2)
    args = parser.parse_args()
    tasks = args.tasks or CIRCUIT_TASKS
    with open(args.results_json) as f:
        protocol_results = json.load(f)

    log("=" * 60)
    log("S05: SEQUENTIAL ENSEMBLE (PARCEL L2b)")
    log("=" * 60)

    results = run_sequential_ensemble(tasks=tasks, protocol_results=protocol_results,
                                       filter_pct=args.filter_pct)
    out = args.out or "meta_p5_sequential_ensemble.json"
    save_results(results, out)
    log(f"\nDone. {len(results)} results.")


if __name__ == "__main__":
    main()
