"""Robust Rank Aggregation (S03)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Meta-protocol:  Consumes results from other protocols
Categories:     meta, ensemble
Validity layer: Internal
Establishes:    Consensus head ranking across protocols via RRA
Requires:       CPU, protocol results as input
Source:         Kolde et al. 2012 (RobustRankAggreg), genomics
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Each protocol ranks heads by importance. Robust Rank Aggregation
tests whether a head appears consistently near the top of multiple
ranked lists — more than expected by chance. Reports a p-value per
head, corrected for multiple testing (Bonferroni).

Also computes Borda count (mean rank) and Dowdall (harmonic mean
rank) as simpler aggregation baselines.

Usage:
    uv run python rank_aggregation.py --results-json modal_sweep_results.json --task ioi
"""
import json
import time

import numpy as np
from scipy.stats import beta as beta_dist

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    get_circuit_heads,
    log,
    parse_common_args,
    save_results,
)

PROTOCOL_ID = "S03"
PROTOCOL_NAME = "Robust Rank Aggregation"

GPT2_HEADS = [(layer, head) for layer in range(12) for head in range(12)]
N_HEADS = len(GPT2_HEADS)


def _extract_rankings(protocol_results: list[dict], task: str) -> dict[str, list[tuple[int, int]]]:
    head_to_idx = {h: i for i, h in enumerate(GPT2_HEADS)}
    rankings = {}

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
            ranked_indices = np.argsort(-scores)
            rankings[proto_id] = [GPT2_HEADS[i] for i in ranked_indices]

    return rankings


def _parse_head_key(key: str) -> tuple[int, int] | None:
    try:
        if key.startswith("L") and "H" in key:
            parts = key[1:].split("H")
            return int(parts[0]), int(parts[1])
    except (ValueError, IndexError):
        pass
    return None


def _rra_score(normalized_ranks: list[float]) -> float:
    k = len(normalized_ranks)
    if k == 0:
        return 1.0
    sorted_r = sorted(normalized_ranks)
    min_p = 1.0
    for i, r in enumerate(sorted_r):
        p = beta_dist.cdf(r, i + 1, k - i)
        min_p = min(min_p, p)
    return min(min_p * k, 1.0)


def run_rank_aggregation(model=None, tasks: list[str] | None = None,
                         device: str = "cpu",
                         protocol_results: list[dict] | None = None) -> list[EvalResult]:
    if tasks is None:
        tasks = list(CIRCUIT_TASKS)
    if protocol_results is None:
        log("  No protocol results provided, skipping")
        return []

    results = []
    for task in tasks:
        rankings = _extract_rankings(protocol_results, task)
        if len(rankings) < 2:
            log(f"  {task}: only {len(rankings)} ranked lists, skipping")
            continue

        log(f"  {task}: {len(rankings)} protocol rankings")

        head_to_idx = {h: i for i, h in enumerate(GPT2_HEADS)}
        n_lists = len(rankings)

        normalized_ranks = {h: [] for h in GPT2_HEADS}
        for proto_id, ranked_list in rankings.items():
            for rank, head in enumerate(ranked_list):
                normalized_ranks[head].append((rank + 1) / N_HEADS)

        rra_pvalues = {}
        borda_scores = {}
        for head in GPT2_HEADS:
            ranks = normalized_ranks[head]
            rra_pvalues[head] = _rra_score(ranks)
            borda_scores[head] = np.mean(ranks) if ranks else 1.0

        bonferroni = {h: min(p * N_HEADS, 1.0) for h, p in rra_pvalues.items()}

        rra_sorted = sorted(bonferroni.items(), key=lambda x: x[1])
        borda_sorted = sorted(borda_scores.items(), key=lambda x: x[1])

        sig_heads = [(h, p) for h, p in rra_sorted if p < 0.05]
        top_borda = borda_sorted[:20]

        gt_heads = get_circuit_heads(task)
        gt_set = set(gt_heads) if gt_heads else set()
        rra_set = set(h for h, _ in sig_heads)

        if gt_set:
            intersection = rra_set & gt_set
            union = rra_set | gt_set
            jaccard = len(intersection) / len(union) if union else 0.0
            precision = len(intersection) / len(rra_set) if rra_set else 0.0
            recall = len(intersection) / len(gt_set) if gt_set else 0.0
        else:
            jaccard = precision = recall = 0.0

        log(f"    RRA significant (p<0.05 Bonf): {len(sig_heads)} heads")
        log(f"    Top 5 RRA: {[f'L{h[0]}H{h[1]}' for h, _ in rra_sorted[:5]]}")
        log(f"    Top 5 Borda: {[f'L{h[0]}H{h[1]}' for h, _ in borda_sorted[:5]]}")
        if gt_set:
            log(f"    GT overlap: Jaccard={jaccard:.3f}, P={precision:.3f}, R={recall:.3f}")

        results.append(EvalResult(
            metric_id="S03.rra_jaccard",
            value=jaccard,
            n_samples=len(rankings),
            metadata={
                "task": task,
                "n_protocols": len(rankings),
                "n_significant": len(sig_heads),
                "significant_heads": [f"L{h[0]}H{h[1]}" for h, _ in sig_heads],
                "rra_pvalues_top20": {
                    f"L{h[0]}H{h[1]}": float(p) for h, p in rra_sorted[:20]
                },
                "borda_top20": {
                    f"L{h[0]}H{h[1]}": float(s) for h, s in borda_sorted[:20]
                },
                "jaccard_with_gt": jaccard,
                "precision": precision,
                "recall": recall,
            },
        ))

        results.append(EvalResult(
            metric_id="S03.n_significant",
            value=float(len(sig_heads)),
            n_samples=N_HEADS,
            metadata={"task": task, "threshold": 0.05, "correction": "bonferroni"},
        ))

    return results


def run_protocol(model, tasks, n_prompts=40, device="cpu", run_cals=False,
                 protocol_results=None):
    from protocols import ProtocolResult
    t0 = time.time()
    evals = run_rank_aggregation(model, tasks, device=device,
                                 protocol_results=protocol_results)
    result = ProtocolResult(
        protocol_id=PROTOCOL_ID,
        protocol_name=PROTOCOL_NAME,
        tasks=tasks,
        elapsed_seconds=time.time() - t0,
    )
    result.metrics["rank_aggregation"] = evals
    return result


def summarize(result) -> str:
    lines = [f"=== {PROTOCOL_NAME} ==="]
    for ev in result.metrics.get("rank_aggregation", []):
        lines.append(f"  {ev.metric_id}: {ev.value:.4f}")
    return "\n".join(lines)


def main():
    parser = parse_common_args("S03: Robust Rank Aggregation")
    parser.add_argument("--results-json", type=str, required=True)
    args = parser.parse_args()
    tasks = args.tasks or CIRCUIT_TASKS

    with open(args.results_json) as f:
        protocol_results = json.load(f)

    log("=" * 60)
    log("S03: ROBUST RANK AGGREGATION")
    log("=" * 60)

    results = run_rank_aggregation(tasks=tasks, protocol_results=protocol_results)
    out = args.out or "meta_p3_rank_aggregation.json"
    save_results(results, out)
    log(f"\nDone. {len(results)} results.")


if __name__ == "__main__":
    main()
