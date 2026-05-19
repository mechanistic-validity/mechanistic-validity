"""Edge Overlap / Jaccard (Metric #41)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     B06 — Template Distance
Categories:     structural
Validity layer: Construct
Criteria:       C3 Task specificity
Establishes:    Circuit edge structure is consistent across discovery methods
Requires:       CPU, data-only
Doc:            /instruments_v2/structural/b06-template-distance
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For each task, compute Jaccard similarity between our circuit's edges
(from roles.py pathways) and EAP-derived edges (from EAP graph data).
If edge data doesn't exist for a task, skip it.

Usage:
    uv run python 27_edge_jaccard.py --tasks ioi sva
    uv run python 27_edge_jaccard.py --device cpu
"""
import json
from pathlib import Path

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    get_circuit_info,
    log,
    parse_common_args,
    save_results,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PILLAR_DATA = SCRIPT_DIR.parents[1] / "unexplored_pillars" / "data"

# Directories that may contain EAP edge/graph data
EAP_DATA_DIRS = [
    SCRIPT_DIR / "data",
    PILLAR_DATA,
    SCRIPT_DIR.parents[4] / "part5_validation_misc" / "data",
    SCRIPT_DIR.parents[4] / "part4_rigorous_circuit_finding" / "experiments"
    / "eap-ablation" / "data",
]


def _load_eap_edges(task: str) -> set[tuple[int, int, int, int]] | None:
    """Try to load EAP-derived edges for a task from available data files.

    Looks for files like eap_{task}*.json that contain head_scores or graph data.
    Converts top EAP head pairs into edges (sender_layer < receiver_layer).
    """
    # Try to find EAP graph data
    for data_dir in EAP_DATA_DIRS:
        if not data_dir.exists():
            continue

        # Look for EAP graph files
        for pattern in [f"eap_{task}*graph*.json", f"*eap*{task}*.json",
                        f"eap_ablation.json"]:
            for path in data_dir.glob(pattern):
                try:
                    with open(path) as f:
                        data = json.load(f)
                except (json.JSONDecodeError, OSError):
                    continue

                edges = _extract_edges_from_eap_data(data, task)
                if edges:
                    log(f"    Loaded EAP edges from {path.name}")
                    return edges

    return None


def _parse_head_key(key: str) -> tuple[int, int] | None:
    """Parse 'L5H3' format into (5, 3)."""
    try:
        if key.startswith("L") and "H" in key:
            parts = key[1:].split("H")
            return int(parts[0]), int(parts[1])
    except (ValueError, IndexError):
        pass
    return None


def _extract_edges_from_eap_data(data: dict, task: str) -> set[tuple[int, int, int, int]] | None:
    """Extract edges from various EAP data formats."""
    edges = set()

    # Format 1: eap_ablation.json with sets containing heads
    if "sets" in data:
        eap_heads_list = []
        for set_name in ["eap_top15", "eap_ig_top15"]:
            if set_name in data["sets"]:
                head_strs = data["sets"][set_name].get("heads", [])
                for hs in head_strs:
                    parsed = _parse_head_key(hs)
                    if parsed:
                        eap_heads_list.append(parsed)

        if eap_heads_list:
            eap_heads_list.sort()
            for i, sender in enumerate(eap_heads_list):
                for receiver in eap_heads_list[i + 1:]:
                    if sender[0] < receiver[0]:
                        edges.add((sender[0], sender[1], receiver[0], receiver[1]))
            return edges if edges else None

    # Format 2: head_scores with per-head EAP scores
    if "head_scores" in data:
        head_scores = data["head_scores"]
        scored = []
        for hk, hv in head_scores.items():
            parsed = _parse_head_key(hk)
            if parsed is None:
                continue
            total = hv.get("total", 0)
            scored.append((parsed, abs(total)))

        scored.sort(key=lambda x: x[1], reverse=True)
        top_heads = [h for h, _ in scored[:15]]
        top_heads.sort()

        for i, sender in enumerate(top_heads):
            for receiver in top_heads[i + 1:]:
                if sender[0] < receiver[0]:
                    edges.add((sender[0], sender[1], receiver[0], receiver[1]))

        return edges if edges else None

    return None


def jaccard_similarity(set_a: set, set_b: set) -> float:
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def run_edge_jaccard(tasks: list[str]) -> list[EvalResult]:
    results = []

    for task in tasks:
        circuit, heads, our_edges = get_circuit_info(task)
        if not our_edges:
            log(f"  {task}: no circuit edges, skipping")
            continue

        eap_edges = _load_eap_edges(task)
        if eap_edges is None:
            log(f"  {task}: no EAP edge data found, skipping")
            continue

        jaccard = jaccard_similarity(our_edges, eap_edges)
        shared = our_edges & eap_edges
        only_ours = our_edges - eap_edges
        only_eap = eap_edges - our_edges

        log(f"  {task}: Jaccard={jaccard:.4f}  "
            f"shared={len(shared)}, ours_only={len(only_ours)}, "
            f"eap_only={len(only_eap)}")

        results.append(EvalResult(
            metric_id="C27.edge_jaccard",
            value=jaccard,
            n_samples=len(our_edges | eap_edges),
            metadata={
                "task": task,
                "jaccard": jaccard,
                "n_shared_edges": len(shared),
                "n_our_edges": len(our_edges),
                "n_eap_edges": len(eap_edges),
                "n_only_ours": len(only_ours),
                "n_only_eap": len(only_eap),
                "shared_edges": sorted(shared),
            },
        ))

    return results


def main():
    parser = parse_common_args("C27: Edge Overlap Jaccard")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS

    log("=" * 60)
    log("C27: EDGE OVERLAP (JACCARD)")
    log("=" * 60)

    results = run_edge_jaccard(tasks)

    out = args.out or "27_edge_jaccard.json"
    save_results(results, out)

    log(f"\nDone. {len(results)} tasks evaluated.")
    for r in results:
        t = r.metadata["task"]
        log(f"  {t}: Jaccard={r.value:.4f}")


if __name__ == "__main__":
    main()
