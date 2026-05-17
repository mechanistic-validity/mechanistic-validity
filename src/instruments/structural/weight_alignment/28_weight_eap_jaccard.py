"""Weight-EAP Head Jaccard (Metric #66)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     B04 — Weight Alignment
Categories:     structural
Validity layer: Construct
Criteria:       C2/C5 Convergent validity
Establishes:    Weight-derived circuits converge with activation-derived circuits
Requires:       CPU, model weights only
Doc:            /instruments_v2/structural/b04-weight-alignment
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For each task, compute Jaccard similarity between our weight-classifier-
derived circuit heads and EAP-derived circuit heads. Loads EAP heads
from available data files in data/ and unexplored_pillars/data/.

Usage:
    uv run python 28_weight_eap_jaccard.py --tasks ioi sva
    uv run python 28_weight_eap_jaccard.py --device cpu
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    CIRCUIT_TASKS,
    EvalResult,
    get_circuit_heads,
    log,
    parse_common_args,
    save_results,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PILLAR_DATA = SCRIPT_DIR.parents[1] / "unexplored_pillars" / "data"

# Directories that may contain EAP head data
EAP_DATA_DIRS = [
    SCRIPT_DIR / "data",
    PILLAR_DATA,
    SCRIPT_DIR.parents[4] / "part5_validation_misc" / "data",
    SCRIPT_DIR.parents[4] / "part4_rigorous_circuit_finding" / "experiments"
    / "eap-ablation" / "data",
]


def _parse_head_key(key: str) -> tuple[int, int] | None:
    """Parse 'L5H3' format into (5, 3)."""
    try:
        if key.startswith("L") and "H" in key:
            parts = key[1:].split("H")
            return int(parts[0]), int(parts[1])
    except (ValueError, IndexError):
        pass
    return None


def _load_eap_heads(task: str) -> set[tuple[int, int]] | None:
    """Try to load EAP-derived heads for a task from available data files."""
    for data_dir in EAP_DATA_DIRS:
        if not data_dir.exists():
            continue

        # Look for EAP data files
        for pattern in [f"eap_{task}*.json", f"*eap*{task}*.json",
                        "eap_ablation.json", "attribution_patching.json"]:
            for path in data_dir.glob(pattern):
                try:
                    with open(path) as f:
                        data = json.load(f)
                except (json.JSONDecodeError, OSError):
                    continue

                heads = _extract_heads_from_eap_data(data, task)
                if heads:
                    log(f"    Loaded EAP heads from {path.name}")
                    return heads

    return None


def _extract_heads_from_eap_data(data: dict, task: str) -> set[tuple[int, int]] | None:
    """Extract head sets from various EAP data formats."""
    heads = set()

    # Format 1: eap_ablation.json with sets containing heads lists
    if "sets" in data:
        for set_name in ["eap_top15", "eap_ig_top15"]:
            if set_name in data["sets"]:
                head_strs = data["sets"][set_name].get("heads", [])
                for hs in head_strs:
                    parsed = _parse_head_key(hs)
                    if parsed:
                        heads.add(parsed)
                if heads:
                    return heads

    # Format 2: head_scores dict with per-head EAP scores
    if "head_scores" in data:
        head_scores = data["head_scores"]
        scored = []
        for hk, hv in head_scores.items():
            parsed = _parse_head_key(hk)
            if parsed is None:
                continue
            total = hv.get("total", 0) if isinstance(hv, dict) else float(hv)
            scored.append((parsed, abs(total)))

        scored.sort(key=lambda x: x[1], reverse=True)
        # Take top-k matching our circuit size (or top 15 by default)
        for h, _ in scored[:15]:
            heads.add(h)
        return heads if heads else None

    # Format 3: per-task dict
    if task in data:
        task_data = data[task]
        if isinstance(task_data, dict):
            for key in ["eap_heads", "heads", "top_heads"]:
                if key in task_data:
                    head_list = task_data[key]
                    for h in head_list:
                        if isinstance(h, list) and len(h) == 2:
                            heads.add(tuple(h))
                        elif isinstance(h, str):
                            parsed = _parse_head_key(h)
                            if parsed:
                                heads.add(parsed)
                    if heads:
                        return heads

    return None


def jaccard_similarity(set_a: set, set_b: set) -> float:
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def run_weight_eap_jaccard(tasks: list[str]) -> list[EvalResult]:
    results = []

    for task in tasks:
        weight_heads = get_circuit_heads(task)
        if not weight_heads:
            log(f"  {task}: no weight circuit, skipping")
            continue

        eap_heads = _load_eap_heads(task)
        if eap_heads is None:
            log(f"  {task}: no EAP head data found, skipping")
            continue

        jaccard = jaccard_similarity(weight_heads, eap_heads)
        shared = weight_heads & eap_heads
        only_weight = weight_heads - eap_heads
        only_eap = eap_heads - weight_heads

        log(f"  {task}: Jaccard={jaccard:.4f}  "
            f"shared={len(shared)}/{len(weight_heads | eap_heads)}  "
            f"weight_only={len(only_weight)}, eap_only={len(only_eap)}")

        results.append(EvalResult(
            metric_id="C28.weight_eap_jaccard",
            value=jaccard,
            n_samples=len(weight_heads | eap_heads),
            metadata={
                "task": task,
                "jaccard": jaccard,
                "n_shared": len(shared),
                "n_weight_heads": len(weight_heads),
                "n_eap_heads": len(eap_heads),
                "n_only_weight": len(only_weight),
                "n_only_eap": len(only_eap),
                "shared_heads": sorted(shared),
                "weight_heads": sorted(weight_heads),
                "eap_heads": sorted(eap_heads),
            },
        ))

    return results


def main():
    parser = parse_common_args("C28: Weight-EAP Head Jaccard")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS

    log("=" * 60)
    log("C28: WEIGHT-EAP HEAD JACCARD")
    log("=" * 60)

    results = run_weight_eap_jaccard(tasks)

    out = args.out or "28_weight_eap_jaccard.json"
    save_results(results, out)

    log(f"\nDone. {len(results)} tasks evaluated.")
    for r in results:
        t = r.metadata["task"]
        log(f"  {t}: Jaccard={r.value:.4f}  "
            f"({r.metadata['n_shared']}/{r.metadata['n_weight_heads']}+{r.metadata['n_eap_heads']})")


if __name__ == "__main__":
    main()
