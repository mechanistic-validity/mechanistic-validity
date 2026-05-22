"""Nomological Validity
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     F02 — Convergent Validity
Categories:     measurement
Validity layer: Measurement
Criteria:       C5 Convergent validity
Establishes:    Circuit structure correlates with expected layer-depth patterns
Requires:       CPU, data-only
Doc:            /instruments_v2/measurement/f02-convergent-validity
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For each task, compute Spearman correlation between layer depth (0-11)
and the proportion of circuit heads at that layer. Also compute
correlation between layer and the tier/role of heads (early=1,
mid=2, late=3). Reports r and p-value.

Usage:
    uv run python 23_nomological_validity.py --tasks ioi sva
    uv run python 23_nomological_validity.py --device cpu
"""

import numpy as np
from scipy import stats

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    get_circuit_heads,
    get_circuit_info,
    log,
    parse_common_args,
    save_results,
)

N_LAYERS = 12
N_HEADS = 12

# Map band names to ordinal tier values for correlation
BAND_ORDINALS = {
    "very_early": 1,
    "early": 1,
    "mid_early": 2,
    "early_mid": 2,
    "mid": 2,
    "midlate": 3,
    "mid_late": 3,
    "late": 4,
}


def _layer_density_correlation(heads: set[tuple[int, int]]) -> tuple[float, float]:
    """Spearman correlation between layer index and head count at that layer."""
    layer_counts = np.zeros(N_LAYERS)
    for layer, _ in heads:
        layer_counts[layer] += 1

    layer_indices = np.arange(N_LAYERS)
    r, p = stats.spearmanr(layer_indices, layer_counts)
    return float(r), float(p)


def _role_depth_correlation(circuit: dict) -> tuple[float, float, dict]:
    """Spearman correlation between head layer and its band ordinal."""
    roles = circuit["roles"]
    bands = circuit["bands"]

    # Map each role to its band ordinal
    role_to_ordinal = {}
    for band_name, (_, role_names) in bands.items():
        ordinal = BAND_ORDINALS.get(band_name, 2)
        for role_name in role_names:
            role_to_ordinal[role_name] = ordinal

    head_layers = []
    head_ordinals = []
    head_role_map = {}

    for role_name, role_heads in roles.items():
        ordinal = role_to_ordinal.get(role_name, 2)
        for layer, head in role_heads:
            head_layers.append(layer)
            head_ordinals.append(ordinal)
            head_role_map[f"L{layer}H{head}"] = {
                "role": role_name,
                "ordinal": ordinal,
            }

    if len(head_layers) < 3:
        return 0.0, 1.0, head_role_map

    r, p = stats.spearmanr(head_layers, head_ordinals)
    return float(r), float(p), head_role_map


def run_nomological_validity(model=None, tasks: list[str] | None = None, device: str = "cpu", n_prompts: int = 40) -> list[EvalResult]:
    if tasks is None:
        tasks = list(CIRCUIT_TASKS)
    results = []

    for task in tasks:
        circuit, heads, edges = get_circuit_info(task)
        if not heads:
            log(f"  {task}: no circuit, skipping")
            continue

        log(f"  {task} ({len(heads)} heads)...")

        # 1. Layer density correlation
        r_density, p_density = _layer_density_correlation(heads)

        layer_counts = {}
        for layer, _ in heads:
            layer_counts[layer] = layer_counts.get(layer, 0) + 1

        log(f"    density: r={r_density:.4f}, p={p_density:.4f}")

        results.append(EvalResult(
            metric_id="C23.layer_density_correlation",
            value=r_density,
            n_samples=N_LAYERS,
            metadata={
                "task": task,
                "spearman_r": r_density,
                "p_value": p_density,
                "layer_counts": layer_counts,
                "n_circuit_heads": len(heads),
            },
        ))

        # 2. Role-depth correlation
        if circuit is not None:
            r_role, p_role, role_map = _role_depth_correlation(circuit)
            n_roles = len(circuit["roles"])
            n_bands = len(circuit["bands"])

            log(f"    role-depth: r={r_role:.4f}, p={p_role:.4f} "
                f"({n_roles} roles, {n_bands} bands)")

            results.append(EvalResult(
                metric_id="C23.role_depth_correlation",
                value=r_role,
                n_samples=len(heads),
                metadata={
                    "task": task,
                    "spearman_r": r_role,
                    "p_value": p_role,
                    "n_roles": n_roles,
                    "n_bands": n_bands,
                    "head_role_map": role_map,
                },
            ))

    return results


def main():
    parser = parse_common_args("C23: Nomological Validity")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS

    log("=" * 60)
    log("C23: NOMOLOGICAL VALIDITY")
    log("=" * 60)

    results = run_nomological_validity(tasks)

    out = args.out or "23_nomological_validity.json"
    save_results(results, out)

    log(f"\nDone. {len(results)} results across {len(tasks)} tasks.")
    for r in results:
        t = r.metadata["task"]
        log(f"  {t}: {r.metric_id} r={r.value:.4f} p={r.metadata['p_value']:.4f}")


if __name__ == "__main__":
    main()
