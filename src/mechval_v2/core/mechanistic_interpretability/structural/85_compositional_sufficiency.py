"""Compositional Sufficiency (Graph Structure G4)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     G04 — Graph Structure
Categories:     structural
Validity layer: Internal
Criteria:       G4 Compositional Sufficiency
Establishes:    Whether the circuit's band composition is superadditive
Requires:       CPU or GPU, model
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Tests whether the circuit's graph structure (edges between bands)
carries real signal beyond the individual bands. Computes:

1. Full circuit faithfulness (keep all circuit heads, ablate rest)
2. Per-band faithfulness (keep one band at a time, ablate rest)
3. Superadditivity = full_circuit - max(individual_bands)

If bands compose meaningfully through their edges, the full circuit
should recover MORE than any single band. This is what distinguishes
G4 from A2: A2 tests whether the heads are sufficient, G4 tests
whether their inter-band wiring adds value.

Pass condition: superadditivity > 0.05 AND full_recovery > 0.2.

Usage:
    uv run python 85_compositional_sufficiency.py --tasks ioi --n-prompts 40
    uv run python 85_compositional_sufficiency.py --tasks ioi sva --device cpu
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
    heads_to_layer_dict,
    load_model,
    log,
    logit_diff_from_logits,
    make_ablation_hook,
    parse_common_args,
    save_incremental,
    save_results,
)


def get_band_heads(circuit: dict, band_name: str) -> set[tuple[int, int]]:
    roles = circuit["roles"]
    bands = circuit.get("bands", {})
    band_def = bands.get(band_name)
    if band_def is None:
        return set()
    if isinstance(band_def, (tuple, list)) and len(band_def) == 2:
        _layer_range, role_names = band_def
    else:
        role_names = band_def
    heads = set()
    for role_name in role_names:
        for head in roles.get(role_name, []):
            heads.add(tuple(head))
    return heads


@torch.no_grad()
def run_compositional_sufficiency(model, tasks: list[str],
                                  n_prompts: int = 40) -> list[EvalResult]:
    tokenizer = model.tokenizer
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
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

        bands = circuit.get("bands", {})
        if len(bands) < 2:
            log(f"  {task}: <2 bands, skipping")
            continue

        log(f"  {task}: {len(all_heads)} heads, {len(all_edges)} edges, "
            f"{len(bands)} bands, {len(prompts)} prompts")

        mean_z = calibrate_mean_z(model, prompts, n_calibration=min(100, len(prompts)))

        full_recovery = float(compute_faithfulness(
            model, prompts, correct_ids, incorrect_ids, all_heads, mean_z))
        log(f"    full_circuit: recovery={full_recovery:.3f}")

        band_scores = {}
        for band_name in bands:
            band_h = get_band_heads(circuit, band_name)
            if not band_h:
                continue
            band_faith = float(compute_faithfulness(
                model, prompts, correct_ids, incorrect_ids, band_h, mean_z))
            band_scores[band_name] = band_faith
            log(f"    band '{band_name}' ({len(band_h)} heads): recovery={band_faith:.3f}")

        max_band = max(band_scores.values()) if band_scores else 0.0
        superadditivity = full_recovery - max_band

        passed = bool(superadditivity > 0.05 and full_recovery > 0.2)

        log(f"    full={full_recovery:.3f}, max_band={max_band:.3f}, "
            f"superadditivity={superadditivity:+.3f}")
        log(f"    [{'PASS' if passed else 'FAIL'}]")

        results.append(EvalResult(
            metric_id="G4.compositional_sufficiency",
            value=float(superadditivity),
            n_samples=len(prompts),
            metadata={
                "task": task,
                "full_recovery": full_recovery,
                "per_band_recovery": {k: float(v) for k, v in band_scores.items()},
                "max_band_recovery": float(max_band),
                "superadditivity": float(superadditivity),
                "n_circuit_heads": len(all_heads),
                "n_circuit_edges": len(all_edges),
                "n_bands": len(bands),
                "passed": passed,
            },
        ))

    return results


def main():
    parser = parse_common_args("G4: Compositional Sufficiency")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("G4: COMPOSITIONAL SUFFICIENCY")
    log("=" * 60)

    out = args.out or "85_compositional_sufficiency.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_compositional_sufficiency(model, [task], args.n_prompts)
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)
            p = "PASS" if r.metadata["passed"] else "FAIL"
            log(f"  {task}: superadditivity={r.value:+.3f}  [{p}]")

    save_results(results, out)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
