"""Compositional Sufficiency (Graph Structure G4)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     G04 — Graph Structure
Categories:     structural
Validity layer: Internal
Criteria:       G4 Compositional Sufficiency
Establishes:    Whether the circuit subgraph reproduces the full computation when isolated
Requires:       CPU or GPU, model
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Ablate all heads except circuit heads and measure how much of the
original logit diff is recovered. This is faithfulness at the
graph/edge level: the circuit subgraph is sufficient if it recovers
a meaningful fraction of the full model's behavior.

Uses compute_faithfulness from _common.py.

Pass condition: recovery > 0.3 (30%).

Usage:
    uv run python 85_compositional_sufficiency.py --tasks ioi --n-prompts 40
    uv run python 85_compositional_sufficiency.py --tasks ioi sva --device cpu
"""
import sys
from pathlib import Path

import numpy as np
import torch

_INSTRUMENTS = Path(__file__).resolve().parents[2]  # up to src/instruments/
sys.path.insert(0, str(_INSTRUMENTS))
from _common import (
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
    save_results,
)


@torch.no_grad()
def compute_per_band_recovery(model, prompts, correct_ids, incorrect_ids,
                              circuit: dict, mean_z: torch.Tensor) -> dict[str, float]:
    """Compute faithfulness for each band (functional role group) in isolation."""
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    roles = circuit["roles"]
    bands = circuit.get("bands", {})

    band_scores = {}
    for band_name, band_def in bands.items():
        band_heads = set()
        # BANDS values are (layer_range, role_name_list) tuples
        if isinstance(band_def, (tuple, list)) and len(band_def) == 2:
            _layer_range, role_names = band_def
        else:
            role_names = band_def
        for role_name in role_names:
            for head in roles.get(role_name, []):
                band_heads.add(tuple(head))
        if not band_heads:
            continue

        non_band = {(L, H) for L in range(n_layers) for H in range(n_heads)} - band_heads
        non_band_by_layer = heads_to_layer_dict(non_band)
        hooks = make_ablation_hook(non_band_by_layer, mean_z, "mean")

        num, den = 0.0, 0.0
        for i, p in enumerate(prompts):
            if i >= len(correct_ids):
                break
            tokens = model.to_tokens(p.text)
            clean_logits = model(tokens)
            clean_ld = logit_diff_from_logits(clean_logits, correct_ids[i], incorrect_ids[i])
            ablated_logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
            ablated_ld = logit_diff_from_logits(ablated_logits, correct_ids[i], incorrect_ids[i])
            num += ablated_ld
            den += clean_ld

        band_scores[band_name] = num / den if abs(den) > 1e-8 else 0.0

    return band_scores


@torch.no_grad()
def run_compositional_sufficiency(model, tasks: list[str],
                                  n_prompts: int = 40) -> list[EvalResult]:
    tokenizer = model.tokenizer
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

        log(f"  {task}: {len(all_heads)} heads, {len(all_edges)} edges, "
            f"{len(prompts)} prompts")

        mean_z = calibrate_mean_z(model, prompts, n_calibration=min(100, len(prompts)))

        # Full circuit faithfulness
        recovery = compute_faithfulness(
            model, prompts, correct_ids, incorrect_ids, all_heads, mean_z)
        log(f"    full_circuit_recovery={recovery:.3f}")

        # Per-band recovery
        band_scores = compute_per_band_recovery(
            model, prompts, correct_ids, incorrect_ids, circuit, mean_z)
        for band_name, score in band_scores.items():
            log(f"    band '{band_name}': recovery={score:.3f}")

        recovery = float(recovery)
        passed = bool(recovery > 0.3)

        log(f"    [{'PASS' if passed else 'FAIL'}]  recovery={recovery:.3f}  threshold=0.30")

        # Ensure per_band_recovery values are plain floats
        clean_band_scores = {k: float(v) for k, v in band_scores.items()}

        results.append(EvalResult(
            metric_id="G4.compositional_sufficiency",
            value=recovery,
            n_samples=len(prompts),
            metadata={
                "task": task,
                "recovery": recovery,
                "n_circuit_heads": len(all_heads),
                "n_circuit_edges": len(all_edges),
                "per_band_recovery": clean_band_scores,
                "passed": passed,
                "threshold": 0.3,
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

    results = run_compositional_sufficiency(model, tasks, args.n_prompts)

    out = args.out or "85_compositional_sufficiency.json"
    save_results(results, out)

    log(f"\nDone. {len(results)} tasks evaluated.")
    for r in results:
        t = r.metadata["task"]
        p = "PASS" if r.metadata["passed"] else "FAIL"
        log(f"  {t}: recovery={r.value:.3f}  [{p}]")


if __name__ == "__main__":
    main()
