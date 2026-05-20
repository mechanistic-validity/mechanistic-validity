"""Phylogenetic Tracking: Circuit Formation Across Layers
Tests how circuit function accumulates across transformer layers,
treating depth as a developmental timeline. Measures cumulative
logit attribution from circuit heads layer by layer and fits a
sigmoid to the formation curve.

Pass: sigmoid R^2 > 0.8
Ref: Haeckel 1866, Generelle Morphologie der Organismen

Usage:
    uv run python GN5_phylogenetic_tracking.py --tasks ioi --n-prompts 40
"""

import numpy as np
import torch

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    InstrumentInfo,
    calibrate_mean_z,
    generate_prompts,
    get_circuit_heads,
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

INSTRUMENT_INFO = InstrumentInfo(
    name="Phylogenetic Tracking",
    paper_ref="Haeckel 1866",
    paper_cite="Haeckel 1866, Generelle Morphologie der Organismen",
    description="Tracks circuit formation across layers as a developmental timeline; fits sigmoid to cumulative attribution",
    category="cross_discipline",
    tier="cross_discipline",
    origin="established",
)

R2_THRESHOLD = 0.8


def _sigmoid(x: np.ndarray, x0: float, k: float) -> np.ndarray:
    """Standard sigmoid: 1 / (1 + exp(-k*(x - x0)))."""
    return 1.0 / (1.0 + np.exp(-k * (x - x0)))


def _fit_sigmoid(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float, float]:
    """Fit sigmoid to data via grid search. Returns (x0, k, r2)."""
    best_r2 = -np.inf
    best_x0, best_k = float(np.mean(xs)), 1.0
    ss_tot = float(np.sum((ys - np.mean(ys)) ** 2))
    if ss_tot < 1e-12:
        return best_x0, best_k, 1.0

    for x0 in np.linspace(float(xs[0]), float(xs[-1]), 20):
        for k in np.logspace(-1, 1.5, 20):
            pred = _sigmoid(xs, x0, k)
            ss_res = float(np.sum((ys - pred) ** 2))
            r2 = 1.0 - ss_res / ss_tot
            if r2 > best_r2:
                best_r2 = r2
                best_x0, best_k = x0, k

    return float(best_x0), float(best_k), float(best_r2)


@torch.no_grad()
def run_phylogenetic_tracking(model, tasks: list[str],
                              n_prompts: int = 40) -> list[EvalResult]:
    """Track cumulative logit attribution across layers for circuit heads."""
    tokenizer = model.tokenizer
    results = []
    n_layers = model.cfg.n_layers
    n_heads_total = model.cfg.n_heads

    for task in tasks:
        circuit_heads = get_circuit_heads(task)
        if not circuit_heads:
            log(f"  {task}: no circuit heads, skipping")
            continue

        prompts = generate_prompts(task, tokenizer, n_prompts)
        if not prompts:
            continue

        correct_ids, incorrect_ids = get_token_ids(prompts, tokenizer)
        if not correct_ids:
            continue

        n_valid = min(len(prompts), len(correct_ids))
        log(f"  {task}: {len(circuit_heads)} heads across {n_layers} layers, {n_valid} prompts")

        mean_z = calibrate_mean_z(model, prompts, n_calibration=min(100, n_valid))

        # Compute clean baseline LD
        clean_ld_sum = 0.0
        for idx in range(n_valid):
            tokens = model.to_tokens(prompts[idx].text)
            logits = model(tokens)
            clean_ld_sum += logit_diff_from_logits(logits, correct_ids[idx], incorrect_ids[idx])
        mean_clean_ld = clean_ld_sum / n_valid

        if abs(mean_clean_ld) < 1e-8:
            log(f"    baseline LD ~ 0, skipping")
            continue

        # For each layer cutoff, ablate all circuit heads ABOVE that layer
        # and measure how much of the circuit function has been accumulated
        cumulative_curve = []

        for cutoff_layer in range(n_layers):
            # Ablate circuit heads in layers > cutoff_layer
            heads_to_ablate = {(L, H) for L, H in circuit_heads if L > cutoff_layer}
            if not heads_to_ablate:
                # All circuit heads are at or below cutoff -> full function
                cumulative_curve.append(1.0)
                continue

            hooks = make_ablation_hook(heads_to_layer_dict(heads_to_ablate), mean_z, "mean")
            ld_sum = 0.0
            for idx in range(n_valid):
                tokens = model.to_tokens(prompts[idx].text)
                logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
                ld_sum += logit_diff_from_logits(logits, correct_ids[idx], incorrect_ids[idx])

            fraction = (ld_sum / n_valid) / mean_clean_ld
            cumulative_curve.append(max(0.0, min(1.0, fraction)))

        xs = np.arange(n_layers, dtype=float)
        ys = np.array(cumulative_curve)

        # Fit sigmoid
        x0, k, r2 = _fit_sigmoid(xs, ys)

        # Formation midpoint: layer at which 50% of attribution is reached
        formation_midpoint = float(x0)
        passed = r2 > R2_THRESHOLD

        log(f"    formation_midpoint=L{formation_midpoint:.1f}  "
            f"sigmoid_R2={r2:.3f}  [{'PASS' if passed else 'FAIL'}]")

        results.append(EvalResult(
            metric_id="GN5.phylogenetic_tracking",
            value=r2,
            n_samples=len(prompts),
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": task,
                "n_heads": len(circuit_heads),
                "n_layers": n_layers,
                "cumulative_curve": cumulative_curve,
                "formation_midpoint": formation_midpoint,
                "sigmoid_k": k,
                "sigmoid_r2": r2,
                "heads_by_layer": {
                    L: [(l, h) for l, h in sorted(circuit_heads) if l == L]
                    for L in sorted({l for l, h in circuit_heads})
                },
                "passed": passed,
                "threshold": R2_THRESHOLD,
            },
        ))

    return results


def main():
    parser = parse_common_args("GN5: Phylogenetic Tracking")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("GN5: PHYLOGENETIC TRACKING")
    log("=" * 60)

    out = args.out or "GN5_phylogenetic_tracking.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_phylogenetic_tracking(model, [task], args.n_prompts)
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)

    save_results(results, out, args=args)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
