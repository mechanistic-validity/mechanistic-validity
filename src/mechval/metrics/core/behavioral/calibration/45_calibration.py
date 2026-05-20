"""Calibration Analysis
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     D06 — Calibration
Categories:     behavioral
Validity layer: Measurement
Criteria:       M5 Calibration
Establishes:    Circuit preserves probability calibration, not just accuracy
Requires:       GPU, model
Doc:            /instruments_v2/behavioral/d06-calibration
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Tests whether the circuit's probability outputs are well-calibrated.
Bins the model's predicted probabilities for the correct answer into
decile bins (0-0.1, 0.1-0.2, ..., 0.9-1.0). In each bin, measures
the actual accuracy. Computes Expected Calibration Error (ECE).

Compares ECE of full model vs circuit-only (complement ablated).
A faithful circuit should preserve calibration, not just accuracy.

Framework reference: Behavioral Pillar D06 -- calibration preservation
as a stronger test of distributional faithfulness.

Usage:
    uv run python 45_calibration.py --tasks ioi sva
    uv run python 45_calibration.py --device cuda --n-prompts 80
"""

import numpy as np
import torch
import torch.nn.functional as F

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    calibrate_mean_z,
    generate_prompts,
    get_circuit_heads,
    get_token_ids,
    heads_to_layer_dict,
    load_model,
    log,
    make_ablation_hook,
    parse_common_args,
    save_results,
)

N_BINS = 10


def compute_ece(predicted_probs, correct_flags):
    """Compute Expected Calibration Error over N_BINS bins."""
    predicted_probs = np.array(predicted_probs)
    correct_flags = np.array(correct_flags)
    bin_edges = np.linspace(0.0, 1.0, N_BINS + 1)
    ece = 0.0
    bin_details = []

    for b in range(N_BINS):
        mask = (predicted_probs >= bin_edges[b]) & (predicted_probs < bin_edges[b + 1])
        if b == N_BINS - 1:
            mask = mask | (predicted_probs == bin_edges[b + 1])
        n_in_bin = mask.sum()
        if n_in_bin == 0:
            bin_details.append({"bin": b, "n": 0, "mean_conf": 0.0, "accuracy": 0.0})
            continue
        mean_conf = float(predicted_probs[mask].mean())
        accuracy = float(correct_flags[mask].mean())
        ece += n_in_bin * abs(accuracy - mean_conf)
        bin_details.append({"bin": b, "n": int(n_in_bin), "mean_conf": mean_conf, "accuracy": accuracy})

    total = len(predicted_probs)
    ece = ece / total if total > 0 else 0.0
    return float(ece), bin_details


@torch.no_grad()
def run_calibration(model, tasks, n_prompts):
    tokenizer = model.tokenizer
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    all_heads = {(L, H) for L in range(n_layers) for H in range(n_heads)}
    results = []

    for task in tasks:
        circuit_heads = get_circuit_heads(task)
        if not circuit_heads:
            log(f"  {task}: no circuit, skipping")
            continue

        prompts = generate_prompts(task, tokenizer, n_prompts)
        if not prompts:
            continue

        correct_ids, incorrect_ids = get_token_ids(prompts, tokenizer)
        if not correct_ids:
            continue

        log(f"  {task} ({len(circuit_heads)} heads, {len(prompts)} prompts)...")
        mean_z = calibrate_mean_z(model, prompts, n_calibration=min(100, len(prompts)))

        complement_heads = all_heads - circuit_heads
        complement_hooks = make_ablation_hook(heads_to_layer_dict(complement_heads), mean_z, "mean")

        full_probs = []
        full_correct = []
        circuit_probs = []
        circuit_correct = []

        for i, p in enumerate(prompts):
            if i >= len(correct_ids):
                break
            tokens = model.to_tokens(p.text)
            cid = correct_ids[i]
            iid = incorrect_ids[i]

            # Full model
            logits = model(tokens)
            probs = F.softmax(logits[0, -1], dim=-1)
            p_correct = probs[cid].item()
            is_correct = 1.0 if logits[0, -1].argmax().item() == cid else 0.0
            full_probs.append(p_correct)
            full_correct.append(is_correct)

            # Circuit-only (complement ablated)
            circuit_logits = model.run_with_hooks(tokens, fwd_hooks=complement_hooks)
            circuit_probs_dist = F.softmax(circuit_logits[0, -1], dim=-1)
            cp_correct = circuit_probs_dist[cid].item()
            cis_correct = 1.0 if circuit_logits[0, -1].argmax().item() == cid else 0.0
            circuit_probs.append(cp_correct)
            circuit_correct.append(cis_correct)

        ece_full, bins_full = compute_ece(full_probs, full_correct)
        ece_circuit, bins_circuit = compute_ece(circuit_probs, circuit_correct)
        ece_ratio = ece_circuit / ece_full if ece_full > 1e-8 else 0.0

        log(f"    ECE_full={ece_full:.4f}, ECE_circuit={ece_circuit:.4f}, ratio={ece_ratio:.3f}")
        log(f"    accuracy_full={np.mean(full_correct):.3f}, accuracy_circuit={np.mean(circuit_correct):.3f}")

        results.append(EvalResult(
            metric_id="D06.calibration_ece",
            value=ece_circuit,
            baseline_random=ece_full,
            n_samples=len(full_probs),
            metadata={
                "task": task,
                "ece_full": ece_full,
                "ece_circuit_only": ece_circuit,
                "ece_ratio": ece_ratio,
                "accuracy_full": float(np.mean(full_correct)),
                "accuracy_circuit_only": float(np.mean(circuit_correct)),
                "mean_prob_full": float(np.mean(full_probs)),
                "mean_prob_circuit": float(np.mean(circuit_probs)),
                "n_circuit_heads": len(circuit_heads),
                "bins_full": bins_full,
                "bins_circuit": bins_circuit,
            },
        ))

    return results


def main():
    parser = parse_common_args("D06: Calibration Analysis")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("D06: CALIBRATION ANALYSIS")
    log("=" * 60)

    results = run_calibration(model, tasks, args.n_prompts)

    out = args.out or "45_calibration.json"
    save_results(results, out)

    log(f"\nDone. {len(results)} tasks evaluated.")
    for r in results:
        t = r.metadata["task"]
        log(f"  {t}: ECE_full={r.metadata['ece_full']:.4f}, "
            f"ECE_circuit={r.value:.4f}, ratio={r.metadata['ece_ratio']:.3f}")


if __name__ == "__main__":
    main()
