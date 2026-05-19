"""Cross-Entropy Delta
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     D04 — CE Delta
Categories:     behavioral
Validity layer: Internal
Criteria:       I1 Necessity
Establishes:    Circuit accounts for majority of model loss sensitivity
Requires:       GPU, model
Doc:            /instruments_v2/behavioral/d04-ce-delta
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Measures the change in cross-entropy loss when ablating circuit vs complement.
For each prompt, computes CE loss of the full model, CE loss with the circuit
ablated, and CE loss with the complement ablated. The ratio of CE increases
tells how much of the model's loss sensitivity is attributable to the circuit.

Framework reference: Behavioral Pillar D04 -- loss-based faithfulness metric
that complements logit-diff by capturing full-distribution effects.

Usage:
    uv run python 43_ce_delta.py --tasks ioi sva
    uv run python 43_ce_delta.py --device cuda --n-prompts 60
"""

import numpy as np
import torch
import torch.nn.functional as F

from mechanistic_validity.metrics.common import (
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


@torch.no_grad()
def compute_ce_at_last(model, tokens, correct_id):
    """Compute cross-entropy loss at the last token position for the correct target."""
    logits = model(tokens)
    last_logits = logits[0, -1]
    log_probs = F.log_softmax(last_logits, dim=-1)
    return -log_probs[correct_id].item()


@torch.no_grad()
def compute_ce_with_hooks(model, tokens, correct_id, hooks):
    """Compute CE at last position with ablation hooks applied."""
    logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
    last_logits = logits[0, -1]
    log_probs = F.log_softmax(last_logits, dim=-1)
    return -log_probs[correct_id].item()


@torch.no_grad()
def run_ce_delta(model, tasks, n_prompts):
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
        circuit_hooks = make_ablation_hook(heads_to_layer_dict(circuit_heads), mean_z, "mean")
        complement_hooks = make_ablation_hook(heads_to_layer_dict(complement_heads), mean_z, "mean")

        ce_full_list = []
        ce_circuit_ablated_list = []
        ce_complement_ablated_list = []

        for i, p in enumerate(prompts):
            if i >= len(correct_ids):
                break
            tokens = model.to_tokens(p.text)
            cid = correct_ids[i]

            ce_full = compute_ce_at_last(model, tokens, cid)
            ce_circuit_ablated = compute_ce_with_hooks(model, tokens, cid, circuit_hooks)
            ce_complement_ablated = compute_ce_with_hooks(model, tokens, cid, complement_hooks)

            ce_full_list.append(ce_full)
            ce_circuit_ablated_list.append(ce_circuit_ablated)
            ce_complement_ablated_list.append(ce_complement_ablated)

        delta_circuit = float(np.mean(ce_circuit_ablated_list) - np.mean(ce_full_list))
        delta_complement = float(np.mean(ce_complement_ablated_list) - np.mean(ce_full_list))
        ratio = delta_circuit / delta_complement if abs(delta_complement) > 1e-8 else 0.0

        log(f"    delta_circuit={delta_circuit:.4f}, delta_complement={delta_complement:.4f}, ratio={ratio:.4f}")

        results.append(EvalResult(
            metric_id="D04.ce_delta",
            value=ratio,
            n_samples=len(ce_full_list),
            metadata={
                "task": task,
                "mean_ce_full": float(np.mean(ce_full_list)),
                "mean_ce_circuit_ablated": float(np.mean(ce_circuit_ablated_list)),
                "mean_ce_complement_ablated": float(np.mean(ce_complement_ablated_list)),
                "delta_circuit": delta_circuit,
                "delta_complement": delta_complement,
                "n_circuit_heads": len(circuit_heads),
                "n_complement_heads": len(complement_heads),
            },
        ))

    return results


def main():
    parser = parse_common_args("D04: Cross-Entropy Delta")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("D04: CROSS-ENTROPY DELTA")
    log("=" * 60)

    results = run_ce_delta(model, tasks, args.n_prompts)

    out = args.out or "43_ce_delta.json"
    save_results(results, out)

    log(f"\nDone. {len(results)} tasks evaluated.")
    for r in results:
        t = r.metadata["task"]
        log(f"  {t}: CE ratio={r.value:.4f} (circuit delta={r.metadata['delta_circuit']:.4f})")


if __name__ == "__main__":
    main()
