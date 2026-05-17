"""MDL Compression Ratio
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     D08 — MDL Compression
Categories:     behavioral
Validity layer: Construct
Criteria:       C4 Minimality
Establishes:    Circuit achieves high faithfulness-to-size efficiency
Requires:       GPU, model
Doc:            /instruments_v2/behavioral/d08-mdl-compression
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Minimum Description Length perspective on circuit quality. Measures:
1. Structural compression ratio: n_circuit_heads / n_total_heads
2. Efficiency: faithfulness / compression_ratio
3. Coding cost: KL(full_model || circuit_only) as the description length
   of the full model's output given the circuit's output.

A good circuit achieves high faithfulness with few components (high
efficiency) and low KL coding cost.

Framework reference: Behavioral Pillar D08 -- information-theoretic
compression quality metric for circuit evaluation.

Usage:
    uv run python 47_mdl_compression.py --tasks ioi sva
    uv run python 47_mdl_compression.py --device cuda --n-prompts 40
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "causal"))
from _common import (
    CIRCUIT_TASKS,
    EvalResult,
    calibrate_mean_z,
    compute_faithfulness,
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

GPT2_TOTAL_HEADS = 144  # 12 layers x 12 heads


@torch.no_grad()
def compute_kl_coding_cost(model, prompts, correct_ids, circuit_heads, mean_z):
    """Compute mean KL(full || circuit_only) over prompts as coding cost."""
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    complement = {(L, H) for L in range(n_layers) for H in range(n_heads)} - circuit_heads
    complement_hooks = make_ablation_hook(heads_to_layer_dict(complement), mean_z, "mean")

    kl_values = []
    for i, p in enumerate(prompts):
        if i >= len(correct_ids):
            break
        tokens = model.to_tokens(p.text)

        full_logits = model(tokens)
        circuit_logits = model.run_with_hooks(tokens, fwd_hooks=complement_hooks)

        # KL(full || circuit) at last position
        full_log_probs = F.log_softmax(full_logits[0, -1], dim=-1)
        circuit_log_probs = F.log_softmax(circuit_logits[0, -1], dim=-1)
        full_probs = full_log_probs.exp()

        kl = (full_probs * (full_log_probs - circuit_log_probs)).sum()
        kl_values.append(max(0.0, kl.item()))

    return float(np.mean(kl_values)) if kl_values else 0.0


@torch.no_grad()
def run_mdl_compression(model, tasks, n_prompts):
    tokenizer = model.tokenizer
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

        log(f"  {task} ({len(circuit_heads)} heads)...")
        mean_z = calibrate_mean_z(model, prompts, n_calibration=min(100, len(prompts)))

        # Structural compression
        n_circuit = len(circuit_heads)
        compression_ratio = n_circuit / GPT2_TOTAL_HEADS

        # Faithfulness
        faithfulness = compute_faithfulness(
            model, prompts, correct_ids, incorrect_ids, circuit_heads, mean_z,
        )

        # Efficiency = faithfulness / compression_ratio
        efficiency = faithfulness / compression_ratio if compression_ratio > 1e-8 else 0.0

        # KL coding cost
        kl_cost = compute_kl_coding_cost(model, prompts, correct_ids, circuit_heads, mean_z)

        # Normalized efficiency: bits saved per component
        bits_per_component = kl_cost / n_circuit if n_circuit > 0 else 0.0

        log(f"    compression={compression_ratio:.4f}, faithfulness={faithfulness:.4f}, "
            f"efficiency={efficiency:.3f}, KL_cost={kl_cost:.4f}")

        results.append(EvalResult(
            metric_id="D08.mdl_compression",
            value=efficiency,
            n_samples=len(prompts),
            metadata={
                "task": task,
                "n_circuit_heads": n_circuit,
                "n_total_heads": GPT2_TOTAL_HEADS,
                "compression_ratio": compression_ratio,
                "faithfulness": faithfulness,
                "efficiency": efficiency,
                "kl_coding_cost": kl_cost,
                "bits_per_component": bits_per_component,
            },
        ))

    return results


def main():
    parser = parse_common_args("D08: MDL Compression Ratio")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("D08: MDL COMPRESSION RATIO")
    log("=" * 60)

    results = run_mdl_compression(model, tasks, args.n_prompts)

    out = args.out or "47_mdl_compression.json"
    save_results(results, out)

    log(f"\nDone. {len(results)} tasks evaluated.")
    for r in results:
        t = r.metadata["task"]
        log(f"  {t}: efficiency={r.value:.3f} (compress={r.metadata['compression_ratio']:.3f}, "
            f"faith={r.metadata['faithfulness']:.3f}, KL={r.metadata['kl_coding_cost']:.4f})")


if __name__ == "__main__":
    main()
