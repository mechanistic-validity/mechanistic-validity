"""Misalignment Score (Metric #75)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     A04 — Woodward Interventionism
Categories:     causal
Validity layer: Internal
Criteria:       I1 Necessity
Establishes:    Noising-denoising misalignment reveals heads with asymmetric necessity vs sufficiency
Requires:       GPU, model
Doc:            /instruments_v2/causal/a04-woodward
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Detect misalignment between noising and denoising directions. For each
circuit head:
  (1) Noising effect (necessity) = drop in logit diff when this head is
      ablated in an otherwise-clean model.
  (2) Denoising effect (sufficiency) = gain in logit diff when this head
      is restored from clean in an otherwise-corrupted model.

Misalignment = cases where necessity is high but sufficiency is low
(or vice versa). Report per-head misalignment and aggregate score.

Usage:
    uv run python 37_misalignment_score.py --tasks ioi sva --n-prompts 40
    uv run python 37_misalignment_score.py --device cuda
"""

import numpy as np
import torch

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    calibrate_mean_z,
    generate_prompts,
    get_circuit_heads,
    get_token_ids,
    load_model,
    log,
    logit_diff_from_logits,
    make_ablation_hook,
    parse_common_args,
    save_results,
)


@torch.no_grad()
def compute_noising_effect(model, tokens, correct_id: int, incorrect_id: int,
                           layer: int, head: int, mean_z: torch.Tensor) -> float:
    """Necessity: drop in LD when ablating this head in a clean model.

    Returns normalized effect in [0, 1] (fraction of clean LD lost).
    """
    clean_logits = model(tokens)
    clean_ld = logit_diff_from_logits(clean_logits, correct_id, incorrect_id)

    hooks = make_ablation_hook({layer: [head]}, mean_z, "mean")
    ablated_logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
    ablated_ld = logit_diff_from_logits(ablated_logits, correct_id, incorrect_id)

    if abs(clean_ld) < 1e-8:
        return 0.0
    return (clean_ld - ablated_ld) / clean_ld


@torch.no_grad()
def compute_denoising_effect(model, tokens, correct_id: int, incorrect_id: int,
                             layer: int, head: int,
                             mean_z: torch.Tensor,
                             clean_z_cache: dict[int, torch.Tensor],
                             n_layers: int) -> float:
    """Sufficiency: gain in LD when restoring this head in a corrupted model.

    Start with all heads mean-ablated, restore just this one head from clean.
    Returns normalized effect in [0, 1] (fraction of gap recovered).
    """
    # Clean LD for normalization
    clean_logits = model(tokens)
    clean_ld = logit_diff_from_logits(clean_logits, correct_id, incorrect_id)

    # Fully corrupted LD
    corrupt_hooks = []
    for lay in range(n_layers):
        def _corrupt_hook(z, hook, _layer=lay):
            for h in range(z.shape[2]):
                z[0, :, h, :] = mean_z[_layer, h].to(z.device)
            return z
        corrupt_hooks.append((f"blocks.{lay}.attn.hook_z", _corrupt_hook))

    corrupt_logits = model.run_with_hooks(tokens, fwd_hooks=corrupt_hooks)
    corrupt_ld = logit_diff_from_logits(corrupt_logits, correct_id, incorrect_id)

    # Restore single head
    restore_hooks = []
    for lay in range(n_layers):
        if lay == layer:
            def _restore_hook(z, hook, _layer=layer, _head=head):
                # Mean-ablate all heads
                for h in range(z.shape[2]):
                    z[0, :, h, :] = mean_z[_layer, h].to(z.device)
                # Restore target head from clean cache
                clean_z = clean_z_cache[_layer]
                seq_len = min(z.shape[1], clean_z.shape[1])
                z[0, :seq_len, _head, :] = clean_z[0, :seq_len, _head, :]
                return z
            restore_hooks.append((f"blocks.{lay}.attn.hook_z", _restore_hook))
        else:
            def _corrupt_hook(z, hook, _layer=lay):
                for h in range(z.shape[2]):
                    z[0, :, h, :] = mean_z[_layer, h].to(z.device)
                return z
            restore_hooks.append((f"blocks.{lay}.attn.hook_z", _corrupt_hook))

    restored_logits = model.run_with_hooks(tokens, fwd_hooks=restore_hooks)
    restored_ld = logit_diff_from_logits(restored_logits, correct_id, incorrect_id)

    gap = clean_ld - corrupt_ld
    if abs(gap) < 1e-8:
        return 0.0
    return (restored_ld - corrupt_ld) / gap


@torch.no_grad()
def run_misalignment(model, tasks: list[str], n_prompts: int = 40) -> list[EvalResult]:
    tokenizer = model.tokenizer
    n_layers = model.cfg.n_layers
    results = []

    for task in tasks:
        circuit_heads = get_circuit_heads(task)
        if not circuit_heads:
            log(f"  {task}: no circuit, skipping")
            continue

        prompts = generate_prompts(task, tokenizer, n_prompts)
        if not prompts:
            log(f"  {task}: no prompts, skipping")
            continue

        correct_ids, incorrect_ids = get_token_ids(prompts, tokenizer)
        if not correct_ids:
            continue

        log(f"  {task} ({len(circuit_heads)} heads, {len(prompts)} prompts)...")
        mean_z = calibrate_mean_z(model, prompts, n_calibration=min(100, len(prompts)))

        # Per-head noising and denoising effects
        per_head_noising = {f"L{L}H{H}": [] for L, H in sorted(circuit_heads)}
        per_head_denoising = {f"L{L}H{H}": [] for L, H in sorted(circuit_heads)}

        for i, p in enumerate(prompts):
            if i >= len(correct_ids):
                break
            tokens = model.to_tokens(p.text)

            # Cache clean z for denoising
            _, cache = model.run_with_cache(tokens, names_filter=lambda n: "hook_z" in n)
            clean_z_cache = {}
            for lay in range(n_layers):
                clean_z_cache[lay] = cache[f"blocks.{lay}.attn.hook_z"].clone()

            for L, H in sorted(circuit_heads):
                noise_eff = compute_noising_effect(
                    model, tokens, correct_ids[i], incorrect_ids[i], L, H, mean_z)
                denoise_eff = compute_denoising_effect(
                    model, tokens, correct_ids[i], incorrect_ids[i],
                    L, H, mean_z, clean_z_cache, n_layers)

                per_head_noising[f"L{L}H{H}"].append(noise_eff)
                per_head_denoising[f"L{L}H{H}"].append(denoise_eff)

        # Aggregate per-head
        per_head_summary = {}
        misalignment_scores = []

        for L, H in sorted(circuit_heads):
            key = f"L{L}H{H}"
            mean_noise = float(np.mean(per_head_noising[key])) if per_head_noising[key] else 0.0
            mean_denoise = float(np.mean(per_head_denoising[key])) if per_head_denoising[key] else 0.0

            # Misalignment = absolute difference between noising and denoising effects
            # High misalignment means the two directions disagree about this head's role
            misalign = abs(mean_noise - mean_denoise)
            misalignment_scores.append(misalign)

            per_head_summary[key] = {
                "noising_necessity": mean_noise,
                "denoising_sufficiency": mean_denoise,
                "misalignment": misalign,
                "direction": "necessity>sufficiency" if mean_noise > mean_denoise else "sufficiency>necessity",
            }

            log(f"    {key}: necessity={mean_noise:.3f}  sufficiency={mean_denoise:.3f}  "
                f"misalign={misalign:.3f}")

        # Aggregate misalignment score
        aggregate_misalignment = float(np.mean(misalignment_scores)) if misalignment_scores else 0.0
        max_misalignment = float(np.max(misalignment_scores)) if misalignment_scores else 0.0

        # Count severely misaligned heads (threshold = 0.3)
        n_severe = sum(1 for m in misalignment_scores if m > 0.3)

        log(f"    aggregate_misalignment={aggregate_misalignment:.3f}  "
            f"max={max_misalignment:.3f}  n_severe={n_severe}/{len(circuit_heads)}")

        results.append(EvalResult(
            metric_id="C37.misalignment_score",
            value=aggregate_misalignment,
            n_samples=len(correct_ids),
            metadata={
                "task": task,
                "aggregate_misalignment": aggregate_misalignment,
                "max_misalignment": max_misalignment,
                "n_severe_misaligned": n_severe,
                "severity_threshold": 0.3,
                "per_head": per_head_summary,
                "n_circuit_heads": len(circuit_heads),
                "circuit_heads": sorted(circuit_heads),
            },
        ))

    return results


def main():
    parser = parse_common_args("C37: Misalignment Score")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("C37: MISALIGNMENT SCORE (Metric #75)")
    log("=" * 60)

    results = run_misalignment(model, tasks, args.n_prompts)

    out = args.out or "37_misalignment_score.json"
    save_results(results, out)

    log(f"\nDone. {len(results)} tasks evaluated.")
    for r in results:
        t = r.metadata["task"]
        log(f"  {t}: misalignment={r.value:.3f}  "
            f"(severe={r.metadata['n_severe_misaligned']}/{r.metadata['n_circuit_heads']})")


if __name__ == "__main__":
    main()
