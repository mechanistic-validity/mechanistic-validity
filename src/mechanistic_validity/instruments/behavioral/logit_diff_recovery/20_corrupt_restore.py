"""Corrupt-Restore Patching (Reverse Direction)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     D01 — Logit Diff Recovery
Categories:     behavioral
Validity layer: Internal
Criteria:       I2 Sufficiency
Establishes:    Circuit alone can restore performance from fully corrupted baseline
Requires:       GPU, model
Doc:            /instruments_v2/behavioral/d01-logit-diff-recovery
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Standard activation patching goes clean->corrupt (ablate and measure
degradation). This script goes the opposite direction: start with a fully
corrupted run (ALL heads mean-ablated), restore specific heads from the
clean cache, and measure recovery.

Tests SUFFICIENCY (can the circuit alone restore performance?) vs the
standard test of NECESSITY (does removing the circuit break performance?).

Usage:
    uv run python 20_corrupt_restore.py --tasks ioi sva
    uv run python 20_corrupt_restore.py --device cuda --n-prompts 60
"""

import numpy as np
import torch

from mechanistic_validity.instruments.common import (
    CIRCUIT_TASKS,
    EvalResult,
    calibrate_mean_z,
    generate_prompts,
    get_circuit_heads,
    get_token_ids,
    heads_to_layer_dict,
    load_model,
    log,
    logit_diff_from_logits,
    parse_common_args,
    save_results,
)


@torch.no_grad()
def cache_clean_z(model, tokens) -> dict[int, torch.Tensor]:
    """Cache hook_z for every layer on clean input."""
    _, cache = model.run_with_cache(tokens, names_filter=lambda n: "hook_z" in n)
    clean_z = {}
    for L in range(model.cfg.n_layers):
        clean_z[L] = cache[f"blocks.{L}.attn.hook_z"].clone()
    return clean_z


def make_corrupt_then_restore_hooks(
    n_layers: int, n_heads: int,
    restore_heads: set[tuple[int, int]],
    mean_z: torch.Tensor,
    clean_z: dict[int, torch.Tensor],
) -> list[tuple[str, callable]]:
    """Ablate ALL heads with mean, then overwrite restored heads from clean cache."""
    hooks = []
    for layer in range(n_layers):
        heads_to_restore = [H for (L, H) in restore_heads if L == layer]

        def _hook(z, hook, _layer=layer, _restore=heads_to_restore):
            # First: mean-ablate every head in this layer
            for H in range(z.shape[2]):
                z[0, :, H, :] = mean_z[_layer, H].to(z.device)
            # Then: restore specific heads from clean cache
            clean = clean_z[_layer]
            seq_len = min(z.shape[1], clean.shape[1])
            for H in _restore:
                z[0, :seq_len, H, :] = clean[0, :seq_len, H, :]
            return z

        hooks.append((f"blocks.{layer}.attn.hook_z", _hook))
    return hooks


def make_full_corrupt_hooks(n_layers: int, mean_z: torch.Tensor) -> list[tuple[str, callable]]:
    """Ablate ALL heads with mean activations."""
    hooks = []
    for layer in range(n_layers):
        def _hook(z, hook, _layer=layer):
            for H in range(z.shape[2]):
                z[0, :, H, :] = mean_z[_layer, H].to(z.device)
            return z
        hooks.append((f"blocks.{layer}.attn.hook_z", _hook))
    return hooks


@torch.no_grad()
def run_corrupt_restore(model, tasks: list[str], n_prompts: int = 40,
                        n_random_baselines: int = 100) -> list[EvalResult]:
    tokenizer = model.tokenizer
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    all_heads = [(L, H) for L in range(n_layers) for H in range(n_heads)]
    results = []
    rng = np.random.RandomState(42)

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

        clean_lds = []
        corrupt_lds = []
        restored_lds = []
        per_head_contributions = {f"L{L}H{H}": [] for L, H in sorted(circuit_heads)}

        hooks_corrupt = make_full_corrupt_hooks(n_layers, mean_z)

        for i, p in enumerate(prompts):
            if i >= len(correct_ids):
                break
            tokens = model.to_tokens(p.text)
            clean_z = cache_clean_z(model, tokens)

            # Clean logit diff
            clean_logits = model(tokens)
            clean_ld = logit_diff_from_logits(clean_logits, correct_ids[i], incorrect_ids[i])
            clean_lds.append(clean_ld)

            # Fully corrupted logit diff
            corrupt_logits = model.run_with_hooks(tokens, fwd_hooks=hooks_corrupt)
            corrupt_ld = logit_diff_from_logits(corrupt_logits, correct_ids[i], incorrect_ids[i])
            corrupt_lds.append(corrupt_ld)

            # Restore ALL circuit heads
            hooks_restore_all = make_corrupt_then_restore_hooks(
                n_layers, n_heads, circuit_heads, mean_z, clean_z,
            )
            restored_logits = model.run_with_hooks(tokens, fwd_hooks=hooks_restore_all)
            restored_ld = logit_diff_from_logits(restored_logits, correct_ids[i], incorrect_ids[i])
            restored_lds.append(restored_ld)

            # Per-head restoration
            gap = clean_ld - corrupt_ld
            for L, H in sorted(circuit_heads):
                hooks_single = make_corrupt_then_restore_hooks(
                    n_layers, n_heads, {(L, H)}, mean_z, clean_z,
                )
                single_logits = model.run_with_hooks(tokens, fwd_hooks=hooks_single)
                single_ld = logit_diff_from_logits(single_logits, correct_ids[i], incorrect_ids[i])
                contribution = (single_ld - corrupt_ld) / gap if abs(gap) > 1e-8 else 0.0
                per_head_contributions[f"L{L}H{H}"].append(contribution)

        # Aggregate
        mean_clean = float(np.mean(clean_lds))
        mean_corrupt = float(np.mean(corrupt_lds))
        mean_restored = float(np.mean(restored_lds))

        gap = mean_clean - mean_corrupt
        restoration_rate = (mean_restored - mean_corrupt) / gap if abs(gap) > 1e-8 else 0.0

        per_head_mean = {k: float(np.mean(v)) if v else 0.0
                         for k, v in per_head_contributions.items()}

        log(f"    clean={mean_clean:.3f} corrupt={mean_corrupt:.3f} "
            f"restored={mean_restored:.3f} rate={restoration_rate:.3f}")

        # Random baseline: restore random k heads
        k = len(circuit_heads)
        random_rates = []
        for _ in range(min(n_random_baselines, 200)):
            rand_heads_idx = rng.choice(len(all_heads), size=k, replace=False)
            rand_heads = {all_heads[j] for j in rand_heads_idx}

            rand_restored_lds = []
            for i, p in enumerate(prompts):
                if i >= len(correct_ids):
                    break
                tokens = model.to_tokens(p.text)
                clean_z = cache_clean_z(model, tokens)
                hooks_rand = make_corrupt_then_restore_hooks(
                    n_layers, n_heads, rand_heads, mean_z, clean_z,
                )
                rand_logits = model.run_with_hooks(tokens, fwd_hooks=hooks_rand)
                rand_ld = logit_diff_from_logits(rand_logits, correct_ids[i], incorrect_ids[i])
                rand_restored_lds.append(rand_ld)

            mean_rand = float(np.mean(rand_restored_lds))
            rand_rate = (mean_rand - mean_corrupt) / gap if abs(gap) > 1e-8 else 0.0
            random_rates.append(rand_rate)

        baseline_random = float(np.mean(random_rates))
        baseline_std = float(np.std(random_rates))
        log(f"    random_baseline={baseline_random:.3f}+/-{baseline_std:.3f}")

        results.append(EvalResult(
            metric_id="C20.corrupt_restore",
            value=restoration_rate,
            baseline_random=baseline_random,
            n_samples=len(clean_lds),
            metadata={
                "task": task,
                "mean_clean_ld": mean_clean,
                "mean_corrupt_ld": mean_corrupt,
                "mean_restored_ld": mean_restored,
                "restoration_rate": restoration_rate,
                "per_head_restoration": per_head_mean,
                "random_baseline_std": baseline_std,
                "n_circuit_heads": k,
                "circuit_heads": sorted(circuit_heads),
            },
        ))

    return results


def main():
    parser = parse_common_args("C20: Corrupt-Restore Patching")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("C20: CORRUPT-RESTORE PATCHING")
    log("=" * 60)

    results = run_corrupt_restore(model, tasks, args.n_prompts, args.n_random_baselines)

    out = args.out or "20_corrupt_restore.json"
    save_results(results, out)

    log(f"\nDone. {len(results)} tasks evaluated.")
    for r in results:
        t = r.metadata["task"]
        log(f"  {t}: restoration={r.value:.3f}  vs random={r.baseline_random:.3f}")


if __name__ == "__main__":
    main()
