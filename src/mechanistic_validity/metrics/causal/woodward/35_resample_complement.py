"""Resample Ablation on Complement (Metric #49)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     A04 — Woodward Interventionism
Categories:     causal
Validity layer: Internal
Criteria:       I1 Necessity
Establishes:    Circuit faithfulness holds under resample ablation (stricter than mean ablation)
Requires:       GPU, model
Doc:            /instruments_v2/causal/a04-woodward
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Instead of mean/zero ablation, replace non-circuit head activations with
activations from a DIFFERENT prompt (resample from the batch). This is a
stricter test: mean ablation can overestimate faithfulness because the
mean is an unrealistically smooth replacement. Resampling preserves the
distributional structure of activations.

Reports faithfulness under resample ablation and comparison to mean
ablation faithfulness.

Usage:
    uv run python 35_resample_complement.py --tasks ioi sva --n-prompts 40
    uv run python 35_resample_complement.py --device cuda
"""

import numpy as np
import torch

from mechanistic_validity.metrics.common import (
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
    logit_diff_from_logits,
    parse_common_args,
    save_results,
)


@torch.no_grad()
def cache_all_z(model, prompts) -> list[dict[int, torch.Tensor]]:
    """Cache hook_z for every layer on every prompt."""
    all_caches = []
    for p in prompts:
        tokens = model.to_tokens(p.text)
        _, cache = model.run_with_cache(tokens, names_filter=lambda n: "hook_z" in n)
        per_layer = {}
        for layer in range(model.cfg.n_layers):
            per_layer[layer] = cache[f"blocks.{layer}.attn.hook_z"][0].cpu().clone()
        all_caches.append(per_layer)
    return all_caches


def make_resample_hooks(heads_by_layer: dict[int, list[int]],
                        donor_z: dict[int, torch.Tensor]) -> list[tuple[str, callable]]:
    """Replace specified heads with activations from a donor prompt."""
    hooks = []
    for layer, head_list in heads_by_layer.items():
        def _hook(z, hook, _layer=layer, _heads=head_list):
            donor = donor_z[_layer].to(z.device)
            seq_len = min(z.shape[1], donor.shape[0])
            for H in _heads:
                z[0, :seq_len, H, :] = donor[:seq_len, H, :]
            return z
        hooks.append((f"blocks.{layer}.attn.hook_z", _hook))
    return hooks


@torch.no_grad()
def compute_resample_faithfulness(model, prompts, correct_ids, incorrect_ids,
                                  circuit_heads: set[tuple[int, int]],
                                  all_z_caches: list[dict[int, torch.Tensor]],
                                  rng) -> float:
    """Faithfulness under resample ablation of the complement."""
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    non_circuit = {(L, H) for L in range(n_layers) for H in range(n_heads)} - circuit_heads
    non_circuit_by_layer = heads_to_layer_dict(non_circuit)

    faith_num, faith_den = 0.0, 0.0
    n = min(len(prompts), len(correct_ids), len(all_z_caches))

    for i in range(n):
        tokens = model.to_tokens(prompts[i].text)

        # Clean logit diff
        clean_logits = model(tokens)
        clean_ld = logit_diff_from_logits(clean_logits, correct_ids[i], incorrect_ids[i])

        # Resample: pick a different prompt's activations as donor
        donor_idx = rng.choice([j for j in range(n) if j != i])
        donor_z = all_z_caches[donor_idx]

        hooks = make_resample_hooks(non_circuit_by_layer, donor_z)
        resampled_logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
        resampled_ld = logit_diff_from_logits(resampled_logits, correct_ids[i], incorrect_ids[i])

        faith_num += resampled_ld
        faith_den += clean_ld

    if abs(faith_den) < 1e-8:
        return 0.0
    return faith_num / faith_den


@torch.no_grad()
def run_resample_complement(model, tasks: list[str], n_prompts: int = 40,
                            n_resample_trials: int = 5) -> list[EvalResult]:
    tokenizer = model.tokenizer
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

        # Cache activations for all prompts
        log(f"    caching activations...")
        all_z_caches = cache_all_z(model, prompts[:len(correct_ids)])

        # Mean ablation faithfulness (for comparison)
        mean_z = calibrate_mean_z(model, prompts, n_calibration=min(100, len(prompts)))
        mean_faith = compute_faithfulness(
            model, prompts, correct_ids, incorrect_ids, circuit_heads, mean_z)

        # Resample ablation faithfulness (average over multiple trials)
        resample_faiths = []
        for trial in range(n_resample_trials):
            rng = np.random.RandomState(42 + trial)
            resample_faith = compute_resample_faithfulness(
                model, prompts, correct_ids, incorrect_ids,
                circuit_heads, all_z_caches, rng)
            resample_faiths.append(resample_faith)

        resample_mean = float(np.mean(resample_faiths))
        resample_std = float(np.std(resample_faiths))
        delta = resample_mean - mean_faith

        log(f"    mean_ablation_faith={mean_faith:.3f}")
        log(f"    resample_faith={resample_mean:.3f} +/- {resample_std:.3f}")
        log(f"    delta (resample - mean) = {delta:.3f}")

        results.append(EvalResult(
            metric_id="C35.resample_complement",
            value=resample_mean,
            n_samples=len(correct_ids),
            metadata={
                "task": task,
                "resample_faithfulness": resample_mean,
                "resample_std": resample_std,
                "mean_ablation_faithfulness": mean_faith,
                "delta": delta,
                "per_trial": resample_faiths,
                "n_resample_trials": n_resample_trials,
                "n_circuit_heads": len(circuit_heads),
                "circuit_heads": sorted(circuit_heads),
            },
        ))

    return results


def main():
    parser = parse_common_args("C35: Resample Complement")
    parser.add_argument("--n-resample-trials", type=int, default=5,
                        help="Number of resampling trials to average")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("C35: RESAMPLE ABLATION ON COMPLEMENT (Metric #49)")
    log("=" * 60)

    results = run_resample_complement(model, tasks, args.n_prompts, args.n_resample_trials)

    out = args.out or "35_resample_complement.json"
    save_results(results, out)

    log(f"\nDone. {len(results)} tasks evaluated.")
    for r in results:
        t = r.metadata["task"]
        log(f"  {t}: resample_faith={r.value:.3f}  mean_faith={r.metadata['mean_ablation_faithfulness']:.3f}")


if __name__ == "__main__":
    main()
