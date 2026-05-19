"""Sigma-Ablation Robustness (Multi-Method)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     A04 — Woodward Interventionism
Categories:     causal
Validity layer: Internal
Criteria:       I1 Necessity
Establishes:    Circuit faithfulness is robust across 8 ablation strategies (not method-dependent)
Requires:       GPU, model
Doc:            /instruments_v2/causal/a04-woodward
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Measures variance of faithfulness score across 8 ablation strategies.
Low CV = robust circuit; high CV = ablation-method-dependent artifact
(Miller et al. 2024 critique).

No new dependencies — uses TransformerLens hooks only.

Usage:
    uv run python 03_sigma_ablation.py --tasks ioi sva
    uv run python 03_sigma_ablation.py --device cuda --n-prompts 60
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
    heads_to_layer_dict,
    load_model,
    log,
    logit_diff_from_logits,
    parse_common_args,
    save_results,
)

ABLATION_METHODS = [
    "zero",         # ABL-0: Replace with zeros
    "mean",         # ABL-1: Replace with dataset mean
    "resample",     # ABL-2: Replace with activation from random other prompt
    "noise",        # ABL-3: Add Gaussian noise sigma=std(activation)
    "causal_resample",  # ABL-4: Resample from same-class prompt
    "soft",         # ABL-5: Multiply by 0.1
    "attn_knockout", # ABL-6: Zero Q projection only
    "mean_last",    # ABL-7: Replace z at last position only with mean z
]


@torch.no_grad()
def faithfulness_with_ablation(model, prompts, correct_ids, incorrect_ids,
                                circuit_heads, mean_z, all_z_cache,
                                ablation_type, rng) -> float:
    """Compute faithfulness using a specific ablation method."""
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    non_circuit = {(L, H) for L in range(n_layers) for H in range(n_heads)} - circuit_heads
    non_circuit_by_layer = heads_to_layer_dict(non_circuit)

    faith_num, faith_den = 0.0, 0.0

    for i, p in enumerate(prompts):
        if i >= len(correct_ids):
            break

        tokens = model.to_tokens(p.text)
        clean_logits = model(tokens)
        clean_ld = logit_diff_from_logits(clean_logits, correct_ids[i], incorrect_ids[i])

        hooks = []
        for layer, head_list in non_circuit_by_layer.items():
            def _hook(z, hook, _layer=layer, _heads=head_list, _i=i):
                for H in _heads:
                    if ablation_type == "zero":
                        z[0, :, H, :] = 0.0
                    elif ablation_type == "mean":
                        z[0, :, H, :] = mean_z[_layer, H].to(z.device)
                    elif ablation_type == "resample":
                        other = rng.choice([j for j in range(len(prompts)) if j != _i])
                        if other < len(all_z_cache) and _layer in all_z_cache[other]:
                            z[0, -1, H, :] = all_z_cache[other][_layer][H].to(z.device)
                    elif ablation_type == "noise":
                        std = z[0, :, H, :].std()
                        z[0, :, H, :] += torch.randn_like(z[0, :, H, :]) * std
                    elif ablation_type == "causal_resample":
                        same_class = [j for j in range(len(correct_ids))
                                      if j != _i and correct_ids[j] == correct_ids[_i]]
                        if same_class and same_class[0] < len(all_z_cache):
                            other = rng.choice(same_class)
                            if _layer in all_z_cache[other]:
                                z[0, -1, H, :] = all_z_cache[other][_layer][H].to(z.device)
                    elif ablation_type == "soft":
                        z[0, :, H, :] *= 0.1
                    elif ablation_type == "attn_knockout":
                        z[0, :, H, :] = 0.0
                    elif ablation_type == "mean_last":
                        z[0, -1, H, :] = mean_z[_layer, H].to(z.device)
                return z
            hooks.append((f"blocks.{layer}.attn.hook_z", _hook))

        ablated_logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
        ablated_ld = logit_diff_from_logits(ablated_logits, correct_ids[i], incorrect_ids[i])

        faith_num += ablated_ld
        faith_den += clean_ld

    if abs(faith_den) < 1e-8:
        return 0.0
    return faith_num / faith_den


@torch.no_grad()
def cache_all_z(model, prompts) -> list[dict]:
    """Cache last-position z for all prompts (for resample ablation)."""
    all_cache = []
    for p in prompts:
        tokens = model.to_tokens(p.text)
        _, cache = model.run_with_cache(tokens, names_filter=lambda n: "hook_z" in n)
        prompt_z = {}
        for L in range(model.cfg.n_layers):
            z = cache[f"blocks.{L}.attn.hook_z"]
            prompt_z[L] = z[0, -1].cpu()
        all_cache.append(prompt_z)
    return all_cache


def run_sigma_ablation(model, tasks: list[str], n_prompts: int = 40) -> list[EvalResult]:
    tokenizer = model.tokenizer
    results = []
    rng = np.random.RandomState(42)

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
        all_z_cache = cache_all_z(model, prompts)

        method_scores = {}
        for method in ABLATION_METHODS:
            score = faithfulness_with_ablation(
                model, prompts, correct_ids, incorrect_ids,
                circuit_heads, mean_z, all_z_cache, method, rng,
            )
            method_scores[method] = score
            log(f"    {method}: {score:.3f}")

        scores = list(method_scores.values())
        mean_f = float(np.mean(scores))
        std_f = float(np.std(scores))
        cv = std_f / abs(mean_f) if abs(mean_f) > 1e-8 else float("inf")

        log(f"    CV={cv:.3f} (mean={mean_f:.3f}, std={std_f:.3f})")

        results.append(EvalResult(
            metric_id="C3.sigma_ablation",
            value=cv,
            n_samples=len(prompts),
            metadata={
                "task": task,
                "method_scores": method_scores,
                "mean_faithfulness": mean_f,
                "std_faithfulness": std_f,
                "n_circuit_heads": len(circuit_heads),
            },
        ))

    return results


def main():
    parser = parse_common_args("C3: Sigma-Ablation Robustness")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("C3: SIGMA-ABLATION ROBUSTNESS")
    log("=" * 60)

    results = run_sigma_ablation(model, tasks, args.n_prompts)

    out = args.out or "03_sigma_ablation.json"
    save_results(results, out)

    log(f"\nDone. {len(results)} tasks evaluated.")
    for r in results:
        t = r.metadata["task"]
        log(f"  {t}: CV={r.value:.3f}")


if __name__ == "__main__":
    main()
