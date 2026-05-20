"""Free Energy Gradient Tracing
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Metric:         SM-01 — Free Energy Gradient Tracing
Categories:     wildcard, self_model
Evidence family: representational
Description mode: implementational-functional

Tests whether surprise signals flow in a structured direction across
layers, consistent with the Free Energy Principle's prediction that
neural systems resolve prediction errors hierarchically.

Background:
    Friston's Free Energy Principle (2010, "The Free-Energy Principle:
    A Unified Brain Theory?", Nature Reviews Neuroscience 11:127-138)
    predicts that hierarchical systems reduce surprise (free energy)
    through top-down predictions that explain away bottom-up errors.
    Rao & Ballard (1999, "Predictive Coding in the Visual Cortex",
    Nature Neuroscience 2:79-87) formalized this as predictive coding:
    each level sends predictions downward and receives prediction
    errors upward.

    In a transformer, if layers implement something like predictive
    coding, then corrupting the input should produce a "surprise
    signal" — the deviation between clean and corrupted residual
    streams — that changes systematically across layers. If later
    layers resolve surprise (top-down prediction dominates), the
    signal should decrease. If later layers amplify errors (bottom-up
    error propagation dominates), it should increase.

    The gradient (slope of surprise vs layer) reveals which mode
    dominates. Additionally, ablating circuit heads and measuring the
    change in surprise reduction identifies which heads contribute
    most to error resolution.

Method:
    1. Run model on correct prompts, cache residual stream at each
       layer
    2. Run model on corrupted prompts (target token replaced with
       incorrect token), cache residual stream at each layer
    3. Compute per-layer "surprise" = ||resid_corrupt - resid_clean||_2
       at the last token position
    4. Test directional structure: does surprise decrease (top-down
       prediction) or increase (bottom-up error)?
    5. Compute gradient direction: fit linear regression of surprise
       vs layer index
    6. FEP predicts: surprise decreases in later layers (predictions
       resolve errors)
    7. Gradient score = -slope (positive = surprise decreases =
       FEP-consistent)
    8. Also compute: which circuit heads contribute most to surprise
       reduction via ablation

Pass condition: gradient_score > 0 (surprise decreases across layers).

Usage:
    mechval.run("free_energy_gradient", tasks=["ioi"], device="cpu")

References:
    - Friston 2010, Nature Reviews Neuroscience 11:127-138
    - Rao & Ballard 1999, Nature Neuroscience 2:79-87
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
    make_ablation_hook,
    parse_common_args,
    save_incremental,
    save_results,
)

INSTRUMENT_INFO = InstrumentInfo(
    name="Free Energy Gradient Tracing",
    paper_ref="Friston 2010, Nature Reviews Neuroscience 11:127-138; Rao & Ballard 1999, Nature Neuroscience 2:79-87",
    paper_cite="Friston 2010 (FEP); Rao & Ballard 1999 (predictive coding)",
    description="Tests whether surprise signals decrease across layers (FEP-consistent gradient)",
    category="wildcard",
    tier="cogsci",
    origin="established",
    subcategory="meta_cognitive",
)


@torch.no_grad()
def run_free_energy_gradient(model, tasks: list[str],
                             n_prompts: int = 40) -> list[EvalResult]:
    tokenizer = model.tokenizer
    results = []
    n_layers = model.cfg.n_layers

    for task in tasks:
        circuit_heads = get_circuit_heads(task)
        prompts = generate_prompts(task, tokenizer, n_prompts)
        if not prompts:
            log(f"  {task}: no prompts, skipping")
            continue

        correct_ids, incorrect_ids = get_token_ids(prompts, tokenizer)
        if not correct_ids:
            log(f"  {task}: no valid token ids, skipping")
            continue

        log(f"  {task}: {len(prompts)} prompts, {n_layers} layers")

        # Collect per-layer surprise for each prompt
        all_surprises = np.zeros((len(prompts), n_layers))

        for idx, p in enumerate(prompts):
            if idx >= len(correct_ids):
                break

            clean_tokens = model.to_tokens(p.text)

            # Create corrupted tokens: replace the answer position token
            # We corrupt by running with the incorrect target in mind:
            # use the same prompt but we measure divergence from clean
            corrupt_tokens = clean_tokens.clone()
            # Swap last token to incorrect target to create corruption
            corrupt_tokens[0, -1] = incorrect_ids[idx]

            _, clean_cache = model.run_with_cache(
                clean_tokens,
                names_filter=lambda n: "hook_resid_post" in n,
            )
            _, corrupt_cache = model.run_with_cache(
                corrupt_tokens,
                names_filter=lambda n: "hook_resid_post" in n,
            )

            for layer in range(n_layers):
                clean_resid = clean_cache[f"blocks.{layer}.hook_resid_post"][0, -1]
                corrupt_resid = corrupt_cache[f"blocks.{layer}.hook_resid_post"][0, -1]
                surprise = torch.norm(corrupt_resid - clean_resid, p=2).item()
                all_surprises[idx, layer] = surprise

        n_valid = min(len(prompts), len(correct_ids))
        mean_surprises = all_surprises[:n_valid].mean(axis=0)
        std_surprises = all_surprises[:n_valid].std(axis=0)

        # Fit linear regression: surprise = slope * layer + intercept
        layers = np.arange(n_layers, dtype=float)
        if n_layers > 1:
            slope, intercept = np.polyfit(layers, mean_surprises, 1)
        else:
            slope, intercept = 0.0, mean_surprises[0]

        # Gradient score: negative slope means surprise decreases
        gradient_score = float(-slope)

        # Compute per-head surprise reduction contribution via ablation
        head_contributions = {}
        if circuit_heads:
            mean_z = calibrate_mean_z(model, prompts[:20])

            for (lh_layer, lh_head) in sorted(circuit_heads):
                ablation_hooks = make_ablation_hook(
                    {lh_layer: [lh_head]}, mean_z, "mean"
                )

                ablated_surprises = np.zeros(n_layers)
                n_ablation_prompts = min(10, n_valid)

                for idx in range(n_ablation_prompts):
                    clean_tokens = model.to_tokens(prompts[idx].text)
                    corrupt_tokens = clean_tokens.clone()
                    corrupt_tokens[0, -1] = incorrect_ids[idx]

                    # Clean with ablation
                    clean_logits = model.run_with_hooks(
                        clean_tokens, fwd_hooks=ablation_hooks
                    )
                    _, clean_cache_abl = model.run_with_cache(
                        clean_tokens,
                        names_filter=lambda n: "hook_resid_post" in n,
                    )
                    _, corrupt_cache_abl = model.run_with_cache(
                        corrupt_tokens,
                        names_filter=lambda n: "hook_resid_post" in n,
                    )

                    for layer in range(n_layers):
                        c = clean_cache_abl[f"blocks.{layer}.hook_resid_post"][0, -1]
                        x = corrupt_cache_abl[f"blocks.{layer}.hook_resid_post"][0, -1]
                        ablated_surprises[layer] += torch.norm(x - c, p=2).item()

                ablated_surprises /= n_ablation_prompts

                if n_layers > 1:
                    abl_slope, _ = np.polyfit(layers, ablated_surprises, 1)
                else:
                    abl_slope = 0.0

                # Contribution = how much this head helps reduce surprise
                contribution = float(-abl_slope) - gradient_score
                head_contributions[f"L{lh_layer}H{lh_head}"] = {
                    "surprise_gradient_without": float(-abl_slope),
                    "contribution_to_reduction": float(-contribution),
                }

        passed = gradient_score > 0

        log(f"    gradient_score={gradient_score:.4f}  slope={slope:.4f}")
        log(f"    surprise: L0={mean_surprises[0]:.3f} -> L{n_layers-1}={mean_surprises[-1]:.3f}")
        if head_contributions:
            top_heads = sorted(head_contributions.items(),
                               key=lambda kv: abs(kv[1]["contribution_to_reduction"]),
                               reverse=True)[:5]
            for k, v in top_heads:
                log(f"      {k}: contribution={v['contribution_to_reduction']:.4f}")
        log(f"    [{'PASS (FEP-consistent)' if passed else 'FAIL (no gradient)'}]")

        surprise_profile = {
            f"layer_{l}": {
                "mean_surprise": float(mean_surprises[l]),
                "std_surprise": float(std_surprises[l]),
            }
            for l in range(n_layers)
        }

        results.append(EvalResult(
            metric_id="SM01.free_energy_gradient",
            value=gradient_score,
            n_samples=n_valid,
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": task,
                "n_layers": n_layers,
                "gradient_score": gradient_score,
                "slope": float(slope),
                "intercept": float(intercept),
                "surprise_profile": surprise_profile,
                "initial_surprise": float(mean_surprises[0]),
                "final_surprise": float(mean_surprises[-1]),
                "head_contributions": head_contributions,
                "n_circuit_heads": len(circuit_heads),
                "passed": passed,
            },
        ))

    return results


def main():
    parser = parse_common_args("SM-01: Free Energy Gradient Tracing")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("SM-01: FREE ENERGY GRADIENT TRACING")
    log("=" * 60)

    out = args.out or "SM01_free_energy_gradient.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_free_energy_gradient(model, [task], args.n_prompts)
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)

    save_results(results, out, args=args)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
