"""Privileged Access / Introspection — Early Layer Prediction of Late Behavior
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Metric:         SM-08 — Privileged Access
Categories:     wildcard, self_model
Evidence family: representational
Description mode: implementational-functional

Tests whether the model has "better-than-chance access to its own
internal states" — specifically, whether early layers already encode
the answer that late layers will produce.

Background:
    Privileged access is a concept from philosophy of mind: the idea
    that a system has direct, non-inferential knowledge of its own
    mental states (Binder et al. 2026, "Introspect-Bench: Can LLMs
    Introspect?"). In transformers, this can be operationalized as:
    do early layers already "know" what the final output will be?

    The logit lens (nostalgebraist 2020) provides a natural probe:
    at each intermediate layer, unembed the residual stream via W_U
    to produce a predicted logit difference. If this predicted logit
    difference correlates highly with the final logit difference
    from early layers, the model has "privileged access" — its early
    representations already contain the answer.

    This is distinct from SM-06 (entropy cascade) which measures
    global entropy reduction. SM-08 specifically measures whether
    the model's prediction of the *correct vs incorrect token* is
    established early, using per-prompt correlation rather than
    aggregate entropy.

    Connections:
    - Binder et al. (2026) — Introspect-Bench
    - nostalgebraist (2020) — logit lens
    - Elhage et al. (2022) "Softmax Linear Units", Anthropic —
      intermediate residual stream contains structured predictions

Method:
    1. For each prompt, run the model and record the final logit
       difference (correct - incorrect token logit)
    2. At each layer l, apply the logit lens:
       predicted_ld_l = (resid_l @ W_U)[correct] - (resid_l @ W_U)[incorrect]
    3. Across all prompts, compute Pearson correlation between
       predicted_ld at layer l and the actual final logit difference
    4. Privileged access score: find the earliest layer where
       correlation exceeds 0.8
    5. Normalize: privileged_layer = earliest_high_corr_layer / n_layers
       - Low value = early privileged access (model "knows" early)
       - High value = late privileged access (model decides late)
    6. Also report the full correlation trajectory across layers

Pass condition: privileged_layer < 0.5 (model knows its answer by
the halfway point).

Usage:
    mechval.run("privileged_access", tasks=["ioi"], device="cpu")

References:
    Binder et al. 2026 "Introspect-Bench";
    nostalgebraist 2020 (logit lens)
"""

import numpy as np
import torch

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    InstrumentInfo,
    generate_prompts,
    get_circuit_heads,
    get_token_ids,
    load_model,
    log,
    parse_common_args,
    save_incremental,
    save_results,
)

INSTRUMENT_INFO = InstrumentInfo(
    name="Privileged Access / Introspection",
    paper_ref="Binder et al. 2026 Introspect-Bench; nostalgebraist 2020 (logit lens)",
    paper_cite="Binder et al. 2026 (privileged access); nostalgebraist 2020 (logit lens)",
    description="Tests whether early layers predict the model's final answer via logit-lens correlation",
    category="wildcard",
    tier="cogsci",
    origin="established",
    subcategory="meta_cognitive",
)

CORRELATION_THRESHOLD = 0.8
PASS_THRESHOLD = 0.5


@torch.no_grad()
def run_privileged_access(model, tasks: list[str],
                          n_prompts: int = 40) -> list[EvalResult]:
    tokenizer = model.tokenizer
    results = []

    n_layers = model.cfg.n_layers
    W_U = model.W_U.detach()
    b_U = model.b_U.detach() if hasattr(model, 'b_U') and model.b_U is not None else None

    for task in tasks:
        circuit_heads = get_circuit_heads(task)

        prompts = generate_prompts(task, tokenizer, n_prompts)
        if not prompts:
            continue

        correct_ids, incorrect_ids = get_token_ids(prompts, tokenizer)

        log(f"  {task}: {n_layers} layers, {len(prompts)} prompts")

        # Collect per-layer predicted logit diffs and final logit diffs
        predicted_lds = np.zeros((len(prompts), n_layers))
        final_lds = np.zeros(len(prompts))
        valid_count = 0

        for idx, p in enumerate(prompts):
            if idx >= len(correct_ids):
                break

            tokens = model.to_tokens(p.text)

            # Full forward pass with cached residual streams
            logits, cache = model.run_with_cache(
                tokens,
                names_filter=lambda n: "hook_resid_post" in n,
            )

            # Final logit diff
            final_logits = logits[0, -1]
            final_ld = (final_logits[correct_ids[idx]] - final_logits[incorrect_ids[idx]]).item()
            final_lds[idx] = final_ld

            # Per-layer logit lens prediction
            for layer in range(n_layers):
                resid = cache[f"blocks.{layer}.hook_resid_post"][0, -1]

                if hasattr(model, 'ln_final'):
                    resid_normed = model.ln_final(resid.unsqueeze(0)).squeeze(0)
                else:
                    resid_normed = resid

                layer_logits = resid_normed @ W_U
                if b_U is not None:
                    layer_logits = layer_logits + b_U

                pred_ld = (layer_logits[correct_ids[idx]] - layer_logits[incorrect_ids[idx]]).item()
                predicted_lds[idx, layer] = pred_ld

            valid_count += 1

        if valid_count < 5:
            log(f"    too few valid prompts ({valid_count}), skipping")
            continue

        predicted_lds = predicted_lds[:valid_count]
        final_lds = final_lds[:valid_count]

        # Compute per-layer correlation with final logit diff
        correlations = np.zeros(n_layers)
        for layer in range(n_layers):
            pred = predicted_lds[:, layer]
            # Pearson correlation
            if np.std(pred) < 1e-10 or np.std(final_lds) < 1e-10:
                correlations[layer] = 0.0
            else:
                corr = np.corrcoef(pred, final_lds)[0, 1]
                correlations[layer] = corr if not np.isnan(corr) else 0.0

        # Find earliest layer with correlation > threshold
        earliest_high_corr = n_layers  # default: never reaches threshold
        for layer in range(n_layers):
            if correlations[layer] >= CORRELATION_THRESHOLD:
                earliest_high_corr = layer
                break

        privileged_layer = earliest_high_corr / n_layers
        passed = privileged_layer < PASS_THRESHOLD

        # Build per-layer profile
        layer_profile = {}
        for layer in range(n_layers):
            layer_profile[f"layer_{layer}"] = {
                "correlation": float(correlations[layer]),
                "mean_predicted_ld": float(predicted_lds[:, layer].mean()),
                "std_predicted_ld": float(predicted_lds[:, layer].std()),
            }

        # Find the layer with the biggest single-step correlation jump
        max_jump_layer = 0
        max_jump = 0.0
        for layer in range(1, n_layers):
            jump = correlations[layer] - correlations[layer - 1]
            if jump > max_jump:
                max_jump = jump
                max_jump_layer = layer

        log(f"    correlation trajectory: L0={correlations[0]:.3f} -> "
            f"L{n_layers-1}={correlations[-1]:.3f}")
        log(f"    earliest high-corr layer: {earliest_high_corr} "
            f"(normalized: {privileged_layer:.3f})")
        log(f"    max correlation jump: layer {max_jump_layer} (+{max_jump:.3f})")
        log(f"    final logit diff: mean={final_lds.mean():.3f} std={final_lds.std():.3f}")
        log(f"    [{'PASS (early access)' if passed else 'FAIL (late access)'}]")

        results.append(EvalResult(
            metric_id="SM08.privileged_access",
            value=privileged_layer,
            n_samples=valid_count,
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": task,
                "n_layers": n_layers,
                "privileged_layer_index": earliest_high_corr,
                "privileged_layer_normalized": privileged_layer,
                "correlation_threshold": CORRELATION_THRESHOLD,
                "correlations": [float(c) for c in correlations],
                "layer_profile": layer_profile,
                "max_jump_layer": max_jump_layer,
                "max_jump_value": float(max_jump),
                "mean_final_ld": float(final_lds.mean()),
                "std_final_ld": float(final_lds.std()),
                "passed": passed,
                "threshold": PASS_THRESHOLD,
            },
        ))

    return results


def main():
    parser = parse_common_args("SM-08: Privileged Access")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("SM-08: PRIVILEGED ACCESS / INTROSPECTION")
    log("=" * 60)

    out = args.out or "SM08_privileged_access.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_privileged_access(model, [task], args.n_prompts)
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)

    save_results(results, out, args=args)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
