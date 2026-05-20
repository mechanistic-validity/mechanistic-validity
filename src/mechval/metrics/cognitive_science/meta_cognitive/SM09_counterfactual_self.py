"""Counterfactual Self-Prediction — Does the Circuit Predict Its Own Behavior Under Intervention?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Metric:         SM-09 — Counterfactual Self-Prediction
Categories:     wildcard, self_model
Evidence family: causal
Description mode: implementational-functional

Tests whether a circuit's baseline activations can predict how the
circuit would behave under a counterfactual intervention (scaling
each head by 0.5x).

Background:
    A system with a good "self-model" should be able to predict its
    own behavior under interventions. Pearl (2009, "Causality", 2nd
    edition, Cambridge University Press) formalizes counterfactual
    reasoning: given the actual state of the world, what would have
    happened under a different intervention? Halpern (2016, "Actual
    Causality", MIT Press) extends this to define actual causation
    via counterfactual dependence.

    For a transformer circuit: if we know a head's baseline activation
    and its output projection (W_O @ W_U), we can *predict* how much
    the logit difference would change if we scaled that head's output
    by 0.5x. The predicted effect is:
        predicted_delta = activation_norm * 0.5 * (W_O[h] @ W_U contribution)
    If this prediction correlates with the actual measured effect of
    the 0.5x scaling, the circuit has a "self-predictable" structure.

    This metric measures the correlation between predicted and actual
    intervention effects across prompts, per head. High correlation
    means the head's contribution is linearly predictable from its
    activation magnitude and output direction.

    Connections:
    - Pearl (2009) — counterfactual reasoning, do-calculus
    - Halpern (2016) — actual causality
    - Vig et al. (2020) "Causal Mediation Analysis for Interpreting
      Neural NLP" — direct/indirect effects in transformers

Method:
    1. Baseline: run model on all prompts, collect logit diffs and
       per-head activations at the last token position
    2. For each circuit head h:
       a. Intervene: scale head h's output by 0.5x using a hook on
          hook_result, re-run, measure intervened logit diff
       b. Actual effect = ld_baseline - ld_intervened (per prompt)
       c. Predicted effect: from baseline activations, compute
          contribution = head_activation @ W_O[h] @ W_U
          predicted_delta = 0.5 * (contribution[correct] - contribution[incorrect])
       d. Self-prediction accuracy = Pearson correlation between
          predicted_delta and actual_delta across prompts
    3. Aggregate: mean self-prediction accuracy across all circuit heads
    4. A model with predictable causal structure scores high.

Pass condition: mean_self_prediction > 0.5

Usage:
    mechval.run("counterfactual_self", tasks=["ioi"], device="cpu")

References:
    Pearl 2009 "Causality" 2nd ed;
    Halpern 2016 "Actual Causality"
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
    logit_diff_from_logits,
    parse_common_args,
    save_incremental,
    save_results,
)

INSTRUMENT_INFO = InstrumentInfo(
    name="Counterfactual Self-Prediction",
    paper_ref="Pearl 2009 Causality 2nd ed; Halpern 2016 Actual Causality",
    paper_cite="Pearl 2009 (counterfactuals); Halpern 2016 (actual causality)",
    description="Tests whether baseline activations predict the circuit's behavior under head-scaling interventions",
    category="wildcard",
    tier="cogsci",
    origin="established",
    subcategory="meta_cognitive",
)

SCALE_FACTOR = 0.5
PASS_THRESHOLD = 0.5


@torch.no_grad()
def run_counterfactual_self(model, tasks: list[str],
                            n_prompts: int = 40) -> list[EvalResult]:
    tokenizer = model.tokenizer
    results = []

    W_U = model.W_U.detach()
    b_U = model.b_U.detach() if hasattr(model, 'b_U') and model.b_U is not None else None

    for task in tasks:
        circuit_heads = get_circuit_heads(task)
        if not circuit_heads:
            log(f"  {task}: no circuit heads, skipping")
            continue

        prompts = generate_prompts(task, tokenizer, n_prompts)
        if not prompts:
            continue

        correct_ids, incorrect_ids = get_token_ids(prompts, tokenizer)
        heads = sorted(circuit_heads)
        log(f"  {task}: {len(heads)} circuit heads, {len(prompts)} prompts")

        # Step 1: Baseline run — collect logit diffs and head activations
        head_hook_names = sorted({f"blocks.{l}.attn.hook_z" for l, _ in heads})

        baseline_lds = []
        baseline_head_acts = {f"L{l}H{h}": [] for l, h in heads}

        for idx, p in enumerate(prompts):
            if idx >= len(correct_ids):
                break

            tokens = model.to_tokens(p.text)
            logits, cache = model.run_with_cache(
                tokens,
                names_filter=lambda n: n in head_hook_names,
            )
            ld = logit_diff_from_logits(logits, correct_ids[idx], incorrect_ids[idx])
            baseline_lds.append(ld)

            for l, h in heads:
                act = cache[f"blocks.{l}.attn.hook_z"][0, -1, h].clone()
                baseline_head_acts[f"L{l}H{h}"].append(act)

        n_valid = len(baseline_lds)
        if n_valid < 5:
            log(f"    too few valid prompts ({n_valid}), skipping")
            continue

        baseline_lds_arr = np.array(baseline_lds)

        # Step 2: Per-head intervention and prediction
        per_head_results = {}

        for l, h in heads:
            key = f"L{l}H{h}"
            W_O_h = model.W_O[l, h].detach().float()  # (d_head, d_model)

            # Predicted effect for each prompt
            predicted_deltas = []
            for i in range(n_valid):
                act = baseline_head_acts[key][i].float()  # (d_head,)
                # Contribution to unembedding: act @ W_O @ W_U
                contribution = act @ W_O_h @ W_U  # (d_vocab,)
                if b_U is not None:
                    contribution = contribution + b_U
                pred_delta = SCALE_FACTOR * (
                    contribution[correct_ids[i]] - contribution[incorrect_ids[i]]
                ).item()
                predicted_deltas.append(pred_delta)

            # Actual effect: scale head h by SCALE_FACTOR
            hook_name = f"blocks.{l}.attn.hook_z"

            def scale_hook(z, hook, _h=h):
                z[0, :, _h, :] = z[0, :, _h, :] * SCALE_FACTOR
                return z

            actual_deltas = []
            for idx in range(n_valid):
                tokens = model.to_tokens(prompts[idx].text)
                intervened_logits = model.run_with_hooks(
                    tokens,
                    fwd_hooks=[(hook_name, scale_hook)],
                )
                intervened_ld = logit_diff_from_logits(
                    intervened_logits, correct_ids[idx], incorrect_ids[idx]
                )
                actual_delta = baseline_lds[idx] - intervened_ld
                actual_deltas.append(actual_delta)

            predicted_arr = np.array(predicted_deltas)
            actual_arr = np.array(actual_deltas)

            # Pearson correlation
            if np.std(predicted_arr) < 1e-10 or np.std(actual_arr) < 1e-10:
                corr = 0.0
            else:
                corr = np.corrcoef(predicted_arr, actual_arr)[0, 1]
                corr = corr if not np.isnan(corr) else 0.0

            per_head_results[key] = {
                "self_prediction_correlation": float(corr),
                "mean_predicted_delta": float(predicted_arr.mean()),
                "mean_actual_delta": float(actual_arr.mean()),
                "std_predicted_delta": float(predicted_arr.std()),
                "std_actual_delta": float(actual_arr.std()),
            }

        # Step 3: Aggregate
        correlations = [v["self_prediction_correlation"] for v in per_head_results.values()]
        mean_corr = float(np.mean(correlations)) if correlations else 0.0
        passed = mean_corr > PASS_THRESHOLD

        # Rank heads by self-prediction accuracy
        ranked_heads = sorted(per_head_results.items(),
                              key=lambda kv: kv[1]["self_prediction_correlation"],
                              reverse=True)

        log(f"    mean self-prediction correlation: {mean_corr:.3f}")
        log(f"    per-head correlations:")
        for key, info in ranked_heads[:5]:
            log(f"      {key}: r={info['self_prediction_correlation']:.3f}  "
                f"pred_delta={info['mean_predicted_delta']:.4f}  "
                f"actual_delta={info['mean_actual_delta']:.4f}")
        if len(ranked_heads) > 5:
            log(f"      ... and {len(ranked_heads) - 5} more heads")
        log(f"    [{'PASS (self-predictable)' if passed else 'FAIL (unpredictable)'}]")

        results.append(EvalResult(
            metric_id="SM09.counterfactual_self",
            value=mean_corr,
            n_samples=n_valid,
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": task,
                "n_circuit_heads": len(heads),
                "scale_factor": SCALE_FACTOR,
                "mean_self_prediction": mean_corr,
                "per_head_results": per_head_results,
                "ranked_heads": [
                    {"head": k, **v} for k, v in ranked_heads
                ],
                "mean_baseline_ld": float(baseline_lds_arr.mean()),
                "std_baseline_ld": float(baseline_lds_arr.std()),
                "passed": passed,
                "threshold": PASS_THRESHOLD,
            },
        ))

    return results


def main():
    parser = parse_common_args("SM-09: Counterfactual Self-Prediction")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("SM-09: COUNTERFACTUAL SELF-PREDICTION")
    log("=" * 60)

    out = args.out or "SM09_counterfactual_self.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_counterfactual_self(model, [task], args.n_prompts)
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)

    save_results(results, out, args=args)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
