"""World Model vs Self-Model Double Dissociation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Metric:         SM-04 — World vs Self-Model Dissociation
Categories:     wildcard, self_model
Evidence family: causal
Description mode: implementational-functional

Tests whether circuit heads specialize into "world model" (computing
the correct answer about external content) vs "self-model" (predicting
the model's own confidence), and whether these functions dissociate.

Background:
    The classic neuropsychological double dissociation (Teuber 1955;
    Shallice 1988, "From Neuropsychology to Mental Structure",
    Cambridge University Press) demonstrates that two cognitive
    functions are supported by distinct neural substrates when damage
    to region A impairs function X but not Y, while damage to region B
    impairs Y but not X.

    Applied to transformer circuits: if some heads contribute to task
    performance (world model) but not to confidence calibration
    (self-model), while other heads contribute to calibration but not
    task performance, this constitutes a double dissociation between
    world-modeling and self-modeling within the circuit.

    "World model" = the circuit computing the correct answer (measured
    by logit-diff). "Self-model" = the circuit's magnitude predicting
    model confidence (measured by correlation between |logit_diff| and
    top-1 softmax probability). If these functions dissociate across
    heads, the circuit has specialized subcomponents.

Method:
    1. "World model" test: standard task performance via logit-diff
    2. "Self-model" test: does the circuit's behavior predict its own
       confidence?
       - Run each prompt, get logit-diff (circuit output)
       - Get model's top-1 confidence (max softmax prob)
       - Self-model score = correlation(|logit_diff|, confidence)
       - If high: the circuit's magnitude predicts model confidence
    3. For each circuit head, compute:
       - World contribution = change in mean logit-diff when ablated
       - Self contribution = change in self-model correlation when
         ablated
    4. Double dissociation score = fraction of head pairs where
       world/self contributions diverge (one head helps world but not
       self, or vice versa)
    5. Pass: dissociation_ratio > 0.2 (at least some heads specialize)

Pass condition: dissociation_ratio > 0.2.

Usage:
    mechval.run("world_vs_self_dissociation", tasks=["ioi"], device="cpu")

References:
    - Teuber 1955 (double dissociation logic)
    - Shallice 1988, "From Neuropsychology to Mental Structure",
      Cambridge University Press
"""

import numpy as np
import torch

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    InstrumentInfo,
    calibrate_mean_z,
    compute_logit_diffs,
    generate_prompts,
    get_circuit_heads,
    get_token_ids,
    heads_to_layer_dict,
    load_model,
    log,
    logit_diff_from_logits,
    make_ablation_hook,
    parse_common_args,
    save_incremental,
    save_results,
)

INSTRUMENT_INFO = InstrumentInfo(
    name="World vs Self-Model Dissociation",
    paper_ref="Teuber 1955; Shallice 1988, From Neuropsychology to Mental Structure",
    paper_cite="Teuber 1955 (double dissociation); Shallice 1988",
    description="Tests whether circuit heads specialize into world-modeling vs self-modeling functions",
    category="wildcard",
    tier="cogsci",
    origin="established",
    subcategory="meta_cognitive",
)

DISSOCIATION_THRESHOLD = 0.2


@torch.no_grad()
def run_world_vs_self_dissociation(model, tasks: list[str],
                                   n_prompts: int = 40) -> list[EvalResult]:
    tokenizer = model.tokenizer
    results = []

    for task in tasks:
        circuit_heads = get_circuit_heads(task)
        if not circuit_heads:
            log(f"  {task}: no circuit heads, skipping")
            continue

        prompts = generate_prompts(task, tokenizer, n_prompts)
        if not prompts:
            log(f"  {task}: no prompts, skipping")
            continue

        correct_ids, incorrect_ids = get_token_ids(prompts, tokenizer)
        if not correct_ids:
            log(f"  {task}: no valid token ids, skipping")
            continue

        n_valid = min(len(prompts), len(correct_ids))
        log(f"  {task}: {n_valid} prompts, {len(circuit_heads)} circuit heads")

        # Step 1: baseline world model (logit-diffs) and self-model (confidence)
        baseline_logit_diffs = np.zeros(n_valid)
        baseline_confidences = np.zeros(n_valid)

        for idx in range(n_valid):
            tokens = model.to_tokens(prompts[idx].text)
            logits = model(tokens)
            ld = logit_diff_from_logits(logits, correct_ids[idx], incorrect_ids[idx])
            baseline_logit_diffs[idx] = ld

            probs = torch.softmax(logits[0, -1], dim=-1)
            baseline_confidences[idx] = probs.max().item()

        mean_ld = float(np.mean(baseline_logit_diffs))
        abs_lds = np.abs(baseline_logit_diffs)

        # Baseline self-model score: correlation(|logit_diff|, confidence)
        if np.std(abs_lds) > 1e-10 and np.std(baseline_confidences) > 1e-10:
            baseline_self_score = float(np.corrcoef(abs_lds, baseline_confidences)[0, 1])
            if np.isnan(baseline_self_score):
                baseline_self_score = 0.0
        else:
            baseline_self_score = 0.0

        log(f"    baseline: mean_ld={mean_ld:.3f}  self_score={baseline_self_score:.3f}")

        # Step 2: per-head ablation to measure world and self contributions
        mean_z = calibrate_mean_z(model, prompts[:20])

        head_world_contrib = {}
        head_self_contrib = {}

        for (l, h) in sorted(circuit_heads):
            ablation_hooks = make_ablation_hook(
                {l: [h]}, mean_z, "mean"
            )

            abl_logit_diffs = np.zeros(n_valid)
            abl_confidences = np.zeros(n_valid)

            for idx in range(n_valid):
                tokens = model.to_tokens(prompts[idx].text)
                abl_logits = model.run_with_hooks(tokens, fwd_hooks=ablation_hooks)
                abl_ld = logit_diff_from_logits(
                    abl_logits, correct_ids[idx], incorrect_ids[idx]
                )
                abl_logit_diffs[idx] = abl_ld

                abl_probs = torch.softmax(abl_logits[0, -1], dim=-1)
                abl_confidences[idx] = abl_probs.max().item()

            # World contribution = change in mean logit-diff
            abl_mean_ld = float(np.mean(abl_logit_diffs))
            world_contrib = mean_ld - abl_mean_ld  # positive = head helps task

            # Self contribution = change in self-model correlation
            abl_abs_lds = np.abs(abl_logit_diffs)
            if np.std(abl_abs_lds) > 1e-10 and np.std(abl_confidences) > 1e-10:
                abl_self_score = float(np.corrcoef(abl_abs_lds, abl_confidences)[0, 1])
                if np.isnan(abl_self_score):
                    abl_self_score = 0.0
            else:
                abl_self_score = 0.0

            self_contrib = baseline_self_score - abl_self_score  # positive = head helps self-model

            head_world_contrib[(l, h)] = world_contrib
            head_self_contrib[(l, h)] = self_contrib

        # Step 3: identify dissociation
        # Normalize contributions for comparison
        world_values = np.array([head_world_contrib[k] for k in sorted(circuit_heads)])
        self_values = np.array([head_self_contrib[k] for k in sorted(circuit_heads)])

        world_std = float(np.std(world_values)) if len(world_values) > 1 else 1.0
        self_std = float(np.std(self_values)) if len(self_values) > 1 else 1.0

        if world_std < 1e-10:
            world_std = 1.0
        if self_std < 1e-10:
            self_std = 1.0

        # Count dissociated pairs: head A has high world but low self,
        # head B has low world but high self (or vice versa)
        heads_sorted = sorted(circuit_heads)
        n_dissociated_pairs = 0
        n_total_pairs = 0

        for i in range(len(heads_sorted)):
            for j in range(i + 1, len(heads_sorted)):
                hi, hj = heads_sorted[i], heads_sorted[j]
                wi = head_world_contrib[hi] / world_std
                si = head_self_contrib[hi] / self_std
                wj = head_world_contrib[hj] / world_std
                sj = head_self_contrib[hj] / self_std

                # Dissociation: one head favors world, other favors self
                # Head i: world > self, Head j: self > world (or vice versa)
                i_world_dominant = wi > si
                j_world_dominant = wj > sj
                if i_world_dominant != j_world_dominant:
                    n_dissociated_pairs += 1
                n_total_pairs += 1

        dissociation_ratio = (n_dissociated_pairs / n_total_pairs
                              if n_total_pairs > 0 else 0.0)

        passed = dissociation_ratio > DISSOCIATION_THRESHOLD

        # Per-head summary
        head_stats = {}
        for (l, h) in sorted(circuit_heads):
            w = head_world_contrib[(l, h)]
            s = head_self_contrib[(l, h)]
            w_norm = w / world_std
            s_norm = s / self_std
            head_stats[f"L{l}H{h}"] = {
                "world_contribution": float(w),
                "self_contribution": float(s),
                "world_normalized": float(w_norm),
                "self_normalized": float(s_norm),
                "dominant_role": "world" if w_norm > s_norm else "self",
            }

        log(f"    dissociation_ratio={dissociation_ratio:.3f} "
            f"({n_dissociated_pairs}/{n_total_pairs} pairs)")
        log(f"    baseline_self_score={baseline_self_score:.3f}")
        for k, v in sorted(head_stats.items(),
                           key=lambda kv: abs(kv[1]["world_contribution"]),
                           reverse=True):
            log(f"      {k}: world={v['world_contribution']:.3f}  "
                f"self={v['self_contribution']:.3f}  "
                f"role={v['dominant_role']}")
        log(f"    [{'PASS (dissociation found)' if passed else 'FAIL (no dissociation)'}]")

        results.append(EvalResult(
            metric_id="SM04.world_vs_self_dissociation",
            value=dissociation_ratio,
            n_samples=n_valid,
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": task,
                "n_circuit_heads": len(circuit_heads),
                "baseline_mean_logit_diff": mean_ld,
                "baseline_self_model_score": baseline_self_score,
                "dissociation_ratio": dissociation_ratio,
                "n_dissociated_pairs": n_dissociated_pairs,
                "n_total_pairs": n_total_pairs,
                "head_stats": head_stats,
                "passed": passed,
                "threshold": DISSOCIATION_THRESHOLD,
            },
        ))

    return results


def main():
    parser = parse_common_args("SM-04: World vs Self-Model Dissociation")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("SM-04: WORLD VS SELF-MODEL DISSOCIATION")
    log("=" * 60)

    out = args.out or "SM04_world_vs_self_dissociation.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_world_vs_self_dissociation(model, [task], args.n_prompts)
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)

    save_results(results, out, args=args)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
