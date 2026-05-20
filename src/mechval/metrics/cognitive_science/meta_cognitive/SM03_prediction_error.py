"""Prediction Error Circuit Lesioning
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Metric:         SM-03 — Prediction Error Circuit Lesioning
Categories:     wildcard, self_model
Evidence family: causal
Description mode: implementational-functional

Tests whether the circuit contains dedicated uncertainty/error
components by measuring correlation between head activation magnitude
and model confidence, then lesioning candidate error heads.

Background:
    Botvinick et al. (2001, "Conflict Monitoring and Cognitive Control",
    Trends in Cognitive Sciences 5:295-300) proposed that the anterior
    cingulate cortex monitors for errors and conflict, adjusting
    cognitive control accordingly. Analogously, a transformer may
    contain heads that function as error monitors: their activations
    anti-correlate with model confidence (high activation when the
    model is uncertain, low when confident).

    If such heads exist, ablating them should degrade the model's
    calibration — its ability to know when it is likely to be wrong.
    This is measured via Expected Calibration Error (ECE): after
    removing error-signal heads, the model's confidence should become
    less aligned with its actual accuracy.

    This metric identifies candidate error heads via activation-
    confidence correlation, then causally validates by measuring ECE
    change under ablation.

    Connections:
    - Botvinick et al. (2001) — conflict monitoring / error detection
    - Niculescu-Mizil & Caruana (2005) — calibration and ECE
    - Guo et al. (2017) "On Calibration of Modern Neural Networks" —
      ECE methodology

Method:
    1. Run model on all prompts, compute per-prompt confidence =
       max(softmax(logits)) at the final token
    2. For each circuit head, compute correlation between head's
       activation magnitude (L2 norm of hook_result at last token)
       and model confidence:
       - Negative correlation = error signal head (active when
         uncertain)
       - Positive correlation = prediction head (active when certain)
    3. Error heads = heads with negative correlation
    4. Progressively ablate error heads (most negative correlation
       first) and measure calibration via ECE:
       ECE = mean |accuracy_in_bin - confidence_in_bin| across
       confidence bins
    5. Error circuit score = ECE increase when error heads ablated
    6. Pass: at least one head has negative correlation (error signal
       exists)

Pass condition: at least one circuit head has negative correlation
with model confidence.

Usage:
    mechval.run("prediction_error", tasks=["ioi"], device="cpu")

References:
    - Botvinick et al. 2001, Trends in Cognitive Sciences 5:295-300
    - Niculescu-Mizil & Caruana 2005 (calibration)
    - Guo et al. 2017, ICML (ECE)
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
    name="Prediction Error Circuit Lesioning",
    paper_ref="Botvinick et al. 2001, Trends in Cognitive Sciences 5:295-300",
    paper_cite="Botvinick et al. 2001 (conflict monitoring); Guo et al. 2017 (ECE)",
    description="Identifies error-signal heads (anti-correlated with confidence) and measures ECE change under ablation",
    category="wildcard",
    tier="cogsci",
    origin="established",
    subcategory="meta_cognitive",
)

N_ECE_BINS = 10


def _compute_ece(confidences: np.ndarray, correctness: np.ndarray,
                 n_bins: int = N_ECE_BINS) -> float:
    """Expected Calibration Error: mean |acc_bin - conf_bin| weighted by bin count."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(confidences)
    if total == 0:
        return 0.0
    for i in range(n_bins):
        mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = mask | (confidences == bin_edges[i + 1])
        n_in_bin = mask.sum()
        if n_in_bin == 0:
            continue
        avg_conf = confidences[mask].mean()
        avg_acc = correctness[mask].mean()
        ece += (n_in_bin / total) * abs(avg_acc - avg_conf)
    return float(ece)


@torch.no_grad()
def run_prediction_error(model, tasks: list[str],
                         n_prompts: int = 40) -> list[EvalResult]:
    tokenizer = model.tokenizer
    results = []
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

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

        # Step 1: collect per-prompt confidence and per-head activation magnitude
        confidences = np.zeros(n_valid)
        correctness = np.zeros(n_valid)
        # head_activations[i][(layer, head)] = L2 norm of hook_result at last token
        head_activations = {(l, h): np.zeros(n_valid) for l, h in circuit_heads}

        for idx in range(n_valid):
            tokens = model.to_tokens(prompts[idx].text)
            logits, cache = model.run_with_cache(
                tokens,
                names_filter=lambda n: "attn.hook_z" in n,
            )

            # Confidence = max softmax probability at last token
            probs = torch.softmax(logits[0, -1], dim=-1)
            confidences[idx] = probs.max().item()

            # Correctness = whether top prediction matches correct answer
            top_pred = probs.argmax().item()
            correctness[idx] = 1.0 if top_pred == correct_ids[idx] else 0.0

            # Per-head activation magnitudes
            for (l, h) in circuit_heads:
                result = cache[f"blocks.{l}.attn.hook_z"]
                head_act = result[0, -1, h]  # (d_head,)
                head_activations[(l, h)][idx] = torch.norm(head_act, p=2).item()

        # Step 2: compute correlation between activation magnitude and confidence
        head_correlations = {}
        for (l, h) in sorted(circuit_heads):
            acts = head_activations[(l, h)]
            if np.std(acts) < 1e-10 or np.std(confidences) < 1e-10:
                corr = 0.0
            else:
                corr = float(np.corrcoef(acts, confidences)[0, 1])
                if np.isnan(corr):
                    corr = 0.0
            head_correlations[(l, h)] = corr

        # Identify error heads (negative correlation)
        error_heads = [(l, h) for (l, h), c in head_correlations.items() if c < 0]
        error_heads_sorted = sorted(error_heads,
                                    key=lambda lh: head_correlations[lh])

        n_error_heads = len(error_heads)
        has_error_heads = n_error_heads > 0

        # Step 3: baseline ECE
        baseline_ece = _compute_ece(confidences, correctness)

        # Step 4: progressive ablation of error heads
        ablation_curve = [{"n_ablated": 0, "ece": baseline_ece}]

        if error_heads_sorted:
            mean_z = calibrate_mean_z(model, prompts[:20])

            heads_to_ablate = set()
            for (l, h) in error_heads_sorted:
                heads_to_ablate.add((l, h))
                ablation_hooks = make_ablation_hook(
                    heads_to_layer_dict(heads_to_ablate), mean_z, "mean"
                )

                abl_confidences = np.zeros(n_valid)
                abl_correctness = np.zeros(n_valid)

                for idx in range(n_valid):
                    tokens = model.to_tokens(prompts[idx].text)
                    abl_logits = model.run_with_hooks(
                        tokens, fwd_hooks=ablation_hooks
                    )
                    abl_probs = torch.softmax(abl_logits[0, -1], dim=-1)
                    abl_confidences[idx] = abl_probs.max().item()
                    top_pred = abl_probs.argmax().item()
                    abl_correctness[idx] = 1.0 if top_pred == correct_ids[idx] else 0.0

                abl_ece = _compute_ece(abl_confidences, abl_correctness)
                ablation_curve.append({
                    "n_ablated": len(heads_to_ablate),
                    "ece": abl_ece,
                    "ablated_head": f"L{l}H{h}",
                })

        # Error circuit score = ECE increase after ablating all error heads
        final_ece = ablation_curve[-1]["ece"] if ablation_curve else baseline_ece
        ece_increase = final_ece - baseline_ece

        passed = has_error_heads

        # Per-head summary
        head_stats = {}
        for (l, h) in sorted(circuit_heads):
            corr = head_correlations[(l, h)]
            head_stats[f"L{l}H{h}"] = {
                "activation_confidence_correlation": corr,
                "is_error_head": corr < 0,
                "mean_activation": float(head_activations[(l, h)].mean()),
            }

        log(f"    baseline_ece={baseline_ece:.4f}  final_ece={final_ece:.4f}")
        log(f"    ece_increase={ece_increase:.4f}")
        log(f"    error_heads={n_error_heads}/{len(circuit_heads)}")
        for k, v in sorted(head_stats.items(),
                           key=lambda kv: kv[1]["activation_confidence_correlation"]):
            marker = " [ERROR]" if v["is_error_head"] else ""
            log(f"      {k}: corr={v['activation_confidence_correlation']:.4f}{marker}")
        log(f"    [{'PASS (error heads found)' if passed else 'FAIL (no error heads)'}]")

        results.append(EvalResult(
            metric_id="SM03.prediction_error",
            value=ece_increase,
            n_samples=n_valid,
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": task,
                "n_circuit_heads": len(circuit_heads),
                "n_error_heads": n_error_heads,
                "baseline_ece": baseline_ece,
                "final_ece": final_ece,
                "ece_increase": ece_increase,
                "mean_confidence": float(confidences.mean()),
                "mean_accuracy": float(correctness.mean()),
                "head_stats": head_stats,
                "ablation_curve": ablation_curve,
                "passed": passed,
            },
        ))

    return results


def main():
    parser = parse_common_args("SM-03: Prediction Error Circuit Lesioning")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("SM-03: PREDICTION ERROR CIRCUIT LESIONING")
    log("=" * 60)

    out = args.out or "SM03_prediction_error.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_prediction_error(model, [task], args.n_prompts)
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)

    save_results(results, out, args=args)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
