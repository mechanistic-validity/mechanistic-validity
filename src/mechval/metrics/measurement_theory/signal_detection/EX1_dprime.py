"""Signal Detection Theory: d-prime for Circuit Discrimination
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Metric:         EX1 — d-prime (Signal Detection Theory)
Categories:     causal, signal_detection
Evidence family: causal
Description mode: implementational-functional

Separates a circuit's SENSITIVITY (d') from its CRITERION (beta),
providing a more principled measure than raw accuracy.

Background:
    Signal Detection Theory (Green & Swets 1966, "Signal Detection
    Theory and Psychophysics", Wiley) decomposes detection performance
    into two independent components:

    - d' (d-prime): sensitivity — the observer's ability to discriminate
      signal from noise, independent of response bias
    - beta (criterion): the threshold at which the observer decides
      "signal present" — conservative (few false alarms, many misses)
      vs liberal (few misses, many false alarms)

    Standard circuit evaluations (ablation accuracy, logit-diff) conflate
    these. A circuit might have high sensitivity but conservative
    criterion (it CAN detect the pattern but only fires when very
    confident), or vice versa.

    Applied to circuits: the "signal" is the correct token being top-
    ranked. "Noise" is the incorrect token being top-ranked. We compute
    hit rate (correct when circuit heads active) and false alarm rate
    (correct when circuit heads ablated — i.e., when the "signal" is
    absent but the model still gets it right by other means).

    d' = Z(hit_rate) - Z(false_alarm_rate)

    where Z is the inverse normal CDF.

    Connections:
    - Green & Swets (1966) — foundational SDT text
    - Macmillan & Creelman (2005) "Detection Theory: A User's Guide"
    - Banks (1970) "Signal Detection Theory and Human Memory",
      Psychological Bulletin 74:81-99
    - Stanislaw & Todorov (1999) "Calculation of Signal Detection
      Theory Measures", Behavior Research Methods 31:137-149

Method:
    1. Run model on task prompts with full circuit → count hits
       (correct predictions where logit_diff > 0)
    2. Mean-ablate all circuit heads → count "false alarms"
       (still correct despite circuit removal)
    3. Compute d' = Z(hit_rate) - Z(false_alarm_rate)
    4. Compute criterion beta = -0.5 * (Z(hit_rate) + Z(false_alarm_rate))
    5. Also compute AUC from an ROC curve by sweeping the logit-diff
       threshold that counts as "detecting the signal"

Pass condition: d' > 1.0 (the circuit discriminates meaningfully above
chance) AND AUC > 0.7.

Usage:
    mechval.run("dprime", tasks=["ioi"], device="cpu")
"""

import numpy as np
import torch
from scipy.stats import norm

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
    logit_diff_from_logits,
    make_ablation_hook,
    parse_common_args,
    save_incremental,
    save_results,
)

INSTRUMENT_INFO = InstrumentInfo(
    name="d-prime (Signal Detection Theory)",
    paper_ref="Green & Swets 1966; Macmillan & Creelman 2005",
    paper_cite="Green & Swets 1966, Signal Detection Theory and Psychophysics",
    description="Separates circuit sensitivity (d') from criterion (beta) using signal detection theory",
    category="causal",
    tier="cogsci",
    origin="established",
)

DPRIME_THRESHOLD = 1.0
AUC_THRESHOLD = 0.7


@torch.no_grad()
def run_dprime(model, tasks: list[str],
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
            continue

        correct_ids, incorrect_ids = get_token_ids(prompts, tokenizer)
        if not correct_ids:
            continue

        log(f"  {task}: {len(circuit_heads)} heads, {len(prompts)} prompts")

        mean_z = calibrate_mean_z(model, prompts, n_calibration=min(100, len(prompts)))

        n_layers = model.cfg.n_layers
        n_model_heads = model.cfg.n_heads
        non_circuit = {(L, H) for L in range(n_layers) for H in range(n_model_heads)} - circuit_heads
        ablation_hooks = make_ablation_hook(
            heads_to_layer_dict(circuit_heads), mean_z, "mean")

        signal_lds = []
        noise_lds = []

        for idx, p in enumerate(prompts):
            if idx >= len(correct_ids):
                break
            tokens = model.to_tokens(p.text)

            logits = model(tokens)
            signal_ld = logit_diff_from_logits(logits, correct_ids[idx], incorrect_ids[idx])
            signal_lds.append(signal_ld)

            logits_ablated = model.run_with_hooks(tokens, fwd_hooks=ablation_hooks)
            noise_ld = logit_diff_from_logits(logits_ablated, correct_ids[idx], incorrect_ids[idx])
            noise_lds.append(noise_ld)

        signal_lds = np.array(signal_lds)
        noise_lds = np.array(noise_lds)

        hit_rate = np.mean(signal_lds > 0)
        false_alarm_rate = np.mean(noise_lds > 0)

        hit_rate_adj = np.clip(hit_rate, 1 / (2 * len(signal_lds)), 1 - 1 / (2 * len(signal_lds)))
        fa_rate_adj = np.clip(false_alarm_rate, 1 / (2 * len(noise_lds)), 1 - 1 / (2 * len(noise_lds)))

        dprime = float(norm.ppf(hit_rate_adj) - norm.ppf(fa_rate_adj))
        criterion = float(-0.5 * (norm.ppf(hit_rate_adj) + norm.ppf(fa_rate_adj)))

        all_lds = np.concatenate([signal_lds, noise_lds])
        thresholds = np.sort(np.unique(all_lds))
        labels = np.concatenate([np.ones(len(signal_lds)), np.zeros(len(noise_lds))])

        tpr_list = [1.0]
        fpr_list = [1.0]
        for thresh in thresholds:
            predictions = (np.concatenate([signal_lds, noise_lds]) > thresh).astype(float)
            tp = np.sum((predictions == 1) & (labels == 1))
            fp = np.sum((predictions == 1) & (labels == 0))
            fn = np.sum((predictions == 0) & (labels == 1))
            tn = np.sum((predictions == 0) & (labels == 0))
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            tpr_list.append(tpr)
            fpr_list.append(fpr)
        tpr_list.append(0.0)
        fpr_list.append(0.0)

        auc = 0.0
        for i in range(len(fpr_list) - 1):
            auc += abs(fpr_list[i] - fpr_list[i + 1]) * (tpr_list[i] + tpr_list[i + 1]) / 2

        passed = dprime >= DPRIME_THRESHOLD and auc >= AUC_THRESHOLD

        log(f"    hit_rate={hit_rate:.3f}  fa_rate={false_alarm_rate:.3f}")
        log(f"    d'={dprime:.3f}  criterion={criterion:.3f}  AUC={auc:.3f}")
        log(f"    signal_mean={signal_lds.mean():.4f}  noise_mean={noise_lds.mean():.4f}")
        log(f"    [{'PASS' if passed else 'FAIL'}]")

        results.append(EvalResult(
            metric_id="EX1.dprime",
            value=dprime,
            n_samples=len(prompts),
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": task,
                "n_heads": len(circuit_heads),
                "hit_rate": float(hit_rate),
                "false_alarm_rate": float(false_alarm_rate),
                "dprime": dprime,
                "criterion": criterion,
                "auc": auc,
                "signal_mean_ld": float(signal_lds.mean()),
                "signal_std_ld": float(signal_lds.std()),
                "noise_mean_ld": float(noise_lds.mean()),
                "noise_std_ld": float(noise_lds.std()),
                "passed": passed,
                "dprime_threshold": DPRIME_THRESHOLD,
                "auc_threshold": AUC_THRESHOLD,
            },
        ))

    return results


def main():
    parser = parse_common_args("EX1: d-prime (Signal Detection Theory)")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("EX1: D-PRIME (SIGNAL DETECTION THEORY)")
    log("=" * 60)

    out = args.out or "EX1_dprime.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_dprime(model, [task], args.n_prompts)
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)

    save_results(results, out, args=args)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
