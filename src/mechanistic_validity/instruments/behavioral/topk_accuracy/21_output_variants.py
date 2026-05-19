"""Output Metric Variants (Top-K Accuracy)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     D05 — Top-K Accuracy
Categories:     behavioral
Validity layer: Internal
Criteria:       I2 Sufficiency
Establishes:    Circuit faithfulness is robust across alternative output metrics
Requires:       GPU, model
Doc:            /instruments_v2/behavioral/d05-topk-accuracy
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The standard metric is logit difference. This script computes circuit
faithfulness under 5 alternative output metrics to test whether the circuit
claim is robust to output metric choice.

Reports sigma_output = std across the 5 metrics (analogous to sigma_ablation
from 03 but for output metric choice).

Usage:
    uv run python 21_output_variants.py --tasks ioi sva
    uv run python 21_output_variants.py --device cuda --n-prompts 60
"""

import numpy as np
import torch
import torch.nn.functional as F

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
    make_ablation_hook,
    parse_common_args,
    save_results,
)


# ---------------------------------------------------------------------------
# Output metric functions
# ---------------------------------------------------------------------------
# Each takes (logits, correct_id, incorrect_id) and returns a scalar float.

def metric_logit_diff(logits: torch.Tensor, correct_id: int, incorrect_id: int) -> float:
    last = logits[0, -1]
    return (last[correct_id] - last[incorrect_id]).item()


def metric_log_prob(logits: torch.Tensor, correct_id: int, incorrect_id: int) -> float:
    last = logits[0, -1]
    log_probs = F.log_softmax(last, dim=-1)
    return log_probs[correct_id].item()


def metric_probability(logits: torch.Tensor, correct_id: int, incorrect_id: int) -> float:
    last = logits[0, -1]
    probs = F.softmax(last, dim=-1)
    return probs[correct_id].item()


def metric_top1_accuracy(logits: torch.Tensor, correct_id: int, incorrect_id: int) -> float:
    last = logits[0, -1]
    return 1.0 if last.argmax().item() == correct_id else 0.0


def metric_kl_divergence(logits: torch.Tensor, correct_id: int, incorrect_id: int,
                         clean_logits: torch.Tensor = None) -> float:
    """KL(clean || ablated). Requires clean_logits to be passed separately."""
    if clean_logits is None:
        return 0.0
    clean_log_probs = F.log_softmax(clean_logits[0, -1], dim=-1)
    ablated_log_probs = F.log_softmax(logits[0, -1], dim=-1)
    clean_probs = clean_log_probs.exp()
    kl = (clean_probs * (clean_log_probs - ablated_log_probs)).sum()
    return kl.item()


METRICS = {
    "logit_diff": metric_logit_diff,
    "log_prob": metric_log_prob,
    "probability": metric_probability,
    "top1_accuracy": metric_top1_accuracy,
    "kl_divergence": None,  # handled specially
}


# ---------------------------------------------------------------------------
# Faithfulness computation per metric
# ---------------------------------------------------------------------------

@torch.no_grad()
def faithfulness_by_metric(model, prompts, correct_ids, incorrect_ids,
                           circuit_heads, mean_z, metric_name: str) -> float:
    """Compute faithfulness using a specific output metric.

    For logit_diff/log_prob/probability: faith = metric(circuit_only) / metric(full).
    For top1_accuracy: faith = accuracy(circuit_only) / accuracy(full).
    For kl_divergence: faith = 1 - KL(clean||circuit_only) / KL(clean||all_ablated).
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    non_circuit = {(L, H) for L in range(n_layers) for H in range(n_heads)} - circuit_heads
    non_circuit_by_layer = heads_to_layer_dict(non_circuit)
    hooks = make_ablation_hook(non_circuit_by_layer, mean_z, "mean")

    all_heads = {(L, H) for L in range(n_layers) for H in range(n_heads)}
    all_by_layer = heads_to_layer_dict(all_heads)
    hooks_all = make_ablation_hook(all_by_layer, mean_z, "mean")

    if metric_name == "kl_divergence":
        kl_circuit_sum, kl_all_sum = 0.0, 0.0
        count = 0
        for i, p in enumerate(prompts):
            if i >= len(correct_ids):
                break
            tokens = model.to_tokens(p.text)
            clean_logits = model(tokens)
            ablated_logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
            all_ablated_logits = model.run_with_hooks(tokens, fwd_hooks=hooks_all)

            kl_circuit = metric_kl_divergence(
                ablated_logits, correct_ids[i], incorrect_ids[i], clean_logits,
            )
            kl_all = metric_kl_divergence(
                all_ablated_logits, correct_ids[i], incorrect_ids[i], clean_logits,
            )
            kl_circuit_sum += kl_circuit
            kl_all_sum += kl_all
            count += 1

        if count == 0 or abs(kl_all_sum) < 1e-8:
            return 0.0
        return 1.0 - (kl_circuit_sum / kl_all_sum)

    metric_fn = METRICS[metric_name]
    score_clean_sum, score_ablated_sum = 0.0, 0.0
    count = 0

    for i, p in enumerate(prompts):
        if i >= len(correct_ids):
            break
        tokens = model.to_tokens(p.text)
        clean_logits = model(tokens)
        ablated_logits = model.run_with_hooks(tokens, fwd_hooks=hooks)

        score_clean = metric_fn(clean_logits, correct_ids[i], incorrect_ids[i])
        score_ablated = metric_fn(ablated_logits, correct_ids[i], incorrect_ids[i])

        score_clean_sum += score_clean
        score_ablated_sum += score_ablated
        count += 1

    if count == 0 or abs(score_clean_sum) < 1e-8:
        return 0.0
    return score_ablated_sum / score_clean_sum


def run_output_variants(model, tasks: list[str], n_prompts: int = 40) -> list[EvalResult]:
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
        mean_z = calibrate_mean_z(model, prompts, n_calibration=min(100, len(prompts)))

        metric_scores = {}
        for metric_name in METRICS:
            score = faithfulness_by_metric(
                model, prompts, correct_ids, incorrect_ids,
                circuit_heads, mean_z, metric_name,
            )
            metric_scores[metric_name] = score
            log(f"    {metric_name}: {score:.4f}")

        scores = list(metric_scores.values())
        mean_f = float(np.mean(scores))
        std_f = float(np.std(scores))
        sigma_output = std_f / abs(mean_f) if abs(mean_f) > 1e-8 else float("inf")

        log(f"    sigma_output={sigma_output:.3f} (mean={mean_f:.3f}, std={std_f:.3f})")

        results.append(EvalResult(
            metric_id="C21.output_variants",
            value=sigma_output,
            n_samples=len(prompts),
            metadata={
                "task": task,
                "metric_scores": metric_scores,
                "mean_faithfulness": mean_f,
                "std_faithfulness": std_f,
                "sigma_output": sigma_output,
                "n_circuit_heads": len(circuit_heads),
                "circuit_heads": sorted(circuit_heads),
            },
        ))

    return results


def main():
    parser = parse_common_args("C21: Output Metric Variants")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("C21: OUTPUT METRIC VARIANTS")
    log("=" * 60)

    results = run_output_variants(model, tasks, args.n_prompts)

    out = args.out or "21_output_variants.json"
    save_results(results, out)

    log(f"\nDone. {len(results)} tasks evaluated.")
    for r in results:
        t = r.metadata["task"]
        scores = r.metadata["metric_scores"]
        log(f"  {t}: sigma={r.value:.3f}  "
            + "  ".join(f"{k}={v:.3f}" for k, v in scores.items()))


if __name__ == "__main__":
    main()
