"""Pairwise Ablation Synergy (Hedonic PAS)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     C10 — Pairwise Ablation Synergy
Categories:     causal
Validity layer: Internal
Criteria:       Second-order interaction effects between circuit components
Establishes:    Whether circuit heads exhibit synergy or redundancy beyond
                first-order effects (i.e., whether EAP-style linear attribution
                fully describes the circuit)
Requires:       CPU or GPU, model
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Inspired by Chowdhury et al. 2025 ("Hedonic Neurons", UMass).

For each pair of circuit heads (i, j):
  1. Clean logit-diff: l(x)
  2. Ablate head i alone (mean-ablate): l_i(x)
  3. Ablate head j alone: l_j(x)
  4. Ablate both i and j: l_{ij}(x)
  5. PAS(i,j) = E_x[ l(x) - l_i(x) - l_j(x) + l_{ij}(x) ]

Positive PAS = synergy (joint effect > sum of individual effects).
Negative PAS = redundancy (individual effects overlap).

Pass condition: mean |PAS| across circuit head pairs > 0.02
(evidence of non-trivial second-order interactions).

Usage:
    uv run python 94_pairwise_synergy.py --tasks ioi --n-prompts 40
    uv run python 94_pairwise_synergy.py --tasks ioi sva --device cpu
"""

import itertools

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
    make_ablation_hook,
    parse_common_args,
    save_incremental,
    save_results,
)

PAS_THRESHOLD = 0.02


@torch.no_grad()
def compute_pairwise_synergy(model, prompts, correct_ids, incorrect_ids,
                             circuit_heads: set[tuple[int, int]],
                             mean_z: torch.Tensor) -> dict:
    """Compute PAS for all pairs of circuit heads.

    Returns dict mapping (head_i, head_j) -> PAS value.
    """
    heads = sorted(circuit_heads)
    pairs = list(itertools.combinations(heads, 2))

    if not pairs:
        return {}

    pas_values: dict[tuple[tuple[int, int], tuple[int, int]], float] = {}

    for head_i, head_j in pairs:
        pas_sum = 0.0
        count = 0

        hooks_i = make_ablation_hook(heads_to_layer_dict({head_i}), mean_z, "mean")
        hooks_j = make_ablation_hook(heads_to_layer_dict({head_j}), mean_z, "mean")
        hooks_ij = make_ablation_hook(heads_to_layer_dict({head_i, head_j}), mean_z, "mean")

        for idx, p in enumerate(prompts):
            if idx >= len(correct_ids):
                break

            tokens = model.to_tokens(p.text)
            cid = correct_ids[idx]
            iid = incorrect_ids[idx]

            # Clean
            clean_logits = model(tokens)
            ld_clean = logit_diff_from_logits(clean_logits, cid, iid)

            # Ablate head i only
            logits_i = model.run_with_hooks(tokens, fwd_hooks=hooks_i)
            ld_i = logit_diff_from_logits(logits_i, cid, iid)

            # Ablate head j only
            logits_j = model.run_with_hooks(tokens, fwd_hooks=hooks_j)
            ld_j = logit_diff_from_logits(logits_j, cid, iid)

            # Ablate both heads
            logits_ij = model.run_with_hooks(tokens, fwd_hooks=hooks_ij)
            ld_ij = logit_diff_from_logits(logits_ij, cid, iid)

            # PAS(i,j) = l(x) - l_i(x) - l_j(x) + l_{ij}(x)
            pas = ld_clean - ld_i - ld_j + ld_ij
            pas_sum += pas
            count += 1

        if count > 0:
            pas_values[(head_i, head_j)] = pas_sum / count

    return pas_values


@torch.no_grad()
def run_pairwise_synergy(model, tasks: list[str],
                         n_prompts: int = 40) -> list[EvalResult]:
    tokenizer = model.tokenizer
    results = []

    for task in tasks:
        circuit_heads = get_circuit_heads(task)
        if not circuit_heads or len(circuit_heads) < 2:
            log(f"  {task}: <2 circuit heads, skipping")
            continue

        prompts = generate_prompts(task, tokenizer, n_prompts)
        if not prompts:
            log(f"  {task}: no prompts, skipping")
            continue

        correct_ids, incorrect_ids = get_token_ids(prompts, tokenizer)
        if not correct_ids:
            continue

        log(f"  {task}: {len(circuit_heads)} heads, "
            f"{len(list(itertools.combinations(sorted(circuit_heads), 2)))} pairs, "
            f"{len(prompts)} prompts")

        mean_z = calibrate_mean_z(model, prompts, n_calibration=min(100, len(prompts)))

        pas_values = compute_pairwise_synergy(
            model, prompts, correct_ids, incorrect_ids, circuit_heads, mean_z)

        if not pas_values:
            log(f"  {task}: no pairs computed, skipping")
            continue

        abs_pas = [abs(v) for v in pas_values.values()]
        mean_abs_pas = float(np.mean(abs_pas))
        max_abs_pas = float(np.max(abs_pas))
        n_synergistic = sum(1 for v in pas_values.values() if v > PAS_THRESHOLD)
        n_redundant = sum(1 for v in pas_values.values() if v < -PAS_THRESHOLD)
        n_independent = len(pas_values) - n_synergistic - n_redundant
        passed = mean_abs_pas > PAS_THRESHOLD

        # Per-pair detail
        per_pair = {}
        for (hi, hj), v in sorted(pas_values.items()):
            key = f"L{hi[0]}H{hi[1]}-L{hj[0]}H{hj[1]}"
            label = "synergy" if v > PAS_THRESHOLD else ("redundancy" if v < -PAS_THRESHOLD else "independent")
            per_pair[key] = {"pas": float(v), "label": label}
            log(f"    {key}: PAS={v:+.4f}  [{label}]")

        log(f"    mean|PAS|={mean_abs_pas:.4f}  max|PAS|={max_abs_pas:.4f}  "
            f"syn={n_synergistic} red={n_redundant} ind={n_independent}  "
            f"[{'PASS' if passed else 'FAIL'}]")

        results.append(EvalResult(
            metric_id="C10.pairwise_ablation_synergy",
            value=mean_abs_pas,
            n_samples=len(prompts),
            metadata={
                "task": task,
                "n_heads": len(circuit_heads),
                "n_pairs": len(pas_values),
                "mean_abs_pas": mean_abs_pas,
                "max_abs_pas": max_abs_pas,
                "n_synergistic": n_synergistic,
                "n_redundant": n_redundant,
                "n_independent": n_independent,
                "per_pair": per_pair,
                "passed": passed,
                "threshold": PAS_THRESHOLD,
            },
        ))

    return results


def main():
    parser = parse_common_args("C10: Pairwise Ablation Synergy (Hedonic PAS)")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("C10: PAIRWISE ABLATION SYNERGY")
    log("=" * 60)

    out = args.out or "94_pairwise_synergy.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_pairwise_synergy(model, [task], args.n_prompts)
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)
            p = "PASS" if r.metadata["passed"] else "FAIL"
            log(f"  {task}: mean|PAS|={r.value:.4f}  [{p}]")

    save_results(results, out)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
