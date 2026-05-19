"""Effect Size Reporting
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     M90 — Effect Size Reporting
Categories:     measurement
Validity layer: Measurement
Criteria:       M90 Standardized effect size
Establishes:    Whether circuit vs non-circuit heads show a large standardized mean difference
Requires:       CPU or GPU, model
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For each task:
  1. Activation-patch each circuit head and each non-circuit head individually
  2. Collect per-head logit-diff recovery scores for both groups
  3. Compute Cohen's d, Glass's delta, and Hedges' g between groups

Pass condition: Cohen's d > 0.8 (large effect size per Cohen 1988).

Usage:
    uv run python 90_effect_size.py --tasks ioi --n-prompts 40
    uv run python 90_effect_size.py --tasks ioi sva greater_than --device cpu
"""

import math

import numpy as np
import torch

from mechanistic_validity.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
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


@torch.no_grad()
def compute_per_head_recovery(model, prompts, correct_ids, incorrect_ids,
                              heads: set[tuple[int, int]]) -> dict[tuple[int, int], float]:
    """Activation-patch each head individually and return its mean logit-diff recovery."""
    n_valid = min(len(prompts), len(correct_ids))
    rng = np.random.RandomState(42)
    corrupt_indices = list(rng.permutation(n_valid))
    for i in range(n_valid):
        if corrupt_indices[i] == i:
            corrupt_indices[i] = (i + 1) % n_valid

    recovery = {h: [] for h in heads}

    for i in range(n_valid):
        ci = corrupt_indices[i]
        clean_tokens = model.to_tokens(prompts[i].text)
        corrupt_tokens = model.to_tokens(prompts[ci].text)

        _, clean_cache = model.run_with_cache(
            clean_tokens, names_filter=lambda n: "hook_z" in n)

        clean_ld = logit_diff_from_logits(
            model(clean_tokens), correct_ids[i], incorrect_ids[i])
        corrupt_ld = logit_diff_from_logits(
            model(corrupt_tokens), correct_ids[i], incorrect_ids[i])
        gap = clean_ld - corrupt_ld
        if abs(gap) < 1e-8:
            continue

        for L, H in heads:
            hook_name = f"blocks.{L}.attn.hook_z"
            clean_z = clean_cache[hook_name]

            def patch_hook(z, hook, _H=H, _cz=clean_z):
                seq_len = min(z.shape[1], _cz.shape[1])
                z[0, :seq_len, _H, :] = _cz[0, :seq_len, _H, :]
                return z

            patched_logits = model.run_with_hooks(
                corrupt_tokens, fwd_hooks=[(hook_name, patch_hook)])
            patched_ld = logit_diff_from_logits(
                patched_logits, correct_ids[i], incorrect_ids[i])
            recovery[(L, H)].append((patched_ld - corrupt_ld) / gap)

    return {h: float(np.mean(v)) if v else 0.0 for h, v in recovery.items()}


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)
    pooled_std = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std < 1e-12:
        return 0.0
    return float((np.mean(group1) - np.mean(group2)) / pooled_std)


def glass_delta(group1: np.ndarray, group2: np.ndarray) -> float:
    """Glass's delta uses the control group (group2 = non-circuit) std as denominator."""
    if len(group2) < 2:
        return 0.0
    std2 = float(np.std(group2, ddof=1))
    if std2 < 1e-12:
        return 0.0
    return float((np.mean(group1) - np.mean(group2)) / std2)


def hedges_g(group1: np.ndarray, group2: np.ndarray) -> float:
    """Hedges' g = Cohen's d with small-sample bias correction."""
    d = cohens_d(group1, group2)
    n = len(group1) + len(group2)
    if n <= 3:
        return d
    correction = 1.0 - 3.0 / (4.0 * (n - 2) - 1.0)
    return d * correction


@torch.no_grad()
def run_effect_size(model, task: str, n_prompts: int = 40) -> EvalResult | None:
    tokenizer = model.tokenizer
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    circuit_heads = get_circuit_heads(task)
    if not circuit_heads:
        log(f"  {task}: no circuit, skipping")
        return None

    all_heads = {(L, H) for L in range(n_layers) for H in range(n_heads)}
    non_circuit_heads = all_heads - circuit_heads

    prompts = generate_prompts(task, tokenizer, n_prompts)
    if not prompts:
        log(f"  {task}: no prompts, skipping")
        return None

    correct_ids, incorrect_ids = get_token_ids(prompts, tokenizer)
    if not correct_ids:
        log(f"  {task}: no valid token pairs, skipping")
        return None

    log(f"  {task}: {len(circuit_heads)} circuit heads, "
        f"{len(non_circuit_heads)} non-circuit heads, {len(prompts)} prompts")

    circuit_recovery = compute_per_head_recovery(
        model, prompts, correct_ids, incorrect_ids, circuit_heads)
    non_circuit_recovery = compute_per_head_recovery(
        model, prompts, correct_ids, incorrect_ids, non_circuit_heads)

    circuit_scores = np.array(list(circuit_recovery.values()))
    non_circuit_scores = np.array(list(non_circuit_recovery.values()))

    d = cohens_d(circuit_scores, non_circuit_scores)
    delta = glass_delta(circuit_scores, non_circuit_scores)
    g = hedges_g(circuit_scores, non_circuit_scores)
    passed = d > 0.8

    log(f"    Cohen's d={d:.4f}  Glass's delta={delta:.4f}  Hedges' g={g:.4f}  "
        f"[{'PASS' if passed else 'FAIL'}]")

    return EvalResult(
        metric_id="M90.effect_size",
        value=d,
        n_samples=len(prompts),
        metadata={
            "task": task,
            "cohens_d": d,
            "glass_delta": delta,
            "hedges_g": g,
            "circuit_mean": float(np.mean(circuit_scores)),
            "circuit_std": float(np.std(circuit_scores, ddof=1)) if len(circuit_scores) > 1 else 0.0,
            "non_circuit_mean": float(np.mean(non_circuit_scores)),
            "non_circuit_std": float(np.std(non_circuit_scores, ddof=1)) if len(non_circuit_scores) > 1 else 0.0,
            "n_circuit_heads": len(circuit_heads),
            "n_non_circuit_heads": len(non_circuit_heads),
            "passed": passed,
            "threshold": 0.8,
        },
    )


def main():
    parser = parse_common_args("M90: Effect Size Reporting")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("M90: EFFECT SIZE REPORTING (Cohen's d / Glass's delta / Hedges' g)")
    log("=" * 60)

    out = args.out or "90_effect_size.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        result = run_effect_size(model, task, args.n_prompts)
        if result is None:
            continue
        results.append(result)
        save_incremental(result, jsonl_out)
        p = "PASS" if result.metadata["passed"] else "FAIL"
        log(f"  {task}: d={result.value:.4f}  [{p}]")

    save_results(results, out)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
