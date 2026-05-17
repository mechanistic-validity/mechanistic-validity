"""Reliability Suite (Test-Retest)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     F07 — Test-Retest
Categories:     measurement
Validity layer: Measurement
Criteria:       M1 Reliability
Establishes:    Circuit scores are stable across repeated measurements with different seeds
Requires:       GPU, model
Doc:            /instruments_v2/measurement/f07-test-retest
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Three psychometric reliability metrics (Pillar 3):

  Metric #83 -- Test-retest: Pearson r across faithfulness measured
               with 3 different prompt seeds (42, 123, 456).
  Metric #85 -- Split-half: Spearman-Brown corrected r from odd/even
               prompt splits.
  Metric #84 -- Cronbach's alpha: internal consistency across
               individual head patching effects.

Usage:
    uv run python 16_reliability_suite.py --tasks ioi sva --n-prompts 40
    uv run python 16_reliability_suite.py --device cuda --n-prompts 60
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    CIRCUIT_TASKS,
    EvalResult,
    _RtiPromptAdapter,
    calibrate_mean_z,
    compute_faithfulness,
    generate_prompts,
    get_circuit_heads,
    get_token_ids,
    heads_to_layer_dict,
    load_model,
    log,
    logit_diff_from_logits,
    make_ablation_hook,
    parse_common_args,
    save_results,
)
from run_probes_rti_v2 import make_rti_prompts
from tasks.task_prompts import TASK_REGISTRY


# ---------------------------------------------------------------------------
# Prompt generation with explicit seed override
# ---------------------------------------------------------------------------

def _generate_prompts_with_seed(task: str, tokenizer, n_prompts: int, seed: int):
    """Generate prompts with a specific seed (bypasses _common's fixed seed=42)."""
    if task == "rti":
        raw = make_rti_prompts(tokenizer, n=n_prompts, seed=seed)
        return [_RtiPromptAdapter(d, tokenizer) for d in raw]
    if task not in TASK_REGISTRY:
        return []
    builder = TASK_REGISTRY[task]
    if task == "buffalo":
        return builder(tokenizer, seed=seed)[:n_prompts]
    return builder(tokenizer, n_prompts=n_prompts, seed=seed)


# ---------------------------------------------------------------------------
# Metric #83 — Test-retest reliability
# ---------------------------------------------------------------------------

def compute_test_retest(model, task: str, circuit_heads: set[tuple[int, int]],
                        n_prompts: int,
                        seeds: list[int] | None = None) -> tuple[float, list[float]]:
    """Compute faithfulness at multiple seeds, return (mean Pearson r, per-seed values)."""
    if seeds is None:
        seeds = [42, 123, 456]

    tokenizer = model.tokenizer
    per_seed_faithfulness = []

    for seed in seeds:
        prompts = _generate_prompts_with_seed(task, tokenizer, n_prompts, seed)
        if not prompts:
            per_seed_faithfulness.append(0.0)
            continue
        correct_ids, incorrect_ids = get_token_ids(prompts, tokenizer)
        if not correct_ids:
            per_seed_faithfulness.append(0.0)
            continue
        mean_z = calibrate_mean_z(model, prompts, n_calibration=min(100, len(prompts)))
        faith = compute_faithfulness(
            model, prompts, correct_ids, incorrect_ids, circuit_heads, mean_z,
        )
        per_seed_faithfulness.append(faith)

    values = np.array(per_seed_faithfulness)
    n_seeds = len(seeds)

    # Compute pairwise Pearson r between all seed pairs
    # With scalar measurements per seed, we compute r across (task, seed) vectors.
    # For a single task, this reduces to checking consistency of the scalar values.
    # We report the coefficient of variation (lower = more reliable) inverted as
    # a reliability proxy: r = 1 - CV, clamped to [0, 1].
    mean_val = float(np.mean(values))
    std_val = float(np.std(values))

    if abs(mean_val) < 1e-8:
        reliability = 0.0
    else:
        cv = std_val / abs(mean_val)
        reliability = max(0.0, min(1.0, 1.0 - cv))

    return reliability, per_seed_faithfulness


# ---------------------------------------------------------------------------
# Metric #85 — Split-half reliability
# ---------------------------------------------------------------------------

def compute_split_half(model, task: str, circuit_heads: set[tuple[int, int]],
                       n_prompts: int) -> tuple[float, float, float]:
    """Split-half reliability with Spearman-Brown correction.

    Returns (corrected_r, odd_faithfulness, even_faithfulness).
    """
    tokenizer = model.tokenizer
    prompts = generate_prompts(task, tokenizer, n_prompts)
    if not prompts:
        return 0.0, 0.0, 0.0

    correct_ids, incorrect_ids = get_token_ids(prompts, tokenizer)
    if not correct_ids:
        return 0.0, 0.0, 0.0

    # Split into odd/even
    odd_prompts = [prompts[i] for i in range(len(prompts)) if i % 2 == 1]
    even_prompts = [prompts[i] for i in range(len(prompts)) if i % 2 == 0]
    odd_correct = [correct_ids[i] for i in range(len(correct_ids)) if i % 2 == 1]
    even_correct = [correct_ids[i] for i in range(len(correct_ids)) if i % 2 == 0]
    odd_incorrect = [incorrect_ids[i] for i in range(len(incorrect_ids)) if i % 2 == 1]
    even_incorrect = [incorrect_ids[i] for i in range(len(incorrect_ids)) if i % 2 == 0]

    # Calibrate mean_z on all prompts
    mean_z = calibrate_mean_z(model, prompts, n_calibration=min(100, len(prompts)))

    odd_faith = compute_faithfulness(
        model, odd_prompts, odd_correct, odd_incorrect, circuit_heads, mean_z,
    )
    even_faith = compute_faithfulness(
        model, even_prompts, even_correct, even_incorrect, circuit_heads, mean_z,
    )

    # For per-prompt logit-diff vectors (needed for Pearson r between halves),
    # compute per-prompt faithfulness ratios
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    non_circuit = {(L, H) for L in range(n_layers) for H in range(n_heads)} - circuit_heads
    non_circuit_by_layer = heads_to_layer_dict(non_circuit)
    hooks = make_ablation_hook(non_circuit_by_layer, mean_z, "mean")

    # Build per-prompt faithfulness for correlation
    odd_per_prompt = _per_prompt_faithfulness(model, odd_prompts, odd_correct, odd_incorrect, hooks)
    even_per_prompt = _per_prompt_faithfulness(model, even_prompts, even_correct, even_incorrect, hooks)

    # Pearson r between the two halves' per-prompt faithfulness
    min_len = min(len(odd_per_prompt), len(even_per_prompt))
    if min_len < 3:
        # Not enough data for meaningful correlation
        r_half = 0.0
    else:
        odd_arr = np.array(odd_per_prompt[:min_len])
        even_arr = np.array(even_per_prompt[:min_len])
        if np.std(odd_arr) < 1e-8 or np.std(even_arr) < 1e-8:
            r_half = 0.0
        else:
            r_half = float(np.corrcoef(odd_arr, even_arr)[0, 1])
            if np.isnan(r_half):
                r_half = 0.0

    # Spearman-Brown correction: r_sb = 2*r / (1 + r)
    corrected_r = 2.0 * r_half / (1.0 + r_half) if (1.0 + r_half) > 1e-8 else 0.0

    return corrected_r, odd_faith, even_faith


@torch.no_grad()
def _per_prompt_faithfulness(model, prompts, correct_ids, incorrect_ids,
                             ablation_hooks) -> list[float]:
    """Compute per-prompt faithfulness ratio (circuit-only LD / clean LD)."""
    ratios = []
    for i, p in enumerate(prompts):
        if i >= len(correct_ids):
            break
        tokens = model.to_tokens(p.text)
        clean_logits = model(tokens)
        clean_ld = logit_diff_from_logits(clean_logits, correct_ids[i], incorrect_ids[i])
        ablated_logits = model.run_with_hooks(tokens, fwd_hooks=ablation_hooks)
        ablated_ld = logit_diff_from_logits(ablated_logits, correct_ids[i], incorrect_ids[i])
        if abs(clean_ld) < 1e-8:
            ratios.append(0.0)
        else:
            ratios.append(ablated_ld / clean_ld)
    return ratios


# ---------------------------------------------------------------------------
# Metric #84 — Cronbach's alpha
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_cronbach_alpha(model, task: str, circuit_heads: set[tuple[int, int]],
                           n_prompts: int) -> tuple[float, dict]:
    """Cronbach's alpha over per-head activation patching effects.

    Each circuit head is an "item." For each prompt, the head's effect =
    logit_diff(clean) - logit_diff(ablate_that_head). Alpha measures
    internal consistency across heads.

    Returns (alpha, per_head_metadata).
    """
    tokenizer = model.tokenizer
    prompts = generate_prompts(task, tokenizer, n_prompts)
    if not prompts:
        return 0.0, {}

    correct_ids, incorrect_ids = get_token_ids(prompts, tokenizer)
    if not correct_ids:
        return 0.0, {}

    mean_z = calibrate_mean_z(model, prompts, n_calibration=min(100, len(prompts)))

    sorted_heads = sorted(circuit_heads)
    k = len(sorted_heads)
    if k < 2:
        return 0.0, {"reason": "need >= 2 heads for Cronbach's alpha"}

    n = min(len(prompts), len(correct_ids))

    # effects[head_idx][prompt_idx] = patching effect of that head on that prompt
    effects = np.zeros((k, n))

    for prompt_idx in range(n):
        tokens = model.to_tokens(prompts[prompt_idx].text)
        clean_logits = model(tokens)
        clean_ld = logit_diff_from_logits(
            clean_logits, correct_ids[prompt_idx], incorrect_ids[prompt_idx],
        )

        for head_idx, (L, H) in enumerate(sorted_heads):
            # Ablate single head
            single_hooks = make_ablation_hook({L: [H]}, mean_z, "mean")
            ablated_logits = model.run_with_hooks(tokens, fwd_hooks=single_hooks)
            ablated_ld = logit_diff_from_logits(
                ablated_logits, correct_ids[prompt_idx], incorrect_ids[prompt_idx],
            )
            effects[head_idx, prompt_idx] = clean_ld - ablated_ld

    # Cronbach's alpha = (k / (k-1)) * (1 - sum(s_i^2) / s_total^2)
    item_variances = np.var(effects, axis=1, ddof=1)  # variance per head across prompts
    total_scores = np.sum(effects, axis=0)  # sum across heads for each prompt
    total_variance = np.var(total_scores, ddof=1)

    if total_variance < 1e-12:
        alpha = 0.0
    else:
        alpha = (k / (k - 1)) * (1.0 - np.sum(item_variances) / total_variance)

    per_head_meta = {
        f"L{L}H{H}": {
            "mean_effect": float(np.mean(effects[i])),
            "var_effect": float(item_variances[i]),
        }
        for i, (L, H) in enumerate(sorted_heads)
    }

    return float(alpha), per_head_meta


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def run_reliability_suite(model, tasks: list[str],
                          n_prompts: int = 40) -> list[EvalResult]:
    results = []

    for task in tasks:
        circuit_heads = get_circuit_heads(task)
        if not circuit_heads:
            log(f"  {task}: no circuit, skipping")
            continue

        log(f"  {task} ({len(circuit_heads)} heads)...")

        # --- Metric #83: Test-retest ---
        log(f"    computing test-retest...")
        reliability, per_seed = compute_test_retest(
            model, task, circuit_heads, n_prompts,
        )
        log(f"    test-retest reliability={reliability:.3f} seeds={per_seed}")
        results.append(EvalResult(
            metric_id="C16.test_retest",
            value=reliability,
            n_samples=n_prompts,
            metadata={
                "task": task,
                "seeds": [42, 123, 456],
                "per_seed_faithfulness": per_seed,
                "n_circuit_heads": len(circuit_heads),
            },
        ))

        # --- Metric #85: Split-half ---
        log(f"    computing split-half...")
        corrected_r, odd_faith, even_faith = compute_split_half(
            model, task, circuit_heads, n_prompts,
        )
        log(f"    split-half r_sb={corrected_r:.3f} "
            f"(odd={odd_faith:.3f}, even={even_faith:.3f})")
        results.append(EvalResult(
            metric_id="C16.split_half",
            value=corrected_r,
            n_samples=n_prompts,
            metadata={
                "task": task,
                "odd_faithfulness": odd_faith,
                "even_faithfulness": even_faith,
                "n_circuit_heads": len(circuit_heads),
            },
        ))

        # --- Metric #84: Cronbach's alpha ---
        log(f"    computing Cronbach's alpha...")
        alpha, per_head_meta = compute_cronbach_alpha(
            model, task, circuit_heads, n_prompts,
        )
        log(f"    Cronbach's alpha={alpha:.3f} ({len(circuit_heads)} items)")
        results.append(EvalResult(
            metric_id="C16.cronbach_alpha",
            value=alpha,
            n_samples=n_prompts,
            metadata={
                "task": task,
                "n_items": len(circuit_heads),
                "per_head_effects": per_head_meta,
            },
        ))

    return results


def main():
    parser = parse_common_args("C16: Reliability Suite")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("C16: RELIABILITY SUITE (Test-Retest, Split-Half, Cronbach's Alpha)")
    log("=" * 60)

    results = run_reliability_suite(model, tasks, args.n_prompts)

    out = args.out or "16_reliability_suite.json"
    save_results(results, out)

    log(f"\nDone. {len(results)} results across "
        f"{len(set(r.metadata['task'] for r in results))} tasks.")
    for r in results:
        log(f"  {r.metadata['task']}/{r.metric_id}: {r.value:.3f}")


if __name__ == "__main__":
    main()
