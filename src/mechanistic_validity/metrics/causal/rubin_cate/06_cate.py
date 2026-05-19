"""Heterogeneous Treatment Effects (CATE by Context)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     A03 — Rubin CATE
Categories:     causal
Validity layer: Internal
Criteria:       I3 Specificity
Establishes:    Circuit heads have context-dependent causal effects (heterogeneous treatment)
Requires:       GPU, model
Doc:            /instruments_v2/causal/a03-rubin-cate
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Tests whether circuit heads have different causal effects across
syntactic contexts (e.g., SVA head on PP-interrupted vs simple sentences).

Optional: pip install econml (for CausalForestDML). Falls back to
manual subgroup analysis if unavailable.

Usage:
    uv run python 06_cate.py --tasks ioi sva --n-prompts 60
"""

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
    save_results,
)

try:
    from econml.dml import CausalForestDML
    HAS_ECONML = True
except ImportError:
    HAS_ECONML = False


def extract_context_features(prompt, tokenizer, clean_ld: float = 0.0) -> dict:
    """Extract context features from a prompt."""
    text = prompt.text
    tokens = tokenizer.encode(text)
    n_tokens = len(tokens)
    n_unique = len(set(tokens))
    has_comma = "," in text
    has_that = " that " in text.lower()
    has_who = " who " in text.lower()
    has_prep = any(w in text.lower().split() for w in ["in", "on", "at", "by", "with", "from", "to"])

    return {
        "n_tokens": n_tokens,
        "n_unique_tokens": n_unique,
        "clean_logit_diff": clean_ld,
        "has_relative_clause": int(has_that or has_who),
        "has_prepositional_phrase": int(has_prep),
        "has_comma": int(has_comma),
        "complexity": int(has_that or has_who) + int(has_prep) + int(has_comma),
    }


@torch.no_grad()
def compute_per_prompt_effects(model, prompts, correct_ids, incorrect_ids,
                                circuit_heads) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-prompt mean-ablation effect for circuit heads.

    Returns (effects, context_features_matrix, feature_names).
    """
    tokenizer = model.tokenizer
    n_prompts_valid = min(len(prompts), len(correct_ids))

    mean_z = {}
    for i in range(n_prompts_valid):
        tokens = model.to_tokens(prompts[i].text)
        _, cache = model.run_with_cache(tokens, names_filter=lambda n: "hook_z" in n)
        for L, H in circuit_heads:
            key = (L, H)
            z_val = cache[f"blocks.{L}.attn.hook_z"][0, -1, H, :].clone()
            if key not in mean_z:
                mean_z[key] = z_val
            else:
                mean_z[key] = mean_z[key] + z_val
    for key in mean_z:
        mean_z[key] = mean_z[key] / n_prompts_valid

    effects = np.zeros(n_prompts_valid)
    features = []

    for i in range(n_prompts_valid):
        tokens = model.to_tokens(prompts[i].text)
        clean_logits = model(tokens)
        clean_ld = logit_diff_from_logits(clean_logits, correct_ids[i], incorrect_ids[i])

        hooks = []
        for L, H in circuit_heads:
            _mean = mean_z[(L, H)]
            def _hook(z, hook, _H=H, _m=_mean):
                z[0, -1, _H, :] = _m.to(z.device)
                return z
            hooks.append((f"blocks.{L}.attn.hook_z", _hook))

        ablated_logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
        ablated_ld = logit_diff_from_logits(ablated_logits, correct_ids[i], incorrect_ids[i])

        effects[i] = clean_ld - ablated_ld
        features.append(extract_context_features(prompts[i], tokenizer, clean_ld=clean_ld))

    feature_names = list(features[0].keys()) if features else []
    X = np.array([[f[k] for k in feature_names] for f in features])
    return effects, X, feature_names


def run_cate(model, tasks: list[str], n_prompts: int = 60) -> list[EvalResult]:
    tokenizer = model.tokenizer
    results = []

    for task in tasks:
        circuit_heads = get_circuit_heads(task)
        if not circuit_heads:
            log(f"  {task}: no circuit, skipping")
            continue

        prompts = generate_prompts(task, tokenizer, n_prompts)
        if not prompts:
            continue

        correct_ids, incorrect_ids = get_token_ids(prompts, tokenizer)
        if not correct_ids:
            continue

        log(f"  {task} ({len(circuit_heads)} heads, {len(prompts)} prompts)...")

        effects, X, feature_names = compute_per_prompt_effects(
            model, prompts, correct_ids, incorrect_ids, circuit_heads)

        ate = float(np.mean(effects))
        ate_std = float(np.std(effects))

        subgroup_results = {}
        BINARY_FEATURES = {"has_relative_clause", "has_prepositional_phrase", "has_comma"}
        for j, fname in enumerate(feature_names):
            col = X[:, j]
            n_unique = len(np.unique(col))
            if n_unique < 2:
                continue
            if fname in BINARY_FEATURES and n_unique == 2:
                low_mask = col == 0
                high_mask = col == 1
            else:
                median = np.median(col)
                low_mask = col <= median
                high_mask = col > median

            if low_mask.sum() >= 3 and high_mask.sum() >= 3:
                low_mean = float(np.mean(effects[low_mask]))
                high_mean = float(np.mean(effects[high_mask]))
                diff = high_mean - low_mean
                pooled_std = float(np.sqrt(
                    (np.var(effects[low_mask]) * (low_mask.sum() - 1) +
                     np.var(effects[high_mask]) * (high_mask.sum() - 1)) /
                    (low_mask.sum() + high_mask.sum() - 2)
                ))
                cohens_d = diff / pooled_std if pooled_std > 1e-8 else 0.0
                subgroup_results[fname] = {
                    "low_mean": low_mean,
                    "high_mean": high_mean,
                    "difference": diff,
                    "cohens_d": cohens_d,
                    "n_low": int(low_mask.sum()),
                    "n_high": int(high_mask.sum()),
                }

        max_heterogeneity = max(
            (abs(v["cohens_d"]) for v in subgroup_results.values()), default=0.0)

        log(f"    ATE={ate:.3f}+/-{ate_std:.3f}  max_d={max_heterogeneity:.3f}")

        results.append(EvalResult(
            metric_id="C6.cate",
            value=max_heterogeneity,
            n_samples=len(effects),
            metadata={
                "task": task,
                "ate": ate,
                "ate_std": ate_std,
                "subgroup_effects": subgroup_results,
                "feature_names": feature_names,
                "n_circuit_heads": len(circuit_heads),
            },
        ))

    return results


def main():
    parser = parse_common_args("C6: CATE (Heterogeneous Treatment Effects)")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("C6: HETEROGENEOUS TREATMENT EFFECTS (CATE)")
    log("=" * 60)

    if not HAS_ECONML:
        log("NOTE: econml not installed. Using manual subgroup analysis.")

    results = run_cate(model, tasks, args.n_prompts)

    out = args.out or "06_cate.json"
    save_results(results, out)

    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
