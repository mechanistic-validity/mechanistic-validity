"""Cross-Model Invariance (Metrics #90-95)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     A12 — Transportability
Categories:     causal
Validity layer: External
Criteria:       E5/E6 Cross-model
Establishes:    Circuit properties are invariant across GPT-2 model sizes (configural/metric/scalar)
Requires:       GPU, model
Doc:            /instruments_v2/causal/a12-transportability
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Tests whether circuit properties are invariant across GPT-2 model sizes
(gpt2, gpt2-medium, gpt2-large, gpt2-xl). Because circuit definitions
only exist for gpt2-small, larger models use activation patching to
identify top-k heads as a proxy circuit (where k = |gpt2_small_circuit|).

Metric #90 — Configural invariance:
    Do the same layers contain circuit heads across model sizes?
    Spearman correlation of layer-distribution histograms.

Metric #91 — Metric invariance:
    Does faithfulness scale similarly? Report faithfulness per size and
    test whether the ranking is preserved.

Metric #92 — Scalar invariance:
    Are absolute faithfulness values comparable? Report mean and std.

Metric #93 — Cross-model weight alignment:
    Cosine similarity of W_OV singular-value distributions between
    matching layers across sizes.

Metric #95 — Scale invariance:
    Faithfulness vs parameter count. Report slope of log-log regression.

Usage:
    uv run python 38_cross_model_invariance.py --tasks ioi --skip-large --device cpu
    uv run python 38_cross_model_invariance.py --tasks ioi sva --device cuda
"""

import numpy as np
import torch
from scipy import stats as sp_stats

from mechanistic_validity.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    calibrate_mean_z,
    compute_faithfulness,
    generate_prompts,
    get_circuit_heads,
    get_token_ids,
    load_model,
    log,
    logit_diff_from_logits,
    parse_common_args,
    save_results,
)

MODEL_PARAM_COUNTS = {
    "gpt2": 124_000_000,
    "gpt2-medium": 355_000_000,
    "gpt2-large": 774_000_000,
    "gpt2-xl": 1_500_000_000,
}


@torch.no_grad()
def discover_circuit_by_patching(model, prompts, correct_ids, incorrect_ids,
                                 k: int) -> set[tuple[int, int]]:
    """Identify top-k heads by activation patching effect for a larger model.

    For each head, patch its z from clean into a corrupted run and measure
    the fraction of the logit-diff gap restored. Return the k heads with
    the largest positive effect.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    rng = np.random.RandomState(42)

    head_effects = np.zeros((n_layers, n_heads))
    n_valid = 0

    n_eval = min(len(prompts), len(correct_ids), 10)

    for i in range(n_eval):
        clean_tokens = model.to_tokens(prompts[i].text)
        other_idx = rng.choice([j for j in range(len(prompts)) if j != i])
        corrupt_tokens = model.to_tokens(prompts[other_idx].text)

        _, clean_cache = model.run_with_cache(
            clean_tokens, names_filter=lambda n: "hook_z" in n
        )

        clean_ld = logit_diff_from_logits(
            model(clean_tokens), correct_ids[i], incorrect_ids[i]
        )
        corrupt_ld = logit_diff_from_logits(
            model(corrupt_tokens), correct_ids[i], incorrect_ids[i]
        )
        gap = clean_ld - corrupt_ld
        if abs(gap) < 1e-8:
            continue

        for layer in range(n_layers):
            for head in range(n_heads):
                def patch_hook(z, hook, _L=layer, _H=head):
                    clean_z = clean_cache[hook.name]
                    seq_len = min(z.shape[1], clean_z.shape[1])
                    z[0, :seq_len, _H, :] = clean_z[0, :seq_len, _H, :]
                    return z

                hook_name = f"blocks.{layer}.attn.hook_z"
                patched_logits = model.run_with_hooks(
                    corrupt_tokens, fwd_hooks=[(hook_name, patch_hook)]
                )
                patched_ld = logit_diff_from_logits(
                    patched_logits, correct_ids[i], incorrect_ids[i]
                )
                head_effects[layer, head] += (patched_ld - corrupt_ld) / gap

        n_valid += 1

    if n_valid > 0:
        head_effects /= n_valid

    flat = []
    for layer in range(n_layers):
        for head in range(n_heads):
            flat.append((head_effects[layer, head], layer, head))
    flat.sort(reverse=True)

    return {(layer, head) for _, layer, head in flat[:k]}


@torch.no_grad()
def compute_layer_histogram(circuit_heads: set[tuple[int, int]],
                            n_layers: int) -> np.ndarray:
    """Return a normalized histogram of circuit head counts per layer."""
    hist = np.zeros(n_layers)
    for layer, _ in circuit_heads:
        hist[layer] += 1
    total = hist.sum()
    if total > 0:
        hist = hist / total
    return hist


@torch.no_grad()
def compute_sv_distribution(model, circuit_heads: set[tuple[int, int]],
                            layer: int) -> np.ndarray:
    """Return normalized singular value distribution of W_OV for circuit heads at a layer."""
    heads_in_layer = [h for l, h in circuit_heads if l == layer]
    if not heads_in_layer:
        return np.array([])

    all_svs = []
    for head in heads_in_layer:
        w_ov = model.W_V[layer, head] @ model.W_O[layer, head]  # (d_model, d_model)
        svs = torch.linalg.svdvals(w_ov.float()).cpu().numpy()
        total = svs.sum()
        if total > 0:
            svs = svs / total
        all_svs.append(svs)

    if not all_svs:
        return np.array([])
    max_len = max(len(s) for s in all_svs)
    padded = [np.pad(s, (0, max_len - len(s))) for s in all_svs]
    return np.mean(padded, axis=0)


@torch.no_grad()
def run_cross_model_invariance(model_names: list[str], tasks: list[str],
                               device: str, n_prompts: int = 40) -> list[EvalResult]:
    results = []

    for task in tasks:
        small_circuit = get_circuit_heads(task)
        if not small_circuit:
            log(f"  {task}: no circuit for gpt2-small, skipping")
            continue

        k = len(small_circuit)
        log(f"  {task}: gpt2-small circuit has {k} heads")

        per_model = {}

        for model_name in model_names:
            log(f"    Loading {model_name}...")
            model = load_model(model_name, device)
            tokenizer = model.tokenizer
            n_layers = model.cfg.n_layers

            prompts = generate_prompts(task, tokenizer, n_prompts)
            if not prompts:
                log(f"    {model_name}: no prompts, skipping")
                continue
            correct_ids, incorrect_ids = get_token_ids(prompts, tokenizer)
            if not correct_ids:
                log(f"    {model_name}: no valid token ids, skipping")
                continue

            if model_name == "gpt2":
                circuit_heads = small_circuit
                log(f"    {model_name}: using known circuit ({len(circuit_heads)} heads)")
            else:
                log(f"    {model_name}: discovering circuit via patching (top-{k})...")
                circuit_heads = discover_circuit_by_patching(
                    model, prompts, correct_ids, incorrect_ids, k
                )
                log(f"    {model_name}: found {len(circuit_heads)} heads")

            mean_z = calibrate_mean_z(model, prompts, n_calibration=min(50, len(prompts)))
            faith = compute_faithfulness(
                model, prompts, correct_ids, incorrect_ids, circuit_heads, mean_z
            )
            log(f"    {model_name}: faithfulness={faith:.3f}")

            layer_hist = compute_layer_histogram(circuit_heads, n_layers)

            per_model[model_name] = {
                "circuit_heads": sorted(circuit_heads),
                "faithfulness": faith,
                "n_layers": n_layers,
                "layer_histogram": layer_hist,
                "n_params": MODEL_PARAM_COUNTS.get(model_name, 0),
            }

        if len(per_model) < 2:
            log(f"  {task}: fewer than 2 models succeeded, skipping")
            continue

        model_order = [m for m in model_names if m in per_model]
        faithfulness_values = [per_model[m]["faithfulness"] for m in model_order]
        param_counts = [per_model[m]["n_params"] for m in model_order]

        # --- Metric #90: Configural invariance ---
        configural_corrs = {}
        if "gpt2" in per_model:
            small_hist = per_model["gpt2"]["layer_histogram"]
            for mname in model_order:
                if mname == "gpt2":
                    continue
                other_hist = per_model[mname]["layer_histogram"]
                n_small = len(small_hist)
                n_other = len(other_hist)
                max_shared = min(n_small, n_other)
                small_trunc = small_hist[:max_shared]
                other_trunc = other_hist[:max_shared]
                if np.std(small_trunc) < 1e-10 or np.std(other_trunc) < 1e-10:
                    corr_val = 0.0
                else:
                    corr_val, _ = sp_stats.spearmanr(small_trunc, other_trunc)
                    if np.isnan(corr_val):
                        corr_val = 0.0
                configural_corrs[mname] = float(corr_val)

        mean_configural = float(np.mean(list(configural_corrs.values()))) if configural_corrs else 0.0

        results.append(EvalResult(
            metric_id="C38.90_configural_invariance",
            value=mean_configural,
            n_samples=len(model_order),
            metadata={
                "task": task,
                "pairwise_correlations": configural_corrs,
                "models": model_order,
            },
        ))
        log(f"    #90 configural: mean_corr={mean_configural:.3f}")

        # --- Metric #91: Metric invariance ---
        if len(faithfulness_values) >= 3:
            ranks = sp_stats.rankdata(faithfulness_values)
            ideal_ranks = np.arange(1, len(ranks) + 1, dtype=float)
            rank_corr, rank_pval = sp_stats.spearmanr(ranks, ideal_ranks)
            if np.isnan(rank_corr):
                rank_corr = 0.0
                rank_pval = 1.0
        else:
            rank_corr = 1.0 if len(faithfulness_values) == 2 else 0.0
            rank_pval = 1.0

        per_model_faith = {m: per_model[m]["faithfulness"] for m in model_order}
        results.append(EvalResult(
            metric_id="C38.91_metric_invariance",
            value=float(rank_corr),
            n_samples=len(model_order),
            metadata={
                "task": task,
                "faithfulness_per_model": per_model_faith,
                "rank_correlation": float(rank_corr),
                "rank_pvalue": float(rank_pval),
                "ranking_preserved": bool(rank_pval < 0.05) if len(faithfulness_values) >= 3 else None,
                "models": model_order,
            },
        ))
        log(f"    #91 metric: rank_corr={rank_corr:.3f}")

        # --- Metric #92: Scalar invariance ---
        faith_mean = float(np.mean(faithfulness_values))
        faith_std = float(np.std(faithfulness_values))
        faith_range = float(max(faithfulness_values) - min(faithfulness_values))

        results.append(EvalResult(
            metric_id="C38.92_scalar_invariance",
            value=faith_mean,
            n_samples=len(model_order),
            metadata={
                "task": task,
                "faithfulness_mean": faith_mean,
                "faithfulness_std": faith_std,
                "faithfulness_range": faith_range,
                "faithfulness_per_model": per_model_faith,
                "models": model_order,
            },
        ))
        log(f"    #92 scalar: mean={faith_mean:.3f} std={faith_std:.3f}")

        # --- Metric #93: Cross-model weight alignment ---
        sv_similarities = {}
        if "gpt2" in per_model:
            small_model = load_model("gpt2", device)
            small_heads = per_model["gpt2"]["circuit_heads"]
            small_layers_used = sorted({l for l, _ in small_heads})

            for mname in model_order:
                if mname == "gpt2":
                    continue
                other_model = load_model(mname, device)
                other_heads = per_model[mname]["circuit_heads"]
                layer_sims = []

                for layer in small_layers_used:
                    if layer >= per_model[mname]["n_layers"]:
                        continue
                    sv_small = compute_sv_distribution(
                        small_model, set(small_heads), layer
                    )
                    sv_other = compute_sv_distribution(
                        other_model, set(other_heads), layer
                    )
                    if len(sv_small) == 0 or len(sv_other) == 0:
                        continue
                    min_len = min(len(sv_small), len(sv_other))
                    sv_s = sv_small[:min_len]
                    sv_o = sv_other[:min_len]
                    norm_s = np.linalg.norm(sv_s)
                    norm_o = np.linalg.norm(sv_o)
                    if norm_s < 1e-10 or norm_o < 1e-10:
                        continue
                    cos_sim = float(np.dot(sv_s, sv_o) / (norm_s * norm_o))
                    layer_sims.append(cos_sim)

                if layer_sims:
                    sv_similarities[mname] = float(np.mean(layer_sims))

        mean_sv_sim = float(np.mean(list(sv_similarities.values()))) if sv_similarities else 0.0

        results.append(EvalResult(
            metric_id="C38.93_cross_model_weight_alignment",
            value=mean_sv_sim,
            n_samples=len(model_order),
            metadata={
                "task": task,
                "sv_similarity_per_model": sv_similarities,
                "models": model_order,
            },
        ))
        log(f"    #93 weight alignment: mean_sv_sim={mean_sv_sim:.3f}")

        # --- Metric #95: Scale invariance ---
        if len(param_counts) >= 2 and all(p > 0 for p in param_counts):
            log_params = np.log10(param_counts)
            log_faith = np.log10([max(f, 1e-10) for f in faithfulness_values])
            if len(log_params) >= 2:
                slope, intercept, r_value, p_value, std_err = sp_stats.linregress(
                    log_params, log_faith
                )
            else:
                slope, intercept, r_value, p_value, std_err = 0.0, 0.0, 0.0, 1.0, 0.0
        else:
            slope, intercept, r_value, p_value, std_err = 0.0, 0.0, 0.0, 1.0, 0.0

        results.append(EvalResult(
            metric_id="C38.95_scale_invariance",
            value=float(slope),
            n_samples=len(model_order),
            metadata={
                "task": task,
                "log_log_slope": float(slope),
                "log_log_intercept": float(intercept),
                "r_squared": float(r_value ** 2),
                "p_value": float(p_value),
                "std_err": float(std_err),
                "param_counts": {m: per_model[m]["n_params"] for m in model_order},
                "faithfulness_values": {m: per_model[m]["faithfulness"] for m in model_order},
                "models": model_order,
            },
        ))
        log(f"    #95 scale: slope={slope:.4f} R2={r_value**2:.3f}")

    return results


def main():
    parser = parse_common_args("C38: Cross-Model Invariance")
    parser.add_argument("--models", nargs="+",
                        default=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],
                        help="Model names to compare")
    parser.add_argument("--skip-large", action="store_true",
                        help="Only run gpt2 + gpt2-medium (for testing)")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model_names = args.models
    if args.skip_large:
        model_names = [m for m in model_names if m in ("gpt2", "gpt2-medium")]

    log("=" * 60)
    log("C38: CROSS-MODEL INVARIANCE (Metrics #90-95)")
    log("=" * 60)
    log(f"Models: {model_names}")
    log(f"Tasks: {tasks}")

    results = run_cross_model_invariance(model_names, tasks, args.device, args.n_prompts)

    out = args.out or "38_cross_model_invariance.json"
    save_results(results, out)

    log(f"\nDone. {len(results)} metrics computed.")
    for r in results:
        task = r.metadata["task"]
        log(f"  {r.metric_id} [{task}]: {r.value:.4f}")


if __name__ == "__main__":
    main()
