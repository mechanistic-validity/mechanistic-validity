"""Information Bottleneck Analysis
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     C05 — Information Bottleneck
Categories:     information
Validity layer: Construct
Criteria:       C4 Minimality
Establishes:    Maps information compression vs task-relevance across layers
Requires:       GPU, model
Doc:            /instruments_v2/information/c05-info-bottleneck
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For each layer, computes how much information the residual stream retains
about the input (I(X; T_l)) vs how much it preserves about the output
(I(T_l; Y)), where X is the input token identity and Y is the correct/
incorrect label.

Uses residual stream activations projected onto top PCA dimensions. Estimates
MI via binned approximation on PCA scores. Plots the information plane
(I(X;T) vs I(T;Y)) across layers. Circuit-critical layers should show
high I(T;Y) — they preserve task-relevant info.

Usage:
    uv run python 57_info_bottleneck.py --tasks ioi sva --n-prompts 80
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "causal"))
from _common import (
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

N_PCA_DIMS = 10
N_BINS = 8


def quantile_bin(values, n_bins=N_BINS):
    """Bin into equal-frequency bins."""
    if len(values) < n_bins:
        return np.zeros(len(values), dtype=int)
    percentiles = np.linspace(0, 100, n_bins + 1)[1:-1]
    thresholds = np.percentile(values, percentiles)
    return np.digitize(values, thresholds)


def mi_continuous_discrete(X_continuous, y_discrete, n_pca=N_PCA_DIMS, n_bins=N_BINS):
    """Estimate MI between continuous representation and discrete label.

    Projects X onto top PCA dims, bins each dim, and computes MI between
    the joint binned representation and y.
    """
    n = len(y_discrete)
    if n < n_bins * 2:
        return 0.0

    # PCA projection
    X_centered = X_continuous - X_continuous.mean(axis=0)
    n_dims = min(n_pca, X_continuous.shape[1], n - 1)
    try:
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        X_pca = U[:, :n_dims] * S[:n_dims]
    except np.linalg.LinAlgError:
        return 0.0

    # Bin each PCA dimension and compute joint MI
    # Use sum of per-dimension MI as lower bound (ignores joint structure)
    total_mi = 0.0
    for d in range(n_dims):
        x_binned = quantile_bin(X_pca[:, d], n_bins)
        # MI(binned_dim; y)
        n_x = x_binned.max() + 1
        n_y = int(y_discrete.max()) + 1
        joint = np.zeros((n_x, n_y))
        for i in range(n):
            joint[x_binned[i], int(y_discrete[i])] += 1
        joint /= joint.sum()
        px = joint.sum(axis=1)
        py = joint.sum(axis=0)
        mi = 0.0
        for i in range(n_x):
            for j in range(n_y):
                if joint[i, j] > 0 and px[i] > 0 and py[j] > 0:
                    mi += joint[i, j] * np.log2(joint[i, j] / (px[i] * py[j]))
        total_mi += max(mi, 0.0)

    return total_mi


@torch.no_grad()
def collect_residual_streams(model, prompts, correct_ids, incorrect_ids):
    """Collect residual stream at each layer and input/output labels.

    Returns:
        residuals: list of (n_prompts, d_model) arrays, one per layer
        input_tokens: (n_prompts,) last input token ID
        output_labels: (n_prompts,) binary correct=1/incorrect=0 based on logit diff sign
    """
    n_layers = model.cfg.n_layers
    n_valid = min(len(prompts), len(correct_ids))
    d_model = model.cfg.d_model

    residuals = [np.zeros((n_valid, d_model)) for _ in range(n_layers)]
    input_tokens = np.zeros(n_valid, dtype=int)
    output_labels = np.zeros(n_valid, dtype=int)

    for i in range(n_valid):
        tokens = model.to_tokens(prompts[i].text)
        input_tokens[i] = tokens[0, -1].item() % 100  # Mod to limit cardinality
        logits, cache = model.run_with_cache(
            tokens, names_filter=lambda n: "hook_resid_post" in n)

        ld = logit_diff_from_logits(logits, correct_ids[i], incorrect_ids[i])
        output_labels[i] = 1 if ld > 0 else 0

        for L in range(n_layers):
            residuals[L][i] = cache[f"blocks.{L}.hook_resid_post"][0, -1].cpu().numpy()

    return residuals, input_tokens, output_labels


def run_info_bottleneck(model, tasks, n_prompts):
    tokenizer = model.tokenizer
    n_layers = model.cfg.n_layers
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
        residuals, input_tokens, output_labels = collect_residual_streams(
            model, prompts, correct_ids, incorrect_ids)

        # Identify circuit-critical layers
        circuit_layers = {L for L, H in circuit_heads}

        info_plane = []
        for L in range(n_layers):
            i_x_t = mi_continuous_discrete(residuals[L], input_tokens)
            i_t_y = mi_continuous_discrete(residuals[L], output_labels)
            info_plane.append({
                "layer": L,
                "I_X_T": float(i_x_t),
                "I_T_Y": float(i_t_y),
                "is_circuit_layer": L in circuit_layers,
            })

        # Compare I(T;Y) at circuit vs non-circuit layers
        circuit_i_t_y = [p["I_T_Y"] for p in info_plane if p["is_circuit_layer"]]
        non_circuit_i_t_y = [p["I_T_Y"] for p in info_plane if not p["is_circuit_layer"]]

        mean_circuit = float(np.mean(circuit_i_t_y)) if circuit_i_t_y else 0.0
        mean_non_circuit = float(np.mean(non_circuit_i_t_y)) if non_circuit_i_t_y else 0.0

        log(f"    I(T;Y) circuit_layers={mean_circuit:.4f} "
            f"non_circuit_layers={mean_non_circuit:.4f}")

        results.append(EvalResult(
            metric_id="C8.info_bottleneck",
            value=mean_circuit,
            baseline_random=mean_non_circuit,
            n_samples=len(prompts),
            metadata={
                "task": task,
                "info_plane": info_plane,
                "mean_circuit_I_T_Y": mean_circuit,
                "mean_non_circuit_I_T_Y": mean_non_circuit,
                "circuit_layers": sorted(circuit_layers),
                "n_pca_dims": N_PCA_DIMS,
            },
        ))

    return results


def main():
    parser = parse_common_args("C8: Information Bottleneck Analysis")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("C8: INFORMATION BOTTLENECK ANALYSIS")
    log("=" * 60)

    results = run_info_bottleneck(model, tasks, args.n_prompts)

    out = args.out or "57_info_bottleneck.json"
    save_results(results, out)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
