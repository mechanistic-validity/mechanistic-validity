"""Mutual Information Between Heads
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     C01 — Mutual Information
Categories:     information
Validity layer: Internal
Criteria:       I3 Specificity
Establishes:    Quantifies pairwise information sharing between circuit heads
Requires:       GPU, model
Doc:            /instruments_v2/information/c01-mutual-information
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Estimates MI between pairs of circuit heads' DLA values across prompts using
binned MI estimation. Discretizes each head's DLA into equal-frequency bins
and computes MI from the joint histogram.

Reports: MI matrix for circuit heads, mean MI within-circuit vs
between-circuit-and-random heads, and MI-weighted graph structure showing
which head pairs share the most information.

Usage:
    uv run python 54_mutual_information.py --tasks ioi sva --n-prompts 100
"""
import itertools

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

N_BINS = 10


def quantile_bin(values, n_bins=N_BINS):
    """Bin values into equal-frequency quantile bins."""
    if len(values) < n_bins:
        return np.zeros(len(values), dtype=int)
    percentiles = np.linspace(0, 100, n_bins + 1)[1:-1]
    thresholds = np.percentile(values, percentiles)
    return np.digitize(values, thresholds)


def binned_mi(x_binned, y_binned):
    """Compute mutual information from binned variables using joint histogram."""
    n_x = x_binned.max() + 1
    n_y = y_binned.max() + 1
    joint = np.zeros((n_x, n_y))
    for i in range(len(x_binned)):
        joint[x_binned[i], y_binned[i]] += 1
    joint /= joint.sum()

    px = joint.sum(axis=1)
    py = joint.sum(axis=0)

    mi = 0.0
    for i in range(n_x):
        for j in range(n_y):
            if joint[i, j] > 0 and px[i] > 0 and py[j] > 0:
                mi += joint[i, j] * np.log2(joint[i, j] / (px[i] * py[j]))
    return max(mi, 0.0)


@torch.no_grad()
def collect_dla_values(model, prompts, correct_ids, incorrect_ids, heads):
    """Collect DLA values for specified heads.

    Returns dict mapping (L, H) -> array of DLA values across prompts.
    """
    n_valid = min(len(prompts), len(correct_ids))
    dla = {h: np.zeros(n_valid) for h in heads}

    for i in range(n_valid):
        tokens = model.to_tokens(prompts[i].text)
        logits, cache = model.run_with_cache(tokens, names_filter=lambda n: "hook_z" in n)
        diff_dir = model.W_U[:, correct_ids[i]] - model.W_U[:, incorrect_ids[i]]
        for L, H in heads:
            z = cache[f"blocks.{L}.attn.hook_z"]
            dla[(L, H)][i] = (z[0, -1, H] @ model.W_O[L, H] @ diff_dir).item()

    return dla


def run_mutual_information(model, tasks, n_prompts):
    tokenizer = model.tokenizer
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
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

        # Select random non-circuit heads for comparison
        all_heads = {(L, H) for L in range(n_layers) for H in range(n_heads)}
        non_circuit = all_heads - circuit_heads
        rng = np.random.RandomState(42)
        n_random = min(len(circuit_heads), len(non_circuit))
        random_heads = set(
            [tuple(x) for x in rng.permutation(sorted(non_circuit))[:n_random]]
        )

        all_needed = circuit_heads | random_heads
        dla = collect_dla_values(model, prompts, correct_ids, incorrect_ids, all_needed)

        # Bin all DLA values
        binned = {h: quantile_bin(dla[h]) for h in all_needed}

        # MI within circuit
        circuit_list = sorted(circuit_heads)
        within_mi = []
        mi_matrix = {}
        for (h1, h2) in itertools.combinations(circuit_list, 2):
            mi_val = binned_mi(binned[h1], binned[h2])
            within_mi.append(mi_val)
            mi_matrix[f"L{h1[0]}H{h1[1]}-L{h2[0]}H{h2[1]}"] = float(mi_val)

        # MI between circuit and random
        between_mi = []
        random_list = sorted(random_heads)
        for h_c in circuit_list:
            for h_r in random_list:
                mi_val = binned_mi(binned[h_c], binned[h_r])
                between_mi.append(mi_val)

        mean_within = float(np.mean(within_mi)) if within_mi else 0.0
        mean_between = float(np.mean(between_mi)) if between_mi else 0.0
        ratio = mean_within / max(mean_between, 1e-8)

        # Top MI edges (MI-weighted graph)
        top_edges = sorted(mi_matrix.items(), key=lambda x: -x[1])[:10]

        log(f"    MI: within={mean_within:.4f} between={mean_between:.4f} ratio={ratio:.2f}")

        results.append(EvalResult(
            metric_id="C4.mutual_information",
            value=ratio,
            baseline_random=1.0,
            n_samples=len(prompts),
            metadata={
                "task": task,
                "mean_within_circuit_mi": mean_within,
                "mean_between_circuit_random_mi": mean_between,
                "ratio": ratio,
                "n_circuit_pairs": len(within_mi),
                "n_between_pairs": len(between_mi),
                "top_mi_edges": [{"edge": e, "mi": v} for e, v in top_edges],
                "n_bins": N_BINS,
            },
        ))

    return results


def main():
    parser = parse_common_args("C4: Mutual Information Between Heads")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("C4: MUTUAL INFORMATION BETWEEN HEADS")
    log("=" * 60)

    results = run_mutual_information(model, tasks, args.n_prompts)

    out = args.out or "54_mutual_information.json"
    save_results(results, out)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
