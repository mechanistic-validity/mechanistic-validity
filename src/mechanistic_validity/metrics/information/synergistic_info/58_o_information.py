"""O-Information (Synergy vs Redundancy)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     C06 — O-Information
Categories:     information
Validity layer: Internal
Criteria:       I3 Specificity
Establishes:    Whether circuit heads are synergy- or redundancy-dominated
Requires:       GPU, model
Doc:            /instruments_v2/information/c06-o-information
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Computes the O-information (Rosas et al. 2019) of circuit head DLA values:
    Omega = (n-2)*H(all) + sum_i H(Xi) - sum_i H(all \ Xi)

Positive Omega indicates redundancy-dominated: heads carry overlapping
information about the task output. Negative Omega indicates synergy-dominated:
heads carry information jointly that no subset carries alone.

Compares Omega for the full circuit head set vs random subsets of the same
size to test whether circuit heads are specifically synergistic or redundant.

Usage:
    uv run python 58_o_information.py --tasks ioi sva --n-prompts 100
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

N_BINS = 6


def quantile_bin(values, n_bins=N_BINS):
    """Bin into equal-frequency bins."""
    if len(values) < n_bins:
        return np.zeros(len(values), dtype=int)
    percentiles = np.linspace(0, 100, n_bins + 1)[1:-1]
    thresholds = np.percentile(values, percentiles)
    return np.digitize(values, thresholds)


def entropy_binned(variables_binned):
    """Compute joint entropy of a set of binned variables.

    variables_binned: list of 1D integer arrays (same length).
    """
    n = len(variables_binned[0])
    # Build joint distribution via tuple hashing
    counts = {}
    for i in range(n):
        key = tuple(v[i] for v in variables_binned)
        counts[key] = counts.get(key, 0) + 1

    h = 0.0
    for c in counts.values():
        p = c / n
        if p > 0:
            h -= p * np.log2(p)
    return h


def o_information(variables_binned):
    """Compute O-information: Omega = (n-2)*H(X1,...,Xn) + sum H(Xi) - sum H(X\\Xi).

    variables_binned: list of n 1D integer arrays.
    Returns Omega. Positive = redundancy, negative = synergy.
    """
    n = len(variables_binned)
    if n < 3:
        return 0.0

    h_all = entropy_binned(variables_binned)

    sum_h_individual = sum(entropy_binned([v]) for v in variables_binned)

    sum_h_leave_one_out = 0.0
    for i in range(n):
        subset = [v for j, v in enumerate(variables_binned) if j != i]
        sum_h_leave_one_out += entropy_binned(subset)

    omega = (n - 2) * h_all + sum_h_individual - sum_h_leave_one_out
    return omega


@torch.no_grad()
def collect_dla_all_heads(model, prompts, correct_ids, incorrect_ids):
    """Collect DLA for all heads. Returns (n_prompts, n_layers * n_heads)."""
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    n_valid = min(len(prompts), len(correct_ids))
    dla = np.zeros((n_valid, n_layers, n_heads))

    for i in range(n_valid):
        tokens = model.to_tokens(prompts[i].text)
        logits, cache = model.run_with_cache(tokens, names_filter=lambda n: "hook_z" in n)
        diff_dir = model.W_U[:, correct_ids[i]] - model.W_U[:, incorrect_ids[i]]
        for L in range(n_layers):
            z = cache[f"blocks.{L}.attn.hook_z"]
            for H in range(n_heads):
                dla[i, L, H] = (z[0, -1, H] @ model.W_O[L, H] @ diff_dir).item()

    return dla


def run_o_information(model, tasks, n_prompts, n_random_baselines=50):
    tokenizer = model.tokenizer
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    results = []

    for task in tasks:
        circuit_heads = get_circuit_heads(task)
        if not circuit_heads or len(circuit_heads) < 3:
            log(f"  {task}: need >= 3 circuit heads, skipping")
            continue

        prompts = generate_prompts(task, tokenizer, n_prompts)
        if not prompts:
            continue
        correct_ids, incorrect_ids = get_token_ids(prompts, tokenizer)
        if not correct_ids:
            continue

        log(f"  {task} ({len(circuit_heads)} heads, {len(prompts)} prompts)...")
        dla = collect_dla_all_heads(model, prompts, correct_ids, incorrect_ids)

        # Bin circuit head DLA values
        head_list = sorted(circuit_heads)
        circuit_binned = [quantile_bin(dla[:, L, H]) for L, H in head_list]

        # Compute O-information for circuit heads
        # Limit to max 12 heads to keep computation tractable
        max_heads = min(len(circuit_binned), 12)
        circuit_subset = circuit_binned[:max_heads]
        omega_circuit = o_information(circuit_subset)

        # Random baselines: same number of heads, random selection
        rng = np.random.RandomState(42)
        all_heads = [(L, H) for L in range(n_layers) for H in range(n_heads)]
        non_circuit = [h for h in all_heads if h not in circuit_heads]

        omega_random_list = []
        for _ in range(n_random_baselines):
            rand_idx = rng.choice(len(non_circuit), max_heads, replace=False)
            rand_heads = [non_circuit[i] for i in rand_idx]
            rand_binned = [quantile_bin(dla[:, L, H]) for L, H in rand_heads]
            omega_random_list.append(o_information(rand_binned))

        mean_omega_random = float(np.mean(omega_random_list))
        std_omega_random = float(np.std(omega_random_list))
        z_score = (omega_circuit - mean_omega_random) / max(std_omega_random, 1e-8)

        interpretation = "redundancy" if omega_circuit > 0 else "synergy"
        log(f"    Omega={omega_circuit:.4f} ({interpretation}) "
            f"random={mean_omega_random:.4f}+/-{std_omega_random:.4f} "
            f"z={z_score:.2f}")

        results.append(EvalResult(
            metric_id="C9.o_information",
            value=float(omega_circuit),
            baseline_random=mean_omega_random,
            n_samples=len(prompts),
            metadata={
                "task": task,
                "omega_circuit": float(omega_circuit),
                "omega_random_mean": mean_omega_random,
                "omega_random_std": std_omega_random,
                "z_score": z_score,
                "interpretation": interpretation,
                "n_heads_used": max_heads,
                "n_circuit_heads": len(circuit_heads),
                "n_random_baselines": n_random_baselines,
            },
        ))

    return results


def main():
    parser = parse_common_args("C9: O-Information (Synergy vs Redundancy)")
    parser.add_argument("--n-random", type=int, default=50,
                        help="Number of random baseline samples")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("C9: O-INFORMATION (SYNERGY vs REDUNDANCY)")
    log("=" * 60)

    results = run_o_information(model, tasks, args.n_prompts, args.n_random)

    out = args.out or "58_o_information.json"
    save_results(results, out)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
