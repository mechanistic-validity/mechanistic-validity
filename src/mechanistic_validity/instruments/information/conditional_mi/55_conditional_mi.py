"""Conditional Mutual Information
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     C02 — Conditional MI
Categories:     information
Validity layer: Internal
Criteria:       I3 Specificity
Establishes:    Distinguishes direct vs mediated dependencies between circuit heads
Requires:       GPU, model
Doc:            /instruments_v2/information/c02-conditional-mi
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For triplets of circuit heads (h1, h2, h3), computes I(h1; h2 | h3) — the MI
between h1 and h2 conditioned on h3. Reveals whether dependencies are direct
or mediated through a third head.

If I(h1;h2) is high but I(h1;h2|h3) is near zero, then h3 mediates the
h1-h2 dependency. Uses residualization: regress out h3's DLA from both h1
and h2, then compute binned MI on residuals.

Reports the fraction of pairwise MI that is mediated vs direct for each task.

Usage:
    uv run python 55_conditional_mi.py --tasks ioi sva --n-prompts 100
"""
import itertools

import numpy as np
import torch

from mechanistic_validity.instruments.common import (
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

N_BINS = 8


def quantile_bin(values, n_bins=N_BINS):
    """Bin values into equal-frequency quantile bins."""
    if len(values) < n_bins:
        return np.zeros(len(values), dtype=int)
    percentiles = np.linspace(0, 100, n_bins + 1)[1:-1]
    thresholds = np.percentile(values, percentiles)
    return np.digitize(values, thresholds)


def binned_mi(x_binned, y_binned):
    """Compute MI from binned variables."""
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


def residualize(x, z):
    """Regress z out of x, return residuals."""
    z_aug = np.column_stack([z.reshape(-1, 1), np.ones(len(z))])
    coef = np.linalg.lstsq(z_aug, x, rcond=None)[0]
    return x - z_aug @ coef


@torch.no_grad()
def collect_dla_values(model, prompts, correct_ids, incorrect_ids, heads):
    """Collect DLA values for specified heads."""
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


def run_conditional_mi(model, tasks, n_prompts, max_triplets=50):
    tokenizer = model.tokenizer
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
        dla = collect_dla_values(model, prompts, correct_ids, incorrect_ids, circuit_heads)

        head_list = sorted(circuit_heads)
        pairs = list(itertools.combinations(head_list, 2))

        # For each pair, find the best mediator among remaining heads
        mediation_results = []
        total_direct_frac = []

        rng = np.random.RandomState(42)
        triplet_count = 0

        for h1, h2 in pairs:
            if triplet_count >= max_triplets:
                break

            x = dla[h1]
            y = dla[h2]
            mi_raw = binned_mi(quantile_bin(x), quantile_bin(y))
            if mi_raw < 0.01:
                continue

            # Test each other head as potential mediator
            other_heads = [h for h in head_list if h != h1 and h != h2]
            best_mediation = 0.0
            best_mediator = None

            for h3 in other_heads:
                z = dla[h3]
                res_x = residualize(x, z)
                res_y = residualize(y, z)
                mi_cond = binned_mi(quantile_bin(res_x), quantile_bin(res_y))
                mediation_frac = 1.0 - mi_cond / max(mi_raw, 1e-8)
                mediation_frac = np.clip(mediation_frac, 0.0, 1.0)

                if mediation_frac > best_mediation:
                    best_mediation = mediation_frac
                    best_mediator = h3

            direct_frac = 1.0 - best_mediation
            total_direct_frac.append(direct_frac)

            mediation_results.append({
                "head_1": f"L{h1[0]}H{h1[1]}",
                "head_2": f"L{h2[0]}H{h2[1]}",
                "mediator": f"L{best_mediator[0]}H{best_mediator[1]}" if best_mediator else None,
                "mi_raw": float(mi_raw),
                "mediation_fraction": float(best_mediation),
                "direct_fraction": float(direct_frac),
            })
            triplet_count += 1

        mean_direct = float(np.mean(total_direct_frac)) if total_direct_frac else 0.0
        mean_mediated = 1.0 - mean_direct

        log(f"    {len(mediation_results)} pairs: direct={mean_direct:.3f} "
            f"mediated={mean_mediated:.3f}")

        results.append(EvalResult(
            metric_id="C5.conditional_mi",
            value=mean_direct,
            n_samples=len(prompts),
            metadata={
                "task": task,
                "mean_direct_fraction": mean_direct,
                "mean_mediated_fraction": mean_mediated,
                "n_pairs_analyzed": len(mediation_results),
                "top_mediated": sorted(
                    mediation_results, key=lambda d: -d["mediation_fraction"]
                )[:5],
                "top_direct": sorted(
                    mediation_results, key=lambda d: -d["direct_fraction"]
                )[:5],
            },
        ))

    return results


def main():
    parser = parse_common_args("C5: Conditional Mutual Information")
    parser.add_argument("--max-triplets", type=int, default=50)
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("C5: CONDITIONAL MUTUAL INFORMATION (Mediation Analysis)")
    log("=" * 60)

    results = run_conditional_mi(model, tasks, args.n_prompts, args.max_triplets)

    out = args.out or "55_conditional_mi.json"
    save_results(results, out)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
