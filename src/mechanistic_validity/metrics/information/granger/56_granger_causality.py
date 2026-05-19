"""Granger Causality Across Layers
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     C07 — Granger Causality
Categories:     information, causal
Validity layer: Internal
Criteria:       I4 Consistency
Establishes:    Whether earlier-layer heads Granger-cause later-layer head activations
Requires:       GPU, model
Doc:            /instruments_v2/information/c07-granger-causality
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Treats the sequence of head activations across layers as a "time series"
(layer = time). For each pair of circuit heads h1 (layer L1) and h2 (layer
L2, L2 > L1), tests whether h1's DLA Granger-causes h2's DLA.

Uses linear regression: does adding h1's DLA improve prediction of h2's DLA
beyond all other circuit heads at layers < L2? Reports F-statistic and
p-value per edge, and compares Granger-significant edges to known circuit
edges.

Usage:
    uv run python 56_granger_causality.py --tasks ioi sva --n-prompts 100
"""

import numpy as np
import torch
from scipy import stats as sp_stats

from mechanistic_validity.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    generate_prompts,
    get_circuit_heads,
    get_circuit_info,
    get_token_ids,
    load_model,
    log,
    logit_diff_from_logits,
    parse_common_args,
    save_results,
)


@torch.no_grad()
def collect_dla_matrix(model, prompts, correct_ids, incorrect_ids):
    """Collect per-head DLA values. Returns (n_prompts, n_layers, n_heads)."""
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


def granger_f_test(y, x_full, x_restricted):
    """Compute F-test for nested linear models.

    Tests whether x_full (includes candidate predictor) explains significantly
    more variance than x_restricted (without candidate predictor).

    Returns (F_statistic, p_value).
    """
    n = len(y)

    # Restricted model
    if x_restricted.shape[1] == 0:
        ss_restricted = np.sum((y - y.mean()) ** 2)
        df_restricted = n - 1
    else:
        X_r = np.column_stack([x_restricted, np.ones(n)])
        coef_r = np.linalg.lstsq(X_r, y, rcond=None)[0]
        resid_r = y - X_r @ coef_r
        ss_restricted = np.sum(resid_r ** 2)
        df_restricted = n - X_r.shape[1]

    # Full model
    X_f = np.column_stack([x_full, np.ones(n)])
    coef_f = np.linalg.lstsq(X_f, y, rcond=None)[0]
    resid_f = y - X_f @ coef_f
    ss_full = np.sum(resid_f ** 2)
    df_full = n - X_f.shape[1]

    # F-statistic
    df_diff = df_restricted - df_full
    if df_diff <= 0 or df_full <= 0 or ss_full <= 0:
        return 0.0, 1.0

    f_stat = ((ss_restricted - ss_full) / df_diff) / (ss_full / df_full)
    p_value = 1.0 - sp_stats.f.cdf(f_stat, df_diff, df_full)
    return float(f_stat), float(p_value)


def run_granger(model, tasks, n_prompts, alpha=0.05):
    tokenizer = model.tokenizer
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    results = []

    for task in tasks:
        circuit, circuit_heads, circuit_edges = get_circuit_info(task)
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
        dla = collect_dla_matrix(model, prompts, correct_ids, incorrect_ids)
        head_list = sorted(circuit_heads)

        edge_results = []
        circuit_significant = 0
        circuit_total = 0
        non_circuit_significant = 0
        non_circuit_total = 0

        for h2 in head_list:
            L2, H2 = h2
            y = dla[:, L2, H2]

            # All circuit heads at earlier layers (the "history")
            earlier_heads = [(L, H) for (L, H) in head_list if L < L2]
            if not earlier_heads:
                continue

            for h1 in earlier_heads:
                L1, H1 = h1

                # Restricted model: all earlier heads except h1
                other_earlier = [(L, H) for (L, H) in earlier_heads if (L, H) != (L1, H1)]
                if other_earlier:
                    x_restricted = np.column_stack([dla[:, L, H] for L, H in other_earlier])
                else:
                    x_restricted = np.empty((len(y), 0))

                # Full model: restricted + h1
                x_full = np.column_stack([x_restricted, dla[:, L1, H1]]) if x_restricted.shape[1] > 0 else dla[:, L1, H1].reshape(-1, 1)

                f_stat, p_val = granger_f_test(y, x_full, x_restricted)
                is_significant = p_val < alpha
                is_edge = (L1, H1, L2, H2) in circuit_edges

                if is_edge:
                    circuit_total += 1
                    if is_significant:
                        circuit_significant += 1
                else:
                    non_circuit_total += 1
                    if is_significant:
                        non_circuit_significant += 1

                edge_results.append({
                    "sender": f"L{L1}H{H1}",
                    "receiver": f"L{L2}H{H2}",
                    "f_stat": f_stat,
                    "p_value": p_val,
                    "significant": is_significant,
                    "is_circuit_edge": is_edge,
                })

        circuit_rate = circuit_significant / max(circuit_total, 1)
        non_circuit_rate = non_circuit_significant / max(non_circuit_total, 1)

        log(f"    Granger significant: circuit={circuit_rate:.2%} "
            f"({circuit_significant}/{circuit_total}) "
            f"non-circuit={non_circuit_rate:.2%} "
            f"({non_circuit_significant}/{non_circuit_total})")

        results.append(EvalResult(
            metric_id="C7.granger_causality",
            value=circuit_rate,
            baseline_random=non_circuit_rate,
            n_samples=len(prompts),
            metadata={
                "task": task,
                "circuit_significance_rate": circuit_rate,
                "non_circuit_significance_rate": non_circuit_rate,
                "n_circuit_edges_tested": circuit_total,
                "n_non_circuit_edges_tested": non_circuit_total,
                "alpha": alpha,
                "top_significant": sorted(
                    [e for e in edge_results if e["significant"]],
                    key=lambda d: d["f_stat"], reverse=True
                )[:10],
            },
        ))

    return results


def main():
    parser = parse_common_args("C7: Granger Causality Across Layers")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Significance level (default: 0.05)")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("C7: GRANGER CAUSALITY ACROSS LAYERS")
    log("=" * 60)

    results = run_granger(model, tasks, args.n_prompts, args.alpha)

    out = args.out or "56_granger_causality.json"
    save_results(results, out)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
