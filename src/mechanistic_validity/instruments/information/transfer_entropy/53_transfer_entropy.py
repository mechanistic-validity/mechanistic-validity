"""Transfer Entropy Between Layers
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     C03 — Transfer Entropy
Categories:     information, causal
Validity layer: Internal
Criteria:       I4 Consistency
Establishes:    Directional information flow between circuit heads across layers
Requires:       GPU, model
Doc:            /instruments_v2/information/c03-transfer-entropy
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Estimates directional information flow between circuit heads across layers.
For head h1 at layer L1 and head h2 at layer L2 (L1 < L2), estimates
TE(h1->h2) as the partial correlation of h1's DLA with h2's DLA, controlling
for h2's input context (other earlier-layer heads). Compares transfer entropy
for known circuit edges vs non-edges.

Uses DLA (dot product of head output with correct-incorrect unembedding
direction) as a scalar summary. Partial correlations serve as a tractable
proxy for true transfer entropy.

Usage:
    uv run python 53_transfer_entropy.py --tasks ioi sva --n-prompts 100
"""
import itertools

import numpy as np
import torch
from scipy import stats as sp_stats

from mechanistic_validity.instruments.common import (
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
    """Collect per-head DLA values across prompts.

    Returns:
        dla: (n_prompts, n_layers, n_heads) array of signed DLA values
    """
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


def partial_correlation(x, y, Z):
    """Compute partial correlation of x and y controlling for columns of Z.

    Uses OLS residualization. Returns (partial_r, p_value).
    """
    if Z is None or Z.shape[1] == 0:
        r, p = sp_stats.pearsonr(x, y)
        return r, p
    Z_aug = np.column_stack([Z, np.ones(len(Z))])
    coef_x = np.linalg.lstsq(Z_aug, x, rcond=None)[0]
    coef_y = np.linalg.lstsq(Z_aug, y, rcond=None)[0]
    res_x = x - Z_aug @ coef_x
    res_y = y - Z_aug @ coef_y
    if res_x.std() < 1e-10 or res_y.std() < 1e-10:
        return 0.0, 1.0
    r, p = sp_stats.pearsonr(res_x, res_y)
    return r, p


def compute_transfer_entropy_proxy(dla, circuit_heads, circuit_edges, n_layers, n_heads):
    """Compute partial-correlation-based TE proxy for circuit and non-circuit edges.

    For each directed pair (h1 at L1) -> (h2 at L2) with L1 < L2:
      TE proxy = partial_corr(h1_dla, h2_dla | all other heads at layers < L2)
    """
    head_list = sorted(circuit_heads)
    all_heads = [(L, H) for L in range(n_layers) for H in range(n_heads)]
    n_prompts = dla.shape[0]

    circuit_te = []
    non_circuit_te = []
    edge_details = []

    for (L1, H1), (L2, H2) in itertools.combinations(head_list, 2):
        if L1 >= L2:
            continue

        x = dla[:, L1, H1]
        y = dla[:, L2, H2]

        # Conditioning set: all circuit heads at layers strictly between L1 and L2
        cond_heads = [(L, H) for (L, H) in head_list if L1 < L < L2]
        if cond_heads:
            Z = np.column_stack([dla[:, L, H] for L, H in cond_heads])
        else:
            Z = None

        r, p = partial_correlation(x, y, Z)
        te_proxy = r ** 2  # Squared partial correlation as TE proxy

        is_edge = (L1, H1, L2, H2) in circuit_edges
        if is_edge:
            circuit_te.append(te_proxy)
        else:
            non_circuit_te.append(te_proxy)

        edge_details.append({
            "sender": f"L{L1}H{H1}",
            "receiver": f"L{L2}H{H2}",
            "te_proxy": float(te_proxy),
            "partial_r": float(r),
            "p_value": float(p),
            "is_circuit_edge": is_edge,
        })

    return circuit_te, non_circuit_te, edge_details


def run_transfer_entropy(model, tasks, n_prompts):
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

        circuit_te, non_circuit_te, details = compute_transfer_entropy_proxy(
            dla, circuit_heads, circuit_edges, n_layers, n_heads)

        mean_circuit = float(np.mean(circuit_te)) if circuit_te else 0.0
        mean_non_circuit = float(np.mean(non_circuit_te)) if non_circuit_te else 0.0
        ratio = mean_circuit / max(mean_non_circuit, 1e-8)

        log(f"    TE proxy: circuit={mean_circuit:.4f} non-circuit={mean_non_circuit:.4f} "
            f"ratio={ratio:.2f}")

        results.append(EvalResult(
            metric_id="C1.transfer_entropy",
            value=ratio,
            baseline_random=1.0,
            n_samples=len(prompts),
            metadata={
                "task": task,
                "mean_circuit_te": mean_circuit,
                "mean_non_circuit_te": mean_non_circuit,
                "ratio": ratio,
                "n_circuit_edges": len(circuit_te),
                "n_non_circuit_edges": len(non_circuit_te),
                "top_edges": sorted(details, key=lambda d: -d["te_proxy"])[:10],
            },
        ))

    return results


def main():
    parser = parse_common_args("C1: Transfer Entropy Between Layers")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("C1: TRANSFER ENTROPY BETWEEN LAYERS")
    log("=" * 60)

    results = run_transfer_entropy(model, tasks, args.n_prompts)

    out = args.out or "53_transfer_entropy.json"
    save_results(results, out)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
