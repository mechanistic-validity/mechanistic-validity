"""Observational Circuit Discovery (oCSE + Stability Selection)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     C08 — OCSE
Categories:     information, causal
Validity layer: Internal
Criteria:       I1 Necessity
Establishes:    Discovers circuit structure from observational data without interventions
Requires:       GPU, model
Doc:            /instruments_v2/information/c08-ocse
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Two complementary discovery methods, both purely observational:
1. Stability selection via bootstrap LassoCV — robust sparse variable
   selection that handles multicollinearity across 144 head DLAs.
2. Greedy oCSE (conditional mutual information) with permutation threshold.

Both use DLA features (signed head contribution to logit diff) rather than
norms, since sign carries essential information about head function.

Usage:
    uv run python 07_ocse.py --tasks ioi sva --n-prompts 200
"""

import numpy as np
import torch
from scipy import stats as sp_stats
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

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


@torch.no_grad()
def collect_head_activations(model, prompts, correct_ids, incorrect_ids):
    """Collect per-head DLA (direct logit attribution) and logit diffs.

    Returns:
        activations: (n_prompts, n_layers * n_heads) signed DLA per head
        logit_diffs: (n_prompts,) logit differences
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    n_valid = min(len(prompts), len(correct_ids))

    activations = np.zeros((n_valid, n_layers * n_heads))
    logit_diffs = np.zeros(n_valid)

    for i in range(n_valid):
        tokens = model.to_tokens(prompts[i].text)
        logits, cache = model.run_with_cache(tokens, names_filter=lambda n: "hook_z" in n)

        logit_diffs[i] = logit_diff_from_logits(logits, correct_ids[i], incorrect_ids[i])

        diff_dir = model.W_U[:, correct_ids[i]] - model.W_U[:, incorrect_ids[i]]
        for L in range(n_layers):
            z = cache[f"blocks.{L}.attn.hook_z"]
            for H in range(n_heads):
                activations[i, L * n_heads + H] = (
                    z[0, -1, H] @ model.W_O[L, H] @ diff_dir
                ).item()

    return activations, logit_diffs


def stability_selection(activations: np.ndarray, target: np.ndarray,
                        n_bootstrap: int = 50,
                        threshold: float = 0.5) -> list[tuple[int, float, float]]:
    """Bootstrap LassoCV stability selection.

    Returns list of (variable_idx, selection_frequency, mean_coefficient)
    for variables selected in >threshold fraction of bootstrap runs.
    """
    n_samples, n_vars = activations.shape
    scaler = StandardScaler()
    X = scaler.fit_transform(activations)
    y = (target - target.mean()) / (target.std() + 1e-8)

    selection_counts = np.zeros(n_vars)
    coef_sum = np.zeros(n_vars)

    rng = np.random.RandomState(42)
    for _ in range(n_bootstrap):
        idx = rng.choice(n_samples, n_samples, replace=True)
        lasso = LassoCV(cv=5, max_iter=10000, n_jobs=1)
        lasso.fit(X[idx], y[idx])
        nonzero = np.abs(lasso.coef_) > 1e-6
        selection_counts += nonzero
        coef_sum += lasso.coef_

    freq = selection_counts / n_bootstrap
    mean_coef = coef_sum / n_bootstrap

    selected = np.where(freq > threshold)[0]
    order = np.argsort(-freq[selected])
    return [(int(selected[i]), float(freq[selected[i]]), float(mean_coef[selected[i]]))
            for i in order]


def gaussian_cmi(x: np.ndarray, y: np.ndarray, z: np.ndarray | None = None) -> float:
    """Estimate conditional mutual information I(X;Y|Z) using Gaussian assumption."""
    if len(x) < 5:
        return 0.0

    if z is None:
        r, _ = sp_stats.pearsonr(x, y)
        r = np.clip(r, -0.999, 0.999)
        return -0.5 * np.log(1 - r ** 2)

    from numpy.linalg import lstsq
    Z = z.reshape(len(z), -1) if z.ndim > 1 else z.reshape(-1, 1)
    Z_aug = np.column_stack([Z, np.ones(len(Z))])

    res_x = x - Z_aug @ lstsq(Z_aug, x, rcond=None)[0]
    res_y = y - Z_aug @ lstsq(Z_aug, y, rcond=None)[0]

    r, _ = sp_stats.pearsonr(res_x, res_y)
    r = np.clip(r, -0.999, 0.999)
    return -0.5 * np.log(1 - r ** 2)


def compute_ocse_parents(activations: np.ndarray, target: np.ndarray,
                         max_parents: int = 20,
                         n_permutations: int = 100) -> list[tuple[int, float]]:
    """Greedy forward selection with permutation-calibrated threshold."""
    n_samples, n_vars = activations.shape
    rng = np.random.RandomState(42)

    perm_cmis = []
    for _ in range(n_permutations):
        perm_target = target[rng.permutation(n_samples)]
        best_perm = max(gaussian_cmi(activations[:, v], perm_target) for v in range(n_vars))
        perm_cmis.append(best_perm)
    threshold = float(np.percentile(perm_cmis, 95))

    selected = []
    remaining = list(range(n_vars))
    scores = []

    for _ in range(max_parents):
        best_gain = -1.0
        best_var = None
        conditioning = activations[:, selected] if selected else None

        for var in remaining:
            cmi = gaussian_cmi(activations[:, var], target, conditioning)
            if cmi > best_gain:
                best_gain = cmi
                best_var = var

        if best_var is None or best_gain < threshold:
            break

        selected.append(best_var)
        remaining.remove(best_var)
        scores.append((best_var, best_gain))

    return scores


def _compute_f1(circuit_heads: set, discovered_indices: set, n_heads: int):
    circuit_indices = {L * n_heads + H for L, H in circuit_heads}
    tp = len(circuit_indices & discovered_indices)
    fp = len(discovered_indices - circuit_indices)
    fn = len(circuit_indices - discovered_indices)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1, precision, recall


def run_ocse(model, tasks: list[str], n_prompts: int = 200) -> list[EvalResult]:
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

        activations, logit_diffs = collect_head_activations(
            model, prompts, correct_ids, incorrect_ids)

        log(f"    Running stability selection (50 bootstrap LassoCV)...")
        stable_vars = stability_selection(activations, logit_diffs)
        stable_indices = {idx for idx, _, _ in stable_vars}
        s_f1, s_p, s_r = _compute_f1(circuit_heads, stable_indices, n_heads)

        log(f"    Running oCSE (permutation-calibrated)...")
        ocse_parents = compute_ocse_parents(activations, logit_diffs)
        ocse_indices = {idx for idx, _ in ocse_parents}
        o_f1, o_p, o_r = _compute_f1(circuit_heads, ocse_indices, n_heads)

        combined_indices = stable_indices | ocse_indices
        c_f1, c_p, c_r = _compute_f1(circuit_heads, combined_indices, n_heads)

        def _idx_to_head(idx):
            return f"L{idx // n_heads}H{idx % n_heads}"

        log(f"    stability: F1={s_f1:.3f} (P={s_p:.3f} R={s_r:.3f}) n={len(stable_indices)}")
        log(f"    oCSE:      F1={o_f1:.3f} (P={o_p:.3f} R={o_r:.3f}) n={len(ocse_indices)}")
        log(f"    combined:  F1={c_f1:.3f} (P={c_p:.3f} R={c_r:.3f}) n={len(combined_indices)}")

        results.append(EvalResult(
            metric_id="C7.ocse",
            value=s_f1,
            n_samples=len(prompts),
            metadata={
                "task": task,
                "precision": s_p,
                "recall": s_r,
                "stability_selection": {
                    "f1": s_f1, "precision": s_p, "recall": s_r,
                    "discovered": [
                        {"head": _idx_to_head(idx), "freq": freq, "coef": coef}
                        for idx, freq, coef in stable_vars
                    ],
                },
                "ocse": {
                    "f1": o_f1, "precision": o_p, "recall": o_r,
                    "discovered": [
                        {"head": _idx_to_head(idx), "score": score}
                        for idx, score in ocse_parents
                    ],
                },
                "combined": {
                    "f1": c_f1, "precision": c_p, "recall": c_r,
                    "n_discovered": len(combined_indices),
                },
                "n_circuit_heads": len(circuit_heads),
                "n_discovered": len(stable_indices),
            },
        ))

    return results


def main():
    parser = parse_common_args("C7: Observational Circuit Discovery")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("C7: OBSERVATIONAL CIRCUIT DISCOVERY (Stability Selection + oCSE)")
    log("=" * 60)

    results = run_ocse(model, tasks, args.n_prompts)

    out = args.out or "07_ocse.json"
    save_results(results, out)

    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
