"""Inter-Rater Reliability
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     F08 — Inter-Rater
Categories:     measurement
Validity layer: Measurement
Criteria:       M1 Reliability
Establishes:    Independent circuit discovery methods agree on circuit membership
Requires:       CPU, data-only
Doc:            /instruments_v2/measurement/f08-inter-rater
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Treats circuit discovery methods as independent "raters" and measures
their agreement on which heads belong to the circuit for each task.

Three methods rank all heads:
  (1) Activation patching -- logit-diff restoration per head.
  (2) DLA (Direct Logit Attribution) -- projection of head output onto
      the correct-incorrect unembedding direction.
  (3) Weight-space OV norm -- Frobenius norm of W_OV projected onto
      the answer direction.

Agreement metrics:
  - Kendall's W (concordance) across all three ranked lists.
  - Pairwise Spearman correlations between raters.
  - Cohen's kappa on binary in/out-of-circuit classification (top-k).

Usage:
    uv run python 59_inter_rater.py --tasks ioi sva --device cpu
    uv run python 59_inter_rater.py --device cuda --n-prompts 40
"""

import numpy as np
import torch
from scipy import stats as sp_stats

from mechval.metrics.common import (
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


def _cohens_kappa(y1: np.ndarray, y2: np.ndarray) -> float:
    """Compute Cohen's kappa for two binary arrays."""
    n = len(y1)
    if n == 0:
        return 0.0
    agree = np.sum(y1 == y2)
    p_o = agree / n
    p1_pos = np.mean(y1)
    p2_pos = np.mean(y2)
    p_e = p1_pos * p2_pos + (1 - p1_pos) * (1 - p2_pos)
    if abs(1 - p_e) < 1e-10:
        return 1.0 if p_o == 1.0 else 0.0
    return (p_o - p_e) / (1 - p_e)


def _kendalls_w(rankings: np.ndarray) -> float:
    """Compute Kendall's W (coefficient of concordance).

    rankings: shape (m_raters, n_items), each row is a rank vector.
    """
    m, n = rankings.shape
    if m < 2 or n < 2:
        return 0.0
    rank_sums = rankings.sum(axis=0)
    mean_rank_sum = rank_sums.mean()
    ss = np.sum((rank_sums - mean_rank_sum) ** 2)
    w = 12 * ss / (m ** 2 * (n ** 3 - n))
    return float(w)


@torch.no_grad()
def rank_by_patching(model, prompts, correct_ids, incorrect_ids) -> np.ndarray:
    """Rank heads by activation patching effect (higher = more important)."""
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    effects = np.zeros(n_layers * n_heads)
    n_eval = min(len(prompts), len(correct_ids), 10)

    for i in range(n_eval):
        tokens = model.to_tokens(prompts[i].text)
        _, cache = model.run_with_cache(tokens, names_filter=lambda n: "hook_z" in n)
        clean_ld = logit_diff_from_logits(model(tokens), correct_ids[i], incorrect_ids[i])

        for L in range(n_layers):
            for H in range(n_heads):
                def ablate_hook(z, hook, _L=L, _H=H):
                    z[0, :, _H, :] = 0.0
                    return z
                abl_logits = model.run_with_hooks(
                    tokens, fwd_hooks=[(f"blocks.{L}.attn.hook_z", ablate_hook)]
                )
                abl_ld = logit_diff_from_logits(abl_logits, correct_ids[i], incorrect_ids[i])
                effects[L * n_heads + H] += clean_ld - abl_ld

    return effects


@torch.no_grad()
def rank_by_dla(model, prompts, correct_ids, incorrect_ids) -> np.ndarray:
    """Rank heads by Direct Logit Attribution onto correct-incorrect direction."""
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    effects = np.zeros(n_layers * n_heads)
    n_eval = min(len(prompts), len(correct_ids), 10)

    W_U = model.W_U  # (d_model, d_vocab)

    for i in range(n_eval):
        tokens = model.to_tokens(prompts[i].text)
        _, cache = model.run_with_cache(
            tokens, names_filter=lambda n: "hook_result" in n
        )
        direction = (W_U[:, correct_ids[i]] - W_U[:, incorrect_ids[i]]).float()
        direction = direction / (direction.norm() + 1e-10)

        for L in range(n_layers):
            result = cache[f"blocks.{L}.attn.hook_result"][0, -1]  # (n_heads, d_head)
            W_O = model.W_O[L]  # (n_heads, d_head, d_model)
            for H in range(n_heads):
                head_out = result[H] @ W_O[H]  # (d_model,)
                dla = (head_out.float() @ direction).item()
                effects[L * n_heads + H] += abs(dla)

    return effects


@torch.no_grad()
def rank_by_ov_norm(model, correct_ids, incorrect_ids) -> np.ndarray:
    """Rank heads by W_OV norm projected onto answer direction."""
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    effects = np.zeros(n_layers * n_heads)

    W_U = model.W_U  # (d_model, d_vocab)

    for i in range(min(len(correct_ids), 10)):
        direction = (W_U[:, correct_ids[i]] - W_U[:, incorrect_ids[i]]).float()
        direction = direction / (direction.norm() + 1e-10)

        for L in range(n_layers):
            W_V = model.W_V[L]  # (n_heads, d_model, d_head)
            W_O = model.W_O[L]  # (n_heads, d_head, d_model)
            for H in range(n_heads):
                W_OV = W_V[H] @ W_O[H]  # (d_model, d_model)
                projected_norm = (W_OV.float() @ direction).norm().item()
                effects[L * n_heads + H] += projected_norm

    return effects


@torch.no_grad()
def main():
    parser = parse_common_args("F59: Inter-Rater Reliability")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)
    tokenizer = model.tokenizer
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    n_total = n_layers * n_heads

    log("=" * 60)
    log("F59: INTER-RATER RELIABILITY")
    log("=" * 60)

    results = []

    for task in tasks:
        circuit_heads = get_circuit_heads(task)
        if not circuit_heads:
            log(f"  {task}: no circuit, skipping")
            continue

        k = len(circuit_heads)
        log(f"  {task}: circuit has {k} heads, computing 3 rater rankings...")

        prompts = generate_prompts(task, tokenizer, args.n_prompts)
        if not prompts:
            continue
        correct_ids, incorrect_ids = get_token_ids(prompts, tokenizer)
        if not correct_ids:
            continue

        # Three rater rankings (raw scores, higher = more important)
        scores_patch = rank_by_patching(model, prompts, correct_ids, incorrect_ids)
        scores_dla = rank_by_dla(model, prompts, correct_ids, incorrect_ids)
        scores_ov = rank_by_ov_norm(model, correct_ids, incorrect_ids)

        # Convert to ranks
        rank_patch = sp_stats.rankdata(scores_patch)
        rank_dla = sp_stats.rankdata(scores_dla)
        rank_ov = sp_stats.rankdata(scores_ov)

        rankings = np.vstack([rank_patch, rank_dla, rank_ov])

        # Kendall's W
        w = _kendalls_w(rankings)

        # Pairwise Spearman
        rho_patch_dla, _ = sp_stats.spearmanr(scores_patch, scores_dla)
        rho_patch_ov, _ = sp_stats.spearmanr(scores_patch, scores_ov)
        rho_dla_ov, _ = sp_stats.spearmanr(scores_dla, scores_ov)
        mean_rho = float(np.nanmean([rho_patch_dla, rho_patch_ov, rho_dla_ov]))

        # Cohen's kappa on binary top-k classification
        binary_patch = (sp_stats.rankdata(-scores_patch) <= k).astype(int)
        binary_dla = (sp_stats.rankdata(-scores_dla) <= k).astype(int)
        binary_ov = (sp_stats.rankdata(-scores_ov) <= k).astype(int)

        kappa_patch_dla = _cohens_kappa(binary_patch, binary_dla)
        kappa_patch_ov = _cohens_kappa(binary_patch, binary_ov)
        kappa_dla_ov = _cohens_kappa(binary_dla, binary_ov)
        mean_kappa = float(np.mean([kappa_patch_dla, kappa_patch_ov, kappa_dla_ov]))

        log(f"    Kendall W={w:.3f}, mean_rho={mean_rho:.3f}, mean_kappa={mean_kappa:.3f}")

        results.append(EvalResult(
            metric_id="F59.kendalls_w",
            value=w,
            n_samples=n_total,
            metadata={"task": task, "n_raters": 3, "n_items": n_total},
        ))
        results.append(EvalResult(
            metric_id="F59.mean_spearman",
            value=mean_rho,
            n_samples=n_total,
            metadata={
                "task": task,
                "rho_patch_dla": float(rho_patch_dla) if not np.isnan(rho_patch_dla) else 0.0,
                "rho_patch_ov": float(rho_patch_ov) if not np.isnan(rho_patch_ov) else 0.0,
                "rho_dla_ov": float(rho_dla_ov) if not np.isnan(rho_dla_ov) else 0.0,
            },
        ))
        results.append(EvalResult(
            metric_id="F59.mean_kappa",
            value=mean_kappa,
            n_samples=n_total,
            metadata={
                "task": task,
                "k": k,
                "kappa_patch_dla": kappa_patch_dla,
                "kappa_patch_ov": kappa_patch_ov,
                "kappa_dla_ov": kappa_dla_ov,
            },
        ))

    out = args.out or "59_inter_rater.json"
    save_results(results, out)
    log(f"\nDone. {len(results)} metrics across {len(tasks)} tasks.")


if __name__ == "__main__":
    main()
