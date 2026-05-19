"""109 weight-space features per attention head.

Feature groups:
  Structural/SVD (10): singular value statistics of W_OV = W_V.T @ W_O and W_QK = W_Q.T @ W_K
  Token Interaction (8): same-token vs diff-token QK scores, OV copy matrix diagonal
  Direction Alignment (91): 15 directions x 6 probes (ov_signed, ov_top_out, ov_top_in,
                            q_align, k_align, k_align_whitened) + tok_sens

All features are deterministic functions of W_Q, W_K, W_V, W_O, W_E — no forward pass needed.
"""
from __future__ import annotations

import numpy as np
import torch

from mechval.lib.features import DirectionSet, HeadFeatures


def compute_head_features(
    W_Q: torch.Tensor,
    W_K: torch.Tensor,
    W_V: torch.Tensor,
    W_O: torch.Tensor,
    W_E: torch.Tensor,
    direction_set: DirectionSet,
    token_ids: list[int],
) -> HeadFeatures:
    """Compute all 109 weight features for every attention head.

    Args:
        W_Q: (n_layers, n_heads, d_head, d_model)
        W_K: (n_layers, n_heads, d_head, d_model)
        W_V: (n_layers, n_heads, d_head, d_model)
        W_O: (n_layers, n_heads, d_head, d_model)
        W_E: (d_vocab, d_model)
        direction_set: DirectionSet with directions + whitening
        token_ids: list of token indices for same-diff and OV copy features

    Returns:
        HeadFeatures with per-(layer, head) feature dicts.
    """
    W_Q = W_Q.cpu().float()
    W_K = W_K.cpu().float()
    W_V = W_V.cpu().float()
    W_O = W_O.cpu().float()
    W_E = W_E.cpu().float()
    whitening = direction_set.whitening.cpu().float()
    directions = {k: v.cpu().float() for k, v in direction_set.directions.items()}

    n_layers, n_heads = W_Q.shape[:2]
    tok_embeds = W_E[token_ids]
    tok_embeds_w = (whitening @ tok_embeds.T).T
    n_tok = len(token_ids)
    W_U_sub = W_E[token_ids]
    tok_dir = tok_embeds.mean(0)
    tok_dir = tok_dir / (tok_dir.norm() + 1e-10)

    whitened_dirs: dict[str, torch.Tensor] = {}
    for name, d in directions.items():
        d_w = whitening @ d
        whitened_dirs[name] = d_w / (d_w.norm() + 1e-10)

    feature_names: list[str] | None = None
    head_feats: dict[tuple[int, int], dict[str, float]] = {}

    for L in range(n_layers):
        for H in range(n_heads):
            wq = W_Q[L, H]
            wk = W_K[L, H]
            wv = W_V[L, H]
            wo = W_O[L, H]
            W_ov = wv.T @ wo
            W_qk = wq.T @ wk

            ov_svs = torch.linalg.svdvals(W_ov)
            qk_svs = torch.linalg.svdvals(W_qk)
            U_ov, _, Vh_ov = torch.linalg.svd(W_ov, full_matrices=False)

            f: dict[str, float] = {}

            # --- Structural/SVD (10) ---
            f["ov_norm"] = ov_svs.sum().item()
            f["ov_concentration"] = (ov_svs[0] / ov_svs.sum()).item()
            f["ov_sv_gap"] = (ov_svs[0] - ov_svs[1]).item()
            f["ov_effective_rank"] = (ov_svs.sum() ** 2 / (ov_svs ** 2).sum()).item()
            f["ov_top2_ratio"] = (ov_svs[0] / (ov_svs[1] + 1e-10)).item()
            f["qk_norm"] = qk_svs.sum().item()
            f["qk_concentration"] = (qk_svs[0] / qk_svs.sum()).item()
            f["qk_sv_gap"] = (qk_svs[0] - qk_svs[1]).item()
            f["qk_effective_rank"] = (qk_svs.sum() ** 2 / (qk_svs ** 2).sum()).item()
            f["qk_top2_ratio"] = (qk_svs[0] / (qk_svs[1] + 1e-10)).item()

            # --- Token Interaction (8) ---
            qk_scores = tok_embeds @ W_qk @ tok_embeds.T
            qk_scores_w = tok_embeds_w @ W_qk @ tok_embeds_w.T
            diag_qk = torch.diag(qk_scores)
            diag_qk_w = torch.diag(qk_scores_w)
            same_mean = diag_qk.mean().item()
            same_mean_w = diag_qk_w.mean().item()
            if n_tok > 1:
                diff_mean = (qk_scores.sum().item() - diag_qk.sum().item()) / (n_tok * (n_tok - 1))
                diff_mean_w = (qk_scores_w.sum().item() - diag_qk_w.sum().item()) / (n_tok * (n_tok - 1))
            else:
                diff_mean = diff_mean_w = 0.0
            f["qk_same_diff_ratio"] = same_mean / (abs(diff_mean) + 1e-10)
            f["qk_same_diff_gap"] = same_mean - diff_mean
            f["qk_same_diff_ratio_w"] = same_mean_w / (abs(diff_mean_w) + 1e-10)
            f["qk_sens_tok"] = (tok_dir @ W_qk @ tok_dir).item()

            ov_logits = (W_U_sub @ W_ov @ tok_embeds.T).cpu().numpy()
            f["ov_tok_diag_mean"] = float(np.diag(ov_logits).mean())
            if n_tok > 1:
                offdiag = (ov_logits.sum() - np.diag(ov_logits).sum()) / (n_tok * (n_tok - 1))
            else:
                offdiag = 0.0
            f["ov_tok_offdiag_mean"] = float(offdiag)
            f["ov_tok_copy_ratio"] = f["ov_tok_diag_mean"] / (abs(f["ov_tok_offdiag_mean"]) + 1e-10)
            f["ov_tok_logit_min"] = float(ov_logits.min())

            # --- Direction Alignment (91 = 15 dirs x 6 probes + qk_sens_tok already counted above) ---
            for dname in direction_set.direction_names:
                d_vec = directions[dname]
                d_w = whitened_dirs[dname]
                f[f"ov_{dname}_signed"] = (d_vec @ W_ov @ d_vec).item()
                f[f"ov_top_out_{dname}"] = (d_vec @ U_ov[:, 0]).item()
                f[f"ov_top_in_{dname}"] = (d_vec @ Vh_ov[0, :]).item()
                f[f"q_align_{dname}"] = (d_vec @ wq.T).norm().item()
                f[f"k_align_{dname}"] = (d_vec @ wk.T).norm().item()
                f[f"k_align_{dname}_w"] = (d_w @ wk.T).norm().item()

            head_feats[(L, H)] = f

            if feature_names is None:
                feature_names = list(f.keys())

    assert feature_names is not None
    return HeadFeatures(
        features=head_feats,
        feature_names=feature_names,
        n_layers=n_layers,
        n_heads=n_heads,
    )


def compute_head_features_from_model(
    model: object,
    direction_set: DirectionSet,
    token_ids: list[int] | None = None,
    top_k_tokens: int = 500,
) -> HeadFeatures:
    """Convenience: extract features directly from a HookedTransformer-like model.

    If token_ids is None, uses the first top_k_tokens token indices.
    """
    if token_ids is None:
        token_ids = list(range(top_k_tokens))
    return compute_head_features(
        W_Q=model.W_Q,  # type: ignore[attr-defined]
        W_K=model.W_K,  # type: ignore[attr-defined]
        W_V=model.W_V,  # type: ignore[attr-defined]
        W_O=model.W_O,  # type: ignore[attr-defined]
        W_E=model.W_E,  # type: ignore[attr-defined]
        direction_set=direction_set,
        token_ids=token_ids,
    )
