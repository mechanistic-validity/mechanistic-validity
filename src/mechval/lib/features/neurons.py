"""~160 weight-space features per MLP neuron.

Feature groups:
  Group 1 — W_out / Vocabulary (25): peakiness, kurtosis, entropy, nullspace fraction, ...
  Group 2 — W_in / Input (18): input selectivity statistics, frequency correlation, ...
  Group 3 — W_in/W_out Composition (18): cosine, Jaccard overlap, polarity, ...
  Group 4 — Layer Context (15): downstream OV/K/V/Q alignment, upstream alignment, gate threshold, ...
  Group 5 — Cross-Neuron (14): nearest/mean W_out cosine, specialist density, rank features, ...
  Group 6 — Direction Alignment (60): 15 directions x 4 probes (wout/win x raw/whitened)
  Group 7 — PCA Projection (10): top-5 PCA components x {W_out, W_in}

All features are deterministic functions of W_in, W_out, W_E, W_U, b_in,
W_Q, W_K, W_V, W_O — no forward pass needed.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats as sp_stats
from sklearn.decomposition import PCA

from mechval.lib.features import DirectionSet, NeuronFeatures

SEMANTIC_CATEGORIES: dict[str, set[str]] = {
    "article": {" a", " an", " the", " The", " A", " An"},
    "preposition": {" in", " on", " at", " to", " of", " for", " with", " from", " by", " into"},
    "conjunction": {" and", " but", " or", " nor", " so", " yet"},
    "pronoun": {" he", " she", " it", " they", " we", " I", " you", " him", " her", " them", " us"},
    "negation": {" not", " no", " never", " neither", " nor", "n't", " nothing", " none"},
    "auxiliary": {" is", " was", " are", " were", " be", " been", " being", " has", " have", " had",
                  " do", " does", " did", " will", " would", " could", " should", " might", " may",
                  " can", " shall"},
    "punctuation": {".", ",", "!", "?", ";", ":", "'", '"', "-", "(", ")", "[", "]"},
    "number_word": {" one", " two", " three", " four", " five", " six", " seven", " eight",
                    " nine", " ten", " zero", " hundred", " thousand", " million"},
    "number_digit": {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"},
}


def classify_token(token_str: str) -> str:
    for cat, members in SEMANTIC_CATEGORIES.items():
        if token_str in members:
            return cat
    if token_str.strip().isdigit():
        return "number_digit"
    return "other"


def _rank_percentile(vals: torch.Tensor) -> torch.Tensor:
    order = vals.argsort()
    ranks = torch.zeros_like(vals)
    n = max(len(vals) - 1, 1)
    for r, idx in enumerate(order):
        ranks[idx] = r / n
    return ranks


def _to_np(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    return x


def compute_log_token_frequencies(
    model: Any,
    n_pile_rows: int = 200,
) -> torch.Tensor:
    """Estimate log token frequencies from Pile-10k (falls back to uniform)."""
    d_vocab = model.cfg.d_vocab
    token_counts = torch.ones(d_vocab)
    try:
        from datasets import load_dataset
        ds = load_dataset("NeelNanda/pile-10k", split=f"train[:{n_pile_rows}]")
        for row in ds:
            toks = model.to_tokens(row["text"], prepend_bos=False)[0]
            for t in toks:
                token_counts[t.item()] += 1
    except Exception:
        token_counts += torch.rand(d_vocab)
    return token_counts.log()


def compute_neuron_features(
    model: Any,
    direction_set: DirectionSet,
    device: str | torch.device = "cpu",
    layers: list[int] | None = None,
    log_fn: Any = None,
) -> NeuronFeatures:
    """Compute ~160 weight-space features for every MLP neuron.

    Args:
        model: HookedTransformer-like model with W_out, W_in, W_E, W_U, W_Q/K/V/O, b_in.
        direction_set: precomputed DirectionSet (from directions.compute_direction_set).
        device: torch device for computation.
        layers: subset of layers to process (default: all).
        log_fn: optional callable(str) for progress messages.

    Returns:
        NeuronFeatures with (n_neurons, n_features) matrix, feature names, and metadata.
    """
    tokenizer = model.tokenizer
    n_layers = model.cfg.n_layers
    d_mlp = model.cfg.d_mlp
    d_model = model.cfg.d_model
    n_heads = model.cfg.n_heads

    if layers is None:
        layers = list(range(n_layers))

    total_neurons = len(layers) * d_mlp
    if log_fn:
        log_fn(f"Feature extraction: {len(layers)} layers x {d_mlp} neurons = {total_neurons}")

    with torch.no_grad():
        W_U = model.W_U.float().to(device)
        W_E = model.W_E.float().to(device)

        mean_embed = W_E.mean(dim=0)
        mean_embed_dir = mean_embed / (mean_embed.norm() + 1e-10)

        dir_names = direction_set.direction_names
        n_dirs = len(dir_names)
        dir_vecs = direction_set.stacked(device)
        whitened_dir_vecs = direction_set.stacked_whitened(device)

        log_freqs = compute_log_token_frequencies(model)
        log_freqs_np = log_freqs.float().cpu().numpy()

        U_unembed, _, _ = torch.linalg.svd(W_U, full_matrices=False)
        null_rank = min(32, d_model)
        unembed_null_proj = (torch.eye(d_model, device=device)
                             - U_unembed[:, :null_rank] @ U_unembed[:, :null_rank].T)

        W_E_np = W_E[:5000].cpu().numpy()
        pca_emb = PCA(n_components=5, random_state=42)
        pca_emb.fit(W_E_np - W_E_np.mean(0))
        pca_dirs = []
        for i in range(5):
            d = torch.tensor(pca_emb.components_[i], dtype=torch.float32, device=device)
            pca_dirs.append(d / (d.norm() + 1e-10))

        # Build feature name list
        base_feat_names = [
            # Group 1: W_out (25)
            "wout_peakiness", "wout_top_score", "wout_score_gap", "wout_top5_gap",
            "wout_entropy", "wout_norm", "wout_kurtosis", "wout_skewness",
            "wout_promote_suppress_gap", "wout_suppression_score",
            "wout_breadth", "wout_effrank", "wout_freq_corr",
            "wout_nullspace_frac", "wout_embed_align",
            "wout_top_token_embed_cos", "wout_unembed_max_align",
            "wout_norm_percentile", "wout_kurtosis_percentile", "wout_category_purity",
            "wout_top10_gap", "wout_median_abs", "wout_gini",
            "wout_l1_l2_ratio", "wout_max_component",
            # Group 2: W_in (18)
            "win_peakiness", "win_top_score", "win_score_gap",
            "win_entropy", "win_norm", "win_kurtosis", "win_skewness",
            "win_embed_align", "win_freq_corr", "win_breadth",
            "win_suppression_score", "win_effrank", "win_top_token_embed_cos",
            "win_nullspace_frac", "win_top5_gap", "win_median_abs",
            "win_l1_l2_ratio", "win_max_component",
            # Group 3: Composition (18)
            "inout_cosine", "inout_dot", "inout_kv_jaccard",
            "inout_overlap_5", "inout_overlap_10", "inout_kurtosis_product",
            "inout_top_token_match", "inout_polarity", "ngram_suppress_score",
            "inout_norm_ratio", "inout_sv_ratio", "concept_vector_score",
            "inout_freq_corr_diff", "inout_entropy_ratio",
            "inout_suppress_match", "inout_breadth_ratio",
            "inout_skewness_product", "ngram_suppress_rank",
            # Group 4: Layer context (15)
            "layer_frac", "layer_from_end",
            "downstream_ov_align", "upstream_ov_align",
            "downstream_k_align", "downstream_v_align",
            "head_inhibition_score", "upstream_wo_align", "upstream_mlp_wout_align",
            "downstream_q_align", "samelayer_qk_self_mod",
            "upstream_k_align", "upstream_v_align",
            "downstream_ov_top_sv", "gate_threshold",
            # Group 5: Cross-neuron (14)
            "nearest_wout_cos", "mean_wout_cos", "nearest_win_cos", "mean_win_cos",
            "layer_specialist_density",
            "entropy_neuron_flag", "frequency_neuron_flag", "ngram_detector_flag",
            "bias",
            "wout_rank_in_layer", "win_rank_in_layer",
            "kurtosis_rank_in_layer", "bias_rank_in_layer",
            "inout_cos_rank_in_layer",
            # Group 7: PCA projection (10)
            "wout_pca0", "wout_pca1", "wout_pca2", "wout_pca3", "wout_pca4",
            "win_pca0", "win_pca1", "win_pca2", "win_pca3", "win_pca4",
        ]

        # Group 6: Direction alignment (15 dirs x 4 = 60)
        dir_feat_names = []
        for dn in dir_names:
            dir_feat_names.extend([
                f"wout_align_{dn}", f"win_align_{dn}",
                f"wout_align_{dn}_w", f"win_align_{dn}_w",
            ])

        all_feat_names = base_feat_names + dir_feat_names
        n_features = len(all_feat_names)

        feat_matrix = np.zeros((total_neurons, n_features), dtype=np.float32)
        neuron_meta: list[dict[str, Any]] = []
        neuron_idx = 0

        for l in layers:
            t0 = time.time()

            W_out_l = model.W_out[l].float().to(device)       # (d_mlp, d_model)
            W_in_l = model.W_in[l].float().to(device).T       # (d_mlp, d_model)
            b_in_l = model.b_in[l].float().to(device)         # (d_mlp,)

            # === GROUP 1: W_out ===
            logit_effects = W_out_l @ W_U                     # (d_mlp, d_vocab)
            top_out_scores, top_out_idx = logit_effects.topk(10, dim=-1)
            bot_out_scores, bot_out_idx = logit_effects.topk(10, dim=-1, largest=False)

            out_max = top_out_scores[:, 0]
            out_mean_abs = logit_effects.abs().mean(dim=-1)
            wout_peakiness = out_max / (out_mean_abs + 1e-10)
            wout_top_score = top_out_scores[:, 0]
            wout_score_gap = top_out_scores[:, 0] - top_out_scores[:, 1]
            wout_top5_gap = top_out_scores[:, 0] - top_out_scores[:, 4]

            out_probs = F.softmax(logit_effects, dim=-1)
            wout_entropy = -(out_probs * torch.log(out_probs + 1e-10)).sum(dim=-1)
            wout_norm = W_out_l.norm(dim=-1)

            le_centered = logit_effects - logit_effects.mean(dim=-1, keepdim=True)
            le_std = logit_effects.std(dim=-1, keepdim=True) + 1e-10
            le_z = le_centered / le_std
            wout_kurtosis = (le_z ** 4).mean(dim=-1) - 3
            wout_skewness = (le_z ** 3).mean(dim=-1)

            wout_promote_suppress_gap = top_out_scores[:, 0] - bot_out_scores[:, 0].abs()
            wout_suppression_score = bot_out_scores[:, 0]
            wout_breadth = (logit_effects > top_out_scores[:, 0:1] * 0.5).float().sum(dim=-1)

            le_abs = logit_effects.abs()
            wout_effrank = (le_abs.sum(dim=-1) ** 2) / ((le_abs ** 2).sum(dim=-1) + 1e-10)

            le_np = logit_effects.cpu().numpy()
            wout_freq_corr = np.zeros(d_mlp, dtype=np.float32)
            for n in range(d_mlp):
                r, _ = sp_stats.pearsonr(le_np[n], log_freqs_np)
                wout_freq_corr[n] = float(r)

            wout_null_proj_norms = (W_out_l @ unembed_null_proj).norm(dim=-1)
            wout_nullspace_frac = wout_null_proj_norms / (wout_norm + 1e-10)

            out_dirs = W_out_l / (wout_norm.unsqueeze(-1) + 1e-10)
            wout_embed_align = (out_dirs @ mean_embed_dir).abs()

            wout_top_token_embed_cos = torch.zeros(d_mlp, device=device)
            for n in range(d_mlp):
                tok_embed = W_E[top_out_idx[n, 0]]
                tok_embed_dir = tok_embed / (tok_embed.norm() + 1e-10)
                wout_top_token_embed_cos[n] = (out_dirs[n] @ tok_embed_dir).item()

            out_unembed_cosines = out_dirs @ W_U
            wout_unembed_max_align = out_unembed_cosines.abs().max(dim=-1).values

            wout_norm_percentile = _rank_percentile(wout_norm)
            wout_kurt_percentile = _rank_percentile(wout_kurtosis)

            wout_top10_gap = top_out_scores[:, 0] - top_out_scores[:, 9]
            wout_median_abs = logit_effects.abs().median(dim=-1).values
            le_abs_sorted, _ = le_abs.sort(dim=-1)
            cumsum = le_abs_sorted.cumsum(dim=-1)
            total_sum = cumsum[:, -1:] + 1e-10
            wout_gini = 1.0 - 2.0 * (cumsum / total_sum).mean(dim=-1)
            wout_l1 = W_out_l.abs().sum(dim=-1)
            wout_l1_l2_ratio = wout_l1 / (wout_norm + 1e-10)
            wout_max_component = out_dirs.abs().max(dim=-1).values

            # === GROUP 2: W_in ===
            input_selectivity = W_in_l @ W_E.T               # (d_mlp, d_vocab)
            top_in_scores, top_in_idx = input_selectivity.topk(10, dim=-1)
            bot_in_scores, _ = input_selectivity.topk(10, dim=-1, largest=False)

            in_mean_abs = input_selectivity.abs().mean(dim=-1)
            win_peakiness = top_in_scores[:, 0] / (in_mean_abs + 1e-10)
            win_top_score = top_in_scores[:, 0]
            win_score_gap = top_in_scores[:, 0] - top_in_scores[:, 1]

            in_probs = F.softmax(input_selectivity, dim=-1)
            win_entropy = -(in_probs * torch.log(in_probs + 1e-10)).sum(dim=-1)
            win_norm = W_in_l.norm(dim=-1)

            in_centered = input_selectivity - input_selectivity.mean(dim=-1, keepdim=True)
            in_std = input_selectivity.std(dim=-1, keepdim=True) + 1e-10
            in_z = in_centered / in_std
            win_kurtosis = (in_z ** 4).mean(dim=-1) - 3
            win_skewness = (in_z ** 3).mean(dim=-1)

            in_dirs = W_in_l / (win_norm.unsqueeze(-1) + 1e-10)
            win_embed_align = (in_dirs @ mean_embed_dir).abs()

            is_np = input_selectivity.cpu().numpy()
            win_freq_corr = np.zeros(d_mlp, dtype=np.float32)
            for n in range(d_mlp):
                r, _ = sp_stats.pearsonr(is_np[n], log_freqs_np)
                win_freq_corr[n] = float(r)

            win_breadth = (input_selectivity > top_in_scores[:, 0:1] * 0.5).float().sum(dim=-1)
            win_suppression_score = bot_in_scores[:, 0]

            is_abs = input_selectivity.abs()
            win_effrank = (is_abs.sum(dim=-1) ** 2) / ((is_abs ** 2).sum(dim=-1) + 1e-10)

            win_top_token_embed_cos = torch.zeros(d_mlp, device=device)
            for n in range(d_mlp):
                tok_embed = W_E[top_in_idx[n, 0]]
                tok_embed_dir = tok_embed / (tok_embed.norm() + 1e-10)
                win_top_token_embed_cos[n] = (in_dirs[n] @ tok_embed_dir).item()

            win_null_proj_norms = (W_in_l @ unembed_null_proj).norm(dim=-1)
            win_nullspace_frac = win_null_proj_norms / (win_norm + 1e-10)
            win_top5_gap = top_in_scores[:, 0] - top_in_scores[:, 4]
            win_median_abs = input_selectivity.abs().median(dim=-1).values
            win_l1 = W_in_l.abs().sum(dim=-1)
            win_l1_l2_ratio = win_l1 / (win_norm + 1e-10)
            win_max_component = in_dirs.abs().max(dim=-1).values

            # === GROUP 3: Composition ===
            inout_cosine = (in_dirs * out_dirs).sum(dim=-1)
            inout_dot = (W_in_l * W_out_l).sum(dim=-1)

            inout_kv_jaccard = torch.zeros(d_mlp, device=device)
            inout_overlap_5 = torch.zeros(d_mlp, device=device)
            inout_overlap_10 = torch.zeros(d_mlp, device=device)
            inout_top_token_match = (top_out_idx[:, 0] == top_in_idx[:, 0]).float()
            inout_polarity = torch.zeros(d_mlp, device=device)
            ngram_suppress = torch.zeros(d_mlp, device=device)
            inout_sv_ratio = torch.zeros(d_mlp, device=device)

            for n in range(d_mlp):
                out_set5 = set(top_out_idx[n, :5].tolist())
                in_set5 = set(top_in_idx[n, :5].tolist())
                union5 = len(out_set5 | in_set5)
                inout_kv_jaccard[n] = len(out_set5 & in_set5) / union5 if union5 > 0 else 0
                inout_overlap_5[n] = len(out_set5 & in_set5)

                out_set10 = set(top_out_idx[n, :10].tolist())
                in_set10 = set(top_in_idx[n, :10].tolist())
                inout_overlap_10[n] = len(out_set10 & in_set10)

                inout_polarity[n] = input_selectivity[n, top_out_idx[n, 0]].item()
                ngram_suppress[n] = logit_effects[n, top_in_idx[n, 0]].item()

                stacked = torch.stack([W_in_l[n], W_out_l[n]], dim=0)
                svs = torch.linalg.svdvals(stacked)
                inout_sv_ratio[n] = (svs[0] / (svs[1] + 1e-10)).item()

            inout_kurtosis_product = wout_kurtosis * win_kurtosis
            inout_norm_ratio = wout_norm / (win_norm + 1e-10)
            concept_score = wout_kurt_percentile * inout_kv_jaccard * inout_cosine.abs()

            inout_freq_corr_diff = np.abs(wout_freq_corr - win_freq_corr)
            inout_entropy_ratio = wout_entropy / (win_entropy + 1e-10)
            bot_out_min_idx = bot_out_idx[:, 0]
            bot_in_min = input_selectivity.topk(1, dim=-1, largest=False).indices.squeeze(-1)
            inout_suppress_match = (bot_out_min_idx == bot_in_min).float()
            inout_breadth_ratio = wout_breadth / (win_breadth + 1e-10)
            inout_skewness_product = wout_skewness * win_skewness
            ngram_sorted = ngram_suppress.argsort()
            ngram_suppress_rank = torch.zeros(d_mlp, device=device)
            for rank, idx in enumerate(ngram_sorted):
                ngram_suppress_rank[idx] = rank / max(d_mlp - 1, 1)

            # === GROUP 4: Layer context ===
            layer_frac_val = l / max(n_layers - 1, 1)
            layer_from_end_val = n_layers - 1 - l

            max_downstream_ov = torch.zeros(d_mlp, device=device)
            for dl in range(l + 1, min(l + 3, n_layers)):
                for h in range(n_heads):
                    W_V_h = model.W_V[dl, h].float().to(device)
                    W_O_h = model.W_O[dl, h].float().to(device)
                    W_OV = W_V_h @ W_O_h
                    _, _, Vh = torch.linalg.svd(W_OV, full_matrices=False)
                    ov_input_dir = Vh[0]
                    align = (out_dirs @ ov_input_dir).abs()
                    max_downstream_ov = torch.max(max_downstream_ov, align)

            max_upstream_ov = torch.zeros(d_mlp, device=device)
            for ul in range(max(0, l - 2), l):
                for h in range(n_heads):
                    W_V_h = model.W_V[ul, h].float().to(device)
                    W_O_h = model.W_O[ul, h].float().to(device)
                    W_OV = W_V_h @ W_O_h
                    U_ov = torch.linalg.svd(W_OV, full_matrices=False)[0]
                    ov_output_dir = U_ov[:, 0]
                    align = (in_dirs @ ov_output_dir).abs()
                    max_upstream_ov = torch.max(max_upstream_ov, align)

            max_downstream_k = torch.zeros(d_mlp, device=device)
            for dl in range(l + 1, min(l + 3, n_layers)):
                for h in range(n_heads):
                    W_K_h = model.W_K[dl, h].float().to(device)
                    align = (out_dirs @ W_K_h).norm(dim=-1)
                    max_downstream_k = torch.max(max_downstream_k, align)

            max_downstream_v = torch.zeros(d_mlp, device=device)
            for dl in range(l + 1, min(l + 3, n_layers)):
                for h in range(n_heads):
                    W_V_h = model.W_V[dl, h].float().to(device)
                    align = (out_dirs @ W_V_h).norm(dim=-1)
                    max_downstream_v = torch.max(max_downstream_v, align)

            max_head_inhibition = torch.zeros(d_mlp, device=device)
            for dl in range(l + 1, min(l + 3, n_layers)):
                for h in range(n_heads):
                    W_V_h = model.W_V[dl, h].float().to(device)
                    W_K_h = model.W_K[dl, h].float().to(device)
                    inhibition = (W_out_l @ W_V_h @ W_K_h.T).norm(dim=-1)
                    max_head_inhibition = torch.max(max_head_inhibition, inhibition)

            max_upstream_wo = torch.zeros(d_mlp, device=device)
            for ul in range(max(0, l - 2), l):
                for h in range(n_heads):
                    W_O_h = model.W_O[ul, h].float().to(device)
                    align = (in_dirs @ W_O_h.T).norm(dim=-1)
                    max_upstream_wo = torch.max(max_upstream_wo, align)

            max_upstream_mlp = torch.zeros(d_mlp, device=device)
            if l > 0:
                W_out_prev = model.W_out[l - 1].float().to(device)
                prev_norms = W_out_prev.norm(dim=-1)
                prev_dirs = W_out_prev / (prev_norms.unsqueeze(-1) + 1e-10)
                cross_cos = in_dirs @ prev_dirs.T
                max_upstream_mlp = cross_cos.abs().max(dim=-1).values

            max_downstream_q = torch.zeros(d_mlp, device=device)
            for dl in range(l + 1, min(l + 3, n_layers)):
                for h in range(n_heads):
                    W_Q_h = model.W_Q[dl, h].float().to(device)
                    align = (out_dirs @ W_Q_h).norm(dim=-1)
                    max_downstream_q = torch.max(max_downstream_q, align)

            samelayer_qk_self_mod = torch.zeros(d_mlp, device=device)
            for h in range(n_heads):
                W_Q_h = model.W_Q[l, h].float().to(device)
                W_K_h = model.W_K[l, h].float().to(device)
                W_QK = W_Q_h @ W_K_h.T
                selfmod = (out_dirs @ W_QK @ out_dirs.T).diag().abs()
                samelayer_qk_self_mod = torch.max(samelayer_qk_self_mod, selfmod)

            max_upstream_k = torch.zeros(d_mlp, device=device)
            max_upstream_v = torch.zeros(d_mlp, device=device)
            for ul in range(max(0, l - 2), l):
                for h in range(n_heads):
                    W_K_h = model.W_K[ul, h].float().to(device)
                    W_V_h = model.W_V[ul, h].float().to(device)
                    max_upstream_k = torch.max(max_upstream_k, (in_dirs @ W_K_h).norm(dim=-1))
                    max_upstream_v = torch.max(max_upstream_v, (in_dirs @ W_V_h).norm(dim=-1))

            downstream_ov_top_sv = torch.zeros(d_mlp, device=device)
            for dl in range(l + 1, min(l + 3, n_layers)):
                for h in range(n_heads):
                    W_V_h = model.W_V[dl, h].float().to(device)
                    W_O_h = model.W_O[dl, h].float().to(device)
                    W_OV = W_V_h @ W_O_h
                    proj = out_dirs @ W_OV
                    sv = proj.norm(dim=-1)
                    downstream_ov_top_sv = torch.max(downstream_ov_top_sv, sv)

            gate_threshold = -b_in_l / (win_norm + 1e-10)

            # === GROUP 5: Cross-neuron ===
            out_cos_matrix = out_dirs @ out_dirs.T
            out_cos_matrix.fill_diagonal_(0)
            nearest_wout_cos = out_cos_matrix.abs().max(dim=-1).values
            mean_wout_cos = out_cos_matrix.abs().mean(dim=-1)

            in_cos_matrix = in_dirs @ in_dirs.T
            in_cos_matrix.fill_diagonal_(0)
            nearest_win_cos = in_cos_matrix.abs().max(dim=-1).values
            mean_win_cos = in_cos_matrix.abs().mean(dim=-1)

            specialist_mask = wout_peakiness > 10
            layer_specialist_density = specialist_mask.float().mean().item()

            entropy_neuron_flag = ((wout_norm_percentile > 0.95) & (wout_nullspace_frac > 0.5)).float()
            frequency_neuron_flag = (torch.from_numpy(wout_freq_corr).to(device).abs() > 0.3).float()
            ngram_flag = ((win_peakiness > 8) & (ngram_suppress < 0)).float()

            wout_rank = _rank_percentile(wout_norm)
            win_rank = _rank_percentile(win_norm)
            kurtosis_rank = _rank_percentile(wout_kurtosis)
            bias_rank = _rank_percentile(b_in_l)
            inout_cos_rank = _rank_percentile(inout_cosine)

            # Group 7: PCA projection features
            wout_pca = torch.stack([(out_dirs @ pd).cpu() for pd in pca_dirs], dim=-1)
            win_pca = torch.stack([(in_dirs @ pd).cpu() for pd in pca_dirs], dim=-1)

            # === GROUP 6: Direction alignment ===
            wout_dir_align = out_dirs @ dir_vecs.T
            win_dir_align = in_dirs @ dir_vecs.T
            wout_dir_align_w = out_dirs @ whitened_dir_vecs.T
            win_dir_align_w = in_dirs @ whitened_dir_vecs.T

            # === ASSEMBLE ===
            cat_purity_arr = np.zeros(d_mlp, dtype=np.float32)
            for n in range(d_mlp):
                top_tok_str = tokenizer.decode([top_out_idx[n, 0].item()])
                cat = classify_token(top_tok_str)
                top5_toks = [tokenizer.decode([top_out_idx[n, k].item()]) for k in range(5)]
                top5_cats = [classify_token(t) for t in top5_toks]
                cat_purity_arr[n] = sum(1 for c in top5_cats if c == cat) / 5.0

            all_layer_feats = [
                # Group 1: W_out (25)
                _to_np(wout_peakiness), _to_np(wout_top_score), _to_np(wout_score_gap),
                _to_np(wout_top5_gap), _to_np(wout_entropy), _to_np(wout_norm),
                _to_np(wout_kurtosis), _to_np(wout_skewness),
                _to_np(wout_promote_suppress_gap), _to_np(wout_suppression_score),
                _to_np(wout_breadth), _to_np(wout_effrank), wout_freq_corr,
                _to_np(wout_nullspace_frac), _to_np(wout_embed_align),
                _to_np(wout_top_token_embed_cos), _to_np(wout_unembed_max_align),
                _to_np(wout_norm_percentile), _to_np(wout_kurt_percentile), cat_purity_arr,
                _to_np(wout_top10_gap), _to_np(wout_median_abs), _to_np(wout_gini),
                _to_np(wout_l1_l2_ratio), _to_np(wout_max_component),
                # Group 2: W_in (18)
                _to_np(win_peakiness), _to_np(win_top_score), _to_np(win_score_gap),
                _to_np(win_entropy), _to_np(win_norm), _to_np(win_kurtosis), _to_np(win_skewness),
                _to_np(win_embed_align), win_freq_corr,
                _to_np(win_breadth), _to_np(win_suppression_score), _to_np(win_effrank),
                _to_np(win_top_token_embed_cos),
                _to_np(win_nullspace_frac), _to_np(win_top5_gap), _to_np(win_median_abs),
                _to_np(win_l1_l2_ratio), _to_np(win_max_component),
                # Group 3: Composition (18)
                _to_np(inout_cosine), _to_np(inout_dot), _to_np(inout_kv_jaccard),
                _to_np(inout_overlap_5), _to_np(inout_overlap_10), _to_np(inout_kurtosis_product),
                _to_np(inout_top_token_match), _to_np(inout_polarity), _to_np(ngram_suppress),
                _to_np(inout_norm_ratio), _to_np(inout_sv_ratio), _to_np(concept_score),
                inout_freq_corr_diff, _to_np(inout_entropy_ratio),
                _to_np(inout_suppress_match), _to_np(inout_breadth_ratio),
                _to_np(inout_skewness_product), _to_np(ngram_suppress_rank),
                # Group 4: Layer context (15)
                np.full(d_mlp, layer_frac_val, dtype=np.float32),
                np.full(d_mlp, layer_from_end_val, dtype=np.float32),
                _to_np(max_downstream_ov), _to_np(max_upstream_ov),
                _to_np(max_downstream_k), _to_np(max_downstream_v),
                _to_np(max_head_inhibition), _to_np(max_upstream_wo), _to_np(max_upstream_mlp),
                _to_np(max_downstream_q), _to_np(samelayer_qk_self_mod),
                _to_np(max_upstream_k), _to_np(max_upstream_v),
                _to_np(downstream_ov_top_sv), _to_np(gate_threshold),
                # Group 5: Cross-neuron (14)
                _to_np(nearest_wout_cos), _to_np(mean_wout_cos),
                _to_np(nearest_win_cos), _to_np(mean_win_cos),
                np.full(d_mlp, layer_specialist_density, dtype=np.float32),
                _to_np(entropy_neuron_flag), _to_np(frequency_neuron_flag), _to_np(ngram_flag),
                _to_np(b_in_l),
                _to_np(wout_rank), _to_np(win_rank), _to_np(kurtosis_rank),
                _to_np(bias_rank), _to_np(inout_cos_rank),
                # Group 7: PCA projection (10)
                *[wout_pca[:, i].numpy() for i in range(5)],
                *[win_pca[:, i].numpy() for i in range(5)],
            ]

            for di in range(n_dirs):
                all_layer_feats.append(wout_dir_align[:, di].cpu().numpy())
                all_layer_feats.append(win_dir_align[:, di].cpu().numpy())
                all_layer_feats.append(wout_dir_align_w[:, di].cpu().numpy())
                all_layer_feats.append(win_dir_align_w[:, di].cpu().numpy())

            assert len(all_layer_feats) == n_features, \
                f"Feature count mismatch: {len(all_layer_feats)} != {n_features}"

            for fi, arr in enumerate(all_layer_feats):
                feat_matrix[neuron_idx:neuron_idx + d_mlp, fi] = arr

            for n in range(d_mlp):
                top_tok_str = tokenizer.decode([top_out_idx[n, 0].item()])
                top_in_tok_str = tokenizer.decode([top_in_idx[n, 0].item()])
                neuron_meta.append({
                    "layer": l,
                    "neuron": n,
                    "name": f"L{l}N{n}",
                    "top_token": top_tok_str,
                    "top_in_token": top_in_tok_str,
                    "category": classify_token(top_tok_str),
                    "top5_out": [{"token": tokenizer.decode([top_out_idx[n, k].item()]),
                                  "score": top_out_scores[n, k].item()} for k in range(5)],
                    "top5_in": [{"token": tokenizer.decode([top_in_idx[n, k].item()]),
                                 "score": top_in_scores[n, k].item()} for k in range(5)],
                    "bottom5_out": [{"token": tokenizer.decode([bot_out_idx[n, k].item()]),
                                     "score": bot_out_scores[n, k].item()} for k in range(5)],
                })

            neuron_idx += d_mlp
            dt = time.time() - t0
            if log_fn:
                log_fn(f"L{l}: peak={wout_peakiness.mean().item():.1f}, "
                       f"kurtosis={wout_kurtosis.mean().item():.1f}, "
                       f"inout_cos={inout_cosine.mean().item():.3f} ({dt:.1f}s)")

        feat_matrix = feat_matrix[:neuron_idx]

    return NeuronFeatures(
        matrix=feat_matrix,
        feature_names=all_feat_names,
        metadata=neuron_meta,
    )
