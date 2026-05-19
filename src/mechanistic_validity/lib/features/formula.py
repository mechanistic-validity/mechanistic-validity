"""Greedy linear formula search + bootstrap stability analysis.

The core circuit-finding algorithm: given a feature matrix over components
(attention heads or MLP neurons) and binary labels (circuit member vs not),
find a sparse linear combination of features that best separates the two classes.

This is the "+/- 1 thing" — each term is +1 or -1 times a standardized feature,
and components scoring above a threshold in the combined score are circuit members.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score


@dataclass
class FormulaResult:
    """Result of greedy formula search."""
    scores: np.ndarray           # per-component combined scores
    selected_features: list[str] # feature names in selection order
    signs: list[int]             # +1 or -1 for each selected feature
    auroc: float                 # AUROC of the combined score vs labels


@dataclass
class BootstrapResult:
    """Result of bootstrap stability analysis."""
    feature_stability: dict[str, float]   # feature -> fraction of bootstrap runs it appeared in
    component_stability: dict[int, float] # component index -> fraction of runs in top-k
    formulas: list[dict[str, Any]]        # per-run formula details


def greedy_formula(
    pool_indices: np.ndarray,
    labels: np.ndarray,
    feat_matrix: np.ndarray,
    feat_names: list[str],
    n_terms: int = 5,
) -> FormulaResult:
    """Greedy forward feature selection maximizing AUROC.

    At each step, try adding each unused feature (with +1 or -1 sign) to the
    running score, and keep the one that maximizes AUROC. All features are
    standardized internally before combination.

    Args:
        pool_indices: indices into feat_matrix to use.
        labels: binary labels (0/1) for each pool index.
        feat_matrix: (n_components, n_features) feature values.
        feat_names: feature name for each column.
        n_terms: maximum number of features to select.

    Returns:
        FormulaResult with scores, selected features, signs, and AUROC.
    """
    n = len(pool_indices)

    pool_feats: dict[str, np.ndarray] = {}
    for i, fname in enumerate(feat_names):
        vals = feat_matrix[pool_indices, i]
        s = vals.std()
        pool_feats[fname] = (vals - vals.mean()) / (s + 1e-10)

    if labels.sum() == 0 or labels.sum() == len(labels):
        return FormulaResult(np.zeros(n), [], [], 0.5)

    selected: list[str] = []
    signs: list[int] = []
    best = np.zeros(n)

    for _ in range(n_terms):
        best_result: tuple[float, float, str, int, np.ndarray] | None = None
        for i, fname in enumerate(feat_names):
            if fname in selected:
                continue
            for sign in [1, -1]:
                cand = best + sign * pool_feats[fname]
                try:
                    auroc = roc_auc_score(labels, cand)
                except ValueError:
                    continue
                n_pos = int(labels.sum())
                topk = set(np.argsort(cand)[-n_pos:])
                gt_idx = set(np.where(labels == 1)[0])
                rec = len(topk & gt_idx) / len(gt_idx) if gt_idx else 0
                if best_result is None or auroc > best_result[0] or (auroc == best_result[0] and rec > best_result[1]):
                    best_result = (auroc, rec, fname, sign, cand)
        if best_result is None:
            break
        _, _, fname, sign, cand = best_result
        selected.append(fname)
        signs.append(sign)
        best = cand

    try:
        final_auroc = roc_auc_score(labels, best)
    except ValueError:
        final_auroc = 0.5

    return FormulaResult(
        scores=best,
        selected_features=selected,
        signs=signs,
        auroc=final_auroc,
    )


def bootstrap_stability(
    feat_matrix: np.ndarray,
    labels: np.ndarray,
    feat_names: list[str],
    n_bootstrap: int = 30,
    feature_frac: float = 0.8,
    n_terms: int = 5,
) -> BootstrapResult:
    """Bootstrap stability: which features consistently predict circuit membership?

    Runs greedy_formula many times, each with a random 80% subset of features.
    Reports how often each feature is selected and how often each component
    lands in the top-k predictions.

    Args:
        feat_matrix: (n_components, n_features) for the validated subset.
        labels: binary labels (0/1).
        feat_names: feature names.
        n_bootstrap: number of bootstrap iterations.
        feature_frac: fraction of features to subsample per iteration.
        n_terms: max terms per formula.

    Returns:
        BootstrapResult with feature stability, component stability, and per-run formulas.
    """
    n = len(labels)
    pool_indices = np.arange(n)
    n_pos = int(labels.sum())

    feature_counts: dict[str, int] = defaultdict(int)
    component_counts: dict[int, int] = defaultdict(int)
    bootstrap_formulas: list[dict[str, Any]] = []

    for b in range(n_bootstrap):
        rng = np.random.RandomState(b)
        n_feats = len(feat_names)
        subset_idx = rng.choice(n_feats, int(n_feats * feature_frac), replace=False)
        subset_names = [feat_names[i] for i in subset_idx]

        result = greedy_formula(
            pool_indices, labels,
            feat_matrix[:, subset_idx], subset_names, n_terms=n_terms,
        )
        for fname in result.selected_features:
            feature_counts[fname] += 1

        k = n_pos * 2
        top_idx = np.argsort(result.scores)[-k:]
        for idx in top_idx:
            component_counts[int(idx)] += 1

        bootstrap_formulas.append({
            "features": result.selected_features,
            "signs": result.signs,
            "auroc": result.auroc,
        })

    feature_stability = {fname: count / n_bootstrap for fname, count in feature_counts.items()}
    component_stability = {idx: count / n_bootstrap for idx, count in component_counts.items()}

    return BootstrapResult(
        feature_stability=feature_stability,
        component_stability=component_stability,
        formulas=bootstrap_formulas,
    )
