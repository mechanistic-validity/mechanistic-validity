import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.distance import pdist

from mechval.metrics.common import EvalResult, load_model

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "mechval"
    / "metrics"
    / "representational"
    / "persistent_homology"
    / "67_persistent_homology.py"
)
_spec = importlib.util.spec_from_file_location("ph_67", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["ph_67"] = _mod
_spec.loader.exec_module(_mod)

compute_h0_persistence = _mod.compute_h0_persistence

TASK = "ioi"


def test_h0_persistence_two_clusters():
    cluster1 = np.random.randn(10, 2) + np.array([0, 0])
    cluster2 = np.random.randn(10, 2) + np.array([100, 100])
    X = np.vstack([cluster1, cluster2])
    dists = pdist(X, metric="euclidean")
    result = compute_h0_persistence(dists, X.shape[0])
    assert result["diameter"] > 0.0
    assert result["mean_merge"] > 0.0
    assert 0.0 <= result["betti_auc_normalized"] <= 1.0


def test_h0_persistence_single_cluster():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 3)) * 0.1
    dists = pdist(X, metric="euclidean")
    result = compute_h0_persistence(dists, X.shape[0])
    assert result["diameter"] > 0.0
    assert result["betti_auc_normalized"] > 0.0


def test_h0_persistence_diameter_positive():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((15, 5))
    dists = pdist(X, metric="euclidean")
    result = compute_h0_persistence(dists, X.shape[0])
    assert result["diameter"] > 0.0


def test_h0_persistence_mean_merge_less_than_diameter():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 4))
    dists = pdist(X, metric="euclidean")
    result = compute_h0_persistence(dists, X.shape[0])
    assert result["mean_merge"] <= result["diameter"]


def test_h0_persistence_betti_auc_normalized_range():
    for seed in range(10):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((15, 3))
        dists = pdist(X, metric="euclidean")
        result = compute_h0_persistence(dists, X.shape[0])
        assert 0.0 <= result["betti_auc_normalized"] <= 1.0


def test_h0_persistence_cosine_distance():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 10))
    dists = pdist(X, metric="cosine")
    dists = np.nan_to_num(dists, nan=2.0)
    result = compute_h0_persistence(dists, X.shape[0])
    assert result["diameter"] > 0.0
