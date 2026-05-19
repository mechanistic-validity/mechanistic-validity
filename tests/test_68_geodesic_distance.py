import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from mechanistic_validity.instruments.common import EvalResult, load_model

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "mechanistic_validity"
    / "instruments"
    / "representational"
    / "geodesic_distance"
    / "68_geodesic_distance.py"
)
_spec = importlib.util.spec_from_file_location("geodesic_68", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["geodesic_68"] = _mod
_spec.loader.exec_module(_mod)

build_knn_graph = _mod.build_knn_graph
compute_distortion_ratio = _mod.compute_distortion_ratio

TASK = "ioi"


def test_build_knn_graph_shape():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 5))
    graph = build_knn_graph(X, k=5)
    assert graph.shape == (20, 20)


def test_build_knn_graph_symmetric():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((15, 4))
    graph = build_knn_graph(X, k=4)
    diff = abs(graph - graph.T)
    assert diff.max() < 1e-10


def test_build_knn_graph_non_negative_weights():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 5))
    graph = build_knn_graph(X, k=5)
    assert graph.min() >= 0.0


def test_distortion_ratio_flat_data():
    rng = np.random.default_rng(0)
    n = 30
    t = np.linspace(0, 1, n)
    X = np.column_stack([t, np.zeros(n)])
    result = compute_distortion_ratio(X, k=5)
    assert result["mean_distortion"] >= 1.0 - 1e-6


def test_distortion_ratio_too_few_points():
    X = np.random.randn(3, 5)
    result = compute_distortion_ratio(X, k=10)
    assert result["mean_distortion"] == pytest.approx(1.0)
    assert result["unreachable_frac"] == pytest.approx(1.0)


def test_distortion_ratio_geodesic_geq_euclidean():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 3))
    result = compute_distortion_ratio(X, k=10)
    assert result["mean_distortion"] >= 1.0 - 1e-6


def test_distortion_ratio_max_geq_mean():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 5))
    result = compute_distortion_ratio(X, k=8)
    assert result["max_distortion"] >= result["mean_distortion"]
