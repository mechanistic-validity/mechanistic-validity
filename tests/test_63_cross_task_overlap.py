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
    / "cross_task_overlap"
    / "63_cross_task_overlap.py"
)
_spec = importlib.util.spec_from_file_location("cross_task_63", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["cross_task_63"] = _mod
_spec.loader.exec_module(_mod)

subspace_overlap = _mod.subspace_overlap
linear_cka = _mod.linear_cka

TASK = "ioi"


def test_subspace_overlap_identical():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 20))
    result = subspace_overlap(X, X, k=5)
    assert result == pytest.approx(1.0, abs=1e-6)


def test_subspace_overlap_orthogonal():
    n = 100
    d = 20
    rng = np.random.default_rng(0)
    M = rng.standard_normal((n, d))
    M_centered = M - M.mean(axis=0, keepdims=True)
    U, _, _ = np.linalg.svd(M_centered, full_matrices=False)
    X = np.zeros((n, d))
    X[:, :5] = U[:, :5]
    Y = np.zeros((n, d))
    Y[:, 5:10] = U[:, 5:10]
    result = subspace_overlap(X, Y, k=5)
    assert result < 0.2


def test_subspace_overlap_range():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((50, 20))
    Y = rng.standard_normal((50, 20))
    result = subspace_overlap(X, Y, k=5)
    assert 0.0 <= result <= 1.0


def test_linear_cka_identical():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 10))
    X_c = X - X.mean(axis=0, keepdims=True)
    assert linear_cka(X_c, X_c) == pytest.approx(1.0, abs=1e-6)


def test_linear_cka_symmetry():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 8))
    Y = rng.standard_normal((50, 6))
    X_c = X - X.mean(axis=0, keepdims=True)
    Y_c = Y - Y.mean(axis=0, keepdims=True)
    assert linear_cka(X_c, Y_c) == pytest.approx(linear_cka(Y_c, X_c), abs=1e-10)


def test_linear_cka_scaled():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 10))
    X_c = X - X.mean(axis=0, keepdims=True)
    Y_c = X_c * 5.0
    assert linear_cka(X_c, Y_c) == pytest.approx(1.0, abs=1e-6)


def test_linear_cka_zero_matrix():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 10))
    X_c = X - X.mean(axis=0, keepdims=True)
    Z = np.zeros((50, 10))
    assert linear_cka(X_c, Z) == pytest.approx(0.0, abs=1e-6)


def test_linear_cka_too_few_samples():
    X = np.random.randn(2, 5)
    Y = np.random.randn(2, 5)
    assert linear_cka(X, Y) == pytest.approx(0.0)
