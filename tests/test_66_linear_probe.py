import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from mechanistic_validity.metrics.common import EvalResult, load_model

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "mechanistic_validity"
    / "metrics"
    / "representational"
    / "linear_probe"
    / "66_linear_probe.py"
)
_spec = importlib.util.spec_from_file_location("lp_66", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["lp_66"] = _mod
_spec.loader.exec_module(_mod)

train_linear_probe = _mod.train_linear_probe

TASK = "ioi"


def test_train_linear_probe_separable():
    rng = np.random.default_rng(0)
    n = 100
    d = 10
    X = rng.standard_normal((n, d))
    w_true = rng.standard_normal(d)
    y = (X @ w_true > 0).astype(float)
    acc, weights = train_linear_probe(X, y)
    assert acc > 0.8


def test_train_linear_probe_random_labels():
    rng = np.random.default_rng(0)
    n = 100
    d = 10
    X = rng.standard_normal((n, d))
    y = rng.integers(0, 2, size=n).astype(float)
    acc, weights = train_linear_probe(X, y)
    assert 0.2 < acc < 0.8


def test_train_linear_probe_too_few_samples():
    X = np.random.randn(3, 5)
    y = np.array([0.0, 1.0, 0.0])
    acc, weights = train_linear_probe(X, y)
    assert acc == pytest.approx(0.0)
    assert weights == pytest.approx(np.zeros(5))


def test_train_linear_probe_returns_correct_weight_shape():
    rng = np.random.default_rng(0)
    n = 50
    d = 8
    X = rng.standard_normal((n, d))
    y = rng.integers(0, 2, size=n).astype(float)
    acc, weights = train_linear_probe(X, y)
    assert weights.shape == (d,)


def test_train_linear_probe_perfect_separation():
    n = 50
    X = np.zeros((n, 2))
    X[:n // 2, 0] = 10.0
    X[n // 2:, 0] = -10.0
    y = np.zeros(n)
    y[:n // 2] = 1.0
    acc, weights = train_linear_probe(X, y)
    assert acc == pytest.approx(1.0)
