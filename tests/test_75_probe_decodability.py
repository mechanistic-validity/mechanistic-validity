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
    / "linear_probe"
    / "75_probe_decodability.py"
)
_spec = importlib.util.spec_from_file_location("probe_75", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["probe_75"] = _mod
_spec.loader.exec_module(_mod)

train_logistic_probe = _mod.train_logistic_probe
generate_control_labels = _mod.generate_control_labels
split_train_test = _mod.split_train_test
_sigmoid = _mod._sigmoid

TASK = "ioi"


def test_sigmoid_zero():
    assert _sigmoid(np.array([0.0]))[0] == pytest.approx(0.5)


def test_sigmoid_large_positive():
    result = _sigmoid(np.array([100.0]))
    assert result[0] == pytest.approx(1.0, abs=1e-6)


def test_sigmoid_large_negative():
    result = _sigmoid(np.array([-100.0]))
    assert result[0] == pytest.approx(0.0, abs=1e-6)


def test_sigmoid_symmetry():
    x = np.linspace(-5, 5, 100)
    s = _sigmoid(x)
    s_neg = _sigmoid(-x)
    assert s + s_neg == pytest.approx(np.ones(100), abs=1e-10)


def test_logistic_probe_separable():
    rng = np.random.default_rng(0)
    n = 200
    d = 10
    X = rng.standard_normal((n, d))
    w_true = np.zeros(d)
    w_true[0] = 5.0
    y = (X @ w_true > 0).astype(float)
    split = int(n * 0.7)
    acc = train_logistic_probe(X[:split], y[:split], X[split:], y[split:],
                               lr=0.1, epochs=200)
    assert acc > 0.85


def test_logistic_probe_random_labels():
    rng = np.random.default_rng(0)
    n = 200
    d = 10
    X = rng.standard_normal((n, d))
    y = rng.integers(0, 2, size=n).astype(float)
    split = int(n * 0.7)
    acc = train_logistic_probe(X[:split], y[:split], X[split:], y[split:])
    assert 0.3 < acc < 0.7


def test_logistic_probe_too_few_samples():
    X_train = np.random.randn(2, 5)
    y_train = np.array([0.0, 1.0])
    X_test = np.random.randn(1, 5)
    y_test = np.array([0.0])
    acc = train_logistic_probe(X_train, y_train, X_test, y_test)
    assert acc == pytest.approx(0.5)


def test_generate_control_labels_binary():
    labels = generate_control_labels(100)
    assert set(labels).issubset({0.0, 1.0})
    assert len(labels) == 100


def test_generate_control_labels_roughly_balanced():
    labels = generate_control_labels(1000, seed=0)
    mean = labels.mean()
    assert 0.4 < mean < 0.6


def test_generate_control_labels_deterministic():
    a = generate_control_labels(50, seed=99)
    b = generate_control_labels(50, seed=99)
    assert np.array_equal(a, b)


def test_split_train_test_sizes():
    X = np.random.randn(100, 5)
    y = np.random.randn(100)
    X_tr, y_tr, X_te, y_te = split_train_test(X, y, frac=0.7)
    assert X_tr.shape[0] == 70
    assert X_te.shape[0] == 30
    assert y_tr.shape[0] == 70
    assert y_te.shape[0] == 30


def test_split_train_test_no_overlap():
    X = np.arange(50).reshape(50, 1).astype(float)
    y = np.arange(50).astype(float)
    X_tr, y_tr, X_te, y_te = split_train_test(X, y, frac=0.6)
    all_vals = set(X_tr[:, 0].tolist()) | set(X_te[:, 0].tolist())
    assert len(all_vals) == 50
    assert len(set(X_tr[:, 0].tolist()) & set(X_te[:, 0].tolist())) == 0
