import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from mechanistic_validity.instruments.common import EvalResult, load_model

_CKA_PATH = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "mechanistic_validity"
    / "instruments"
    / "representational"
    / "cka"
    / "92_cka.py"
)
_spec = importlib.util.spec_from_file_location("_cka", _CKA_PATH)
_cka_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _cka_mod
_spec.loader.exec_module(_cka_mod)

linear_cka = _cka_mod.linear_cka
run_cka = _cka_mod.run_cka


def test_linear_cka_identical_matrices():
    X = np.random.randn(50, 10)
    X_centered = X - X.mean(axis=0, keepdims=True)
    assert linear_cka(X_centered, X_centered) == pytest.approx(1.0, abs=1e-6)


def test_linear_cka_orthogonal_representations():
    n = 200
    d = 6
    M = np.random.randn(n, d)
    M_centered = M - M.mean(axis=0, keepdims=True)
    U, _, _ = np.linalg.svd(M_centered, full_matrices=False)
    X = U[:, :3]
    Y = U[:, 3:]
    assert linear_cka(X, Y) == pytest.approx(0.0, abs=1e-6)


def test_linear_cka_random_matrices_between_zero_and_one():
    results = []
    for _ in range(20):
        X = np.random.randn(50, 8)
        Y = np.random.randn(50, 8)
        X_centered = X - X.mean(axis=0, keepdims=True)
        Y_centered = Y - Y.mean(axis=0, keepdims=True)
        results.append(linear_cka(X_centered, Y_centered))
    for cka in results:
        assert 0.0 <= cka <= 1.0
    assert min(results) < 0.5
    assert max(results) > 0.0


def test_linear_cka_symmetry():
    for _ in range(20):
        X = np.random.randn(50, 8)
        Y = np.random.randn(50, 6)
        X_centered = X - X.mean(axis=0, keepdims=True)
        Y_centered = Y - Y.mean(axis=0, keepdims=True)
        assert linear_cka(X_centered, Y_centered) == pytest.approx(
            linear_cka(Y_centered, X_centered), abs=1e-10
        )


def test_linear_cka_scaled_matrix():
    X = np.random.randn(50, 10)
    X_centered = X - X.mean(axis=0, keepdims=True)
    Y_centered = X_centered * 3.7
    assert linear_cka(X_centered, Y_centered) == pytest.approx(1.0, abs=1e-6)


def test_linear_cka_zero_matrix():
    X = np.random.randn(50, 10)
    X_centered = X - X.mean(axis=0, keepdims=True)
    Z = np.zeros((50, 10))
    assert linear_cka(X_centered, Z) == pytest.approx(0.0, abs=1e-6)


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


TASK = "ioi"


def test_run_cka_returns_eval_result(gpt2_model):
    result = run_cka(gpt2_model, TASK, n_prompts=5)
    assert result is not None
    assert isinstance(result, EvalResult)
    assert result.metric_id == "E92.cka"
    assert 0.0 <= result.value <= 1.0
    assert result.n_samples == 5
    assert "mean_cka_circuit_layers" in result.metadata
    assert 0.0 <= result.metadata["mean_cka_circuit_layers"] <= 1.0
    assert "per_layer_cka" in result.metadata
    for cka_val in result.metadata["per_layer_cka"]:
        assert 0.0 <= cka_val <= 1.0
