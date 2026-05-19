import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "mechanistic_validity"
    / "instruments"
    / "representational"
    / "cka"
    / "E6b_cka_cross_arch.py"
)
_spec = importlib.util.spec_from_file_location("cka_e6b", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["cka_e6b"] = _mod
_spec.loader.exec_module(_mod)

linear_cka = _mod.linear_cka
run_cka_analysis = _mod.run_cka_analysis

from mechanistic_validity.instruments.common import EvalResult, load_model

TASK = "ioi"


def test_linear_cka_identical_matrices():
    X = np.random.randn(50, 10)
    assert linear_cka(X, X) == pytest.approx(1.0, abs=1e-6)


def test_linear_cka_scaled_matrix():
    X = np.random.randn(50, 10)
    Y = X * 4.2
    assert linear_cka(X, Y) == pytest.approx(1.0, abs=1e-6)


def test_linear_cka_orthogonal_representations():
    n = 200
    d = 6
    M = np.random.randn(n, d)
    M_centered = M - M.mean(axis=0, keepdims=True)
    U, _, _ = np.linalg.svd(M_centered, full_matrices=False)
    X = U[:, :3]
    Y = U[:, 3:]
    assert linear_cka(X, Y) == pytest.approx(0.0, abs=1e-6)


def test_linear_cka_symmetry():
    for _ in range(20):
        X = np.random.randn(50, 8)
        Y = np.random.randn(50, 6)
        assert linear_cka(X, Y) == pytest.approx(linear_cka(Y, X), abs=1e-10)


def test_linear_cka_random_between_zero_and_one():
    results = []
    for _ in range(20):
        X = np.random.randn(50, 8)
        Y = np.random.randn(50, 8)
        results.append(linear_cka(X, Y))
    for cka in results:
        assert 0.0 <= cka <= 1.0 + 1e-10
    assert min(results) < 0.5


def test_linear_cka_zero_matrix():
    X = np.random.randn(50, 10)
    Z = np.zeros((50, 10))
    assert linear_cka(X, Z) == pytest.approx(0.0, abs=1e-6)


def test_linear_cka_single_sample():
    X = np.random.randn(1, 10)
    Y = np.random.randn(1, 10)
    assert linear_cka(X, Y) == pytest.approx(0.0, abs=1e-6)


def test_linear_cka_different_dimensions():
    X = np.random.randn(50, 4)
    Y = np.random.randn(50, 12)
    cka = linear_cka(X, Y)
    assert 0.0 <= cka <= 1.0 + 1e-10


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


@pytest.fixture(scope="module")
def circuit_results(gpt2_model):
    return run_cka_analysis(gpt2_model, tasks=[TASK], n_prompts=5)


def test_run_cka_analysis_returns_eval_results(circuit_results):
    assert len(circuit_results) >= 1
    for r in circuit_results:
        assert isinstance(r, EvalResult)


def test_run_cka_analysis_metric_id(circuit_results):
    for r in circuit_results:
        assert r.metric_id == "E6b.cka_cross_layer"


def test_run_cka_analysis_value_in_range(circuit_results):
    for r in circuit_results:
        assert 0.0 <= r.value <= 1.0 + 1e-10


def test_run_cka_analysis_metadata_has_task(circuit_results):
    for r in circuit_results:
        assert r.metadata["task"] == TASK


def test_run_cka_analysis_metadata_has_cka_matrix(circuit_results):
    for r in circuit_results:
        assert "cka_matrix" in r.metadata
        matrix = r.metadata["cka_matrix"]
        assert isinstance(matrix, list)
        n = len(matrix)
        for row in matrix:
            assert len(row) == n


def test_run_cka_analysis_metadata_has_circuit_layers(circuit_results):
    for r in circuit_results:
        assert "circuit_layers" in r.metadata
        assert isinstance(r.metadata["circuit_layers"], list)
        assert len(r.metadata["circuit_layers"]) >= 2


def test_run_cka_analysis_metadata_has_first_last_cka(circuit_results):
    for r in circuit_results:
        assert "first_last_cka" in r.metadata
        assert 0.0 <= r.metadata["first_last_cka"] <= 1.0 + 1e-10


def test_run_cka_analysis_n_samples_positive(circuit_results):
    for r in circuit_results:
        assert r.n_samples > 0
