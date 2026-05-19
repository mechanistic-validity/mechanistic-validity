import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechanistic_validity" / "instruments"
    / "causal" / "causal_discovery" / "09_notears.py"
)
_spec = importlib.util.spec_from_file_location("notears_09", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["notears_09"] = _mod
_spec.loader.exec_module(_mod)

run_notears = _mod.run_notears
notears_linear = _mod.notears_linear
compute_shd = _mod.compute_shd

from mechanistic_validity.instruments.common import EvalResult, load_model

TASK = "ioi"


def test_notears_linear_returns_square_matrix():
    rng = np.random.RandomState(42)
    n, d = 50, 4
    X = rng.randn(n, d)
    W = notears_linear(X, lambda1=0.1, max_iter=10, w_threshold=0.1)
    assert W.shape == (d, d)


def test_notears_linear_zero_diagonal():
    rng = np.random.RandomState(42)
    n, d = 50, 4
    X = rng.randn(n, d)
    W = notears_linear(X, lambda1=0.1, max_iter=10, w_threshold=0.5)
    for i in range(d):
        assert W[i, i] == pytest.approx(0.0, abs=0.5)


def test_notears_linear_sparse_output():
    rng = np.random.RandomState(42)
    n, d = 100, 5
    X = rng.randn(n, d)
    W = notears_linear(X, lambda1=1.0, max_iter=10, w_threshold=0.5)
    n_nonzero = np.count_nonzero(W)
    assert n_nonzero < d * d


def test_compute_shd_identical():
    W = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    assert compute_shd(W, W) == 0


def test_compute_shd_completely_different():
    W_true = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
    W_est = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]])
    assert compute_shd(W_true, W_est) == 2


def test_compute_shd_empty():
    W = np.zeros((3, 3))
    assert compute_shd(W, W) == 0


def test_compute_shd_symmetric():
    W1 = np.array([[0, 1], [0, 0]])
    W2 = np.array([[0, 0], [1, 0]])
    assert compute_shd(W1, W2) == compute_shd(W2, W1)


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


@pytest.fixture(scope="module")
def circuit_results(gpt2_model):
    return run_notears(gpt2_model, tasks=[TASK], n_prompts=5)


def test_run_returns_eval_results(circuit_results):
    assert len(circuit_results) == 1
    assert isinstance(circuit_results[0], EvalResult)


def test_metric_id(circuit_results):
    assert circuit_results[0].metric_id == "C9.notears"


def test_f1_in_range(circuit_results):
    assert 0.0 <= circuit_results[0].value <= 1.0


def test_metadata_keys(circuit_results):
    meta = circuit_results[0].metadata
    expected = {"task", "precision", "recall", "discovered_heads",
                "n_parents", "n_circuit_heads", "n_total_heads_tested"}
    assert set(meta.keys()) == expected


def test_precision_recall_in_range(circuit_results):
    meta = circuit_results[0].metadata
    assert 0.0 <= meta["precision"] <= 1.0
    assert 0.0 <= meta["recall"] <= 1.0


def test_n_samples_positive(circuit_results):
    assert circuit_results[0].n_samples > 0
