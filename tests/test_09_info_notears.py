import importlib
import math

import numpy as np
import pytest

from mechanistic_validity.metrics.common import EvalResult, load_model

_mod = importlib.import_module(
    "mechanistic_validity.metrics.information.notears_dag.09_notears"
)
notears_linear = _mod.notears_linear
compute_shd = _mod.compute_shd
run_notears = _mod.run_notears

TASK = "ioi"


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


@pytest.fixture(scope="module")
def circuit_results(gpt2_model):
    return run_notears(gpt2_model, [TASK], n_prompts=5)


def test_notears_linear_output_shape():
    rng = np.random.RandomState(42)
    X = rng.randn(100, 5)
    W = notears_linear(X, lambda1=0.1, max_iter=10)
    assert W.shape == (5, 5)


def test_notears_linear_approximately_acyclic():
    rng = np.random.RandomState(42)
    n, d = 200, 4
    W_true = np.zeros((d, d))
    W_true[0, 1] = 1.0
    W_true[1, 2] = 0.8
    X = rng.randn(n, d)
    for j in range(d):
        X[:, j] += X @ W_true[:, j]

    W_est = notears_linear(X, lambda1=0.01, max_iter=50)
    from scipy.linalg import expm
    h = np.trace(expm(W_est * W_est)) - d
    assert h < 1.0


def test_notears_linear_sparse_output():
    rng = np.random.RandomState(42)
    X = rng.randn(100, 5)
    W = notears_linear(X, lambda1=0.1, max_iter=20, w_threshold=0.1)
    n_nonzero = np.count_nonzero(W)
    assert n_nonzero < 25


def test_notears_linear_identity_noise():
    rng = np.random.RandomState(42)
    X = rng.randn(200, 3)
    W = notears_linear(X, lambda1=0.1, max_iter=30, w_threshold=0.1)
    assert np.count_nonzero(W) <= 3


def test_compute_shd_identical():
    W = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    assert compute_shd(W, W) == 0


def test_compute_shd_no_overlap():
    W1 = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
    W2 = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]])
    assert compute_shd(W1, W2) == 2


def test_compute_shd_extra_edges():
    W_true = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
    W_est = np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]])
    assert compute_shd(W_true, W_est) == 1


def test_compute_shd_missing_edges():
    W_true = np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]])
    W_est = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
    assert compute_shd(W_true, W_est) == 1


def test_compute_shd_symmetric():
    W1 = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    W2 = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 0]])
    assert compute_shd(W1, W2) == compute_shd(W2, W1)


def test_run_notears_returns_eval_results(circuit_results):
    assert len(circuit_results) >= 1
    assert isinstance(circuit_results[0], EvalResult)


def test_run_notears_metric_id(circuit_results):
    assert circuit_results[0].metric_id == "C9.notears"


def test_run_notears_value_between_zero_and_one(circuit_results):
    assert 0.0 <= circuit_results[0].value <= 1.0


def test_run_notears_value_is_finite(circuit_results):
    assert math.isfinite(circuit_results[0].value)


def test_run_notears_metadata_keys(circuit_results):
    expected = {"task", "precision", "recall", "discovered_heads",
                "n_parents", "n_circuit_heads", "n_total_heads_tested"}
    assert expected <= set(circuit_results[0].metadata.keys())


def test_run_notears_precision_recall_bounded(circuit_results):
    m = circuit_results[0].metadata
    assert 0.0 <= m["precision"] <= 1.0
    assert 0.0 <= m["recall"] <= 1.0


def test_run_notears_task_matches(circuit_results):
    assert circuit_results[0].metadata["task"] == TASK


def test_run_notears_n_samples(circuit_results):
    assert circuit_results[0].n_samples > 0
    assert circuit_results[0].n_samples <= 5


def test_run_notears_discovered_heads_format(circuit_results):
    heads = circuit_results[0].metadata["discovered_heads"]
    assert isinstance(heads, list)
    for h in heads:
        assert isinstance(h, str)
        assert h.startswith("L")
        assert "H" in h
