import importlib
import math

import numpy as np
import pytest

from mechval.metrics.common import EvalResult, load_model

_mod = importlib.import_module(
    "mechval.metrics.information.osce.07_ocse"
)
stability_selection = _mod.stability_selection
gaussian_cmi = _mod.gaussian_cmi
compute_ocse_parents = _mod.compute_ocse_parents
_compute_f1 = _mod._compute_f1
run_ocse = _mod.run_ocse

TASK = "ioi"


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


@pytest.fixture(scope="module")
def circuit_results(gpt2_model):
    return run_ocse(gpt2_model, [TASK], n_prompts=5)


def test_gaussian_cmi_perfect_correlation():
    x = np.arange(100, dtype=float)
    y = x * 2.0 + 1.0
    cmi = gaussian_cmi(x, y)
    assert cmi > 2.0


def test_gaussian_cmi_independent():
    rng = np.random.RandomState(42)
    x = rng.randn(200)
    y = rng.randn(200)
    cmi = gaussian_cmi(x, y)
    assert cmi < 0.1


def test_gaussian_cmi_conditional():
    rng = np.random.RandomState(42)
    n = 200
    z = rng.randn(n)
    x = z + rng.randn(n) * 0.1
    y = z + rng.randn(n) * 0.1
    cmi_marginal = gaussian_cmi(x, y)
    cmi_conditional = gaussian_cmi(x, y, z)
    assert cmi_conditional < cmi_marginal


def test_gaussian_cmi_too_few_samples():
    x = np.array([1.0, 2.0])
    y = np.array([3.0, 4.0])
    cmi = gaussian_cmi(x, y)
    assert cmi == pytest.approx(0.0)


def test_compute_f1_perfect():
    circuit_heads = {(0, 0), (1, 1)}
    n_heads = 4
    discovered_indices = {0 * 4 + 0, 1 * 4 + 1}
    f1, precision, recall = _compute_f1(circuit_heads, discovered_indices, n_heads)
    assert f1 == pytest.approx(1.0)
    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(1.0)


def test_compute_f1_no_overlap():
    circuit_heads = {(0, 0)}
    n_heads = 4
    discovered_indices = {1 * 4 + 1}
    f1, precision, recall = _compute_f1(circuit_heads, discovered_indices, n_heads)
    assert f1 == pytest.approx(0.0)
    assert precision == pytest.approx(0.0)
    assert recall == pytest.approx(0.0)


def test_compute_f1_partial_overlap():
    circuit_heads = {(0, 0), (1, 0)}
    n_heads = 4
    discovered_indices = {0 * 4 + 0, 2 * 4 + 0}
    f1, precision, recall = _compute_f1(circuit_heads, discovered_indices, n_heads)
    assert precision == pytest.approx(0.5)
    assert recall == pytest.approx(0.5)
    assert f1 == pytest.approx(0.5)


def test_compute_f1_empty_discovered():
    circuit_heads = {(0, 0)}
    n_heads = 4
    discovered_indices = set()
    f1, precision, recall = _compute_f1(circuit_heads, discovered_indices, n_heads)
    assert f1 == pytest.approx(0.0)


def test_stability_selection_returns_list():
    rng = np.random.RandomState(42)
    n = 50
    X = rng.randn(n, 10)
    y = X[:, 0] * 3.0 + rng.randn(n) * 0.5
    result = stability_selection(X, y, n_bootstrap=5, threshold=0.3)
    assert isinstance(result, list)
    for idx, freq, coef in result:
        assert isinstance(idx, int)
        assert 0.0 <= freq <= 1.0
        assert math.isfinite(coef)


def test_stability_selection_finds_true_predictor():
    rng = np.random.RandomState(42)
    n = 100
    X = rng.randn(n, 10)
    y = X[:, 0] * 5.0 + rng.randn(n) * 0.1
    result = stability_selection(X, y, n_bootstrap=10, threshold=0.5)
    selected_indices = {idx for idx, _, _ in result}
    assert 0 in selected_indices


def test_run_ocse_returns_eval_results(circuit_results):
    assert len(circuit_results) >= 1
    assert isinstance(circuit_results[0], EvalResult)


def test_run_ocse_metric_id(circuit_results):
    assert circuit_results[0].metric_id == "C7.ocse"


def test_run_ocse_value_is_finite(circuit_results):
    assert math.isfinite(circuit_results[0].value)


def test_run_ocse_value_between_zero_and_one(circuit_results):
    assert 0.0 <= circuit_results[0].value <= 1.0


def test_run_ocse_metadata_keys(circuit_results):
    expected = {"task", "precision", "recall", "stability_selection",
                "ocse", "combined", "n_circuit_heads", "n_discovered"}
    assert expected <= set(circuit_results[0].metadata.keys())


def test_run_ocse_task_matches(circuit_results):
    assert circuit_results[0].metadata["task"] == TASK


def test_run_ocse_n_samples(circuit_results):
    assert circuit_results[0].n_samples > 0
    assert circuit_results[0].n_samples <= 5
