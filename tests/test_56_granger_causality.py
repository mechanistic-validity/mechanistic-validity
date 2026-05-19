import importlib
import math

import numpy as np
import pytest

from mechanistic_validity.instruments.common import EvalResult, load_model

_mod = importlib.import_module(
    "mechanistic_validity.instruments.information.granger.56_granger_causality"
)
granger_f_test = _mod.granger_f_test
run_granger = _mod.run_granger

TASK = "ioi"


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


@pytest.fixture(scope="module")
def circuit_results(gpt2_model):
    return run_granger(gpt2_model, [TASK], n_prompts=3, alpha=0.05)


def test_granger_f_test_significant_predictor():
    rng = np.random.RandomState(42)
    n = 100
    x = rng.randn(n)
    y = x * 3.0 + rng.randn(n) * 0.1

    x_full = x.reshape(-1, 1)
    x_restricted = np.empty((n, 0))

    f_stat, p_val = granger_f_test(y, x_full, x_restricted)
    assert f_stat > 10.0
    assert p_val < 0.01


def test_granger_f_test_irrelevant_predictor():
    rng = np.random.RandomState(42)
    n = 100
    x = rng.randn(n)
    y = rng.randn(n)

    x_full = x.reshape(-1, 1)
    x_restricted = np.empty((n, 0))

    f_stat, p_val = granger_f_test(y, x_full, x_restricted)
    assert p_val > 0.01


def test_granger_f_test_incremental_above_confound():
    rng = np.random.RandomState(42)
    n = 200
    z = rng.randn(n)
    x = rng.randn(n)
    y = z * 2.0 + x * 1.0 + rng.randn(n) * 0.5

    x_restricted = z.reshape(-1, 1)
    x_full = np.column_stack([z, x])

    f_stat, p_val = granger_f_test(y, x_full, x_restricted)
    assert f_stat > 5.0
    assert p_val < 0.05


def test_granger_f_test_returns_floats():
    rng = np.random.RandomState(42)
    n = 50
    y = rng.randn(n)
    x_full = rng.randn(n, 2)
    x_restricted = rng.randn(n, 1)

    f_stat, p_val = granger_f_test(y, x_full, x_restricted)
    assert isinstance(f_stat, float)
    assert isinstance(p_val, float)
    assert f_stat >= 0.0
    assert 0.0 <= p_val <= 1.0


def test_granger_f_test_degenerate_empty_restricted():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    x_full = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    x_restricted = np.empty((5, 0))

    f_stat, p_val = granger_f_test(y, x_full, x_restricted)
    assert math.isfinite(f_stat)
    assert math.isfinite(p_val)


def test_run_granger_returns_eval_results(circuit_results):
    assert len(circuit_results) >= 1
    assert isinstance(circuit_results[0], EvalResult)


def test_run_granger_metric_id(circuit_results):
    assert circuit_results[0].metric_id == "C7.granger_causality"


def test_run_granger_value_between_zero_and_one(circuit_results):
    assert 0.0 <= circuit_results[0].value <= 1.0


def test_run_granger_value_is_finite(circuit_results):
    assert math.isfinite(circuit_results[0].value)


def test_run_granger_metadata_keys(circuit_results):
    expected = {"task", "circuit_significance_rate", "non_circuit_significance_rate",
                "n_circuit_edges_tested", "n_non_circuit_edges_tested",
                "alpha", "top_significant"}
    assert expected <= set(circuit_results[0].metadata.keys())


def test_run_granger_task_matches(circuit_results):
    assert circuit_results[0].metadata["task"] == TASK


def test_run_granger_n_samples(circuit_results):
    assert circuit_results[0].n_samples > 0
    assert circuit_results[0].n_samples <= 3
