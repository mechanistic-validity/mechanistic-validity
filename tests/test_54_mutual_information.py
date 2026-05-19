import importlib
import math

import numpy as np
import pytest

from mechanistic_validity.metrics.common import EvalResult, load_model

_mod = importlib.import_module(
    "mechanistic_validity.metrics.information.mutual_info.54_mutual_information"
)
quantile_bin = _mod.quantile_bin
binned_mi = _mod.binned_mi
run_mutual_information = _mod.run_mutual_information

TASK = "ioi"


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


@pytest.fixture(scope="module")
def circuit_results(gpt2_model):
    return run_mutual_information(gpt2_model, [TASK], n_prompts=3)


def test_quantile_bin_output_range():
    values = np.arange(100, dtype=float)
    binned = quantile_bin(values, n_bins=10)
    assert binned.min() >= 0
    assert binned.max() <= 10


def test_quantile_bin_few_values():
    values = np.array([1.0, 2.0])
    binned = quantile_bin(values, n_bins=10)
    assert (binned == 0).all()


def test_binned_mi_identical_variables():
    x = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    mi = binned_mi(x, x.copy())
    assert mi > 0.0
    expected_h = -4 * (0.25 * np.log2(0.25))
    assert mi == pytest.approx(expected_h, abs=1e-6)


def test_binned_mi_independent_variables():
    rng = np.random.RandomState(42)
    n = 10000
    x = rng.randint(0, 5, size=n)
    y = rng.randint(0, 5, size=n)
    mi = binned_mi(x, y)
    assert mi < 0.05


def test_binned_mi_nonnegative():
    rng = np.random.RandomState(123)
    for _ in range(10):
        x = rng.randint(0, 4, size=50)
        y = rng.randint(0, 4, size=50)
        assert binned_mi(x, y) >= 0.0


def test_binned_mi_deterministic_relationship():
    x = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    y = x * 2
    mi = binned_mi(x, y)
    assert mi > 1.0


def test_run_mutual_information_returns_eval_results(circuit_results):
    assert len(circuit_results) >= 1
    assert isinstance(circuit_results[0], EvalResult)


def test_run_mutual_information_metric_id(circuit_results):
    assert circuit_results[0].metric_id == "C4.mutual_information"


def test_run_mutual_information_value_is_finite(circuit_results):
    assert math.isfinite(circuit_results[0].value)


def test_run_mutual_information_metadata_keys(circuit_results):
    expected = {"task", "mean_within_circuit_mi", "mean_between_circuit_random_mi",
                "ratio", "n_circuit_pairs", "n_between_pairs", "top_mi_edges", "n_bins"}
    assert expected <= set(circuit_results[0].metadata.keys())


def test_run_mutual_information_task_matches(circuit_results):
    assert circuit_results[0].metadata["task"] == TASK


def test_run_mutual_information_n_samples(circuit_results):
    assert circuit_results[0].n_samples > 0
    assert circuit_results[0].n_samples <= 3


def test_run_mutual_information_baseline_is_one(circuit_results):
    assert circuit_results[0].baseline_random == pytest.approx(1.0)
