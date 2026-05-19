import importlib
import math

import numpy as np
import pytest

from mechval.metrics.common import EvalResult, load_model

_mod = importlib.import_module(
    "mechval.metrics.information.conditional_mi.55_conditional_mi"
)
quantile_bin = _mod.quantile_bin
binned_mi = _mod.binned_mi
residualize = _mod.residualize
run_conditional_mi = _mod.run_conditional_mi

TASK = "ioi"


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


@pytest.fixture(scope="module")
def circuit_results(gpt2_model):
    return run_conditional_mi(gpt2_model, [TASK], n_prompts=3, max_triplets=5)


def test_residualize_removes_linear_dependence():
    rng = np.random.RandomState(42)
    z = rng.randn(100)
    x = z * 3.0 + 1.0
    residuals = residualize(x, z)
    assert np.abs(np.corrcoef(residuals, z)[0, 1]) < 1e-6


def test_residualize_preserves_orthogonal_component():
    rng = np.random.RandomState(42)
    z = rng.randn(100)
    noise = rng.randn(100)
    x = z * 2.0 + noise
    residuals = residualize(x, z)
    assert np.std(residuals) > 0.5


def test_residualize_constant_z():
    z = np.ones(50)
    x = np.arange(50, dtype=float)
    residuals = residualize(x, z)
    x_centered = x - x.mean()
    assert np.allclose(residuals, x_centered, atol=1e-10)


def test_binned_mi_after_residualization_drops():
    rng = np.random.RandomState(42)
    n = 200
    z = rng.randn(n)
    x = z + rng.randn(n) * 0.1
    y = z + rng.randn(n) * 0.1

    x_b = quantile_bin(x)
    y_b = quantile_bin(y)
    mi_raw = binned_mi(x_b, y_b)

    res_x = residualize(x, z)
    res_y = residualize(y, z)
    rx_b = quantile_bin(res_x)
    ry_b = quantile_bin(res_y)
    mi_cond = binned_mi(rx_b, ry_b)

    assert mi_cond < mi_raw


def test_run_conditional_mi_returns_eval_results(circuit_results):
    assert len(circuit_results) >= 1
    assert isinstance(circuit_results[0], EvalResult)


def test_run_conditional_mi_metric_id(circuit_results):
    assert circuit_results[0].metric_id == "C5.conditional_mi"


def test_run_conditional_mi_value_is_finite(circuit_results):
    assert math.isfinite(circuit_results[0].value)


def test_run_conditional_mi_value_between_zero_and_one(circuit_results):
    assert 0.0 <= circuit_results[0].value <= 1.0


def test_run_conditional_mi_metadata_keys(circuit_results):
    expected = {"task", "mean_direct_fraction", "mean_mediated_fraction",
                "n_pairs_analyzed", "top_mediated", "top_direct"}
    assert expected <= set(circuit_results[0].metadata.keys())


def test_run_conditional_mi_direct_plus_mediated_is_one(circuit_results):
    m = circuit_results[0].metadata
    assert m["mean_direct_fraction"] + m["mean_mediated_fraction"] == pytest.approx(1.0)


def test_run_conditional_mi_task_matches(circuit_results):
    assert circuit_results[0].metadata["task"] == TASK


def test_run_conditional_mi_n_samples(circuit_results):
    assert circuit_results[0].n_samples > 0
    assert circuit_results[0].n_samples <= 3
