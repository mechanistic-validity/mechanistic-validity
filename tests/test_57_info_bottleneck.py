import importlib
import math

import numpy as np
import pytest

from mechval.metrics.common import EvalResult, load_model

_mod = importlib.import_module(
    "mechval.metrics.information.info_bottleneck.57_info_bottleneck"
)
quantile_bin = _mod.quantile_bin
mi_continuous_discrete = _mod.mi_continuous_discrete
run_info_bottleneck = _mod.run_info_bottleneck

TASK = "ioi"


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


@pytest.fixture(scope="module")
def circuit_results(gpt2_model):
    return run_info_bottleneck(gpt2_model, [TASK], n_prompts=3)


def test_mi_continuous_discrete_informative_features():
    rng = np.random.RandomState(42)
    n = 200
    y = rng.randint(0, 2, size=n)
    X = rng.randn(n, 20)
    X[:, 0] = y * 5.0 + rng.randn(n) * 0.1

    mi = mi_continuous_discrete(X, y)
    assert mi > 0.1


def test_mi_continuous_discrete_independent():
    rng = np.random.RandomState(42)
    n = 200
    X = rng.randn(n, 20)
    y = rng.randint(0, 2, size=n)

    mi = mi_continuous_discrete(X, y)
    assert mi < 2.0


def test_mi_continuous_discrete_nonnegative():
    rng = np.random.RandomState(42)
    X = rng.randn(100, 10)
    y = rng.randint(0, 3, size=100)
    mi = mi_continuous_discrete(X, y)
    assert mi >= 0.0


def test_mi_continuous_discrete_too_few_samples():
    X = np.array([[1.0], [2.0]])
    y = np.array([0, 1])
    mi = mi_continuous_discrete(X, y, n_bins=8)
    assert mi == pytest.approx(0.0)


def test_quantile_bin_preserves_ordering():
    values = np.arange(100, dtype=float)
    binned = quantile_bin(values, n_bins=8)
    for i in range(len(values) - 1):
        assert binned[i] <= binned[i + 1]


def test_run_info_bottleneck_returns_eval_results(circuit_results):
    assert len(circuit_results) >= 1
    assert isinstance(circuit_results[0], EvalResult)


def test_run_info_bottleneck_metric_id(circuit_results):
    assert circuit_results[0].metric_id == "C8.info_bottleneck"


def test_run_info_bottleneck_value_is_finite(circuit_results):
    assert math.isfinite(circuit_results[0].value)


def test_run_info_bottleneck_value_nonnegative(circuit_results):
    assert circuit_results[0].value >= 0.0


def test_run_info_bottleneck_metadata_keys(circuit_results):
    expected = {"task", "info_plane", "mean_circuit_I_T_Y",
                "mean_non_circuit_I_T_Y", "circuit_layers", "n_pca_dims"}
    assert expected <= set(circuit_results[0].metadata.keys())


def test_run_info_bottleneck_info_plane_has_all_layers(circuit_results, gpt2_model):
    info_plane = circuit_results[0].metadata["info_plane"]
    assert len(info_plane) == gpt2_model.cfg.n_layers
    for entry in info_plane:
        assert "layer" in entry
        assert "I_X_T" in entry
        assert "I_T_Y" in entry
        assert "is_circuit_layer" in entry


def test_run_info_bottleneck_task_matches(circuit_results):
    assert circuit_results[0].metadata["task"] == TASK


def test_run_info_bottleneck_n_samples(circuit_results):
    assert circuit_results[0].n_samples > 0
    assert circuit_results[0].n_samples <= 3
