import importlib
import math

import numpy as np
import pytest

from mechanistic_validity.metrics.common import EvalResult, load_model

_mod = importlib.import_module(
    "mechanistic_validity.metrics.information.synergistic_info.58_o_information"
)
quantile_bin = _mod.quantile_bin
entropy_binned = _mod.entropy_binned
o_information = _mod.o_information
run_o_information = _mod.run_o_information

TASK = "ioi"


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


@pytest.fixture(scope="module")
def circuit_results(gpt2_model):
    return run_o_information(gpt2_model, [TASK], n_prompts=3, n_random_baselines=5)


def test_entropy_binned_uniform():
    n = 1000
    x = np.tile(np.arange(4), n // 4)
    h = entropy_binned([x])
    assert h == pytest.approx(2.0, abs=0.01)


def test_entropy_binned_deterministic():
    x = np.zeros(100, dtype=int)
    h = entropy_binned([x])
    assert h == pytest.approx(0.0)


def test_entropy_binned_joint_increases():
    rng = np.random.RandomState(42)
    x = rng.randint(0, 4, size=1000)
    y = rng.randint(0, 4, size=1000)
    h_x = entropy_binned([x])
    h_joint = entropy_binned([x, y])
    assert h_joint >= h_x - 1e-10


def test_o_information_fewer_than_three():
    x = [np.array([0, 1, 0, 1])]
    y = [np.array([0, 1, 0, 1]), np.array([1, 0, 1, 0])]
    assert o_information(x) == pytest.approx(0.0)
    assert o_information(y) == pytest.approx(0.0)


def test_o_information_redundant_copies():
    n = 200
    x = np.tile(np.arange(4), n // 4)
    variables = [x.copy(), x.copy(), x.copy()]
    omega = o_information(variables)
    assert omega > 0.0


def test_o_information_independent_variables():
    rng = np.random.RandomState(42)
    n = 2000
    variables = [rng.randint(0, 4, size=n) for _ in range(4)]
    omega = o_information(variables)
    assert abs(omega) < 0.5


def test_o_information_returns_float():
    rng = np.random.RandomState(42)
    variables = [rng.randint(0, 3, size=100) for _ in range(4)]
    result = o_information(variables)
    assert isinstance(result, float)


def test_run_o_information_returns_eval_results(circuit_results):
    assert len(circuit_results) >= 1
    assert isinstance(circuit_results[0], EvalResult)


def test_run_o_information_metric_id(circuit_results):
    assert circuit_results[0].metric_id == "C9.o_information"


def test_run_o_information_value_is_finite(circuit_results):
    assert math.isfinite(circuit_results[0].value)


def test_run_o_information_metadata_keys(circuit_results):
    expected = {"task", "omega_circuit", "omega_random_mean", "omega_random_std",
                "z_score", "interpretation", "n_heads_used",
                "n_circuit_heads", "n_random_baselines"}
    assert expected <= set(circuit_results[0].metadata.keys())


def test_run_o_information_interpretation_valid(circuit_results):
    assert circuit_results[0].metadata["interpretation"] in {"synergy", "redundancy"}


def test_run_o_information_task_matches(circuit_results):
    assert circuit_results[0].metadata["task"] == TASK


def test_run_o_information_n_samples(circuit_results):
    assert circuit_results[0].n_samples > 0
    assert circuit_results[0].n_samples <= 3


def test_run_o_information_z_score_is_finite(circuit_results):
    assert math.isfinite(circuit_results[0].metadata["z_score"])
