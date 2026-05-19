import importlib
import math

import numpy as np
import pytest

from mechval.metrics.common import EvalResult, load_model

_mod = importlib.import_module(
    "mechval.metrics.information.pid.08_pid"
)
quantile_bin = _mod.quantile_bin
compute_pid_manual = _mod.compute_pid_manual
run_pid = _mod.run_pid

TASK = "ioi"


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


@pytest.fixture(scope="module")
def circuit_results(gpt2_model):
    return run_pid(gpt2_model, [TASK], n_prompts=3, max_pairs=5)


def test_compute_pid_manual_redundant_sources():
    n = 200
    x = np.tile(np.arange(5), n // 5)
    y = x.copy()
    z = x.copy()
    pid = compute_pid_manual(x, y, z)
    assert pid["redundancy"] > 0.0
    assert pid["unique_x"] == pytest.approx(0.0, abs=0.01)
    assert pid["unique_y"] == pytest.approx(0.0, abs=0.01)


def test_compute_pid_manual_independent_sources():
    rng = np.random.RandomState(42)
    n = 1000
    x = rng.randint(0, 5, size=n)
    y = rng.randint(0, 5, size=n)
    z = rng.randint(0, 5, size=n)
    pid = compute_pid_manual(x, y, z)
    assert pid["redundancy"] < 0.2
    assert pid["synergy"] < 0.2


def test_compute_pid_manual_nonnegative_components():
    rng = np.random.RandomState(42)
    n = 200
    x = rng.randint(0, 4, size=n)
    y = rng.randint(0, 4, size=n)
    z = rng.randint(0, 4, size=n)
    pid = compute_pid_manual(x, y, z)
    assert pid["redundancy"] >= 0.0
    assert pid["unique_x"] >= 0.0
    assert pid["unique_y"] >= 0.0
    assert pid["synergy"] >= 0.0


def test_compute_pid_manual_keys():
    x = np.array([0, 1, 0, 1, 0, 1])
    y = np.array([0, 0, 1, 1, 0, 0])
    z = np.array([0, 1, 1, 0, 0, 1])
    pid = compute_pid_manual(x, y, z)
    expected_keys = {"redundancy", "unique_x", "unique_y", "synergy", "i_xz", "i_yz"}
    assert set(pid.keys()) == expected_keys


def test_compute_pid_manual_unique_x_when_only_x_informative():
    rng = np.random.RandomState(42)
    n = 500
    z = rng.randint(0, 4, size=n)
    x = z.copy()
    y = rng.randint(0, 4, size=n)
    pid = compute_pid_manual(x, y, z)
    assert pid["i_xz"] > pid["i_yz"]


def test_quantile_bin_output_range():
    values = np.arange(100, dtype=float)
    binned = quantile_bin(values, n_bins=5)
    assert binned.min() >= 0
    assert binned.max() <= 5


def test_run_pid_returns_eval_results(circuit_results):
    assert len(circuit_results) >= 1
    assert isinstance(circuit_results[0], EvalResult)


def test_run_pid_metric_id(circuit_results):
    assert circuit_results[0].metric_id == "C8.pid"


def test_run_pid_value_is_finite(circuit_results):
    assert math.isfinite(circuit_results[0].value)


def test_run_pid_value_nonnegative(circuit_results):
    assert circuit_results[0].value >= 0.0


def test_run_pid_metadata_keys(circuit_results):
    expected = {"task", "mean_redundancy", "mean_synergy",
                "pair_decompositions", "n_pairs", "n_circuit_heads", "used_dit"}
    assert expected <= set(circuit_results[0].metadata.keys())


def test_run_pid_task_matches(circuit_results):
    assert circuit_results[0].metadata["task"] == TASK


def test_run_pid_n_samples(circuit_results):
    assert circuit_results[0].n_samples > 0
    assert circuit_results[0].n_samples <= 3


def test_run_pid_pair_decompositions_have_components(circuit_results):
    pairs = circuit_results[0].metadata["pair_decompositions"]
    if len(pairs) > 0:
        p = pairs[0]
        assert "head_a" in p
        assert "head_b" in p
        assert "redundancy" in p
        assert "synergy" in p
        assert "unique_x" in p
        assert "unique_y" in p


def test_run_pid_mean_values_consistent(circuit_results):
    m = circuit_results[0].metadata
    if m["n_pairs"] > 0:
        assert math.isfinite(m["mean_redundancy"])
        assert math.isfinite(m["mean_synergy"])
        assert m["mean_redundancy"] >= 0.0
        assert m["mean_synergy"] >= 0.0
