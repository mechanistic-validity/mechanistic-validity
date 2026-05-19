import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechval" / "metrics"
    / "causal" / "pid" / "08_pid.py"
)
_spec = importlib.util.spec_from_file_location("pid_08", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["pid_08"] = _mod
_spec.loader.exec_module(_mod)

run_pid = _mod.run_pid
quantile_bin = _mod.quantile_bin
compute_pid_manual = _mod.compute_pid_manual
N_BINS = _mod.N_BINS

from mechval.metrics.common import EvalResult, load_model

TASK = "ioi"


def test_quantile_bin_correct_range():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    binned = quantile_bin(values, n_bins=5)
    assert binned.min() >= 0
    assert binned.max() <= 4


def test_quantile_bin_few_samples():
    values = np.array([1.0, 2.0])
    binned = quantile_bin(values, n_bins=5)
    assert (binned == 0).all()


def test_quantile_bin_constant_values():
    values = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
    binned = quantile_bin(values, n_bins=3)
    assert len(binned) == 6


def test_compute_pid_manual_identical_sources():
    rng = np.random.RandomState(42)
    n = 100
    x = rng.randint(0, 3, n)
    z = x.copy()
    y = x.copy()
    pid = compute_pid_manual(x, y, z)
    assert pid["redundancy"] >= 0.0
    assert pid["unique_x"] >= 0.0
    assert pid["unique_y"] >= 0.0
    assert pid["synergy"] >= 0.0


def test_compute_pid_manual_independent_sources():
    rng = np.random.RandomState(42)
    n = 200
    x = rng.randint(0, 3, n)
    y = rng.randint(0, 3, n)
    z = rng.randint(0, 3, n)
    pid = compute_pid_manual(x, y, z)
    assert pid["redundancy"] >= 0.0
    assert pid["i_xz"] >= 0.0
    assert pid["i_yz"] >= 0.0


def test_compute_pid_manual_keys():
    x = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    z = np.array([0, 1, 1, 0, 0, 1, 1, 0])
    pid = compute_pid_manual(x, y, z)
    expected_keys = {"redundancy", "unique_x", "unique_y", "synergy", "i_xz", "i_yz"}
    assert set(pid.keys()) == expected_keys


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


@pytest.fixture(scope="module")
def circuit_results(gpt2_model):
    return run_pid(gpt2_model, tasks=[TASK], n_prompts=5, max_pairs=3)


def test_run_returns_eval_results(circuit_results):
    assert len(circuit_results) == 1
    assert isinstance(circuit_results[0], EvalResult)


def test_metric_id(circuit_results):
    assert circuit_results[0].metric_id == "C8.pid"


def test_value_is_mean_synergy(circuit_results):
    r = circuit_results[0]
    assert r.value == pytest.approx(r.metadata["mean_synergy"])


def test_metadata_keys(circuit_results):
    meta = circuit_results[0].metadata
    expected = {"task", "mean_redundancy", "mean_synergy", "pair_decompositions",
                "n_pairs", "n_circuit_heads", "used_dit"}
    assert set(meta.keys()) == expected


def test_pair_decompositions_have_correct_fields(circuit_results):
    for pair in circuit_results[0].metadata["pair_decompositions"]:
        assert "head_a" in pair
        assert "head_b" in pair
        assert "redundancy" in pair
        assert "synergy" in pair


def test_n_samples_positive(circuit_results):
    assert circuit_results[0].n_samples > 0


def test_synergy_non_negative(circuit_results):
    assert circuit_results[0].value >= 0.0
