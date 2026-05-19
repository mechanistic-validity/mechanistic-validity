import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from mechval.metrics.common import EvalResult, load_model

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechval" / "calibrations" / "bootstrap_stability" / "73_distributional_characterization.py"
)
_spec = importlib.util.spec_from_file_location("_distchar_73", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)

_bootstrap_ci = _mod._bootstrap_ci
_skewness = _mod._skewness
_kurtosis = _mod._kurtosis
_effective_rank = _mod._effective_rank
compute_head_stats = _mod.compute_head_stats
run_distributional_characterization = _mod.run_distributional_characterization

TASK = "ioi"


def test_bootstrap_ci_single_value():
    vals = np.array([5.0])
    lo, hi = _bootstrap_ci(vals)
    assert lo == pytest.approx(5.0)
    assert hi == pytest.approx(0.0)


def test_bootstrap_ci_constant_array():
    vals = np.array([3.0, 3.0, 3.0, 3.0, 3.0])
    lo, hi = _bootstrap_ci(vals, n_bootstrap=200)
    assert lo == pytest.approx(3.0, abs=1e-6)
    assert hi == pytest.approx(3.0, abs=1e-6)


def test_bootstrap_ci_interval_contains_mean():
    rng = np.random.RandomState(42)
    vals = rng.normal(10.0, 2.0, size=100)
    lo, hi = _bootstrap_ci(vals, n_bootstrap=500)
    mean = vals.mean()
    assert lo <= mean
    assert hi >= mean


def test_skewness_symmetric_distribution():
    vals = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    skew = _skewness(vals)
    assert skew == pytest.approx(0.0, abs=0.01)


def test_skewness_right_skewed():
    vals = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 100.0])
    skew = _skewness(vals)
    assert skew > 0


def test_skewness_too_few_values():
    assert _skewness(np.array([1.0, 2.0])) == pytest.approx(0.0)


def test_skewness_zero_std():
    assert _skewness(np.array([5.0, 5.0, 5.0, 5.0])) == pytest.approx(0.0)


def test_kurtosis_normal_distribution_near_zero():
    rng = np.random.RandomState(42)
    vals = rng.normal(0, 1, size=10000)
    kurt = _kurtosis(vals)
    assert kurt == pytest.approx(0.0, abs=0.15)


def test_kurtosis_too_few_values():
    assert _kurtosis(np.array([1.0, 2.0, 3.0])) == pytest.approx(0.0)


def test_kurtosis_uniform_distribution_is_negative():
    rng = np.random.RandomState(42)
    vals = rng.uniform(0, 1, size=10000)
    kurt = _kurtosis(vals)
    assert kurt < 0


def test_effective_rank_identity_matrix():
    mat = np.eye(5)
    rank = _effective_rank(mat)
    assert rank == pytest.approx(5.0, abs=0.01)


def test_effective_rank_rank_one_matrix():
    mat = np.outer(np.ones(5), np.array([1.0, 2.0, 3.0]))
    rank = _effective_rank(mat)
    assert rank == pytest.approx(1.0, abs=0.01)


def test_effective_rank_small_matrix():
    mat = np.array([[1.0]])
    rank = _effective_rank(mat)
    assert rank == pytest.approx(1.0)


def test_compute_head_stats_returns_expected_keys():
    vals = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    stats = compute_head_stats(vals)
    expected_keys = {"mean", "ci_low", "ci_high", "variance", "std",
                     "skewness", "kurtosis", "sparsity"}
    assert set(stats.keys()) == expected_keys


def test_compute_head_stats_mean_is_correct():
    vals = np.array([2.0, 4.0, 6.0, 8.0])
    stats = compute_head_stats(vals)
    assert stats["mean"] == pytest.approx(5.0)


def test_compute_head_stats_sparsity_all_below_threshold():
    vals = np.array([0.001, -0.005, 0.002, 0.0, -0.001])
    stats = compute_head_stats(vals)
    assert stats["sparsity"] == pytest.approx(1.0)


def test_compute_head_stats_sparsity_none_below_threshold():
    vals = np.array([1.0, 2.0, 3.0, -5.0])
    stats = compute_head_stats(vals)
    assert stats["sparsity"] == pytest.approx(0.0)


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


@pytest.fixture(scope="module")
def circuit_results(gpt2_model):
    return run_distributional_characterization(gpt2_model, [TASK], n_prompts=3)


def test_run_distributional_characterization_returns_results(circuit_results):
    assert len(circuit_results) > 0
    assert all(isinstance(r, EvalResult) for r in circuit_results)


def test_run_distributional_characterization_metric_id(circuit_results):
    for r in circuit_results:
        assert r.metric_id == "S1.distributional_characterization"


def test_run_distributional_characterization_metadata(circuit_results):
    for r in circuit_results:
        assert r.metadata["task"] == TASK
        assert r.metadata["n_circuit_heads"] > 0
        assert "circuit_mean_magnitude" in r.metadata
        assert "non_circuit_mean_magnitude" in r.metadata
        assert "effective_ranks" in r.metadata


def test_run_distributional_characterization_value_is_positive(circuit_results):
    for r in circuit_results:
        assert r.value > 0
