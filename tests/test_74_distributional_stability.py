import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from mechanistic_validity.metrics.common import EvalResult, load_model

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechanistic_validity" / "calibrations" / "bootstrap_stability" / "74_distributional_stability.py"
)
_spec = importlib.util.spec_from_file_location("_diststab_74", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)

_skewness = _mod._skewness
_cv = _mod._cv
compute_subset_stats = _mod.compute_subset_stats
run_distributional_stability = _mod.run_distributional_stability

TASK = "ioi"


def test_cv_positive_mean():
    vals = np.array([10.0, 12.0, 11.0, 9.0, 13.0])
    cv = _cv(vals)
    expected = float(vals.std(ddof=1) / abs(vals.mean()))
    assert cv == pytest.approx(expected)


def test_cv_zero_mean_returns_inf():
    vals = np.array([-1.0, 1.0])
    cv = _cv(vals)
    assert cv == float("inf")


def test_cv_constant_returns_zero():
    vals = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
    cv = _cv(vals)
    assert cv == pytest.approx(0.0)


def test_skewness_symmetric():
    vals = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    skew = _skewness(vals)
    assert skew == pytest.approx(0.0, abs=0.01)


def test_skewness_too_few():
    assert _skewness(np.array([1.0])) == pytest.approx(0.0)


def test_compute_subset_stats_structure():
    attributions = {
        (0, 0): np.random.RandomState(42).randn(25),
        (0, 1): np.random.RandomState(43).randn(25),
        (1, 0): np.random.RandomState(44).randn(25),
    }
    circuit_heads = {(0, 0), (1, 0)}
    per_head = compute_subset_stats(attributions, circuit_heads, n_subsets=5)
    assert len(per_head) == 2
    for key, stats in per_head.items():
        assert "cv_mean" in stats
        assert "cv_variance" in stats
        assert "cv_skewness" in stats
        assert "pass_mean" in stats
        assert "pass_variance" in stats
        assert "subset_means" in stats
        assert len(stats["subset_means"]) == 5


def test_compute_subset_stats_constant_attribution_zero_cv():
    attributions = {
        (0, 0): np.ones(25) * 3.0,
    }
    circuit_heads = {(0, 0)}
    per_head = compute_subset_stats(attributions, circuit_heads, n_subsets=5)
    stats = per_head["L0H0"]
    assert stats["cv_mean"] == pytest.approx(0.0, abs=1e-10)
    assert stats["pass_mean"] is True


def test_compute_subset_stats_pass_criteria():
    rng = np.random.RandomState(42)
    attributions = {(0, 0): rng.normal(10.0, 0.01, size=50)}
    circuit_heads = {(0, 0)}
    per_head = compute_subset_stats(attributions, circuit_heads, n_subsets=5)
    stats = per_head["L0H0"]
    assert stats["pass_mean"] is True


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


@pytest.fixture(scope="module")
def circuit_results(gpt2_model):
    return run_distributional_stability(gpt2_model, [TASK], n_prompts=5)


def test_run_distributional_stability_returns_results(circuit_results):
    assert len(circuit_results) > 0
    assert all(isinstance(r, EvalResult) for r in circuit_results)


def test_run_distributional_stability_metric_id(circuit_results):
    for r in circuit_results:
        assert r.metric_id == "S3.distributional_stability"


def test_run_distributional_stability_value_range(circuit_results):
    for r in circuit_results:
        assert 0.0 <= r.value <= 1.0


def test_run_distributional_stability_metadata(circuit_results):
    for r in circuit_results:
        assert r.metadata["task"] == TASK
        assert r.metadata["n_subsets"] == 5
        assert "per_head" in r.metadata
        assert r.metadata["n_circuit_heads"] > 0
        assert "mean_cv_mean" in r.metadata
        assert "mean_cv_variance" in r.metadata
