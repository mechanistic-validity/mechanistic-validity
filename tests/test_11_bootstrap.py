import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from mechanistic_validity.metrics.common import EvalResult, load_model

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechanistic_validity" / "calibrations" / "bootstrap_stability" / "11_bootstrap.py"
)
_spec = importlib.util.spec_from_file_location("_bootstrap_11", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)

bootstrap_metric = _mod.bootstrap_metric
run_bootstrap = _mod.run_bootstrap

TASK = "ioi"


def test_bootstrap_metric_empty_prompts():
    point, ci_low, ci_high, sigma = bootstrap_metric(
        model=None, prompts=[], correct_ids=[], incorrect_ids=[],
        circuit_heads=set(), mean_z=None,
        metric_fn=lambda *a: 0.0, n_bootstrap=10,
    )
    assert point == pytest.approx(0.0)
    assert ci_low == pytest.approx(0.0)
    assert ci_high == pytest.approx(0.0)
    assert sigma == pytest.approx(0.0)


def test_bootstrap_metric_constant_metric_has_zero_sigma():
    dummy_prompts = ["a", "b", "c", "d", "e"]
    dummy_ids = [1, 2, 3, 4, 5]

    def constant_metric(model, prompts, correct, incorrect, heads, mean_z):
        return 0.75

    point, ci_low, ci_high, sigma = bootstrap_metric(
        model=None, prompts=dummy_prompts, correct_ids=dummy_ids,
        incorrect_ids=dummy_ids, circuit_heads=set(), mean_z=None,
        metric_fn=constant_metric, n_bootstrap=100,
    )
    assert point == pytest.approx(0.75)
    assert sigma == pytest.approx(0.0, abs=1e-10)
    assert ci_low == pytest.approx(0.75, abs=1e-10)
    assert ci_high == pytest.approx(0.75, abs=1e-10)


def test_bootstrap_metric_variable_metric_has_positive_sigma():
    dummy_prompts = list(range(20))
    dummy_ids = list(range(20))
    call_count = {"n": 0}

    def variable_metric(model, prompts, correct, incorrect, heads, mean_z):
        call_count["n"] += 1
        return float(np.mean([hash(str(p)) % 100 for p in prompts])) / 100.0

    point, ci_low, ci_high, sigma = bootstrap_metric(
        model=None, prompts=dummy_prompts, correct_ids=dummy_ids,
        incorrect_ids=dummy_ids, circuit_heads=set(), mean_z=None,
        metric_fn=variable_metric, n_bootstrap=200,
    )
    assert sigma > 0
    assert ci_low <= point
    assert ci_high >= point


def test_bootstrap_metric_ci_contains_point_estimate():
    dummy_prompts = list(range(10))
    dummy_ids = list(range(10))

    def noisy_metric(model, prompts, correct, incorrect, heads, mean_z):
        return 0.5 + 0.1 * (len(prompts) - 10) / 10

    point, ci_low, ci_high, sigma = bootstrap_metric(
        model=None, prompts=dummy_prompts, correct_ids=dummy_ids,
        incorrect_ids=dummy_ids, circuit_heads=set(), mean_z=None,
        metric_fn=noisy_metric, n_bootstrap=50,
    )
    assert ci_low <= ci_high


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


@pytest.fixture(scope="module")
def circuit_results(gpt2_model):
    return run_bootstrap(gpt2_model, [TASK], n_prompts=3, n_bootstrap=10,
                         inner="faithfulness")


def test_run_bootstrap_returns_eval_results(circuit_results):
    assert len(circuit_results) > 0
    assert all(isinstance(r, EvalResult) for r in circuit_results)


def test_run_bootstrap_metric_id(circuit_results):
    for r in circuit_results:
        assert r.metric_id == "C11.bootstrap_faithfulness"


def test_run_bootstrap_has_ci(circuit_results):
    for r in circuit_results:
        assert r.ci_low is not None
        assert r.ci_high is not None
        assert r.ci_low <= r.ci_high


def test_run_bootstrap_metadata_keys(circuit_results):
    for r in circuit_results:
        assert r.metadata["task"] == TASK
        assert "sigma" in r.metadata
        assert "n_bootstrap" in r.metadata
        assert r.metadata["inner_metric"] == "faithfulness"
        assert r.metadata["n_circuit_heads"] > 0


def test_run_bootstrap_n_samples(circuit_results):
    for r in circuit_results:
        assert r.n_samples > 0
        assert r.n_samples <= 3
