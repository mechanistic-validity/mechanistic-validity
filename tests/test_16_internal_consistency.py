import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from mechanistic_validity.instruments.common import EvalResult, load_model

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechanistic_validity" / "instruments"
    / "measurement" / "internal_consistency" / "16_reliability_suite.py"
)
_spec = importlib.util.spec_from_file_location("_ic_16", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)

compute_test_retest = _mod.compute_test_retest
compute_split_half = _mod.compute_split_half
compute_cronbach_alpha = _mod.compute_cronbach_alpha
run_reliability_suite = _mod.run_reliability_suite

TASK = "ioi"


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


@pytest.fixture(scope="module")
def circuit_results(gpt2_model):
    return run_reliability_suite(gpt2_model, [TASK], n_prompts=4)


def test_run_reliability_suite_produces_three_metrics_per_task(circuit_results):
    assert len(circuit_results) == 3
    metric_ids = {r.metric_id for r in circuit_results}
    assert metric_ids == {"C16.test_retest", "C16.split_half", "C16.cronbach_alpha"}


def test_run_reliability_suite_all_are_eval_results(circuit_results):
    assert all(isinstance(r, EvalResult) for r in circuit_results)


def test_test_retest_metric(circuit_results):
    r = [r for r in circuit_results if r.metric_id == "C16.test_retest"][0]
    assert 0.0 <= r.value <= 1.0
    assert r.metadata["task"] == TASK
    assert len(r.metadata["per_seed_faithfulness"]) == 3
    assert r.metadata["seeds"] == [42, 123, 456]


def test_split_half_metadata(circuit_results):
    r = [r for r in circuit_results if r.metric_id == "C16.split_half"][0]
    assert r.metadata["task"] == TASK
    assert "odd_faithfulness" in r.metadata
    assert "even_faithfulness" in r.metadata


def test_cronbach_alpha_metadata(circuit_results):
    r = [r for r in circuit_results if r.metric_id == "C16.cronbach_alpha"][0]
    assert r.metadata["task"] == TASK
    assert r.metadata["n_items"] > 0
    assert "per_head_effects" in r.metadata


def test_cronbach_alpha_formula_with_synthetic_data():
    k = 4
    n = 10
    rng = np.random.RandomState(42)
    common = rng.randn(n) * 3
    effects = np.array([common + rng.randn(n) * 0.1 for _ in range(k)])

    item_variances = np.var(effects, axis=1, ddof=1)
    total_scores = np.sum(effects, axis=0)
    total_variance = np.var(total_scores, ddof=1)
    alpha = (k / (k - 1)) * (1.0 - np.sum(item_variances) / total_variance)
    assert alpha > 0.9


def test_run_reliability_suite_no_circuit_returns_empty(gpt2_model):
    results = run_reliability_suite(gpt2_model, ["nonexistent_task"], n_prompts=4)
    assert results == []
