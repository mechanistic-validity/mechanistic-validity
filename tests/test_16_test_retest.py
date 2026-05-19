import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from mechanistic_validity.instruments.common import EvalResult, load_model

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechanistic_validity" / "instruments"
    / "measurement" / "test_retest" / "16_reliability_suite.py"
)
_spec = importlib.util.spec_from_file_location("_tr_16", _MOD_PATH)
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


def test_run_reliability_suite_produces_three_metrics(circuit_results):
    assert len(circuit_results) == 3
    metric_ids = {r.metric_id for r in circuit_results}
    assert metric_ids == {"C16.test_retest", "C16.split_half", "C16.cronbach_alpha"}


def test_all_results_are_eval_result(circuit_results):
    assert all(isinstance(r, EvalResult) for r in circuit_results)


def test_test_retest_value_bounded(circuit_results):
    r = [r for r in circuit_results if r.metric_id == "C16.test_retest"][0]
    assert 0.0 <= r.value <= 1.0


def test_test_retest_has_seed_scores(circuit_results):
    r = [r for r in circuit_results if r.metric_id == "C16.test_retest"][0]
    assert len(r.metadata["per_seed_faithfulness"]) == 3


def test_split_half_returns_corrected_r(circuit_results):
    r = [r for r in circuit_results if r.metric_id == "C16.split_half"][0]
    assert "odd_faithfulness" in r.metadata
    assert "even_faithfulness" in r.metadata
    assert isinstance(r.value, float)


def test_cronbach_alpha_needs_at_least_two_heads(circuit_results):
    r = [r for r in circuit_results if r.metric_id == "C16.cronbach_alpha"][0]
    assert r.metadata["n_items"] >= 2


def test_all_results_have_task_metadata(circuit_results):
    for r in circuit_results:
        assert r.metadata["task"] == TASK


def test_test_retest_reliability_formula():
    values = np.array([0.80, 0.82, 0.79])
    mean_val = np.mean(values)
    std_val = np.std(values)
    cv = std_val / abs(mean_val)
    reliability = max(0.0, min(1.0, 1.0 - cv))
    assert 0.0 <= reliability <= 1.0
    assert reliability > 0.95


def test_spearman_brown_formula():
    r_half = 0.6
    corrected = 2.0 * r_half / (1.0 + r_half)
    assert corrected == pytest.approx(0.75)


def test_run_reliability_suite_skips_unknown_task(gpt2_model):
    results = run_reliability_suite(gpt2_model, ["not_a_real_task"], n_prompts=4)
    assert results == []
