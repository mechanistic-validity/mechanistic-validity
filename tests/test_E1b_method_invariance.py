import importlib.util
import math
import sys
from pathlib import Path

import pytest

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechanistic_validity" / "instruments"
    / "measurement" / "ablation_invariance" / "E1b_method_invariance.py"
)
_spec = importlib.util.spec_from_file_location("E1b_method_invariance", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["E1b_method_invariance"] = _mod
_spec.loader.exec_module(_mod)

run_ablation_method_invariance = _mod.run_ablation_method_invariance
ABLATION_METHODS = _mod.ABLATION_METHODS
DIVERGENCE_THRESHOLD = _mod.DIVERGENCE_THRESHOLD

from mechanistic_validity.instruments.common import EvalResult, load_model

TASK = "ioi"


# -- Constants --

def test_ablation_methods_are_three():
    assert len(ABLATION_METHODS) == 3
    assert "zero" in ABLATION_METHODS
    assert "mean" in ABLATION_METHODS
    assert "noise" in ABLATION_METHODS


def test_divergence_threshold():
    assert DIVERGENCE_THRESHOLD == pytest.approx(0.20)


# -- Integration --

@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


@pytest.fixture(scope="module")
def circuit_results(gpt2_model):
    return run_ablation_method_invariance(gpt2_model, [TASK], n_prompts=5)


def test_run_returns_nonempty(circuit_results):
    assert len(circuit_results) == 1


def test_result_is_eval_result(circuit_results):
    assert isinstance(circuit_results[0], EvalResult)


def test_metric_id(circuit_results):
    assert circuit_results[0].metric_id == "E1b.method_invariance"


def test_n_samples_positive(circuit_results):
    assert circuit_results[0].n_samples > 0
    assert circuit_results[0].n_samples <= 5


# -- Per-method faithfulness --

def test_per_method_faithfulness_has_all_methods(circuit_results):
    scores = circuit_results[0].metadata["per_method_faithfulness"]
    for method in ABLATION_METHODS:
        assert method in scores


def test_per_method_faithfulness_finite(circuit_results):
    scores = circuit_results[0].metadata["per_method_faithfulness"]
    for method, val in scores.items():
        assert math.isfinite(val), f"{method} score is not finite: {val}"


def test_per_method_faithfulness_bounded(circuit_results):
    scores = circuit_results[0].metadata["per_method_faithfulness"]
    for method, val in scores.items():
        assert -2.0 <= val <= 2.0, f"{method} score {val} out of range"


# -- Pairwise divergences --

def test_pairwise_divergences_present(circuit_results):
    pairwise = circuit_results[0].metadata["pairwise_divergences"]
    assert len(pairwise) == 3


def test_pairwise_divergences_non_negative(circuit_results):
    pairwise = circuit_results[0].metadata["pairwise_divergences"]
    for key, val in pairwise.items():
        assert val >= 0.0, f"{key} divergence is negative: {val}"


def test_pairwise_divergences_consistent_with_scores(circuit_results):
    r = circuit_results[0]
    scores = r.metadata["per_method_faithfulness"]
    pairwise = r.metadata["pairwise_divergences"]
    for key, val in pairwise.items():
        m1, _, m2 = key.partition("_vs_")
        expected = abs(scores[m1] - scores[m2])
        assert val == pytest.approx(expected)


# -- Max divergence and value --

def test_max_divergence_non_negative(circuit_results):
    assert circuit_results[0].metadata["max_divergence"] >= 0.0


def test_max_divergence_finite(circuit_results):
    assert math.isfinite(circuit_results[0].metadata["max_divergence"])


def test_value_equals_one_minus_max_divergence(circuit_results):
    r = circuit_results[0]
    assert r.value == pytest.approx(1.0 - r.metadata["max_divergence"])


def test_max_divergence_equals_max_minus_min_scores(circuit_results):
    r = circuit_results[0]
    vals = list(r.metadata["per_method_faithfulness"].values())
    expected = max(vals) - min(vals)
    assert r.metadata["max_divergence"] == pytest.approx(expected)


# -- Metadata structure --

def test_metadata_keys(circuit_results):
    expected = {
        "task", "n_circuit_heads", "per_method_faithfulness",
        "pairwise_divergences", "max_divergence", "passed", "threshold",
    }
    assert expected <= set(circuit_results[0].metadata.keys())


def test_metadata_task(circuit_results):
    assert circuit_results[0].metadata["task"] == TASK


def test_metadata_n_circuit_heads_positive(circuit_results):
    assert circuit_results[0].metadata["n_circuit_heads"] > 0


def test_metadata_threshold(circuit_results):
    assert circuit_results[0].metadata["threshold"] == pytest.approx(DIVERGENCE_THRESHOLD)


def test_metadata_passed_is_bool(circuit_results):
    assert isinstance(circuit_results[0].metadata["passed"], bool)


def test_passed_consistent_with_divergence(circuit_results):
    r = circuit_results[0]
    expected = r.metadata["max_divergence"] < r.metadata["threshold"]
    assert r.metadata["passed"] == expected
