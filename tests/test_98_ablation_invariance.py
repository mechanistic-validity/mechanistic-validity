import importlib.util
import math
import sys
from pathlib import Path

import pytest

_AI_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechanistic_validity" / "calibrations" / "ablation_invariance" / "98_ablation_invariance.py"
)
_spec = importlib.util.spec_from_file_location("ablation_invariance_98", _AI_PATH)
_ai_mod = importlib.util.module_from_spec(_spec)
sys.modules["ablation_invariance_98"] = _ai_mod
_spec.loader.exec_module(_ai_mod)

run_ablation_invariance = _ai_mod.run_ablation_invariance
ABLATION_METHODS = _ai_mod.ABLATION_METHODS
DIVERGENCE_THRESHOLD = _ai_mod.DIVERGENCE_THRESHOLD

from mechanistic_validity.metrics.common import EvalResult, load_model


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


TASK = "ioi"


@pytest.fixture(scope="module")
def circuit_results(gpt2_model):
    return run_ablation_invariance(gpt2_model, [TASK], n_prompts=5)


# ── Integration ────────────────────────────────────────────────────

def test_run_returns_nonempty_list(circuit_results):
    assert len(circuit_results) == 1


def test_result_is_eval_result(circuit_results):
    assert isinstance(circuit_results[0], EvalResult)


def test_metric_id_is_correct(circuit_results):
    assert circuit_results[0].metric_id == "M98.ablation_invariance"


def test_n_samples_is_positive(circuit_results):
    assert circuit_results[0].n_samples > 0
    assert circuit_results[0].n_samples <= 5


def test_value_equals_divergence_in_metadata(circuit_results):
    r = circuit_results[0]
    assert r.value == pytest.approx(r.metadata["divergence"])


# ── Faithfulness scores ─────────────────────────────────────────────

def test_scores_contains_all_three_methods(circuit_results):
    scores = circuit_results[0].metadata["scores"]
    for method in ABLATION_METHODS:
        assert method in scores, f"Missing method: {method}"


def test_all_faithfulness_scores_are_finite(circuit_results):
    scores = circuit_results[0].metadata["scores"]
    for method, val in scores.items():
        assert math.isfinite(val), f"{method} score is not finite: {val}"


def test_all_faithfulness_scores_are_bounded(circuit_results):
    scores = circuit_results[0].metadata["scores"]
    for method, val in scores.items():
        assert -2.0 <= val <= 2.0, (
            f"{method} score {val} outside reasonable range [-2, 2]"
        )


# ── Divergence ──────────────────────────────────────────────────────

def test_divergence_is_non_negative(circuit_results):
    assert circuit_results[0].metadata["divergence"] >= 0.0


def test_divergence_is_finite(circuit_results):
    assert math.isfinite(circuit_results[0].metadata["divergence"])


def test_divergence_equals_max_minus_min_of_scores(circuit_results):
    r = circuit_results[0]
    vals = list(r.metadata["scores"].values())
    expected = max(vals) - min(vals)
    assert r.metadata["divergence"] == pytest.approx(expected)


# ── Metadata structure ──────────────────────────────────────────────

def test_metadata_contains_expected_keys(circuit_results):
    expected_keys = {
        "task", "n_circuit_heads", "scores", "divergence",
        "passed", "threshold",
    }
    assert expected_keys <= set(circuit_results[0].metadata.keys())


def test_metadata_task_matches(circuit_results):
    assert circuit_results[0].metadata["task"] == TASK


def test_metadata_n_circuit_heads_is_positive(circuit_results):
    assert circuit_results[0].metadata["n_circuit_heads"] > 0


def test_metadata_threshold_matches_module_constant(circuit_results):
    assert circuit_results[0].metadata["threshold"] == pytest.approx(DIVERGENCE_THRESHOLD)


def test_metadata_passed_is_bool(circuit_results):
    assert isinstance(circuit_results[0].metadata["passed"], bool)


def test_passed_consistent_with_divergence_and_threshold(circuit_results):
    r = circuit_results[0]
    expected_pass = r.metadata["divergence"] < r.metadata["threshold"]
    assert r.metadata["passed"] == expected_pass
