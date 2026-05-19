import importlib.util
import math
import sys
from pathlib import Path

import pytest

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechanistic_validity" / "calibrations" / "certified_stability" / "M3b_certified_stable.py"
)
_spec = importlib.util.spec_from_file_location("certified_m3b", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["certified_m3b"] = _mod
_spec.loader.exec_module(_mod)

classify_head = _mod.classify_head
run_certified_stability = _mod.run_certified_stability

from mechanistic_validity.metrics.common import EvalResult, get_circuit_heads, load_model

TASK = "ioi"


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


@pytest.fixture(scope="module")
def circuit_results(gpt2_model):
    return run_certified_stability(gpt2_model, [TASK], n_prompts=5, n_subsamples=5)


def test_returns_eval_result_list(circuit_results):
    assert len(circuit_results) == 1
    assert isinstance(circuit_results[0], EvalResult)


def test_metric_id(circuit_results):
    assert circuit_results[0].metric_id == "M3b.certified_stability"


def test_n_samples_positive(circuit_results):
    assert circuit_results[0].n_samples > 0
    assert circuit_results[0].n_samples <= 5


def test_value_is_fraction(circuit_results):
    assert 0.0 <= circuit_results[0].value <= 1.0


def test_value_equals_frac_certified(circuit_results):
    r = circuit_results[0]
    assert r.value == pytest.approx(r.metadata["frac_certified"])


def test_metadata_has_expected_keys(circuit_results):
    expected_keys = {
        "task", "n_heads", "n_subsamples", "subsample_fraction",
        "stability_scores", "certified_heads", "contingent_heads",
        "unstable_heads", "frac_certified", "passed",
    }
    assert set(circuit_results[0].metadata.keys()) == expected_keys


def test_task_matches(circuit_results):
    assert circuit_results[0].metadata["task"] == TASK


def test_head_count_matches_circuit(circuit_results):
    circuit_heads = get_circuit_heads(TASK)
    assert len(circuit_heads) == circuit_results[0].metadata["n_heads"]


def test_stability_scores_all_between_0_and_1(circuit_results):
    scores = circuit_results[0].metadata["stability_scores"]
    assert len(scores) > 0
    for key, v in scores.items():
        assert 0.0 <= v <= 1.0, f"Score out of range for {key}: {v}"


def test_head_lists_partition_all_heads(circuit_results):
    m = circuit_results[0].metadata
    total = len(m["certified_heads"]) + len(m["contingent_heads"]) + len(m["unstable_heads"])
    assert total == m["n_heads"]


def test_stability_scores_count_matches_n_heads(circuit_results):
    scores = circuit_results[0].metadata["stability_scores"]
    assert len(scores) == circuit_results[0].metadata["n_heads"]


def test_passed_consistent_with_frac(circuit_results):
    m = circuit_results[0].metadata
    assert m["passed"] == (m["frac_certified"] >= 0.50)


def test_classify_head_certified():
    assert classify_head(1.0) == "certified"
    assert classify_head(0.95) == "certified"


def test_classify_head_contingent():
    assert classify_head(0.94) == "contingent"
    assert classify_head(0.50) == "contingent"
    assert classify_head(0.75) == "contingent"


def test_classify_head_unstable():
    assert classify_head(0.49) == "unstable"
    assert classify_head(0.0) == "unstable"


def test_classify_head_boundary_values():
    assert classify_head(0.95) == "certified"
    assert classify_head(0.50) == "contingent"
    assert classify_head(0.4999) == "unstable"
