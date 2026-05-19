import importlib.util
import sys
from pathlib import Path
import pytest

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechval" / "metrics"
    / "causal" / "minimality" / "C4b_minimality_class.py"
)
_spec = importlib.util.spec_from_file_location("C4b_minimality_class", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["C4b_minimality_class"] = _mod
_spec.loader.exec_module(_mod)

classify_minimality = _mod.classify_minimality
run_minimality_class = _mod.run_minimality_class
CLASS_VALUES = _mod.CLASS_VALUES
FAITHFULNESS_THRESHOLD = _mod.FAITHFULNESS_THRESHOLD

from mechval.metrics.common import EvalResult, load_model

TASK = "ioi"


# -- CLASS_VALUES encoding --

def test_class_values_subset_minimal():
    assert CLASS_VALUES["subset_minimal"] == pytest.approx(1.0)


def test_class_values_locally_minimal():
    assert CLASS_VALUES["locally_minimal"] == pytest.approx(0.75)


def test_class_values_quasi_minimal():
    assert CLASS_VALUES["quasi_minimal"] == pytest.approx(0.5)


def test_class_values_not_minimal():
    assert CLASS_VALUES["not_minimal"] == pytest.approx(0.0)


def test_class_values_ordering():
    assert CLASS_VALUES["subset_minimal"] > CLASS_VALUES["locally_minimal"]
    assert CLASS_VALUES["locally_minimal"] > CLASS_VALUES["quasi_minimal"]
    assert CLASS_VALUES["quasi_minimal"] > CLASS_VALUES["not_minimal"]


def test_faithfulness_threshold():
    assert FAITHFULNESS_THRESHOLD == pytest.approx(0.5)


# -- Integration tests --

@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


@pytest.fixture(scope="module")
def circuit_results(gpt2_model):
    return run_minimality_class(gpt2_model, [TASK], n_prompts=3)


def test_run_returns_nonempty(circuit_results):
    assert len(circuit_results) == 1


def test_result_is_eval_result(circuit_results):
    assert isinstance(circuit_results[0], EvalResult)


def test_metric_id(circuit_results):
    assert circuit_results[0].metric_id == "C4b.minimality_class"


def test_n_samples(circuit_results):
    assert circuit_results[0].n_samples == 3


def test_value_in_class_values(circuit_results):
    assert circuit_results[0].value in CLASS_VALUES.values()


def test_metadata_task(circuit_results):
    assert circuit_results[0].metadata["task"] == TASK


def test_metadata_minimality_class_is_valid(circuit_results):
    cls = circuit_results[0].metadata["minimality_class"]
    assert cls in CLASS_VALUES


def test_metadata_passed_is_bool(circuit_results):
    assert isinstance(circuit_results[0].metadata["passed"], bool)


def test_value_matches_class(circuit_results):
    r = circuit_results[0]
    cls = r.metadata["minimality_class"]
    assert r.value == pytest.approx(CLASS_VALUES[cls])


def test_passed_consistent_with_class(circuit_results):
    r = circuit_results[0]
    cls = r.metadata["minimality_class"]
    expected = cls in ("subset_minimal", "locally_minimal")
    assert r.metadata["passed"] == expected


def test_unknown_task(gpt2_model):
    results = run_minimality_class(gpt2_model, ["nonexistent_task_xyz"], n_prompts=3)
    assert len(results) == 1
    assert results[0].value == pytest.approx(0.0)
    assert results[0].metadata["minimality_class"] == "not_minimal"
