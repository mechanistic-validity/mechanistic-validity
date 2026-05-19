import importlib.util
import sys
from pathlib import Path

import pytest

from mechval.metrics.common import EvalResult, load_model

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechval" / "metrics"
    / "behavioral" / "minimal_pairs" / "A5_epistemic_gradient.py"
)
_spec = importlib.util.spec_from_file_location("_epistemic_gradient", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)

run_epistemic_gradient = _mod.run_epistemic_gradient
compute_monotonicity = _mod.compute_monotonicity

TASK = "ioi"


def test_monotonicity_perfectly_ordered():
    assert compute_monotonicity([1.0, 2.0, 3.0, 4.0]) == pytest.approx(1.0)


def test_monotonicity_perfectly_reversed():
    assert compute_monotonicity([4.0, 3.0, 2.0, 1.0]) == pytest.approx(0.0)


def test_monotonicity_flat():
    assert compute_monotonicity([5.0, 5.0, 5.0]) == pytest.approx(1.0)


def test_monotonicity_single_element():
    assert compute_monotonicity([42.0]) == pytest.approx(1.0)


def test_monotonicity_empty():
    assert compute_monotonicity([]) == pytest.approx(1.0)


def test_monotonicity_two_ordered():
    assert compute_monotonicity([1.0, 2.0]) == pytest.approx(1.0)


def test_monotonicity_two_reversed():
    assert compute_monotonicity([2.0, 1.0]) == pytest.approx(0.0)


def test_monotonicity_mixed():
    assert compute_monotonicity([1.0, 3.0, 2.0, 4.0]) == pytest.approx(2.0 / 3.0)


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


@pytest.fixture(scope="module")
def circuit_results(gpt2_model):
    return run_epistemic_gradient(gpt2_model, TASK, n_templates=3)


def test_returns_eval_result(circuit_results):
    assert isinstance(circuit_results, EvalResult)
    assert circuit_results.metric_id == "A5.epistemic_gradient"


def test_value_is_monotonicity(circuit_results):
    assert circuit_results.value == pytest.approx(circuit_results.metadata["monotonicity"])


def test_value_in_range(circuit_results):
    assert 0.0 <= circuit_results.value <= 1.0


def test_per_level_means_has_four_levels(circuit_results):
    per_level = circuit_results.metadata["per_level_means"]
    assert len(per_level) == 4
    for level in [0, 1, 2, 3]:
        assert str(level) in per_level or level in per_level


def test_per_level_means_nonnegative(circuit_results):
    for lv, val in circuit_results.metadata["per_level_means"].items():
        assert val >= 0.0


def test_metadata_has_required_fields(circuit_results):
    meta = circuit_results.metadata
    assert meta["task"] == TASK
    assert meta["n_circuit_heads"] > 0
    assert isinstance(meta["circuit_heads"], list)
    assert isinstance(meta["passed"], bool)
    assert meta["threshold_monotonicity"] == pytest.approx(0.75)
    assert isinstance(meta["gradient_slope"], float)
    assert isinstance(meta["templates_used"], list)
    assert len(meta["templates_used"]) == 3


def test_no_circuit_returns_zero(gpt2_model):
    r = run_epistemic_gradient(gpt2_model, "nonexistent_task_xyz", n_templates=2)
    assert r.value == pytest.approx(0.0)
    assert r.metadata["error"] == "no circuit"
