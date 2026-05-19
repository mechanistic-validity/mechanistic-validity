import importlib.util
import sys
from pathlib import Path

import pytest

from mechanistic_validity.metrics.common import EvalResult, load_model

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechanistic_validity" / "metrics"
    / "behavioral" / "construct_boundary" / "RI3b_boundary_sweep.py"
)
_spec = importlib.util.spec_from_file_location("_boundary_sweep", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)

run_boundary_sweep = _mod.run_boundary_sweep
measure_circuit_activation = _mod.measure_circuit_activation
PROMPT_CATEGORIES = _mod.PROMPT_CATEGORIES

TASK = "ioi"


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


@pytest.fixture(scope="module")
def circuit_results(gpt2_model):
    return run_boundary_sweep(gpt2_model, TASK, n_prompts_per_category=2)


def test_returns_eval_result(circuit_results):
    assert isinstance(circuit_results, EvalResult)
    assert circuit_results.metric_id == "RI3b.boundary_sweep"


def test_value_is_activation_ratio(circuit_results):
    assert circuit_results.value == pytest.approx(circuit_results.metadata["activation_ratio"])


def test_six_categories_present(circuit_results):
    per_cat = circuit_results.metadata["per_category"]
    assert len(per_cat) == 6
    for cat in PROMPT_CATEGORIES:
        assert cat in per_cat


def test_category_activations_nonnegative(circuit_results):
    for cat, val in circuit_results.metadata["per_category"].items():
        assert val >= 0.0


def test_category_ranking_sorted_descending(circuit_results):
    meta = circuit_results.metadata
    ranking = meta["category_ranking"]
    per_cat = meta["per_category"]
    for i in range(len(ranking) - 1):
        assert per_cat[ranking[i]] >= per_cat[ranking[i + 1]]


def test_activation_ratio_gte_one(circuit_results):
    assert circuit_results.value >= 1.0


def test_metadata_has_required_fields(circuit_results):
    meta = circuit_results.metadata
    assert meta["task"] == TASK
    assert meta["n_circuit_heads"] > 0
    assert isinstance(meta["circuit_heads"], list)
    assert isinstance(meta["passed"], bool)
    assert meta["threshold_ratio"] == pytest.approx(2.0)


def test_no_circuit_returns_zero(gpt2_model):
    r = run_boundary_sweep(gpt2_model, "nonexistent_task_xyz", n_prompts_per_category=2)
    assert r.value == pytest.approx(0.0)
    assert r.metadata["error"] == "no circuit"


def test_measure_circuit_activation_positive(gpt2_model):
    from mechanistic_validity.metrics.common import get_circuit_heads
    heads = get_circuit_heads(TASK)
    if heads:
        act = measure_circuit_activation(gpt2_model, "The answer is", heads)
        assert act > 0.0
