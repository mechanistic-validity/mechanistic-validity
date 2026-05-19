import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from mechanistic_validity.instruments.common import EvalResult, load_model

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechanistic_validity" / "instruments"
    / "behavioral" / "cross_task_transfer" / "E6b_cross_task_generalization.py"
)
_spec = importlib.util.spec_from_file_location("_cross_task_gen", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)

run_cross_task_generalization = _mod.run_cross_task_generalization
compute_attribution = _mod.compute_attribution

TASK = "ioi"


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


@pytest.fixture(scope="module")
def circuit_results(gpt2_model):
    return run_cross_task_generalization(
        gpt2_model, TASK, test_tasks=["ioi", "greater_than"], n_prompts=5,
    )


def test_returns_eval_result(circuit_results):
    assert isinstance(circuit_results, EvalResult)
    assert circuit_results.metric_id == "E6b.cross_task_generalization"


def test_value_is_transfer_score(circuit_results):
    assert circuit_results.value == pytest.approx(circuit_results.metadata["transfer_score"])


def test_metadata_has_required_fields(circuit_results):
    meta = circuit_results.metadata
    assert meta["source_task"] == TASK
    assert isinstance(meta["per_task_attribution"], dict)
    assert isinstance(meta["selectivity"], float)
    assert isinstance(meta["circuit_heads"], list)
    assert meta["n_circuit_heads"] > 0
    assert isinstance(meta["passed"], bool)


def test_per_task_attribution_nonnegative(circuit_results):
    for task, attr in circuit_results.metadata["per_task_attribution"].items():
        assert attr >= 0.0


def test_source_task_in_attributions(circuit_results):
    assert TASK in circuit_results.metadata["per_task_attribution"]


def test_n_samples(circuit_results):
    assert circuit_results.n_samples == 5


def test_no_circuit_returns_zero(gpt2_model):
    r = run_cross_task_generalization(
        gpt2_model, "nonexistent_task_xyz", n_prompts=5,
    )
    assert r.value == pytest.approx(0.0)
    assert r.metadata["error"] == "no circuit"
