import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechanistic_validity" / "instruments"
    / "causal" / "granger_te" / "32_cross_task_iia_transfer.py"
)
_spec = importlib.util.spec_from_file_location("cross_task_iia_32", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["cross_task_iia_32"] = _mod
_spec.loader.exec_module(_mod)

run_cross_task_transfer = _mod.run_cross_task_transfer

from mechanistic_validity.instruments.common import EvalResult, load_model

TASK = "ioi"


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


@pytest.fixture(scope="module")
def circuit_results(gpt2_model):
    return run_cross_task_transfer(gpt2_model, tasks=[TASK, "greater_than"], n_prompts=4, d_sub=2)


def test_run_returns_eval_results(circuit_results):
    assert len(circuit_results) >= 1
    for r in circuit_results:
        assert isinstance(r, EvalResult)


def test_transfer_matrix_result_present(circuit_results):
    transfer_results = [r for r in circuit_results if r.metric_id == "C32.cross_task_iia_transfer"]
    assert len(transfer_results) == 1


def test_self_iia_results_present(circuit_results):
    self_results = [r for r in circuit_results if r.metric_id == "C32.self_iia"]
    assert len(self_results) >= 1


def test_transfer_matrix_shape(circuit_results):
    transfer_r = [r for r in circuit_results if r.metric_id == "C32.cross_task_iia_transfer"][0]
    matrix = transfer_r.metadata["transfer_matrix"]
    n_tasks = len(transfer_r.metadata["tasks"])
    assert len(matrix) == n_tasks
    for row in matrix:
        assert len(row) == n_tasks


def test_specificity_is_diag_minus_offdiag(circuit_results):
    transfer_r = [r for r in circuit_results if r.metric_id == "C32.cross_task_iia_transfer"][0]
    meta = transfer_r.metadata
    expected = meta["diagonal_mean"] - meta["offdiagonal_mean"]
    assert transfer_r.value == pytest.approx(expected)


def test_transfer_details_has_all_pairs(circuit_results):
    transfer_r = [r for r in circuit_results if r.metric_id == "C32.cross_task_iia_transfer"][0]
    meta = transfer_r.metadata
    n_tasks = len(meta["tasks"])
    assert len(meta["transfer_details"]) == n_tasks * n_tasks


def test_self_iia_value_in_range(circuit_results):
    for r in circuit_results:
        if r.metric_id == "C32.self_iia":
            assert 0.0 <= r.value <= 1.0
