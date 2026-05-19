import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Load the 01_das_iia dependency first
_DAS_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechval" / "metrics"
    / "causal" / "counterfactual_das" / "01_das_iia.py"
)
_das_spec = importlib.util.spec_from_file_location("01_das_iia", _DAS_PATH)
_das_mod = importlib.util.module_from_spec(_das_spec)
sys.modules["01_das_iia"] = _das_mod
_das_spec.loader.exec_module(_das_mod)

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechval" / "metrics"
    / "causal" / "counterfactual_das" / "31_multi_axis_iia.py"
)
_spec = importlib.util.spec_from_file_location("multi_axis_iia_31", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["multi_axis_iia_31"] = _mod
_spec.loader.exec_module(_mod)

split_prompts_by_axis = _mod.split_prompts_by_axis
compute_joint_iia = _mod.compute_joint_iia
compute_residual_iia = _mod.compute_residual_iia
run_multi_axis_iia = _mod.run_multi_axis_iia
MULTI_AXIS_TASKS = _mod.MULTI_AXIS_TASKS

from mechval.metrics.common import EvalResult, load_model

TASK = "ioi"


def test_split_prompts_by_axis_returns_correct_length():
    n = 6
    correct_ids = [0, 1, 0, 1, 0, 1]
    incorrect_ids = list(range(n))

    class FakePrompt:
        def __init__(self):
            self.metadata = {}

    prompts = [FakePrompt() for _ in range(n)]
    pairs = split_prompts_by_axis(prompts, correct_ids, incorrect_ids, TASK, axis=0)
    assert len(pairs) == n


def test_split_prompts_no_self_pairing():
    n = 8
    correct_ids = [0, 1, 0, 1, 0, 1, 0, 1]
    incorrect_ids = list(range(n))

    class FakePrompt:
        def __init__(self):
            self.metadata = {}

    prompts = [FakePrompt() for _ in range(n)]
    for axis in [0, 1]:
        pairs = split_prompts_by_axis(prompts, correct_ids, incorrect_ids, TASK, axis=axis)
        for i, j in pairs:
            assert i != j


def test_ioi_is_multi_axis():
    assert "ioi" in MULTI_AXIS_TASKS


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


@pytest.fixture(scope="module")
def circuit_results(gpt2_model):
    return run_multi_axis_iia(gpt2_model, tasks=[TASK], n_prompts=4, d_sub=1)


def test_run_multi_axis_iia_returns_results(circuit_results):
    assert len(circuit_results) >= 1
    for r in circuit_results:
        assert isinstance(r, EvalResult)


def test_run_multi_axis_iia_has_joint_metric(circuit_results):
    joint = [r for r in circuit_results if r.metric_id == "C31.multi_axis_iia"]
    assert len(joint) >= 1


def test_run_multi_axis_iia_has_control_metric(circuit_results):
    control = [r for r in circuit_results if r.metric_id == "C31.multi_axis_control"]
    assert len(control) >= 1


def test_joint_iia_value_in_range(circuit_results):
    for r in circuit_results:
        if r.metric_id == "C31.multi_axis_iia":
            assert 0.0 <= r.value <= 1.0


def test_control_value_in_range(circuit_results):
    for r in circuit_results:
        if r.metric_id == "C31.multi_axis_control":
            assert 0.0 <= r.value <= 1.0


def test_metadata_has_task(circuit_results):
    for r in circuit_results:
        assert r.metadata["task"] == TASK


def test_joint_iia_metadata_has_n_axes(circuit_results):
    for r in circuit_results:
        if r.metric_id == "C31.multi_axis_iia":
            assert "n_axes" in r.metadata
            assert r.metadata["n_axes"] in (1, 2)
