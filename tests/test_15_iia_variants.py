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
    / "causal" / "counterfactual_das" / "15_iia_variants.py"
)
_spec = importlib.util.spec_from_file_location("iia_variants_15", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["iia_variants_15"] = _mod
_spec.loader.exec_module(_mod)

compute_neuron_iia = _mod.compute_neuron_iia
run_iia_variants = _mod.run_iia_variants

from mechval.metrics.common import EvalResult, load_model

TASK = "ioi"


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


@pytest.fixture(scope="module")
def circuit_results(gpt2_model):
    return run_iia_variants(
        gpt2_model, tasks=[TASK], n_prompts=4, subspace_dim=1, iia_k_values=[1, 2],
    )


def test_run_iia_variants_returns_eval_results(circuit_results):
    assert len(circuit_results) >= 1
    for r in circuit_results:
        assert isinstance(r, EvalResult)


def test_run_iia_variants_has_neuron_iia(circuit_results):
    neuron_results = [r for r in circuit_results if r.metric_id == "C15.neuron_iia"]
    assert len(neuron_results) >= 1


def test_run_iia_variants_has_iia_at_k(circuit_results):
    iia_k_results = [r for r in circuit_results if r.metric_id.startswith("C15.iia_at_k")]
    assert len(iia_k_results) >= 1


def test_run_iia_variants_has_cross_layer(circuit_results):
    cross_results = [r for r in circuit_results if r.metric_id == "C15.cross_layer_iia"]
    assert len(cross_results) >= 1


def test_neuron_iia_value_in_range(circuit_results):
    for r in circuit_results:
        if r.metric_id == "C15.neuron_iia":
            assert 0.0 <= r.value <= 1.0


def test_iia_at_k_value_in_range(circuit_results):
    for r in circuit_results:
        if r.metric_id.startswith("C15.iia_at_k"):
            assert 0.0 <= r.value <= 1.0


def test_cross_layer_iia_value_in_range(circuit_results):
    for r in circuit_results:
        if r.metric_id == "C15.cross_layer_iia":
            assert 0.0 <= r.value <= 1.0


def test_all_results_have_task_metadata(circuit_results):
    for r in circuit_results:
        assert r.metadata["task"] == TASK


def test_neuron_iia_has_per_head_data(circuit_results):
    for r in circuit_results:
        if r.metric_id == "C15.neuron_iia":
            assert "per_head_iia" in r.metadata
            assert isinstance(r.metadata["per_head_iia"], dict)
