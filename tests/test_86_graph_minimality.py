import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from mechval.metrics.common import EvalResult, load_model

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechval" / "metrics"
    / "structural" / "edge_analysis" / "86_graph_minimality.py"
)
_spec = importlib.util.spec_from_file_location("_graph_min_86", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)

test_edge_minimality = _mod.test_edge_minimality
run_graph_minimality = _mod.run_graph_minimality

TASK = "ioi"


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


@pytest.fixture(scope="module")
def circuit_results(gpt2_model):
    return run_graph_minimality(gpt2_model, [TASK], n_prompts=3)


def test_test_edge_minimality_returns_dict(gpt2_model):
    from mechval.metrics.common import (
        calibrate_mean_z, generate_prompts, get_circuit_info, get_token_ids,
    )
    prompts = generate_prompts(TASK, gpt2_model.tokenizer, 3)
    correct_ids, incorrect_ids = get_token_ids(prompts, gpt2_model.tokenizer)
    mean_z = calibrate_mean_z(gpt2_model, prompts, n_calibration=3)
    _circuit, _all_heads, all_edges = get_circuit_info(TASK)

    subset_edges = set(sorted(all_edges)[:2])
    result = test_edge_minimality(
        gpt2_model, prompts, correct_ids, incorrect_ids, subset_edges, mean_z)
    assert isinstance(result, dict)
    assert len(result) == len(subset_edges)
    for edge, data in result.items():
        assert "drop_frac" in data
        assert "magnitude_necessary" in data
        assert "directional" in data
        assert "necessary" in data
        assert isinstance(data["necessary"], bool)


def test_run_graph_minimality_returns_results(circuit_results):
    assert len(circuit_results) >= 1
    r = circuit_results[0]
    assert isinstance(r, EvalResult)
    assert r.metric_id == "G5.graph_minimality"
    assert r.n_samples >= 1


def test_run_graph_minimality_value_is_ratio(circuit_results):
    r = circuit_results[0]
    assert 0.0 <= r.value <= 1.0


def test_run_graph_minimality_metadata_fields(circuit_results):
    r = circuit_results[0]
    meta = r.metadata
    assert meta["task"] == TASK
    assert "n_edges" in meta
    assert meta["n_edges"] > 0
    assert "n_necessary" in meta
    assert isinstance(meta["n_necessary"], int)
    assert "n_wrong_direction" in meta
    assert isinstance(meta["n_wrong_direction"], int)
    assert "minimality_ratio" in meta
    assert r.value == pytest.approx(meta["minimality_ratio"])
    assert "per_edge" in meta
    assert isinstance(meta["per_edge"], dict)
    assert "passed" in meta
    assert isinstance(meta["passed"], bool)
    assert meta["threshold_minimality"] == pytest.approx(0.8)
    assert meta["threshold_drop"] == pytest.approx(0.05)


def test_run_graph_minimality_per_edge_has_all_fields(circuit_results):
    r = circuit_results[0]
    for edge_name, data in r.metadata["per_edge"].items():
        assert "drop_frac" in data
        assert "magnitude_necessary" in data
        assert "directional" in data
        assert "necessary" in data
