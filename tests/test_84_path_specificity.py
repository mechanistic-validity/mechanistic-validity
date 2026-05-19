import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from mechanistic_validity.instruments.common import EvalResult, load_model

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechanistic_validity" / "instruments"
    / "structural" / "edge_analysis" / "84_path_specificity.py"
)
_spec = importlib.util.spec_from_file_location("_path_spec_84", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)

compute_edge_effects_for_prompt = _mod.compute_edge_effects_for_prompt
run_path_specificity = _mod.run_path_specificity

TASK = "ioi"


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


@pytest.fixture(scope="module")
def circuit_results(gpt2_model):
    return run_path_specificity(gpt2_model, [TASK], n_prompts=5)


def test_compute_edge_effects_for_prompt_returns_dict(gpt2_model):
    from mechanistic_validity.instruments.common import (
        calibrate_mean_z, generate_prompts, get_circuit_info, get_token_ids,
    )
    prompts = generate_prompts(TASK, gpt2_model.tokenizer, 3)
    correct_ids, incorrect_ids = get_token_ids(prompts, gpt2_model.tokenizer)
    mean_z = calibrate_mean_z(gpt2_model, prompts, n_calibration=3)
    tokens = gpt2_model.to_tokens(prompts[0].text)

    _circuit, _all_heads, all_edges = get_circuit_info(TASK)
    sorted_edges = sorted(all_edges)[:3]

    effects = compute_edge_effects_for_prompt(
        gpt2_model, tokens, correct_ids[0], incorrect_ids[0],
        sorted_edges, mean_z)
    assert isinstance(effects, dict)
    assert len(effects) == len(sorted_edges)
    for edge, val in effects.items():
        assert isinstance(val, float)


def test_run_path_specificity_returns_results(circuit_results):
    assert len(circuit_results) >= 1
    r = circuit_results[0]
    assert isinstance(r, EvalResult)
    assert r.metric_id == "G3.path_specificity"
    assert r.n_samples >= 1


def test_run_path_specificity_value_is_rho(circuit_results):
    r = circuit_results[0]
    assert -1.0 <= r.value <= 1.0


def test_run_path_specificity_metadata_fields(circuit_results):
    r = circuit_results[0]
    meta = r.metadata
    assert meta["task"] == TASK
    assert "n_edges" in meta
    assert meta["n_edges"] > 0
    assert "spearman_rho" in meta
    assert r.value == pytest.approx(meta["spearman_rho"])
    assert "spearman_p" in meta
    assert "pattern_task" in meta
    assert "pattern_control" in meta
    assert isinstance(meta["pattern_task"], list)
    assert isinstance(meta["pattern_control"], list)
    assert len(meta["pattern_task"]) == meta["n_edges"]
    assert "edge_names" in meta
    assert isinstance(meta["edge_names"], list)
    assert "passed" in meta
    assert isinstance(meta["passed"], bool)
    assert meta["threshold_rho"] == pytest.approx(0.5)
