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
    / "structural" / "edge_analysis" / "82_path_identification.py"
)
_spec = importlib.util.spec_from_file_location("_path_id_82", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)

compute_edge_effect = _mod.compute_edge_effect
run_path_identification = _mod.run_path_identification

TASK = "ioi"


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


@pytest.fixture(scope="module")
def circuit_results(gpt2_model):
    return run_path_identification(gpt2_model, [TASK], n_prompts=3)


def test_compute_edge_effect_returns_float(gpt2_model):
    from mechanistic_validity.instruments.common import generate_prompts, get_token_ids
    prompts = generate_prompts(TASK, gpt2_model.tokenizer, 2)
    correct_ids, incorrect_ids = get_token_ids(prompts, gpt2_model.tokenizer)
    tokens = gpt2_model.to_tokens(prompts[0].text)
    _, cache_clean = gpt2_model.run_with_cache(tokens)
    corrupt_tokens = gpt2_model.to_tokens(prompts[1].text)
    _, cache_corrupt = gpt2_model.run_with_cache(corrupt_tokens)

    effect = compute_edge_effect(
        gpt2_model, tokens, correct_ids[0], incorrect_ids[0],
        up_layer=0, up_head=0, down_layer=1, cache_clean=cache_clean, cache_corrupt=cache_corrupt)
    assert isinstance(effect, float)


def test_run_path_identification_returns_results(circuit_results):
    assert len(circuit_results) >= 1
    r = circuit_results[0]
    assert isinstance(r, EvalResult)
    assert r.metric_id == "G1.path_identification"
    assert r.n_samples >= 1


def test_run_path_identification_metadata_fields(circuit_results):
    r = circuit_results[0]
    meta = r.metadata
    assert meta["task"] == TASK
    assert "n_edges" in meta
    assert meta["n_edges"] > 0
    assert "max_specificity" in meta
    assert isinstance(meta["max_specificity"], float)
    assert "n_specific_edges" in meta
    assert isinstance(meta["n_specific_edges"], int)
    assert "edge_specificities" in meta
    assert isinstance(meta["edge_specificities"], dict)
    assert "passed" in meta
    assert isinstance(meta["passed"], bool)
    assert meta["threshold_specificity"] == pytest.approx(5.0)


def test_run_path_identification_value_equals_max_specificity(circuit_results):
    r = circuit_results[0]
    assert r.value == pytest.approx(r.metadata["max_specificity"])


def test_run_path_identification_specificity_nonnegative(circuit_results):
    r = circuit_results[0]
    for spec in r.metadata["edge_specificities"].values():
        assert spec >= 0.0
