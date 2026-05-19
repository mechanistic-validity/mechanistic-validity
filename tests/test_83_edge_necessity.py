import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from mechanistic_validity.metrics.common import EvalResult, load_model

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechanistic_validity" / "metrics"
    / "structural" / "edge_analysis" / "83_edge_necessity.py"
)
_spec = importlib.util.spec_from_file_location("_edge_nec_83", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)

ablate_single_edge = _mod.ablate_single_edge
run_edge_necessity = _mod.run_edge_necessity

TASK = "ioi"


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


@pytest.fixture(scope="module")
def circuit_results(gpt2_model):
    return run_edge_necessity(gpt2_model, [TASK], n_prompts=3)


def test_ablate_single_edge_returns_two_floats(gpt2_model):
    from mechanistic_validity.metrics.common import (
        calibrate_mean_z, generate_prompts, get_token_ids,
    )
    prompts = generate_prompts(TASK, gpt2_model.tokenizer, 3)
    correct_ids, incorrect_ids = get_token_ids(prompts, gpt2_model.tokenizer)
    mean_z = calibrate_mean_z(gpt2_model, prompts, n_calibration=3)
    tokens = gpt2_model.to_tokens(prompts[0].text)

    clean_ld, ablated_ld = ablate_single_edge(
        gpt2_model, tokens, correct_ids[0], incorrect_ids[0],
        up_layer=0, up_head=0, down_layer=1, mean_z=mean_z)
    assert isinstance(clean_ld, float)
    assert isinstance(ablated_ld, float)


def test_run_edge_necessity_returns_results(circuit_results):
    assert len(circuit_results) >= 1
    r = circuit_results[0]
    assert isinstance(r, EvalResult)
    assert r.metric_id == "G2.edge_necessity"
    assert r.n_samples >= 1


def test_run_edge_necessity_value_is_fraction(circuit_results):
    r = circuit_results[0]
    assert 0.0 <= r.value <= 1.0


def test_run_edge_necessity_metadata_fields(circuit_results):
    r = circuit_results[0]
    meta = r.metadata
    assert meta["task"] == TASK
    assert "n_edges" in meta
    assert meta["n_edges"] > 0
    assert "n_necessary" in meta
    assert isinstance(meta["n_necessary"], int)
    assert "frac_necessary" in meta
    assert r.value == pytest.approx(meta["frac_necessary"])
    assert "per_edge" in meta
    assert isinstance(meta["per_edge"], dict)
    assert "passed" in meta
    assert isinstance(meta["passed"], bool)
    assert meta["threshold_frac"] == pytest.approx(0.5)
    assert meta["threshold_drop"] == pytest.approx(0.05)


def test_run_edge_necessity_per_edge_has_drop_frac(circuit_results):
    r = circuit_results[0]
    for edge_name, data in r.metadata["per_edge"].items():
        assert "drop_frac" in data
        assert "necessary" in data
        assert isinstance(data["necessary"], bool)
