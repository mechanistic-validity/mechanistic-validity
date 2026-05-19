import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechanistic_validity" / "instruments"
    / "behavioral" / "mdl_compression" / "47_mdl_compression.py"
)
_spec = importlib.util.spec_from_file_location("mdl_compression_47", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["mdl_compression_47"] = _mod
_spec.loader.exec_module(_mod)

compute_kl_coding_cost = _mod.compute_kl_coding_cost
run_mdl_compression = _mod.run_mdl_compression
GPT2_TOTAL_HEADS = _mod.GPT2_TOTAL_HEADS

from mechanistic_validity.instruments.common import EvalResult, load_model

TASK = "ioi"


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


def test_gpt2_total_heads_constant():
    assert GPT2_TOTAL_HEADS == 144


def test_run_mdl_compression_returns_results(gpt2_model):
    results = run_mdl_compression(gpt2_model, [TASK], n_prompts=3)
    assert len(results) >= 1
    r = results[0]
    assert isinstance(r, EvalResult)
    assert r.metric_id == "D08.mdl_compression"
    assert r.n_samples >= 1


def test_run_mdl_compression_metadata_keys(gpt2_model):
    results = run_mdl_compression(gpt2_model, [TASK], n_prompts=3)
    r = results[0]
    expected_keys = {
        "task", "n_circuit_heads", "n_total_heads",
        "compression_ratio", "faithfulness", "efficiency",
        "kl_coding_cost", "bits_per_component",
    }
    assert expected_keys.issubset(r.metadata.keys())


def test_run_mdl_compression_compression_ratio_bounded(gpt2_model):
    results = run_mdl_compression(gpt2_model, [TASK], n_prompts=3)
    r = results[0]
    ratio = r.metadata["compression_ratio"]
    assert 0.0 < ratio < 1.0


def test_run_mdl_compression_efficiency_is_value(gpt2_model):
    results = run_mdl_compression(gpt2_model, [TASK], n_prompts=3)
    r = results[0]
    if r.metadata["compression_ratio"] > 1e-8:
        expected = r.metadata["faithfulness"] / r.metadata["compression_ratio"]
        assert r.value == pytest.approx(expected)


def test_run_mdl_compression_total_heads_correct(gpt2_model):
    results = run_mdl_compression(gpt2_model, [TASK], n_prompts=3)
    r = results[0]
    assert r.metadata["n_total_heads"] == GPT2_TOTAL_HEADS


def test_run_mdl_compression_kl_nonnegative(gpt2_model):
    results = run_mdl_compression(gpt2_model, [TASK], n_prompts=3)
    r = results[0]
    assert r.metadata["kl_coding_cost"] >= 0.0


def test_run_mdl_compression_compression_ratio_matches(gpt2_model):
    results = run_mdl_compression(gpt2_model, [TASK], n_prompts=3)
    r = results[0]
    expected = r.metadata["n_circuit_heads"] / r.metadata["n_total_heads"]
    assert r.metadata["compression_ratio"] == pytest.approx(expected)


def test_run_mdl_compression_unknown_task_returns_empty(gpt2_model):
    results = run_mdl_compression(gpt2_model, ["nonexistent_task_xyz"], n_prompts=3)
    assert results == []
