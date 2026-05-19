import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from mechanistic_validity.metrics.common import EvalResult, load_model, get_circuit_heads

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechanistic_validity" / "metrics"
    / "structural" / "llc_rllc" / "10_llc.py"
)
_spec = importlib.util.spec_from_file_location("_llc_10", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)

estimate_llc_hessian = _mod.estimate_llc_hessian
run_llc = _mod.run_llc

TASK = "ioi"


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


@pytest.fixture(scope="module")
def circuit_results(gpt2_model):
    return run_llc(gpt2_model, [TASK], n_prompts=3)


def test_estimate_llc_hessian_returns_float(gpt2_model):
    from mechanistic_validity.metrics.common import generate_prompts, get_token_ids
    prompts = generate_prompts(TASK, gpt2_model.tokenizer, 3)
    correct_ids, incorrect_ids = get_token_ids(prompts, gpt2_model.tokenizer)
    llc = estimate_llc_hessian(gpt2_model, prompts, correct_ids, incorrect_ids,
                                layer=0, head=0, n_samples=3)
    assert isinstance(llc, float)
    assert llc >= 0.0


def test_run_llc_returns_results(circuit_results):
    assert len(circuit_results) >= 1
    r = circuit_results[0]
    assert isinstance(r, EvalResult)
    assert r.metric_id == "C10.llc"
    assert r.n_samples >= 1


def test_run_llc_value_nonnegative(circuit_results):
    r = circuit_results[0]
    assert r.value >= 0.0


def test_run_llc_metadata_fields(circuit_results):
    r = circuit_results[0]
    meta = r.metadata
    assert meta["task"] == TASK
    assert "per_head_llc" in meta
    assert isinstance(meta["per_head_llc"], dict)
    assert "mean_circuit_llc" in meta
    assert "mean_non_circuit_llc" in meta
    assert "ratio" in meta
    assert "n_circuit_heads" in meta
    assert meta["n_circuit_heads"] > 0
    assert meta["interpretation"] == "lower LLC = more specialized/degenerate"


def test_run_llc_baseline_random_present(circuit_results):
    r = circuit_results[0]
    assert r.baseline_random is not None
    assert r.baseline_random >= 0.0


def test_run_llc_value_matches_mean_circuit(circuit_results):
    r = circuit_results[0]
    assert r.value == pytest.approx(r.metadata["mean_circuit_llc"])
