import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechanistic_validity" / "metrics"
    / "behavioral" / "ce_delta" / "43_ce_delta.py"
)
_spec = importlib.util.spec_from_file_location("ce_delta_43", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["ce_delta_43"] = _mod
_spec.loader.exec_module(_mod)

compute_ce_at_last = _mod.compute_ce_at_last
compute_ce_with_hooks = _mod.compute_ce_with_hooks
run_ce_delta = _mod.run_ce_delta

from mechanistic_validity.metrics.common import EvalResult, load_model

TASK = "ioi"


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


def test_compute_ce_at_last_returns_positive_nll(gpt2_model):
    tokens = gpt2_model.to_tokens("The cat sat on the")
    vocab_size = gpt2_model.cfg.d_vocab
    target_id = 262  # " the"
    ce = compute_ce_at_last(gpt2_model, tokens, target_id)
    assert isinstance(ce, float)
    assert ce > 0.0


def test_compute_ce_at_last_correct_token_lower_than_rare(gpt2_model):
    tokens = gpt2_model.to_tokens("The cat sat on the")
    common_id = 262  # " the"
    rare_id = 48223  # some rare token
    ce_common = compute_ce_at_last(gpt2_model, tokens, common_id)
    ce_rare = compute_ce_at_last(gpt2_model, tokens, rare_id)
    assert ce_common < ce_rare


def test_compute_ce_at_last_matches_manual(gpt2_model):
    tokens = gpt2_model.to_tokens("Hello world")
    target_id = 995
    ce = compute_ce_at_last(gpt2_model, tokens, target_id)
    with torch.no_grad():
        logits = gpt2_model(tokens)
        log_probs = F.log_softmax(logits[0, -1], dim=-1)
        expected = -log_probs[target_id].item()
    assert ce == pytest.approx(expected)


def test_run_ce_delta_returns_eval_results(gpt2_model):
    results = run_ce_delta(gpt2_model, [TASK], n_prompts=3)
    assert len(results) >= 1
    r = results[0]
    assert isinstance(r, EvalResult)
    assert r.metric_id == "D04.ce_delta"
    assert r.n_samples >= 1
    assert "task" in r.metadata
    assert r.metadata["task"] == TASK


def test_run_ce_delta_metadata_has_expected_keys(gpt2_model):
    results = run_ce_delta(gpt2_model, [TASK], n_prompts=3)
    r = results[0]
    expected_keys = {
        "task", "mean_ce_full", "mean_ce_circuit_ablated",
        "mean_ce_complement_ablated", "delta_circuit", "delta_complement",
        "n_circuit_heads", "n_complement_heads",
    }
    assert expected_keys.issubset(r.metadata.keys())


def test_run_ce_delta_ce_full_is_positive(gpt2_model):
    results = run_ce_delta(gpt2_model, [TASK], n_prompts=3)
    r = results[0]
    assert r.metadata["mean_ce_full"] > 0.0


def test_run_ce_delta_circuit_ablated_increases_ce(gpt2_model):
    results = run_ce_delta(gpt2_model, [TASK], n_prompts=3)
    r = results[0]
    assert r.metadata["delta_circuit"] > -1.0


def test_run_ce_delta_value_is_ratio(gpt2_model):
    results = run_ce_delta(gpt2_model, [TASK], n_prompts=3)
    r = results[0]
    if abs(r.metadata["delta_complement"]) > 1e-8:
        expected_ratio = r.metadata["delta_circuit"] / r.metadata["delta_complement"]
        assert r.value == pytest.approx(expected_ratio)


def test_run_ce_delta_unknown_task_returns_empty(gpt2_model):
    results = run_ce_delta(gpt2_model, ["nonexistent_task_xyz"], n_prompts=3)
    assert results == []
