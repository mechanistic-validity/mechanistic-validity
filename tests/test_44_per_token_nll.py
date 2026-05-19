import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechanistic_validity" / "instruments"
    / "behavioral" / "per_token_nll" / "44_per_token_nll.py"
)
_spec = importlib.util.spec_from_file_location("per_token_nll_44", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["per_token_nll_44"] = _mod
_spec.loader.exec_module(_mod)

per_position_nll = _mod.per_position_nll
run_per_token_nll = _mod.run_per_token_nll

from mechanistic_validity.instruments.common import EvalResult, load_model

TASK = "ioi"


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


def test_per_position_nll_output_length(gpt2_model):
    tokens = gpt2_model.to_tokens("The cat sat on the mat")
    seq_len = tokens.shape[1]
    with torch.no_grad():
        logits = gpt2_model(tokens)
    nll = per_position_nll(logits, tokens)
    assert len(nll) == seq_len - 1


def test_per_position_nll_all_positive(gpt2_model):
    tokens = gpt2_model.to_tokens("The quick brown fox jumps")
    with torch.no_grad():
        logits = gpt2_model(tokens)
    nll = per_position_nll(logits, tokens)
    assert np.all(nll >= 0.0)


def test_per_position_nll_matches_manual(gpt2_model):
    tokens = gpt2_model.to_tokens("Hello world")
    with torch.no_grad():
        logits = gpt2_model(tokens)
    nll = per_position_nll(logits, tokens)
    shift_logits = logits[0, :-1, :]
    shift_targets = tokens[0, 1:]
    log_probs = F.log_softmax(shift_logits, dim=-1)
    expected = -log_probs[torch.arange(shift_targets.shape[0]), shift_targets].cpu().numpy()
    np.testing.assert_allclose(nll, expected, atol=1e-5)


def test_per_position_nll_with_synthetic_logits():
    vocab_size = 100
    seq_len = 5
    logits = torch.randn(1, seq_len, vocab_size)
    tokens = torch.randint(0, vocab_size, (1, seq_len))
    nll = per_position_nll(logits, tokens)
    assert len(nll) == seq_len - 1
    assert np.all(nll >= 0.0)


def test_run_per_token_nll_returns_results(gpt2_model):
    results = run_per_token_nll(gpt2_model, [TASK], n_prompts=3)
    assert len(results) >= 1
    r = results[0]
    assert isinstance(r, EvalResult)
    assert r.metric_id == "D05.per_token_nll"
    assert r.n_samples >= 1


def test_run_per_token_nll_metadata_keys(gpt2_model):
    results = run_per_token_nll(gpt2_model, [TASK], n_prompts=3)
    r = results[0]
    expected_keys = {
        "task", "mean_nll_increase_all_pos", "max_nll_increase",
        "mean_last_pos_increase", "last_pos_fraction", "n_circuit_heads",
        "n_prompts_used",
    }
    assert expected_keys.issubset(r.metadata.keys())


def test_run_per_token_nll_value_is_last_pos_increase(gpt2_model):
    results = run_per_token_nll(gpt2_model, [TASK], n_prompts=3)
    r = results[0]
    assert r.value == pytest.approx(r.metadata["mean_last_pos_increase"])


def test_run_per_token_nll_last_fraction_bounded(gpt2_model):
    results = run_per_token_nll(gpt2_model, [TASK], n_prompts=3)
    r = results[0]
    assert -10.0 <= r.metadata["last_pos_fraction"] <= 10.0


def test_run_per_token_nll_unknown_task_returns_empty(gpt2_model):
    results = run_per_token_nll(gpt2_model, ["nonexistent_task_xyz"], n_prompts=3)
    assert results == []
