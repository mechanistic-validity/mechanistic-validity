import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechanistic_validity" / "instruments"
    / "behavioral" / "logit_diff_recovery" / "22_mean_centered_logit.py"
)
_spec = importlib.util.spec_from_file_location("mean_centered_logit_22", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["mean_centered_logit_22"] = _mod
_spec.loader.exec_module(_mod)

logit_diff_standard = _mod.logit_diff_standard
logit_diff_mean_centered = _mod.logit_diff_mean_centered
run_mean_centered_logit = _mod.run_mean_centered_logit

from mechanistic_validity.instruments.common import EvalResult, load_model

TASK = "ioi"


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


def test_logit_diff_standard_basic():
    logits = torch.zeros(1, 5, 100)
    logits[0, -1, 10] = 7.0
    logits[0, -1, 20] = 3.0
    diff = logit_diff_standard(logits, correct_id=10, incorrect_id=20)
    assert diff == pytest.approx(4.0)


def test_logit_diff_mean_centered_basic():
    logits = torch.zeros(1, 5, 100)
    logits[0, -1, 10] = 7.0
    logits[0, -1, 20] = 3.0
    diff = logit_diff_mean_centered(logits, correct_id=10, incorrect_id=20)
    assert diff == pytest.approx(4.0)


def test_mean_centering_equals_standard_for_diff():
    logits = torch.randn(1, 10, 500)
    correct_id, incorrect_id = 42, 99
    std_diff = logit_diff_standard(logits, correct_id, incorrect_id)
    ctr_diff = logit_diff_mean_centered(logits, correct_id, incorrect_id)
    assert std_diff == pytest.approx(ctr_diff)


def test_mean_centering_invariant_to_constant_shift():
    logits = torch.randn(1, 5, 100)
    correct_id, incorrect_id = 10, 20
    diff_original = logit_diff_mean_centered(logits, correct_id, incorrect_id)
    shifted = logits.clone()
    shifted[0, -1, :] += 1000.0
    diff_shifted = logit_diff_mean_centered(shifted, correct_id, incorrect_id)
    assert diff_original == pytest.approx(diff_shifted)


def test_standard_also_invariant_to_constant_shift():
    logits = torch.randn(1, 5, 100)
    correct_id, incorrect_id = 10, 20
    diff_original = logit_diff_standard(logits, correct_id, incorrect_id)
    shifted = logits.clone()
    shifted[0, -1, :] += 1000.0
    diff_shifted = logit_diff_standard(shifted, correct_id, incorrect_id)
    assert diff_original == pytest.approx(diff_shifted)


def test_run_mean_centered_logit_returns_results(gpt2_model):
    results = run_mean_centered_logit(gpt2_model, [TASK], n_prompts=3)
    assert len(results) >= 1
    r = results[0]
    assert isinstance(r, EvalResult)
    assert r.metric_id == "C22.mean_centered_logit"
    assert r.n_samples >= 1


def test_run_mean_centered_logit_metadata_keys(gpt2_model):
    results = run_mean_centered_logit(gpt2_model, [TASK], n_prompts=3)
    r = results[0]
    expected_keys = {
        "task", "mean_standard_ld", "mean_centered_ld",
        "abs_diff", "ratio", "per_prompt_mean_ratio",
        "per_prompt_std_ratio", "note",
    }
    assert expected_keys.issubset(r.metadata.keys())


def test_run_mean_centered_logit_ratio_near_one(gpt2_model):
    results = run_mean_centered_logit(gpt2_model, [TASK], n_prompts=5)
    r = results[0]
    assert r.value == pytest.approx(1.0, abs=1e-4)


def test_run_mean_centered_logit_abs_diff_near_zero(gpt2_model):
    results = run_mean_centered_logit(gpt2_model, [TASK], n_prompts=5)
    r = results[0]
    assert r.metadata["abs_diff"] == pytest.approx(0.0, abs=1e-4)


def test_run_mean_centered_logit_unknown_task_returns_empty(gpt2_model):
    results = run_mean_centered_logit(gpt2_model, ["nonexistent_task_xyz"], n_prompts=3)
    assert results == []
