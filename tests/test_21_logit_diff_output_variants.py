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
    / "behavioral" / "logit_diff_recovery" / "21_output_variants.py"
)
_spec = importlib.util.spec_from_file_location("logit_diff_output_variants_21", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["logit_diff_output_variants_21"] = _mod
_spec.loader.exec_module(_mod)

metric_logit_diff = _mod.metric_logit_diff
metric_log_prob = _mod.metric_log_prob
metric_probability = _mod.metric_probability
metric_top1_accuracy = _mod.metric_top1_accuracy
metric_kl_divergence = _mod.metric_kl_divergence
run_output_variants = _mod.run_output_variants
METRICS = _mod.METRICS

from mechanistic_validity.instruments.common import EvalResult, load_model

TASK = "ioi"


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


def test_metric_logit_diff_sign():
    logits = torch.zeros(1, 5, 100)
    logits[0, -1, 10] = 5.0
    logits[0, -1, 20] = 2.0
    diff = metric_logit_diff(logits, correct_id=10, incorrect_id=20)
    assert diff == pytest.approx(3.0)


def test_metric_logit_diff_negative():
    logits = torch.zeros(1, 5, 100)
    logits[0, -1, 10] = 1.0
    logits[0, -1, 20] = 5.0
    diff = metric_logit_diff(logits, correct_id=10, incorrect_id=20)
    assert diff == pytest.approx(-4.0)


def test_metric_log_prob_range():
    logits = torch.randn(1, 5, 100)
    lp = metric_log_prob(logits, correct_id=10, incorrect_id=20)
    assert lp <= 0.0


def test_metric_probability_range():
    logits = torch.randn(1, 5, 100)
    prob = metric_probability(logits, correct_id=10, incorrect_id=20)
    assert 0.0 <= prob <= 1.0


def test_metric_top1_accuracy_correct():
    logits = torch.zeros(1, 5, 100)
    logits[0, -1, 42] = 100.0
    acc = metric_top1_accuracy(logits, correct_id=42, incorrect_id=10)
    assert acc == pytest.approx(1.0)


def test_metric_top1_accuracy_incorrect():
    logits = torch.zeros(1, 5, 100)
    logits[0, -1, 10] = 100.0
    acc = metric_top1_accuracy(logits, correct_id=42, incorrect_id=10)
    assert acc == pytest.approx(0.0)


def test_metric_kl_divergence_identical():
    logits = torch.randn(1, 5, 100)
    kl = metric_kl_divergence(logits, 10, 20, clean_logits=logits)
    assert kl == pytest.approx(0.0, abs=1e-5)


def test_metric_kl_divergence_positive():
    clean = torch.zeros(1, 5, 100)
    clean[0, -1, 10] = 10.0
    ablated = torch.zeros(1, 5, 100)
    ablated[0, -1, 20] = 10.0
    kl = metric_kl_divergence(ablated, 10, 20, clean_logits=clean)
    assert kl > 0.0


def test_metric_kl_divergence_no_clean():
    logits = torch.randn(1, 5, 100)
    kl = metric_kl_divergence(logits, 10, 20, clean_logits=None)
    assert kl == pytest.approx(0.0)


def test_metrics_dict_has_five_entries():
    assert len(METRICS) == 5
    assert "logit_diff" in METRICS
    assert "log_prob" in METRICS
    assert "probability" in METRICS
    assert "top1_accuracy" in METRICS
    assert "kl_divergence" in METRICS


def test_run_output_variants_returns_results(gpt2_model):
    results = run_output_variants(gpt2_model, [TASK], n_prompts=3)
    assert len(results) >= 1
    r = results[0]
    assert isinstance(r, EvalResult)
    assert r.metric_id == "C21.output_variants"
    assert r.n_samples >= 1


def test_run_output_variants_metadata_keys(gpt2_model):
    results = run_output_variants(gpt2_model, [TASK], n_prompts=3)
    r = results[0]
    expected_keys = {
        "task", "metric_scores", "mean_faithfulness",
        "std_faithfulness", "sigma_output", "n_circuit_heads",
        "circuit_heads",
    }
    assert expected_keys.issubset(r.metadata.keys())


def test_run_output_variants_all_metrics_present(gpt2_model):
    results = run_output_variants(gpt2_model, [TASK], n_prompts=3)
    r = results[0]
    scores = r.metadata["metric_scores"]
    for metric_name in METRICS:
        assert metric_name in scores


def test_run_output_variants_sigma_is_value(gpt2_model):
    results = run_output_variants(gpt2_model, [TASK], n_prompts=3)
    r = results[0]
    assert r.value == pytest.approx(r.metadata["sigma_output"])


def test_run_output_variants_sigma_nonnegative(gpt2_model):
    results = run_output_variants(gpt2_model, [TASK], n_prompts=3)
    r = results[0]
    assert r.value >= 0.0


def test_run_output_variants_unknown_task_returns_empty(gpt2_model):
    results = run_output_variants(gpt2_model, ["nonexistent_task_xyz"], n_prompts=3)
    assert results == []
