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
    / "behavioral" / "topk_accuracy" / "21_output_variants.py"
)
_spec = importlib.util.spec_from_file_location("topk_accuracy_21", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["topk_accuracy_21"] = _mod
_spec.loader.exec_module(_mod)

metric_logit_diff = _mod.metric_logit_diff
metric_log_prob = _mod.metric_log_prob
metric_probability = _mod.metric_probability
metric_top1_accuracy = _mod.metric_top1_accuracy
metric_kl_divergence = _mod.metric_kl_divergence
faithfulness_by_metric = _mod.faithfulness_by_metric
run_output_variants = _mod.run_output_variants
METRICS = _mod.METRICS

from mechanistic_validity.instruments.common import EvalResult, load_model

TASK = "ioi"


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


def test_metric_logit_diff_symmetric():
    logits = torch.zeros(1, 3, 50)
    logits[0, -1, 10] = 4.0
    logits[0, -1, 20] = 1.0
    d1 = metric_logit_diff(logits, 10, 20)
    d2 = metric_logit_diff(logits, 20, 10)
    assert d1 == pytest.approx(-d2)


def test_metric_log_prob_highest_logit_has_highest_log_prob():
    logits = torch.zeros(1, 3, 50)
    logits[0, -1, 10] = 100.0
    lp_high = metric_log_prob(logits, 10, 20)
    lp_low = metric_log_prob(logits, 20, 10)
    assert lp_high > lp_low


def test_metric_probability_highest_logit_near_one():
    logits = torch.zeros(1, 3, 50)
    logits[0, -1, 10] = 100.0
    p = metric_probability(logits, 10, 20)
    assert p == pytest.approx(1.0, abs=1e-3)


def test_metric_top1_accuracy_binary():
    logits = torch.randn(1, 3, 50)
    acc = metric_top1_accuracy(logits, 0, 1)
    assert acc in (0.0, 1.0)


def test_metric_kl_divergence_none_clean_returns_zero():
    logits = torch.randn(1, 3, 50)
    kl = metric_kl_divergence(logits, 0, 1, clean_logits=None)
    assert kl == pytest.approx(0.0)


def test_metric_kl_divergence_nonnegative():
    clean = torch.randn(1, 3, 50)
    ablated = torch.randn(1, 3, 50)
    kl = metric_kl_divergence(ablated, 0, 1, clean_logits=clean)
    assert kl >= -1e-5


def test_run_output_variants_returns_results(gpt2_model):
    results = run_output_variants(gpt2_model, [TASK], n_prompts=3)
    assert len(results) >= 1
    r = results[0]
    assert isinstance(r, EvalResult)
    assert r.metric_id == "C21.output_variants"


def test_run_output_variants_five_metrics_scored(gpt2_model):
    results = run_output_variants(gpt2_model, [TASK], n_prompts=3)
    r = results[0]
    assert len(r.metadata["metric_scores"]) == 5


def test_run_output_variants_sigma_consistent(gpt2_model):
    results = run_output_variants(gpt2_model, [TASK], n_prompts=3)
    r = results[0]
    scores = list(r.metadata["metric_scores"].values())
    mean_f = np.mean(scores)
    std_f = np.std(scores)
    if abs(mean_f) > 1e-8:
        expected = std_f / abs(mean_f)
        assert r.value == pytest.approx(expected)


def test_run_output_variants_circuit_heads_present(gpt2_model):
    results = run_output_variants(gpt2_model, [TASK], n_prompts=3)
    r = results[0]
    assert "circuit_heads" in r.metadata
    assert r.metadata["n_circuit_heads"] > 0


def test_run_output_variants_unknown_task_returns_empty(gpt2_model):
    results = run_output_variants(gpt2_model, ["nonexistent_task_xyz"], n_prompts=3)
    assert results == []
