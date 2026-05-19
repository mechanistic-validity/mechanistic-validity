import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechval" / "metrics"
    / "behavioral" / "kl_divergence" / "21_output_variants.py"
)
_spec = importlib.util.spec_from_file_location("kl_divergence_21", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["kl_divergence_21"] = _mod
_spec.loader.exec_module(_mod)

metric_logit_diff = _mod.metric_logit_diff
metric_log_prob = _mod.metric_log_prob
metric_probability = _mod.metric_probability
metric_top1_accuracy = _mod.metric_top1_accuracy
metric_kl_divergence = _mod.metric_kl_divergence
faithfulness_by_metric = _mod.faithfulness_by_metric
run_output_variants = _mod.run_output_variants
METRICS = _mod.METRICS

from mechval.metrics.common import EvalResult, load_model

TASK = "ioi"


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


def test_metric_logit_diff_basic():
    logits = torch.zeros(1, 3, 50)
    logits[0, -1, 5] = 8.0
    logits[0, -1, 15] = 2.0
    assert metric_logit_diff(logits, 5, 15) == pytest.approx(6.0)


def test_metric_log_prob_is_negative():
    logits = torch.randn(1, 3, 50)
    lp = metric_log_prob(logits, 5, 15)
    assert lp <= 0.0


def test_metric_probability_sums_ok():
    logits = torch.randn(1, 3, 50)
    p = metric_probability(logits, 5, 15)
    assert 0.0 <= p <= 1.0


def test_metric_top1_accuracy_values():
    logits = torch.zeros(1, 3, 50)
    logits[0, -1, 7] = 100.0
    assert metric_top1_accuracy(logits, 7, 0) == pytest.approx(1.0)
    assert metric_top1_accuracy(logits, 0, 7) == pytest.approx(0.0)


def test_metric_kl_divergence_self_is_zero():
    logits = torch.randn(1, 3, 50)
    kl = metric_kl_divergence(logits, 5, 15, clean_logits=logits)
    assert kl == pytest.approx(0.0, abs=1e-5)


def test_metric_kl_divergence_different_is_positive():
    clean = torch.zeros(1, 3, 50)
    clean[0, -1, 5] = 10.0
    ablated = torch.zeros(1, 3, 50)
    ablated[0, -1, 15] = 10.0
    kl = metric_kl_divergence(ablated, 5, 15, clean_logits=clean)
    assert kl > 0.0


def test_run_output_variants_returns_results(gpt2_model):
    results = run_output_variants(gpt2_model, [TASK], n_prompts=3)
    assert len(results) >= 1
    r = results[0]
    assert isinstance(r, EvalResult)
    assert r.metric_id == "C21.output_variants"
    assert r.n_samples >= 1


def test_run_output_variants_has_all_metrics(gpt2_model):
    results = run_output_variants(gpt2_model, [TASK], n_prompts=3)
    r = results[0]
    scores = r.metadata["metric_scores"]
    for name in METRICS:
        assert name in scores
        assert isinstance(scores[name], float)


def test_run_output_variants_sigma_is_coefficient_of_variation(gpt2_model):
    results = run_output_variants(gpt2_model, [TASK], n_prompts=3)
    r = results[0]
    scores = list(r.metadata["metric_scores"].values())
    mean_f = float(np.mean(scores))
    std_f = float(np.std(scores))
    if abs(mean_f) > 1e-8:
        expected_sigma = std_f / abs(mean_f)
        assert r.value == pytest.approx(expected_sigma)


def test_run_output_variants_metadata_structure(gpt2_model):
    results = run_output_variants(gpt2_model, [TASK], n_prompts=3)
    r = results[0]
    expected_keys = {
        "task", "metric_scores", "mean_faithfulness",
        "std_faithfulness", "sigma_output", "n_circuit_heads",
    }
    assert expected_keys.issubset(r.metadata.keys())


def test_run_output_variants_unknown_task_returns_empty(gpt2_model):
    results = run_output_variants(gpt2_model, ["nonexistent_task_xyz"], n_prompts=3)
    assert results == []
