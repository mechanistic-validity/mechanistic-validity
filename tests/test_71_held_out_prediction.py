import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechanistic_validity" / "instruments"
    / "causal" / "mdc_glennan" / "71_held_out_prediction.py"
)
_spec = importlib.util.spec_from_file_location("held_out_71", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["held_out_71"] = _mod
_spec.loader.exec_module(_mod)

pearson_correlation = _mod.pearson_correlation
held_out_prediction_correlation = _mod.held_out_prediction_correlation
collect_head_z = _mod.collect_head_z
run_held_out_prediction = _mod.run_held_out_prediction

from mechanistic_validity.instruments.common import EvalResult, load_model

TASK = "ioi"


def test_pearson_correlation_perfect_positive():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    assert pearson_correlation(x, y) == pytest.approx(1.0)


def test_pearson_correlation_perfect_negative():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([10.0, 8.0, 6.0, 4.0, 2.0])
    assert pearson_correlation(x, y) == pytest.approx(-1.0)


def test_pearson_correlation_zero_for_uncorrelated():
    x = np.array([1.0, -1.0, 1.0, -1.0])
    y = np.array([1.0, 1.0, -1.0, -1.0])
    assert pearson_correlation(x, y) == pytest.approx(0.0)


def test_pearson_correlation_too_few_samples():
    x = np.array([1.0, 2.0])
    y = np.array([3.0, 4.0])
    assert pearson_correlation(x, y) == pytest.approx(0.0)


def test_pearson_correlation_zero_variance():
    x = np.array([5.0, 5.0, 5.0])
    y = np.array([1.0, 2.0, 3.0])
    assert pearson_correlation(x, y) == pytest.approx(0.0)


def test_held_out_prediction_too_few_samples():
    z = torch.randn(4, 16)
    assert held_out_prediction_correlation(z) == pytest.approx(0.0)


def test_held_out_prediction_returns_finite():
    z = torch.randn(20, 16)
    corr = held_out_prediction_correlation(z)
    assert np.isfinite(corr)
    assert -1.0 <= corr <= 1.0


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


def test_collect_head_z_shape(gpt2_model):
    from mechanistic_validity.instruments.common import generate_prompts

    prompts = generate_prompts(TASK, gpt2_model.tokenizer, n_prompts=4)
    z = collect_head_z(gpt2_model, prompts, layer=0, head=0)
    assert z.shape == (len(prompts), gpt2_model.cfg.d_head)


@pytest.fixture(scope="module")
def circuit_results(gpt2_model):
    return run_held_out_prediction(gpt2_model, tasks=[TASK], n_prompts=8)


def test_run_held_out_prediction_returns_results(circuit_results):
    assert len(circuit_results) >= 1
    for r in circuit_results:
        assert isinstance(r, EvalResult)


def test_metric_id_is_correct(circuit_results):
    for r in circuit_results:
        assert r.metric_id == "F2.held_out_prediction"


def test_value_is_finite(circuit_results):
    for r in circuit_results:
        assert np.isfinite(r.value)


def test_has_baseline_random(circuit_results):
    for r in circuit_results:
        assert r.baseline_random is not None


def test_metadata_has_per_head_correlation(circuit_results):
    for r in circuit_results:
        assert "per_head_correlation" in r.metadata
        assert isinstance(r.metadata["per_head_correlation"], dict)


def test_metadata_has_frac_above_baseline(circuit_results):
    for r in circuit_results:
        assert "frac_above_baseline_median" in r.metadata
        frac = r.metadata["frac_above_baseline_median"]
        assert 0.0 <= frac <= 1.0


def test_metadata_task(circuit_results):
    for r in circuit_results:
        assert r.metadata["task"] == TASK
