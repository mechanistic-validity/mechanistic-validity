import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechval" / "metrics"
    / "causal" / "mdc_glennan" / "70_operation_specification.py"
)
_spec = importlib.util.spec_from_file_location("operation_spec_70", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["operation_spec_70"] = _mod
_spec.loader.exec_module(_mod)

collect_head_outputs = _mod.collect_head_outputs
r_squared = _mod.r_squared
run_operation_specification = _mod.run_operation_specification

from mechval.metrics.common import EvalResult, load_model

TASK = "ioi"


def test_r_squared_perfect_prediction():
    actual = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    r2 = r_squared(actual, actual)
    assert r2 == pytest.approx(1.0)


def test_r_squared_zero_for_mean_prediction():
    actual = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    mean_pred = actual.mean(dim=0, keepdim=True).expand_as(actual)
    r2 = r_squared(mean_pred, actual)
    assert r2 == pytest.approx(0.0, abs=1e-6)


def test_r_squared_negative_for_bad_prediction():
    actual = torch.tensor([[1.0], [2.0], [3.0]])
    predicted = torch.tensor([[10.0], [20.0], [30.0]])
    r2 = r_squared(predicted, actual)
    assert r2 < 0.0


def test_r_squared_zero_variance_returns_zero():
    actual = torch.tensor([[5.0], [5.0], [5.0]])
    predicted = torch.tensor([[1.0], [2.0], [3.0]])
    r2 = r_squared(predicted, actual)
    assert r2 == pytest.approx(0.0)


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


def test_collect_head_outputs_returns_correct_shapes(gpt2_model):
    from mechval.metrics.common import generate_prompts

    prompts = generate_prompts(TASK, gpt2_model.tokenizer, n_prompts=4)
    actual_z, attn_ov_pred, consistency = collect_head_outputs(
        gpt2_model, prompts, layer=0, head=0,
    )
    assert actual_z.shape == (len(prompts), gpt2_model.cfg.d_head)
    assert attn_ov_pred.shape == actual_z.shape
    assert 0.0 <= consistency <= 1.0


@pytest.fixture(scope="module")
def circuit_results(gpt2_model):
    return run_operation_specification(gpt2_model, tasks=[TASK], n_prompts=5)


def test_run_operation_specification_returns_results(circuit_results):
    assert len(circuit_results) >= 1
    for r in circuit_results:
        assert isinstance(r, EvalResult)


def test_metric_id_is_correct(circuit_results):
    for r in circuit_results:
        assert r.metric_id == "F1.operation_specification"


def test_value_is_finite(circuit_results):
    for r in circuit_results:
        assert np.isfinite(r.value)


def test_has_baseline_random(circuit_results):
    for r in circuit_results:
        assert r.baseline_random is not None


def test_metadata_has_per_head(circuit_results):
    for r in circuit_results:
        assert "per_head" in r.metadata
        assert isinstance(r.metadata["per_head"], dict)
        for head_key, scores in r.metadata["per_head"].items():
            assert "consistency" in scores
            assert "attn_ov_r2" in scores
            assert "combined" in scores


def test_metadata_has_passed_flag(circuit_results):
    for r in circuit_results:
        assert "passed" in r.metadata
        assert isinstance(r.metadata["passed"], bool)


def test_metadata_task(circuit_results):
    for r in circuit_results:
        assert r.metadata["task"] == TASK
