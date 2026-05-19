import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechval" / "metrics"
    / "causal" / "mdc_glennan" / "72_replacement_test.py"
)
_spec = importlib.util.spec_from_file_location("replacement_test_72", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["replacement_test_72"] = _mod
_spec.loader.exec_module(_mod)

recovery_constant_replacement = _mod.recovery_constant_replacement
recovery_linear_ov_replacement = _mod.recovery_linear_ov_replacement
run_replacement_test = _mod.run_replacement_test

from mechval.metrics.common import EvalResult, load_model

TASK = "ioi"


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


def test_recovery_constant_returns_finite(gpt2_model):
    from mechval.metrics.common import (
        calibrate_mean_z,
        generate_prompts,
        get_token_ids,
    )

    prompts = generate_prompts(TASK, gpt2_model.tokenizer, n_prompts=3)
    correct_ids, incorrect_ids = get_token_ids(prompts, gpt2_model.tokenizer)
    mean_z = calibrate_mean_z(gpt2_model, prompts, n_calibration=3)

    rec = recovery_constant_replacement(
        gpt2_model, prompts, correct_ids, incorrect_ids, (0, 0), mean_z,
    )
    assert np.isfinite(rec)


def test_recovery_linear_ov_returns_finite(gpt2_model):
    from mechval.metrics.common import (
        generate_prompts,
        get_token_ids,
    )

    prompts = generate_prompts(TASK, gpt2_model.tokenizer, n_prompts=3)
    correct_ids, incorrect_ids = get_token_ids(prompts, gpt2_model.tokenizer)

    rec = recovery_linear_ov_replacement(
        gpt2_model, prompts, correct_ids, incorrect_ids, (0, 0),
    )
    assert np.isfinite(rec)


@pytest.fixture(scope="module")
def circuit_results(gpt2_model):
    return run_replacement_test(gpt2_model, tasks=[TASK], n_prompts=3)


def test_run_replacement_test_returns_two_metrics(circuit_results):
    metric_ids = [r.metric_id for r in circuit_results]
    assert "F3.replacement_constant" in metric_ids
    assert "F3.replacement_linear_ov" in metric_ids


def test_run_replacement_test_all_eval_results(circuit_results):
    for r in circuit_results:
        assert isinstance(r, EvalResult)


def test_run_replacement_test_values_are_finite(circuit_results):
    for r in circuit_results:
        assert np.isfinite(r.value)


def test_run_replacement_test_has_per_head_recovery(circuit_results):
    for r in circuit_results:
        assert "per_head_recovery" in r.metadata
        assert isinstance(r.metadata["per_head_recovery"], dict)


def test_run_replacement_test_has_replacement_type(circuit_results):
    for r in circuit_results:
        assert "replacement_type" in r.metadata


def test_run_replacement_test_metadata_task(circuit_results):
    for r in circuit_results:
        assert r.metadata["task"] == TASK


def test_run_replacement_test_baseline_is_one(circuit_results):
    for r in circuit_results:
        assert r.baseline_random == pytest.approx(1.0)
