import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechanistic_validity" / "metrics"
    / "causal" / "mediation" / "24_pse.py"
)
_spec = importlib.util.spec_from_file_location("pse_24", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["pse_24"] = _mod
_spec.loader.exec_module(_mod)

compute_pse = _mod.compute_pse
run_pse = _mod.run_pse

from mechanistic_validity.metrics.common import EvalResult, load_model

TASK = "ioi"


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


def test_compute_pse_returns_per_head_and_clean_lds(gpt2_model):
    from mechanistic_validity.metrics.common import (
        calibrate_mean_z,
        generate_prompts,
        get_circuit_heads,
        get_token_ids,
    )

    prompts = generate_prompts(TASK, gpt2_model.tokenizer, n_prompts=4)
    correct_ids, incorrect_ids = get_token_ids(prompts, gpt2_model.tokenizer)
    circuit_heads = get_circuit_heads(TASK)
    mean_z = calibrate_mean_z(gpt2_model, prompts, n_calibration=4)

    per_head_pse, clean_lds = compute_pse(
        gpt2_model, prompts, correct_ids, incorrect_ids, circuit_heads, mean_z,
    )
    assert isinstance(per_head_pse, dict)
    assert len(per_head_pse) > 0
    assert isinstance(clean_lds, list)
    assert len(clean_lds) > 0


def test_compute_pse_values_are_finite(gpt2_model):
    from mechanistic_validity.metrics.common import (
        calibrate_mean_z,
        generate_prompts,
        get_circuit_heads,
        get_token_ids,
    )

    prompts = generate_prompts(TASK, gpt2_model.tokenizer, n_prompts=4)
    correct_ids, incorrect_ids = get_token_ids(prompts, gpt2_model.tokenizer)
    circuit_heads = get_circuit_heads(TASK)
    mean_z = calibrate_mean_z(gpt2_model, prompts, n_calibration=4)

    per_head_pse, clean_lds = compute_pse(
        gpt2_model, prompts, correct_ids, incorrect_ids, circuit_heads, mean_z,
    )
    for key, vals in per_head_pse.items():
        for v in vals:
            assert np.isfinite(v)
    for ld in clean_lds:
        assert np.isfinite(ld)


@pytest.fixture(scope="module")
def circuit_results(gpt2_model):
    return run_pse(gpt2_model, tasks=[TASK], n_prompts=4)


def test_run_pse_returns_results(circuit_results):
    assert len(circuit_results) >= 1
    for r in circuit_results:
        assert isinstance(r, EvalResult)


def test_run_pse_metric_id(circuit_results):
    for r in circuit_results:
        assert r.metric_id == "C24.pse"


def test_run_pse_value_is_finite(circuit_results):
    for r in circuit_results:
        assert np.isfinite(r.value)


def test_run_pse_metadata_has_per_head_pse(circuit_results):
    for r in circuit_results:
        assert "per_head_pse" in r.metadata
        assert isinstance(r.metadata["per_head_pse"], dict)


def test_run_pse_metadata_has_sum_and_total(circuit_results):
    for r in circuit_results:
        assert "sum_pse" in r.metadata
        assert "total_effect" in r.metadata
        assert np.isfinite(r.metadata["sum_pse"])
        assert np.isfinite(r.metadata["total_effect"])


def test_run_pse_ratio_equals_value(circuit_results):
    for r in circuit_results:
        te = r.metadata["total_effect"]
        sp = r.metadata["sum_pse"]
        expected = sp / te if abs(te) > 1e-8 else 0.0
        assert r.value == pytest.approx(expected, abs=1e-4)


def test_run_pse_metadata_task(circuit_results):
    for r in circuit_results:
        assert r.metadata["task"] == TASK
