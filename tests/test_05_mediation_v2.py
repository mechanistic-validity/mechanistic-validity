import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechval" / "metrics"
    / "causal" / "mediation" / "05_mediation_v2.py"
)
_spec = importlib.util.spec_from_file_location("mediation_v2_05", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["mediation_v2_05"] = _mod
_spec.loader.exec_module(_mod)

compute_mediation_effects = _mod.compute_mediation_effects
run_mediation_v2 = _mod.run_mediation_v2

from mechval.metrics.common import EvalResult, load_model

TASK = "ioi"


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


def test_compute_mediation_effects_returns_per_head(gpt2_model):
    from mechval.metrics.common import (
        generate_prompts,
        get_circuit_heads,
        get_token_ids,
    )

    prompts = generate_prompts(TASK, gpt2_model.tokenizer, n_prompts=4)
    correct_ids, incorrect_ids = get_token_ids(prompts, gpt2_model.tokenizer)
    circuit_heads = get_circuit_heads(TASK)
    rng = np.random.RandomState(42)
    per_head = compute_mediation_effects(
        gpt2_model, prompts, correct_ids, incorrect_ids, circuit_heads, rng,
    )
    assert isinstance(per_head, dict)
    assert len(per_head) > 0
    for key, vals in per_head.items():
        assert "total_effect" in vals
        assert "nie" in vals
        assert "nde" in vals
        assert "proportion_mediated" in vals


def test_v2_nie_nde_sum_to_te(gpt2_model):
    from mechval.metrics.common import (
        generate_prompts,
        get_circuit_heads,
        get_token_ids,
    )

    prompts = generate_prompts(TASK, gpt2_model.tokenizer, n_prompts=4)
    correct_ids, incorrect_ids = get_token_ids(prompts, gpt2_model.tokenizer)
    circuit_heads = get_circuit_heads(TASK)
    rng = np.random.RandomState(42)
    per_head = compute_mediation_effects(
        gpt2_model, prompts, correct_ids, incorrect_ids, circuit_heads, rng,
    )
    for key, vals in per_head.items():
        te = vals["total_effect"]
        nie = vals["nie"]
        nde = vals["nde"]
        assert nie + nde == pytest.approx(te, abs=1e-4)


def test_v2_proportion_mediated_is_nie_over_te(gpt2_model):
    from mechval.metrics.common import (
        generate_prompts,
        get_circuit_heads,
        get_token_ids,
    )

    prompts = generate_prompts(TASK, gpt2_model.tokenizer, n_prompts=4)
    correct_ids, incorrect_ids = get_token_ids(prompts, gpt2_model.tokenizer)
    circuit_heads = get_circuit_heads(TASK)
    rng = np.random.RandomState(42)
    per_head = compute_mediation_effects(
        gpt2_model, prompts, correct_ids, incorrect_ids, circuit_heads, rng,
    )
    for key, vals in per_head.items():
        te = vals["total_effect"]
        nie = vals["nie"]
        expected = nie / (abs(te) + 1e-8)
        assert vals["proportion_mediated"] == pytest.approx(expected, abs=1e-4)


@pytest.fixture(scope="module")
def circuit_results(gpt2_model):
    return run_mediation_v2(gpt2_model, tasks=[TASK], n_prompts=4)


def test_run_mediation_v2_returns_results(circuit_results):
    assert len(circuit_results) >= 1
    for r in circuit_results:
        assert isinstance(r, EvalResult)


def test_run_mediation_v2_metric_id(circuit_results):
    for r in circuit_results:
        assert r.metric_id == "C5v2.mediation"


def test_run_mediation_v2_value_is_finite(circuit_results):
    for r in circuit_results:
        assert np.isfinite(r.value)


def test_run_mediation_v2_metadata_has_per_head(circuit_results):
    for r in circuit_results:
        assert "per_head" in r.metadata
        assert isinstance(r.metadata["per_head"], dict)


def test_run_mediation_v2_metadata_has_mean_nie(circuit_results):
    for r in circuit_results:
        assert "mean_nie" in r.metadata
        assert np.isfinite(r.metadata["mean_nie"])


def test_run_mediation_v2_metadata_task(circuit_results):
    for r in circuit_results:
        assert r.metadata["task"] == TASK
