import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechanistic_validity" / "metrics"
    / "causal" / "counterfactual_das" / "79_intermediate_state_prediction.py"
)
_spec = importlib.util.spec_from_file_location("intermediate_state_79", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["intermediate_state_79"] = _mod
_spec.loader.exec_module(_mod)

collect_logit_attributions = _mod.collect_logit_attributions
run_intermediate_state_prediction = _mod.run_intermediate_state_prediction
PATHWAY_CORR_THRESHOLD = _mod.PATHWAY_CORR_THRESHOLD
UPLIFT_THRESHOLD = _mod.UPLIFT_THRESHOLD

from mechanistic_validity.metrics.common import EvalResult, load_model

TASK = "ioi"


def test_thresholds_are_positive():
    assert PATHWAY_CORR_THRESHOLD > 0
    assert UPLIFT_THRESHOLD > 0


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


def test_collect_logit_attributions_shape(gpt2_model):
    from mechanistic_validity.metrics.common import generate_prompts, get_token_ids

    prompts = generate_prompts(TASK, gpt2_model.tokenizer, n_prompts=5)
    correct_ids, incorrect_ids = get_token_ids(prompts, gpt2_model.tokenizer)
    heads = {(0, 0), (1, 1)}
    attrs = collect_logit_attributions(gpt2_model, prompts, heads, correct_ids, incorrect_ids)
    assert isinstance(attrs, dict)
    for h, arr in attrs.items():
        assert h in heads
        assert isinstance(arr, np.ndarray)
        assert len(arr) > 0


def test_collect_logit_attributions_values_are_finite(gpt2_model):
    from mechanistic_validity.metrics.common import generate_prompts, get_token_ids

    prompts = generate_prompts(TASK, gpt2_model.tokenizer, n_prompts=5)
    correct_ids, incorrect_ids = get_token_ids(prompts, gpt2_model.tokenizer)
    heads = {(0, 0)}
    attrs = collect_logit_attributions(gpt2_model, prompts, heads, correct_ids, incorrect_ids)
    for arr in attrs.values():
        assert np.all(np.isfinite(arr))


@pytest.fixture(scope="module")
def circuit_results(gpt2_model):
    return run_intermediate_state_prediction(gpt2_model, tasks=[TASK], n_prompts=5)


def test_run_returns_results(circuit_results):
    assert len(circuit_results) >= 1
    for r in circuit_results:
        assert isinstance(r, EvalResult)


def test_metric_id_is_correct(circuit_results):
    for r in circuit_results:
        assert r.metric_id == "A4.intermediate_state_prediction"


def test_value_is_finite(circuit_results):
    for r in circuit_results:
        assert np.isfinite(r.value)


def test_has_baseline_random(circuit_results):
    for r in circuit_results:
        assert r.baseline_random is not None


def test_metadata_has_edge_results(circuit_results):
    for r in circuit_results:
        assert "edge_results" in r.metadata
        assert isinstance(r.metadata["edge_results"], dict)


def test_metadata_has_passed_flag(circuit_results):
    for r in circuit_results:
        assert "passed" in r.metadata
        assert isinstance(r.metadata["passed"], bool)


def test_metadata_has_uplift(circuit_results):
    for r in circuit_results:
        assert "uplift" in r.metadata
        assert np.isfinite(r.metadata["uplift"])


def test_metadata_task(circuit_results):
    for r in circuit_results:
        assert r.metadata["task"] == TASK
