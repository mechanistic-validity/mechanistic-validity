import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechanistic_validity" / "metrics"
    / "causal" / "counterfactual_das" / "34_counterfactual_consistency.py"
)
_spec = importlib.util.spec_from_file_location("counterfactual_consistency_34", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["counterfactual_consistency_34"] = _mod
_spec.loader.exec_module(_mod)

compute_logit_diffs_for_prompts = _mod.compute_logit_diffs_for_prompts
run_counterfactual_consistency = _mod.run_counterfactual_consistency
PARAPHRASE_SEEDS = _mod.PARAPHRASE_SEEDS

from mechanistic_validity.metrics.common import EvalResult, load_model

TASK = "ioi"


def test_paraphrase_seeds_has_multiple_entries():
    assert len(PARAPHRASE_SEEDS) >= 2


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


def test_compute_logit_diffs_returns_list(gpt2_model):
    from mechanistic_validity.metrics.common import generate_prompts, get_token_ids

    prompts = generate_prompts(TASK, gpt2_model.tokenizer, n_prompts=3)
    correct_ids, incorrect_ids = get_token_ids(prompts, gpt2_model.tokenizer)
    diffs = compute_logit_diffs_for_prompts(gpt2_model, prompts, correct_ids, incorrect_ids)
    assert isinstance(diffs, list)
    assert len(diffs) == len(correct_ids)
    for d in diffs:
        assert np.isfinite(d)


@pytest.fixture(scope="module")
def circuit_results(gpt2_model):
    return run_counterfactual_consistency(gpt2_model, tasks=[TASK], n_prompts=4)


def test_run_counterfactual_consistency_returns_results(circuit_results):
    assert len(circuit_results) >= 1
    for r in circuit_results:
        assert isinstance(r, EvalResult)


def test_metric_id_is_correct(circuit_results):
    for r in circuit_results:
        assert r.metric_id == "C34.counterfactual_consistency"


def test_value_in_valid_range(circuit_results):
    for r in circuit_results:
        assert 0.0 <= r.value <= 1.0


def test_metadata_has_consistency_components(circuit_results):
    for r in circuit_results:
        assert "faithfulness_consistency" in r.metadata
        assert "logit_diff_consistency" in r.metadata
        assert "faithfulness_mean" in r.metadata
        assert "faithfulness_std" in r.metadata


def test_metadata_has_seeds(circuit_results):
    for r in circuit_results:
        assert "seeds" in r.metadata
        assert r.metadata["seeds"] == PARAPHRASE_SEEDS


def test_metadata_has_task(circuit_results):
    for r in circuit_results:
        assert r.metadata["task"] == TASK


def test_overall_consistency_is_average_of_components(circuit_results):
    for r in circuit_results:
        expected = (
            r.metadata["faithfulness_consistency"]
            + r.metadata["logit_diff_consistency"]
        ) / 2.0
        assert r.value == pytest.approx(expected)
