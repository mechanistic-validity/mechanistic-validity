import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechanistic_validity" / "metrics"
    / "causal" / "regularity_inus" / "39_inus_conditions.py"
)
_spec = importlib.util.spec_from_file_location("inus_39", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["inus_39"] = _mod
_spec.loader.exec_module(_mod)

classify_inus = _mod.classify_inus
find_minimal_sufficient_sets = _mod.find_minimal_sufficient_sets

from mechanistic_validity.metrics.common import EvalResult, load_model

TASK = "ioi"


def test_classify_inus_necessary():
    heads = {(0, 0), (1, 0), (2, 0)}
    minimal_sets = [
        {(0, 0), (1, 0)},
        {(0, 0), (2, 0)},
    ]
    result = classify_inus(heads, minimal_sets)
    assert result[(0, 0)]["status"] == "necessary"
    assert result[(0, 0)]["in_n_sets"] == 2
    assert result[(0, 0)]["redundancy_index"] == pytest.approx(1.0)


def test_classify_inus_inus_status():
    heads = {(0, 0), (1, 0), (2, 0)}
    minimal_sets = [
        {(0, 0), (1, 0)},
        {(0, 0), (2, 0)},
    ]
    result = classify_inus(heads, minimal_sets)
    assert result[(1, 0)]["status"] == "inus"
    assert result[(2, 0)]["status"] == "inus"


def test_classify_inus_redundant():
    heads = {(0, 0), (1, 0), (2, 0)}
    minimal_sets = [
        {(0, 0), (1, 0)},
    ]
    result = classify_inus(heads, minimal_sets)
    assert result[(2, 0)]["status"] == "redundant"


def test_classify_inus_no_sets():
    heads = {(0, 0)}
    result = classify_inus(heads, [])
    assert result[(0, 0)]["status"] == "undetermined"


def test_classify_inus_single_set_single_member():
    heads = {(0, 0), (1, 0)}
    minimal_sets = [
        {(0, 0), (1, 0)},
    ]
    result = classify_inus(heads, minimal_sets)
    assert result[(0, 0)]["status"] == "non-redundant_necessary"
    assert result[(1, 0)]["status"] == "non-redundant_necessary"


def test_classify_inus_redundancy_index():
    heads = {(0, 0), (1, 0)}
    minimal_sets = [
        {(0, 0)},
        {(1, 0)},
        {(0, 0)},
    ]
    result = classify_inus(heads, minimal_sets)
    assert result[(0, 0)]["redundancy_index"] == pytest.approx(2.0 / 3.0)
    assert result[(1, 0)]["redundancy_index"] == pytest.approx(1.0 / 3.0)


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


def test_find_minimal_sufficient_sets_runs(gpt2_model):
    from mechanistic_validity.metrics.common import (
        calibrate_mean_z,
        generate_prompts,
        get_circuit_heads,
        get_token_ids,
    )
    circuit_heads = get_circuit_heads(TASK)
    prompts = generate_prompts(TASK, gpt2_model.tokenizer, 3)
    correct_ids, incorrect_ids = get_token_ids(prompts, gpt2_model.tokenizer)
    mean_z = calibrate_mean_z(gpt2_model, prompts, n_calibration=3)
    result = find_minimal_sufficient_sets(
        gpt2_model, prompts, correct_ids, incorrect_ids,
        circuit_heads, mean_z, threshold=0.3, max_set_size=3,
    )
    assert isinstance(result, list)
    for s in result:
        assert isinstance(s, set)
