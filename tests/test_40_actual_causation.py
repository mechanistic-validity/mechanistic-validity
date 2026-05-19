import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechanistic_validity" / "instruments"
    / "causal" / "actual_cause" / "40_actual_causation.py"
)
_spec = importlib.util.spec_from_file_location("actual_causation_40", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["actual_causation_40"] = _mod
_spec.loader.exec_module(_mod)

compute_effect_in_context = _mod.compute_effect_in_context
find_witness = _mod.find_witness
compute_standard_necessity = _mod.compute_standard_necessity

from mechanistic_validity.instruments.common import (
    EvalResult,
    calibrate_mean_z,
    generate_prompts,
    get_circuit_heads,
    get_token_ids,
    load_model,
)

TASK = "ioi"


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


@pytest.fixture(scope="module")
def task_data(gpt2_model):
    circuit_heads = get_circuit_heads(TASK)
    prompts = generate_prompts(TASK, gpt2_model.tokenizer, 3)
    correct_ids, incorrect_ids = get_token_ids(prompts, gpt2_model.tokenizer)
    mean_z = calibrate_mean_z(gpt2_model, prompts, n_calibration=3)
    return circuit_heads, prompts, correct_ids, incorrect_ids, mean_z


def test_compute_effect_in_context_returns_two_floats(gpt2_model, task_data):
    circuit_heads, prompts, correct_ids, incorrect_ids, mean_z = task_data
    head = sorted(circuit_heads)[0]
    tokens = gpt2_model.to_tokens(prompts[0].text)
    ld_with, ld_without = compute_effect_in_context(
        gpt2_model, tokens, correct_ids[0], incorrect_ids[0],
        head, set(), mean_z,
    )
    assert isinstance(ld_with, float)
    assert isinstance(ld_without, float)


def test_compute_effect_in_context_with_ablated_context(gpt2_model, task_data):
    circuit_heads, prompts, correct_ids, incorrect_ids, mean_z = task_data
    heads_list = sorted(circuit_heads)
    if len(heads_list) < 2:
        pytest.skip("Need at least 2 circuit heads")
    target = heads_list[0]
    context = {heads_list[1]}
    tokens = gpt2_model.to_tokens(prompts[0].text)
    ld_with, ld_without = compute_effect_in_context(
        gpt2_model, tokens, correct_ids[0], incorrect_ids[0],
        target, context, mean_z,
    )
    assert np.isfinite(ld_with)
    assert np.isfinite(ld_without)


def test_compute_standard_necessity_returns_float(gpt2_model, task_data):
    circuit_heads, prompts, correct_ids, incorrect_ids, mean_z = task_data
    head = sorted(circuit_heads)[0]
    necessity = compute_standard_necessity(
        gpt2_model, prompts, correct_ids, incorrect_ids, head, mean_z,
    )
    assert isinstance(necessity, float)
    assert necessity >= 0.0


def test_find_witness_returns_dict_or_none(gpt2_model, task_data):
    circuit_heads, prompts, correct_ids, incorrect_ids, mean_z = task_data
    head = sorted(circuit_heads)[0]
    witness = find_witness(
        gpt2_model, prompts, correct_ids, incorrect_ids,
        head, circuit_heads, mean_z,
        max_witness_size=1, effect_threshold=0.1,
    )
    assert witness is None or isinstance(witness, dict)
    if witness is not None:
        assert "witness" in witness
        assert "witness_size" in witness
        assert "normalized_effect" in witness
        assert "raw_effect" in witness
