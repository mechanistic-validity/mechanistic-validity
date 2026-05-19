import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from mechanistic_validity.metrics.common import EvalResult, load_model

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechanistic_validity" / "calibrations" / "measurement_invariance" / "13_measurement_invariance.py"
)
_spec = importlib.util.spec_from_file_location("_mi_13", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)

split_by_template = _mod.split_by_template
split_by_metadata = _mod.split_by_metadata
run_measurement_invariance = _mod.run_measurement_invariance

TASK = "ioi"


class _FakePrompt:
    def __init__(self, text, metadata=None):
        self.text = text
        self.metadata = metadata or {}
        self.target_correct = " the"
        self.target_incorrect = " a"


def test_split_by_template_too_few_prompts():
    prompts = [_FakePrompt("short")] * 5
    result = split_by_template(prompts, [1] * 5, [2] * 5, n_groups=3)
    assert result is None


def test_split_by_template_groups_by_length():
    short = [_FakePrompt("a b c") for _ in range(5)]
    medium = [_FakePrompt("a b c d e f g h") for _ in range(5)]
    long = [_FakePrompt("a b c d e f g h i j k l m n o") for _ in range(5)]
    prompts = short + medium + long
    correct = list(range(15))
    incorrect = list(range(15))
    groups = split_by_template(prompts, correct, incorrect, n_groups=3)
    assert groups is not None
    assert len(groups) == 3
    assert set(groups.keys()) == {"short", "medium", "long"}
    for g in groups.values():
        assert len(g["prompts"]) >= 3


def test_split_by_metadata_groups_by_template_key():
    prompts = [
        _FakePrompt("text1", {"template": "A"}),
        _FakePrompt("text2", {"template": "A"}),
        _FakePrompt("text3", {"template": "A"}),
        _FakePrompt("text4", {"template": "B"}),
        _FakePrompt("text5", {"template": "B"}),
        _FakePrompt("text6", {"template": "B"}),
    ]
    correct = list(range(6))
    incorrect = list(range(6))
    groups = split_by_metadata(prompts, correct, incorrect)
    assert groups is not None
    assert set(groups.keys()) == {"A", "B"}


def test_split_by_metadata_needs_two_groups():
    prompts = [_FakePrompt("text", {"template": "A"}) for _ in range(5)]
    correct = list(range(5))
    incorrect = list(range(5))
    groups = split_by_metadata(prompts, correct, incorrect)
    assert groups is None


def test_split_by_metadata_filters_small_groups():
    prompts = [
        _FakePrompt("t1", {"template": "A"}),
        _FakePrompt("t2", {"template": "A"}),
        _FakePrompt("t3", {"template": "A"}),
        _FakePrompt("t4", {"template": "B"}),
        _FakePrompt("t5", {"template": "C"}),
    ]
    correct = list(range(5))
    incorrect = list(range(5))
    groups = split_by_metadata(prompts, correct, incorrect)
    assert groups is None


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


@pytest.fixture(scope="module")
def circuit_results(gpt2_model):
    return run_measurement_invariance(gpt2_model, [TASK], n_prompts=15)


def test_run_measurement_invariance_returns_results(circuit_results):
    assert len(circuit_results) > 0
    assert all(isinstance(r, EvalResult) for r in circuit_results)


def test_run_measurement_invariance_metric_id(circuit_results):
    for r in circuit_results:
        assert r.metric_id == "C13.measurement_invariance"


def test_run_measurement_invariance_metadata(circuit_results):
    for r in circuit_results:
        assert r.metadata["task"] == TASK
        assert "group_scores" in r.metadata
        assert "invariance_verdict" in r.metadata
        assert r.metadata["n_groups"] >= 2
        assert r.metadata["invariance_verdict"] in {
            "invariant", "moderate", "template_sensitive"
        }


def test_run_measurement_invariance_value_is_nonnegative(circuit_results):
    for r in circuit_results:
        assert r.value >= 0.0
