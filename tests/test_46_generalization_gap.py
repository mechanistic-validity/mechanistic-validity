import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechanistic_validity" / "metrics"
    / "behavioral" / "generalization_gap" / "46_generalization_gap.py"
)
_spec = importlib.util.spec_from_file_location("generalization_gap_46", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["generalization_gap_46"] = _mod
_spec.loader.exec_module(_mod)

perturb_prompts = _mod.perturb_prompts
run_generalization_gap = _mod.run_generalization_gap
PADDING_PREFIXES = _mod.PADDING_PREFIXES
PADDING_SUFFIXES = _mod.PADDING_SUFFIXES

from mechanistic_validity.metrics.common import EvalResult, load_model

TASK = "ioi"


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


class _FakePrompt:
    def __init__(self, text):
        self.text = text
        self.target_correct = " Mary"
        self.target_incorrect = " John"
        self.metadata = {}


def test_perturb_prompts_same_length():
    prompts = [_FakePrompt("Hello world"), _FakePrompt("Foo bar")]
    perturbed = perturb_prompts(prompts, seed=42)
    assert len(perturbed) == len(prompts)


def test_perturb_prompts_changes_text():
    prompts = [_FakePrompt("The cat sat on the mat")]
    perturbed = perturb_prompts(prompts, seed=42)
    assert perturbed[0].text != prompts[0].text


def test_perturb_prompts_preserves_targets():
    prompts = [_FakePrompt("Some text here")]
    perturbed = perturb_prompts(prompts, seed=42)
    assert perturbed[0].target_correct == " Mary"
    assert perturbed[0].target_incorrect == " John"


def test_perturb_prompts_adds_prefix_or_suffix():
    prompts = [_FakePrompt("Base text") for _ in range(20)]
    perturbed = perturb_prompts(prompts, seed=42)
    has_prefix = any(
        any(p.text.startswith(prefix) for prefix in PADDING_PREFIXES)
        for p in perturbed
    )
    has_suffix = any(
        any(p.text.endswith(suffix.strip()) for suffix in PADDING_SUFFIXES)
        for p in perturbed
    )
    assert has_prefix or has_suffix


def test_perturb_prompts_deterministic():
    prompts = [_FakePrompt("Test"), _FakePrompt("Another")]
    p1 = perturb_prompts(prompts, seed=99)
    p2 = perturb_prompts(prompts, seed=99)
    assert [p.text for p in p1] == [p.text for p in p2]


def test_perturb_prompts_different_seeds_differ():
    prompts = [_FakePrompt("Test") for _ in range(10)]
    p1 = perturb_prompts(prompts, seed=1)
    p2 = perturb_prompts(prompts, seed=2)
    texts1 = [p.text for p in p1]
    texts2 = [p.text for p in p2]
    assert texts1 != texts2


def test_run_generalization_gap_returns_results(gpt2_model):
    results = run_generalization_gap(gpt2_model, [TASK], n_prompts=3)
    assert len(results) >= 1
    r = results[0]
    assert isinstance(r, EvalResult)
    assert r.metric_id == "D07.generalization_gap"
    assert r.n_samples >= 1


def test_run_generalization_gap_metadata_keys(gpt2_model):
    results = run_generalization_gap(gpt2_model, [TASK], n_prompts=3)
    r = results[0]
    expected_keys = {
        "task", "faithfulness_in_dist", "faithfulness_ood",
        "absolute_gap", "relative_gap", "n_circuit_heads",
        "n_prompts_id", "n_prompts_ood", "perturbation_type",
    }
    assert expected_keys.issubset(r.metadata.keys())


def test_run_generalization_gap_value_is_gap(gpt2_model):
    results = run_generalization_gap(gpt2_model, [TASK], n_prompts=3)
    r = results[0]
    expected_gap = r.metadata["faithfulness_in_dist"] - r.metadata["faithfulness_ood"]
    assert r.value == pytest.approx(expected_gap)


def test_run_generalization_gap_unknown_task_returns_empty(gpt2_model):
    results = run_generalization_gap(gpt2_model, ["nonexistent_task_xyz"], n_prompts=3)
    assert results == []
