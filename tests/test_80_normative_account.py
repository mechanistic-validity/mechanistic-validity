import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechanistic_validity" / "instruments"
    / "behavioral" / "generalization_gap" / "80_normative_account.py"
)
_spec = importlib.util.spec_from_file_location("normative_account_80", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["normative_account_80"] = _mod
_spec.loader.exec_module(_mod)

count_feature = _mod.count_feature
has_feature = _mod.has_feature
compute_circuit_activation_magnitude = _mod.compute_circuit_activation_magnitude
run_normative_account = _mod.run_normative_account
LINGUISTIC_FEATURES = _mod.LINGUISTIC_FEATURES
DIVERSE_PROMPTS = _mod.DIVERSE_PROMPTS

from mechanistic_validity.instruments.common import EvalResult, load_model

TASK = "ioi"


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


def test_count_feature_epistemic_verbs():
    text = "I think and believe that this is true"
    count = count_feature(text, "epistemic_verbs")
    assert count == 2


def test_count_feature_no_match():
    text = "The cat sat on the mat"
    count = count_feature(text, "epistemic_verbs")
    assert count == 0


def test_count_feature_modal_verbs():
    text = "She could and should do it"
    count = count_feature(text, "modal_verbs")
    assert count == 2


def test_has_feature_true():
    assert has_feature("I think it is good", "epistemic_verbs")


def test_has_feature_false():
    assert not has_feature("The dog ran quickly", "epistemic_verbs")


def test_has_feature_all_categories():
    for feature_name in LINGUISTIC_FEATURES:
        words = list(LINGUISTIC_FEATURES[feature_name])
        text = f"Some text with {words[0]} in it"
        assert has_feature(text, feature_name)


def test_diverse_prompts_not_empty():
    assert len(DIVERSE_PROMPTS) > 20


def test_compute_circuit_activation_magnitude_returns_float(gpt2_model):
    tokens = gpt2_model.to_tokens("The cat sat on the mat")
    circuit_heads = {(0, 0), (1, 1)}
    mag = compute_circuit_activation_magnitude(gpt2_model, tokens, circuit_heads)
    assert isinstance(mag, float)
    assert mag >= 0.0


def test_compute_circuit_activation_magnitude_more_heads_larger(gpt2_model):
    tokens = gpt2_model.to_tokens("Hello world, how are you today")
    small_circuit = {(0, 0)}
    large_circuit = {(0, 0), (1, 1), (2, 2), (3, 3)}
    mag_small = compute_circuit_activation_magnitude(gpt2_model, tokens, small_circuit)
    mag_large = compute_circuit_activation_magnitude(gpt2_model, tokens, large_circuit)
    assert mag_large >= mag_small


def test_run_normative_account_returns_results(gpt2_model):
    results = run_normative_account(gpt2_model, [TASK], n_prompts=10)
    assert len(results) >= 1
    r = results[0]
    assert isinstance(r, EvalResult)
    assert r.metric_id == "CM1.normative_account"
    assert r.n_samples >= 1


def test_run_normative_account_metadata_keys(gpt2_model):
    results = run_normative_account(gpt2_model, [TASK], n_prompts=10)
    r = results[0]
    expected_keys = {
        "task", "activation_rate", "max_separation_ratio",
        "best_feature", "feature_analysis", "n_circuit_heads",
        "median_activation", "p75_activation", "passed",
    }
    assert expected_keys.issubset(r.metadata.keys())


def test_run_normative_account_separation_positive(gpt2_model):
    results = run_normative_account(gpt2_model, [TASK], n_prompts=10)
    r = results[0]
    assert r.value >= 0.0


def test_run_normative_account_activation_rate_bounded(gpt2_model):
    results = run_normative_account(gpt2_model, [TASK], n_prompts=10)
    r = results[0]
    assert 0.0 <= r.metadata["activation_rate"] <= 1.0


def test_run_normative_account_feature_analysis_has_all_features(gpt2_model):
    results = run_normative_account(gpt2_model, [TASK], n_prompts=10)
    r = results[0]
    fa = r.metadata["feature_analysis"]
    for feature_name in LINGUISTIC_FEATURES:
        assert feature_name in fa
        assert "high_rate" in fa[feature_name]
        assert "low_rate" in fa[feature_name]
        assert "separation_ratio" in fa[feature_name]


def test_run_normative_account_unknown_task_returns_empty(gpt2_model):
    results = run_normative_account(gpt2_model, ["nonexistent_task_xyz"], n_prompts=10)
    assert results == []
