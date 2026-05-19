import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechanistic_validity" / "instruments"
    / "causal" / "rubin_cate" / "06_cate.py"
)
_spec = importlib.util.spec_from_file_location("cate_06", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["cate_06"] = _mod
_spec.loader.exec_module(_mod)

run_cate = _mod.run_cate
extract_context_features = _mod.extract_context_features
compute_per_prompt_effects = _mod.compute_per_prompt_effects

from mechanistic_validity.instruments.common import EvalResult, load_model

TASK = "ioi"


class _FakePrompt:
    def __init__(self, text):
        self.text = text


def test_extract_context_features_basic():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    prompt = _FakePrompt("The cat that sat on the mat")
    feats = extract_context_features(prompt, tok, clean_ld=2.5)
    assert "n_tokens" in feats
    assert "has_relative_clause" in feats
    assert feats["clean_logit_diff"] == pytest.approx(2.5)
    assert feats["has_relative_clause"] == 1  # contains "that"


def test_extract_context_features_no_clause():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    prompt = _FakePrompt("Hello world")
    feats = extract_context_features(prompt, tok)
    assert feats["has_relative_clause"] == 0
    assert feats["has_comma"] == 0


def test_extract_context_features_comma():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    prompt = _FakePrompt("Hello, world")
    feats = extract_context_features(prompt, tok)
    assert feats["has_comma"] == 1


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


@pytest.fixture(scope="module")
def circuit_results(gpt2_model):
    return run_cate(gpt2_model, tasks=[TASK], n_prompts=5)


def test_run_returns_eval_results(circuit_results):
    assert len(circuit_results) == 1
    assert isinstance(circuit_results[0], EvalResult)


def test_metric_id(circuit_results):
    assert circuit_results[0].metric_id == "C6.cate"


def test_value_is_max_heterogeneity(circuit_results):
    meta = circuit_results[0].metadata
    if meta["subgroup_effects"]:
        max_d = max(abs(v["cohens_d"]) for v in meta["subgroup_effects"].values())
        assert circuit_results[0].value == pytest.approx(max_d)
    else:
        assert circuit_results[0].value == pytest.approx(0.0)


def test_metadata_keys(circuit_results):
    meta = circuit_results[0].metadata
    expected = {"task", "ate", "ate_std", "subgroup_effects",
                "feature_names", "n_circuit_heads"}
    assert set(meta.keys()) == expected


def test_ate_is_finite(circuit_results):
    assert np.isfinite(circuit_results[0].metadata["ate"])


def test_n_samples_positive(circuit_results):
    assert circuit_results[0].n_samples > 0


def test_feature_names_non_empty(circuit_results):
    assert len(circuit_results[0].metadata["feature_names"]) > 0
