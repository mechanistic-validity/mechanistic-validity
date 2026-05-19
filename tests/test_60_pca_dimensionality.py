import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from mechanistic_validity.metrics.common import EvalResult, load_model

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "mechanistic_validity"
    / "metrics"
    / "representational"
    / "pca_dimensionality"
    / "60_pca_dimensionality.py"
)
_spec = importlib.util.spec_from_file_location("pca_dim_60", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["pca_dim_60"] = _mod
_spec.loader.exec_module(_mod)

effective_dimensionality = _mod.effective_dimensionality
participation_ratio = _mod.participation_ratio
collect_residual_activations = _mod.collect_residual_activations

TASK = "ioi"


def test_effective_dimensionality_single_dominant():
    eigenvalues = np.array([100.0, 0.0, 0.0, 0.0])
    assert effective_dimensionality(eigenvalues) == 1


def test_effective_dimensionality_uniform():
    eigenvalues = np.ones(10)
    result = effective_dimensionality(eigenvalues, threshold=0.9)
    assert result == 9


def test_effective_dimensionality_threshold_respected():
    eigenvalues = np.array([0.5, 0.3, 0.15, 0.05])
    dim_50 = effective_dimensionality(eigenvalues, threshold=0.5)
    dim_90 = effective_dimensionality(eigenvalues, threshold=0.9)
    assert dim_50 <= dim_90


def test_effective_dimensionality_zero_input():
    eigenvalues = np.zeros(5)
    assert effective_dimensionality(eigenvalues) == 0


def test_participation_ratio_single_dominant():
    eigenvalues = np.array([1.0, 0.0, 0.0, 0.0])
    assert participation_ratio(eigenvalues) == pytest.approx(1.0)


def test_participation_ratio_uniform():
    n = 10
    eigenvalues = np.ones(n)
    assert participation_ratio(eigenvalues) == pytest.approx(float(n))


def test_participation_ratio_intermediate():
    eigenvalues = np.array([4.0, 1.0, 1.0, 1.0])
    expected = (4 + 1 + 1 + 1) ** 2 / (16 + 1 + 1 + 1)
    assert participation_ratio(eigenvalues) == pytest.approx(expected)


def test_participation_ratio_zero_input():
    eigenvalues = np.zeros(5)
    assert participation_ratio(eigenvalues) == pytest.approx(0.0)


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


def test_collect_residual_activations_shape(gpt2_model):
    from mechanistic_validity.metrics.common import generate_prompts

    prompts = generate_prompts(TASK, gpt2_model.tokenizer, 3)
    acts = collect_residual_activations(gpt2_model, prompts, layer=0)
    assert acts.shape == (len(prompts), gpt2_model.cfg.d_model)
