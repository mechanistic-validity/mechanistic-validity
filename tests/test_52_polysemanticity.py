import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from mechanistic_validity.instruments.common import EvalResult, load_model, get_circuit_heads

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechanistic_validity" / "instruments"
    / "structural" / "polysemanticity" / "52_polysemanticity.py"
)
_spec = importlib.util.spec_from_file_location("_poly_52", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)

effective_rank = _mod.effective_rank
participation_ratio = _mod.participation_ratio
compute_polysemanticity = _mod.compute_polysemanticity

TASK = "ioi"


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


@pytest.fixture(scope="module")
def poly_metrics(gpt2_model):
    return compute_polysemanticity(gpt2_model, cosine_threshold=0.1)


def test_effective_rank_uniform():
    sv = np.array([1.0, 1.0, 1.0, 1.0])
    er = effective_rank(sv)
    assert er == pytest.approx(4.0, abs=1e-4)


def test_effective_rank_single_dominant():
    sv = np.array([100.0, 0.0, 0.0])
    er = effective_rank(sv)
    assert er == pytest.approx(1.0, abs=1e-4)


def test_effective_rank_zero_input():
    sv = np.array([0.0, 0.0])
    er = effective_rank(sv)
    assert er == pytest.approx(0.0)


def test_participation_ratio_uniform():
    sv = np.array([1.0, 1.0, 1.0, 1.0])
    pr = participation_ratio(sv)
    assert pr == pytest.approx(4.0, abs=1e-4)


def test_participation_ratio_single():
    sv = np.array([5.0, 0.0, 0.0])
    pr = participation_ratio(sv)
    assert pr == pytest.approx(1.0, abs=1e-4)


def test_participation_ratio_zero_input():
    sv = np.array([0.0, 0.0])
    pr = participation_ratio(sv)
    assert pr == pytest.approx(0.0)


def test_participation_ratio_bounded():
    n = 10
    sv = np.random.rand(n) + 0.1
    pr = participation_ratio(sv)
    assert 1.0 <= pr <= n


def test_compute_polysemanticity_covers_all_heads(gpt2_model, poly_metrics):
    n_layers = gpt2_model.cfg.n_layers
    n_heads = gpt2_model.cfg.n_heads
    assert len(poly_metrics) == n_layers * n_heads
    for (L, H), data in poly_metrics.items():
        assert "eff_rank" in data
        assert "participation" in data
        assert "fan_out" in data


def test_compute_polysemanticity_eff_rank_positive(poly_metrics):
    for data in poly_metrics.values():
        assert data["eff_rank"] > 0


def test_compute_polysemanticity_fan_out_nonnegative(poly_metrics):
    for data in poly_metrics.values():
        assert data["fan_out"] >= 0
        assert isinstance(data["fan_out"], int)


def test_compute_polysemanticity_participation_positive(poly_metrics):
    for data in poly_metrics.values():
        assert data["participation"] > 0
