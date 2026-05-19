import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from mechanistic_validity.metrics.common import EvalResult, load_model, get_circuit_heads

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechanistic_validity" / "metrics"
    / "structural" / "norm_trajectory" / "51_norm_trajectory.py"
)
_spec = importlib.util.spec_from_file_location("_norm_traj_51", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)

compute_ov_norms = _mod.compute_ov_norms
linear_slope = _mod.linear_slope

TASK = "ioi"


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


def test_linear_slope_constant_is_zero():
    values = np.array([5.0, 5.0, 5.0, 5.0])
    assert linear_slope(values) == pytest.approx(0.0)


def test_linear_slope_increasing():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    slope = linear_slope(values)
    assert slope == pytest.approx(1.0)


def test_linear_slope_decreasing():
    values = np.array([10.0, 8.0, 6.0, 4.0, 2.0])
    slope = linear_slope(values)
    assert slope == pytest.approx(-2.0)


def test_linear_slope_single_element():
    values = np.array([42.0])
    assert linear_slope(values) == pytest.approx(0.0)


def test_linear_slope_empty():
    values = np.array([])
    assert linear_slope(values) == pytest.approx(0.0)


def test_compute_ov_norms_shape(gpt2_model):
    norms = compute_ov_norms(gpt2_model)
    assert norms.shape == (gpt2_model.cfg.n_layers, gpt2_model.cfg.n_heads)


def test_compute_ov_norms_positive(gpt2_model):
    norms = compute_ov_norms(gpt2_model)
    assert (norms > 0).all()


def test_compute_ov_norms_reasonable_magnitude(gpt2_model):
    norms = compute_ov_norms(gpt2_model)
    assert norms.max() < 1000.0
    assert norms.min() > 0.0
