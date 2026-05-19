import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from mechval.metrics.common import EvalResult, load_model

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "mechval"
    / "metrics"
    / "representational"
    / "intrinsic_dimension"
    / "62_intrinsic_dimension.py"
)
_spec = importlib.util.spec_from_file_location("intrinsic_dim_62", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["intrinsic_dim_62"] = _mod
_spec.loader.exec_module(_mod)

two_nn_intrinsic_dimension = _mod.two_nn_intrinsic_dimension

TASK = "ioi"


def test_two_nn_too_few_samples():
    X = np.random.randn(2, 10)
    assert two_nn_intrinsic_dimension(X) == pytest.approx(0.0)


def test_two_nn_1d_data():
    t = np.linspace(0, 10, 100)
    X = np.column_stack([t, np.zeros(100)])
    result = two_nn_intrinsic_dimension(X)
    assert 0.5 < result < 2.5


def test_two_nn_2d_data():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 2))
    X_embedded = np.zeros((200, 10))
    X_embedded[:, :2] = X
    result = two_nn_intrinsic_dimension(X_embedded)
    assert 1.0 < result < 4.0


def test_two_nn_positive_result():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((50, 5))
    result = two_nn_intrinsic_dimension(X)
    assert result > 0.0


def test_two_nn_duplicate_points():
    X = np.ones((10, 5))
    result = two_nn_intrinsic_dimension(X)
    assert result == pytest.approx(0.0)
