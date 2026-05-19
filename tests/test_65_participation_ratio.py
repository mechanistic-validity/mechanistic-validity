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
    / "participation_ratio"
    / "65_participation_ratio.py"
)
_spec = importlib.util.spec_from_file_location("pr_65", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["pr_65"] = _mod
_spec.loader.exec_module(_mod)

compute_participation_ratio = _mod.compute_participation_ratio

TASK = "ioi"


def test_participation_ratio_single_dominant():
    eigenvalues = np.array([1.0, 0.0, 0.0, 0.0])
    assert compute_participation_ratio(eigenvalues) == pytest.approx(1.0)


def test_participation_ratio_uniform():
    n = 8
    eigenvalues = np.ones(n)
    assert compute_participation_ratio(eigenvalues) == pytest.approx(float(n))


def test_participation_ratio_intermediate():
    eigenvalues = np.array([9.0, 1.0])
    expected = (9 + 1) ** 2 / (81 + 1)
    assert compute_participation_ratio(eigenvalues) == pytest.approx(expected)


def test_participation_ratio_zero_input():
    eigenvalues = np.zeros(5)
    assert compute_participation_ratio(eigenvalues) == pytest.approx(1.0)


def test_participation_ratio_negative_clamped():
    eigenvalues = np.array([5.0, -1.0, -2.0, 3.0])
    result = compute_participation_ratio(eigenvalues)
    clamped = np.maximum(eigenvalues, 0.0)
    expected = clamped.sum() ** 2 / (clamped ** 2).sum()
    assert result == pytest.approx(expected)


def test_participation_ratio_bounded():
    for _ in range(50):
        n = 10
        eigenvalues = np.abs(np.random.randn(n)) + 0.01
        pr = compute_participation_ratio(eigenvalues)
        assert 1.0 <= pr <= float(n) + 1e-6
