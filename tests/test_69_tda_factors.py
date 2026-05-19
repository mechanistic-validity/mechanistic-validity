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
    / "tda_factors"
    / "69_tda_factors.py"
)
_spec = importlib.util.spec_from_file_location("tda_69", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["tda_69"] = _mod
_spec.loader.exec_module(_mod)

effective_rank_90 = _mod.effective_rank_90

TASK = "ioi"


def test_effective_rank_single_value():
    sv = np.array([10.0, 0.0, 0.0])
    assert effective_rank_90(sv) == 1


def test_effective_rank_uniform():
    sv = np.ones(10)
    rank = effective_rank_90(sv)
    assert rank == 9


def test_effective_rank_empty():
    sv = np.array([])
    assert effective_rank_90(sv) == 0


def test_effective_rank_bounded_by_length():
    sv = np.ones(5) * 0.01
    rank = effective_rank_90(sv)
    assert rank <= 5


def test_effective_rank_monotonic_with_spread():
    sv_concentrated = np.array([10.0, 0.1, 0.01, 0.001])
    sv_spread = np.array([3.0, 2.5, 2.0, 1.5])
    rank_conc = effective_rank_90(sv_concentrated)
    rank_spread = effective_rank_90(sv_spread)
    assert rank_conc <= rank_spread


def test_effective_rank_zero_values():
    sv = np.zeros(5)
    rank = effective_rank_90(sv)
    assert rank == 5


def test_effective_rank_decreasing_spectrum():
    sv = np.array([10.0, 5.0, 2.0, 1.0, 0.5])
    rank = effective_rank_90(sv)
    assert 1 <= rank <= 5
    variance = sv ** 2
    cum = np.cumsum(variance) / variance.sum()
    expected = int(np.searchsorted(cum, 0.9)) + 1
    assert rank == expected
