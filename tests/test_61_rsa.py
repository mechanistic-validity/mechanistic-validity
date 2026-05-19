import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from mechanistic_validity.instruments.common import EvalResult, load_model

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "mechanistic_validity"
    / "instruments"
    / "representational"
    / "rsa"
    / "61_rsa.py"
)
_spec = importlib.util.spec_from_file_location("rsa_61", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["rsa_61"] = _mod
_spec.loader.exec_module(_mod)

cosine_rdm = _mod.cosine_rdm
build_target_rdm = _mod.build_target_rdm
rdm_to_upper_tri = _mod.rdm_to_upper_tri

TASK = "ioi"


def test_cosine_rdm_identical_vectors():
    X = np.tile(np.array([1.0, 0.0, 0.0]), (5, 1))
    rdm = cosine_rdm(X)
    assert rdm.shape == (5, 5)
    assert rdm.diagonal() == pytest.approx(np.zeros(5), abs=1e-6)
    assert rdm[0, 1] == pytest.approx(0.0, abs=1e-6)


def test_cosine_rdm_orthogonal_vectors():
    X = np.eye(4)
    rdm = cosine_rdm(X)
    for i in range(4):
        for j in range(4):
            if i == j:
                assert rdm[i, j] == pytest.approx(0.0, abs=1e-6)
            else:
                assert rdm[i, j] == pytest.approx(1.0, abs=1e-6)


def test_cosine_rdm_symmetric():
    X = np.random.randn(10, 5)
    rdm = cosine_rdm(X)
    assert rdm == pytest.approx(rdm.T, abs=1e-10)


def test_rdm_to_upper_tri_size():
    n = 5
    rdm = np.random.randn(n, n)
    vec = rdm_to_upper_tri(rdm)
    assert len(vec) == n * (n - 1) // 2


def test_rdm_to_upper_tri_values():
    rdm = np.array([[0.0, 0.3, 0.7],
                     [0.3, 0.0, 0.5],
                     [0.7, 0.5, 0.0]])
    vec = rdm_to_upper_tri(rdm)
    assert list(vec) == pytest.approx([0.3, 0.7, 0.5])


def test_build_target_rdm_same_target():
    class FakePrompt:
        def __init__(self, target):
            self.target_correct = target

    prompts = [FakePrompt("A"), FakePrompt("A"), FakePrompt("B")]
    rdm = build_target_rdm(prompts, TASK)
    assert rdm[0, 1] == pytest.approx(0.0)
    assert rdm[0, 2] == pytest.approx(1.0)
    assert rdm[1, 2] == pytest.approx(1.0)


def test_build_target_rdm_diagonal_zero():
    class FakePrompt:
        def __init__(self, target):
            self.target_correct = target

    prompts = [FakePrompt("X"), FakePrompt("Y"), FakePrompt("Z")]
    rdm = build_target_rdm(prompts, TASK)
    assert rdm.diagonal() == pytest.approx(np.zeros(3))
