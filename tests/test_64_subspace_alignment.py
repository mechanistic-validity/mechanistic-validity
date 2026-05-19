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
    / "subspace_alignment"
    / "64_subspace_alignment.py"
)
_spec = importlib.util.spec_from_file_location("subspace_align_64", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["subspace_align_64"] = _mod
_spec.loader.exec_module(_mod)

principal_angles = _mod.principal_angles
grassmann_distance = _mod.grassmann_distance
subspace_alignment_score = _mod.subspace_alignment_score
get_head_subspace = _mod.get_head_subspace

TASK = "ioi"


def test_principal_angles_identical_subspaces():
    U = np.eye(5, 3)
    cos_angles = principal_angles(U, U)
    assert cos_angles == pytest.approx(np.ones(3), abs=1e-6)


def test_principal_angles_orthogonal_subspaces():
    U = np.zeros((6, 3))
    U[:3, :] = np.eye(3)
    V = np.zeros((6, 3))
    V[3:, :] = np.eye(3)
    cos_angles = principal_angles(U, V)
    assert cos_angles == pytest.approx(np.zeros(3), abs=1e-6)


def test_grassmann_distance_identical():
    U = np.eye(5, 3)
    assert grassmann_distance(U, U) == pytest.approx(0.0, abs=1e-6)


def test_grassmann_distance_orthogonal():
    U = np.zeros((6, 3))
    U[:3, :] = np.eye(3)
    V = np.zeros((6, 3))
    V[3:, :] = np.eye(3)
    dist = grassmann_distance(U, V)
    expected = np.sqrt(3 * (np.pi / 2) ** 2)
    assert dist == pytest.approx(expected, abs=1e-6)


def test_grassmann_distance_non_negative():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((10, 3))
    B = rng.standard_normal((10, 3))
    U, _, _ = np.linalg.svd(A, full_matrices=False)
    V, _, _ = np.linalg.svd(B, full_matrices=False)
    assert grassmann_distance(U[:, :3], V[:, :3]) >= 0.0


def test_subspace_alignment_score_identical():
    U = np.eye(5, 3)
    assert subspace_alignment_score(U, U) == pytest.approx(1.0, abs=1e-6)


def test_subspace_alignment_score_orthogonal():
    U = np.zeros((6, 3))
    U[:3, :] = np.eye(3)
    V = np.zeros((6, 3))
    V[3:, :] = np.eye(3)
    assert subspace_alignment_score(U, V) == pytest.approx(0.0, abs=1e-6)


def test_subspace_alignment_score_range():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((10, 3))
    B = rng.standard_normal((10, 3))
    U, _, _ = np.linalg.svd(A, full_matrices=False)
    V, _, _ = np.linalg.svd(B, full_matrices=False)
    score = subspace_alignment_score(U[:, :3], V[:, :3])
    assert 0.0 <= score <= 1.0


def test_get_head_subspace_shape():
    rng = np.random.default_rng(0)
    head_outputs = rng.standard_normal((20, 64))
    subspace = get_head_subspace(head_outputs, k=5)
    assert subspace.shape == (64, 5)


def test_get_head_subspace_orthonormal():
    rng = np.random.default_rng(0)
    head_outputs = rng.standard_normal((20, 64))
    subspace = get_head_subspace(head_outputs, k=5)
    gram = subspace.T @ subspace
    assert gram == pytest.approx(np.eye(5), abs=1e-6)


def test_get_head_subspace_k_clamped():
    rng = np.random.default_rng(0)
    head_outputs = rng.standard_normal((3, 10))
    subspace = get_head_subspace(head_outputs, k=10)
    assert subspace.shape[1] <= 3
