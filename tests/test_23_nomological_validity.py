import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from mechanistic_validity.instruments.common import EvalResult

_NV_PATH = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "mechanistic_validity"
    / "instruments"
    / "measurement"
    / "convergent_validity"
    / "23_nomological_validity.py"
)
_spec = importlib.util.spec_from_file_location("_nv23", _NV_PATH)
_nv_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _nv_mod
_spec.loader.exec_module(_nv_mod)

_layer_density_correlation = _nv_mod._layer_density_correlation
_role_depth_correlation = _nv_mod._role_depth_correlation
run_nomological_validity = _nv_mod.run_nomological_validity
BAND_ORDINALS = _nv_mod.BAND_ORDINALS
N_LAYERS = _nv_mod.N_LAYERS

TASK = "ioi"


def test_layer_density_correlation_concentrated_early():
    heads = {(0, 0), (0, 1), (1, 0), (1, 1)}
    r, p = _layer_density_correlation(heads)
    assert r < 0


def test_layer_density_correlation_concentrated_late():
    heads = {(10, 0), (10, 1), (11, 0), (11, 1)}
    r, p = _layer_density_correlation(heads)
    assert r > 0


def test_layer_density_correlation_uniform():
    heads = {(i, 0) for i in range(N_LAYERS)}
    r, p = _layer_density_correlation(heads)
    assert np.isnan(r) or abs(r) < 0.01


def test_layer_density_correlation_empty():
    heads = set()
    r, p = _layer_density_correlation(heads)
    layer_counts = np.zeros(N_LAYERS)
    expected_r, expected_p = stats.spearmanr(np.arange(N_LAYERS), layer_counts)
    assert r == pytest.approx(float(expected_r), nan_ok=True)


def test_role_depth_correlation_positive():
    circuit = {
        "roles": {
            "early_role": [(0, 0), (1, 0)],
            "late_role": [(10, 0), (11, 0)],
        },
        "bands": {
            "early": (None, ["early_role"]),
            "late": (None, ["late_role"]),
        },
    }
    r, p, role_map = _role_depth_correlation(circuit)
    assert r > 0
    assert "L0H0" in role_map
    assert role_map["L0H0"]["ordinal"] == BAND_ORDINALS["early"]
    assert "L10H0" in role_map
    assert role_map["L10H0"]["ordinal"] == BAND_ORDINALS["late"]


def test_role_depth_correlation_all_same_band():
    circuit = {
        "roles": {
            "role_a": [(0, 0), (5, 0), (11, 0)],
        },
        "bands": {
            "mid": (None, ["role_a"]),
        },
    }
    r, p, role_map = _role_depth_correlation(circuit)
    assert np.isnan(r) or r == pytest.approx(0.0, abs=1e-10)


def test_role_depth_correlation_too_few_heads():
    circuit = {
        "roles": {
            "role_a": [(0, 0), (1, 0)],
        },
        "bands": {
            "early": (None, ["role_a"]),
        },
    }
    r, p, role_map = _role_depth_correlation(circuit)
    assert r == pytest.approx(0.0)
    assert p == pytest.approx(1.0)


def test_band_ordinals_expected_values():
    assert BAND_ORDINALS["early"] == 1
    assert BAND_ORDINALS["mid"] == 2
    assert BAND_ORDINALS["late"] == 4


def test_run_nomological_validity_real_task():
    results = run_nomological_validity([TASK])
    assert len(results) >= 1
    density_results = [r for r in results if r.metric_id == "C23.layer_density_correlation"]
    assert len(density_results) == 1
    dr = density_results[0]
    assert isinstance(dr, EvalResult)
    assert dr.n_samples == N_LAYERS
    assert dr.metadata["task"] == TASK
    assert "spearman_r" in dr.metadata
    assert "p_value" in dr.metadata
    assert "layer_counts" in dr.metadata
    assert dr.metadata["n_circuit_heads"] > 0


def test_run_nomological_validity_includes_role_depth():
    results = run_nomological_validity([TASK])
    role_results = [r for r in results if r.metric_id == "C23.role_depth_correlation"]
    assert len(role_results) == 1
    rr = role_results[0]
    assert "n_roles" in rr.metadata
    assert "n_bands" in rr.metadata
    assert "head_role_map" in rr.metadata


def test_run_nomological_validity_unknown_task():
    results = run_nomological_validity(["nonexistent_xyz"])
    assert results == []


def test_run_nomological_validity_multiple_tasks():
    results = run_nomological_validity(["ioi", "sva"])
    ioi_density = [r for r in results if r.metadata["task"] == "ioi" and r.metric_id == "C23.layer_density_correlation"]
    sva_density = [r for r in results if r.metadata["task"] == "sva" and r.metric_id == "C23.layer_density_correlation"]
    assert len(ioi_density) == 1
    assert len(sva_density) == 1
