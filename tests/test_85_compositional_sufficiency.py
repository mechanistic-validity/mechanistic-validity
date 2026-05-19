import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from mechanistic_validity.metrics.common import EvalResult, load_model

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechanistic_validity" / "metrics"
    / "structural" / "edge_analysis" / "85_compositional_sufficiency.py"
)
_spec = importlib.util.spec_from_file_location("_comp_suff_85", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)

get_band_heads = _mod.get_band_heads
run_compositional_sufficiency = _mod.run_compositional_sufficiency

TASK = "ioi"


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


@pytest.fixture(scope="module")
def circuit_results(gpt2_model):
    return run_compositional_sufficiency(gpt2_model, [TASK], n_prompts=3)


def test_get_band_heads_returns_set():
    from mechanistic_validity.metrics.common import get_circuit
    circuit = get_circuit(TASK)
    bands = circuit.get("bands", {})
    assert len(bands) >= 2
    for band_name in bands:
        heads = get_band_heads(circuit, band_name)
        assert isinstance(heads, set)
        for h in heads:
            assert isinstance(h, tuple)
            assert len(h) == 2


def test_get_band_heads_nonexistent_band_returns_empty():
    from mechanistic_validity.metrics.common import get_circuit
    circuit = get_circuit(TASK)
    heads = get_band_heads(circuit, "nonexistent_band_xyz")
    assert heads == set()


def test_run_compositional_sufficiency_returns_results(circuit_results):
    assert len(circuit_results) >= 1
    r = circuit_results[0]
    assert isinstance(r, EvalResult)
    assert r.metric_id == "G4.compositional_sufficiency"
    assert r.n_samples >= 1


def test_run_compositional_sufficiency_metadata_fields(circuit_results):
    r = circuit_results[0]
    meta = r.metadata
    assert meta["task"] == TASK
    assert "full_recovery" in meta
    assert isinstance(meta["full_recovery"], float)
    assert "per_band_recovery" in meta
    assert isinstance(meta["per_band_recovery"], dict)
    assert len(meta["per_band_recovery"]) >= 2
    assert "max_band_recovery" in meta
    assert "superadditivity" in meta
    assert r.value == pytest.approx(meta["superadditivity"])
    assert "n_circuit_heads" in meta
    assert "n_circuit_edges" in meta
    assert "n_bands" in meta
    assert meta["n_bands"] >= 2
    assert "passed" in meta
    assert isinstance(meta["passed"], bool)


def test_run_compositional_sufficiency_superadditivity_consistent(circuit_results):
    r = circuit_results[0]
    meta = r.metadata
    expected = meta["full_recovery"] - meta["max_band_recovery"]
    assert meta["superadditivity"] == pytest.approx(expected)
