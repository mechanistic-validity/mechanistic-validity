import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechanistic_validity" / "metrics"
    / "causal" / "mdc_glennan" / "19_logic_gates.py"
)
_spec = importlib.util.spec_from_file_location("logic_gates_19", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["logic_gates_19"] = _mod
_spec.loader.exec_module(_mod)

classify_gate = _mod.classify_gate
completeness_denoising = _mod.completeness_denoising
run_logic_gates = _mod.run_logic_gates

from mechanistic_validity.metrics.common import EvalResult, load_model

TASK = "ioi"


def test_classify_gate_and():
    assert classify_gate(0.3, 0.3, 0.9) == "AND"


def test_classify_gate_or():
    assert classify_gate(0.5, 0.3, 0.2) == "OR"


def test_classify_gate_not_i():
    assert classify_gate(-0.2, 0.3, 0.1) == "NOT_i"


def test_classify_gate_not_j():
    assert classify_gate(0.3, -0.2, 0.1) == "NOT_j"


def test_classify_gate_additive():
    assert classify_gate(0.3, 0.2, 0.5) == "ADDITIVE"


def test_classify_gate_exact_additive():
    assert classify_gate(0.5, 0.5, 1.0) == "ADDITIVE"


def test_classify_gate_not_takes_priority():
    assert classify_gate(-0.1, -0.1, 0.5) == "NOT_i"


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


@pytest.fixture(scope="module")
def circuit_results(gpt2_model):
    return run_logic_gates(gpt2_model, tasks=[TASK], n_prompts=3)


def test_run_logic_gates_returns_results(circuit_results):
    assert len(circuit_results) >= 1
    for r in circuit_results:
        assert isinstance(r, EvalResult)


def test_run_logic_gates_metric_id(circuit_results):
    for r in circuit_results:
        assert r.metric_id == "C19.logic_gates"


def test_run_logic_gates_value_is_finite(circuit_results):
    for r in circuit_results:
        assert np.isfinite(r.value)


def test_run_logic_gates_metadata_has_gate_counts(circuit_results):
    for r in circuit_results:
        assert "gate_counts" in r.metadata
        counts = r.metadata["gate_counts"]
        assert "AND" in counts
        assert "OR" in counts


def test_run_logic_gates_metadata_has_gate_proportions(circuit_results):
    for r in circuit_results:
        assert "gate_proportions" in r.metadata
        props = r.metadata["gate_proportions"]
        total = sum(props.values())
        assert total == pytest.approx(1.0, abs=1e-6)


def test_run_logic_gates_metadata_has_completeness(circuit_results):
    for r in circuit_results:
        assert "completeness_noising" in r.metadata
        assert "completeness_denoising" in r.metadata


def test_run_logic_gates_delta_equals_difference(circuit_results):
    for r in circuit_results:
        expected = (
            r.metadata["completeness_noising"]
            - r.metadata["completeness_denoising"]
        )
        assert r.value == pytest.approx(expected, abs=1e-6)


def test_run_logic_gates_metadata_task(circuit_results):
    for r in circuit_results:
        assert r.metadata["task"] == TASK


def test_run_logic_gates_per_head_effects(circuit_results):
    for r in circuit_results:
        assert "per_head_effects" in r.metadata
        assert isinstance(r.metadata["per_head_effects"], dict)
