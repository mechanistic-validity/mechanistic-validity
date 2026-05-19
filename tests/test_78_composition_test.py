import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechanistic_validity" / "metrics"
    / "causal" / "mdc_glennan" / "78_composition_test.py"
)
_spec = importlib.util.spec_from_file_location("composition_test_78", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["composition_test_78"] = _mod
_spec.loader.exec_module(_mod)

compute_pathway_faithfulness = _mod.compute_pathway_faithfulness
extract_pathway_head_sets = _mod.extract_pathway_head_sets
run_composition_test = _mod.run_composition_test

from mechanistic_validity.metrics.common import EvalResult, load_model

TASK = "ioi"


def test_extract_pathway_head_sets_simple():
    circuit = {
        "roles": {
            "detector": [(0, 0), (0, 1)],
            "integrator": [(1, 0)],
            "executor": [(2, 0)],
        },
        "pathways": [("detector", "integrator"), ("integrator", "executor")],
    }
    result = extract_pathway_head_sets(circuit)
    assert "detector->integrator" in result
    assert "integrator->executor" in result
    expected_heads = {(0, 0), (0, 1), (1, 0)}
    assert result["detector->integrator"] == expected_heads


def test_extract_pathway_head_sets_builds_chains():
    circuit = {
        "roles": {
            "a": [(0, 0)],
            "b": [(1, 0)],
            "c": [(2, 0)],
        },
        "pathways": [("a", "b"), ("b", "c")],
    }
    result = extract_pathway_head_sets(circuit)
    chain_key = "a->b->c"
    assert chain_key in result
    assert result[chain_key] == {(0, 0), (1, 0), (2, 0)}


def test_extract_pathway_head_sets_no_pathways():
    circuit = {
        "roles": {"only": [(0, 0)]},
        "pathways": [],
    }
    result = extract_pathway_head_sets(circuit)
    assert len(result) == 0


def test_extract_pathway_head_sets_single_edge_no_chain():
    circuit = {
        "roles": {
            "a": [(0, 0)],
            "b": [(1, 0)],
        },
        "pathways": [("a", "b")],
    }
    result = extract_pathway_head_sets(circuit)
    assert "a->b" in result
    chain_keys = [k for k in result if k.count("->") > 1]
    assert len(chain_keys) == 0


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


@pytest.fixture(scope="module")
def circuit_results(gpt2_model):
    return run_composition_test(gpt2_model, tasks=[TASK], n_prompts=4)


def test_run_composition_test_returns_results(circuit_results):
    assert len(circuit_results) >= 1
    for r in circuit_results:
        assert isinstance(r, EvalResult)


def test_metric_id_is_correct(circuit_results):
    for r in circuit_results:
        assert r.metric_id == "A2.composition_test"


def test_value_is_finite(circuit_results):
    for r in circuit_results:
        assert np.isfinite(r.value)


def test_metadata_has_full_circuit_faithfulness(circuit_results):
    for r in circuit_results:
        assert "full_circuit_faithfulness" in r.metadata
        assert r.value == pytest.approx(r.metadata["full_circuit_faithfulness"])


def test_metadata_has_per_pathway_faithfulness(circuit_results):
    for r in circuit_results:
        assert "per_pathway_faithfulness" in r.metadata
        assert isinstance(r.metadata["per_pathway_faithfulness"], dict)


def test_metadata_has_max_single_pathway(circuit_results):
    for r in circuit_results:
        assert "max_single_pathway_faithfulness" in r.metadata
        assert np.isfinite(r.metadata["max_single_pathway_faithfulness"])


def test_metadata_has_passed_flag(circuit_results):
    for r in circuit_results:
        assert "passed" in r.metadata
        assert isinstance(r.metadata["passed"], bool)


def test_passed_matches_threshold_logic(circuit_results):
    for r in circuit_results:
        full = r.metadata["full_circuit_faithfulness"]
        max_pw = r.metadata["max_single_pathway_faithfulness"]
        expected = full > 0.30 or max_pw > 0.20
        assert r.metadata["passed"] == expected


def test_metadata_task(circuit_results):
    for r in circuit_results:
        assert r.metadata["task"] == TASK
