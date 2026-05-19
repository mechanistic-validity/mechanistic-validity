import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechanistic_validity" / "instruments"
    / "causal" / "woodward" / "37_misalignment_score.py"
)
_spec = importlib.util.spec_from_file_location("misalignment_37", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["misalignment_37"] = _mod
_spec.loader.exec_module(_mod)

run_misalignment = _mod.run_misalignment
compute_noising_effect = _mod.compute_noising_effect
compute_denoising_effect = _mod.compute_denoising_effect

from mechanistic_validity.instruments.common import EvalResult, load_model

TASK = "ioi"


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


@pytest.fixture(scope="module")
def circuit_results(gpt2_model):
    return run_misalignment(gpt2_model, tasks=[TASK], n_prompts=3)


def test_run_returns_eval_results(circuit_results):
    assert len(circuit_results) == 1
    assert isinstance(circuit_results[0], EvalResult)


def test_metric_id(circuit_results):
    assert circuit_results[0].metric_id == "C37.misalignment_score"


def test_value_equals_aggregate(circuit_results):
    r = circuit_results[0]
    assert r.value == pytest.approx(r.metadata["aggregate_misalignment"])


def test_metadata_keys(circuit_results):
    meta = circuit_results[0].metadata
    expected = {"task", "aggregate_misalignment", "max_misalignment",
                "n_severe_misaligned", "severity_threshold", "per_head",
                "n_circuit_heads", "circuit_heads"}
    assert set(meta.keys()) == expected


def test_per_head_has_entries(circuit_results):
    meta = circuit_results[0].metadata
    assert len(meta["per_head"]) == meta["n_circuit_heads"]


def test_per_head_fields(circuit_results):
    meta = circuit_results[0].metadata
    for key, info in meta["per_head"].items():
        assert "noising_necessity" in info
        assert "denoising_sufficiency" in info
        assert "misalignment" in info
        assert "direction" in info


def test_misalignment_non_negative(circuit_results):
    assert circuit_results[0].value >= 0.0


def test_max_misalignment_gte_aggregate(circuit_results):
    meta = circuit_results[0].metadata
    assert meta["max_misalignment"] >= meta["aggregate_misalignment"] - 1e-8


def test_severity_threshold(circuit_results):
    assert circuit_results[0].metadata["severity_threshold"] == pytest.approx(0.3)


def test_n_samples_positive(circuit_results):
    assert circuit_results[0].n_samples > 0
