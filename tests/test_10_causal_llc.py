import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechanistic_validity" / "instruments"
    / "causal" / "mdl_slt" / "10_llc.py"
)
_spec = importlib.util.spec_from_file_location("llc_10", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["llc_10"] = _mod
_spec.loader.exec_module(_mod)

run_llc = _mod.run_llc
estimate_llc_hessian = _mod.estimate_llc_hessian

from mechanistic_validity.instruments.common import EvalResult, load_model

TASK = "ioi"


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


@pytest.fixture(scope="module")
def circuit_results(gpt2_model):
    return run_llc(gpt2_model, tasks=[TASK], n_prompts=3)


def test_run_returns_eval_results(circuit_results):
    assert len(circuit_results) == 1
    assert isinstance(circuit_results[0], EvalResult)


def test_metric_id(circuit_results):
    assert circuit_results[0].metric_id == "C10.llc"


def test_value_equals_mean_circuit_llc(circuit_results):
    r = circuit_results[0]
    assert r.value == pytest.approx(r.metadata["mean_circuit_llc"])


def test_metadata_keys(circuit_results):
    meta = circuit_results[0].metadata
    expected = {"task", "per_head_llc", "mean_circuit_llc",
                "mean_non_circuit_llc", "ratio", "n_circuit_heads",
                "interpretation"}
    assert set(meta.keys()) == expected


def test_per_head_llc_has_entries(circuit_results):
    meta = circuit_results[0].metadata
    assert len(meta["per_head_llc"]) == meta["n_circuit_heads"]


def test_llc_values_non_negative(circuit_results):
    meta = circuit_results[0].metadata
    for key, llc in meta["per_head_llc"].items():
        assert llc >= 0.0, f"{key} has negative LLC"


def test_n_samples_positive(circuit_results):
    assert circuit_results[0].n_samples > 0


def test_interpretation_field(circuit_results):
    assert circuit_results[0].metadata["interpretation"] == "lower LLC = more specialized/degenerate"
