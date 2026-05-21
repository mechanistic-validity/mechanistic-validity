import importlib.util
import sys
from pathlib import Path

import pytest

from mechval.metrics.common import EvalResult, load_model

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechval" / "metrics"
    / "core" / "representational" / "attention_entropy" / "E11_attention_entropy.py"
)
_spec = importlib.util.spec_from_file_location("_e11_attention_entropy", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)

compute_attention_entropy = _mod.compute_attention_entropy
run = _mod.run

TASK = "ioi"


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


@pytest.fixture(scope="module")
def circuit_results(gpt2_model, tmp_path_factory):
    out = tmp_path_factory.mktemp("e11")
    return run(model=gpt2_model, tasks=[TASK], device="cpu", n_prompts=5,
               save=False, resume=False, output_dir=str(out))


def test_run_returns_eval_result(circuit_results):
    assert len(circuit_results) >= 1
    r = circuit_results[0]
    assert isinstance(r, EvalResult)
    assert r.metric_id == "E11.attention_entropy"


def test_run_entropy_positive(circuit_results):
    r = circuit_results[0]
    assert r.value > 0


def test_run_metadata_fields(circuit_results):
    r = circuit_results[0]
    meta = r.metadata
    assert meta["task"] == TASK
    assert "circuit_mean" in meta
    assert "circuit_min" in meta
    assert "circuit_max" in meta
    assert "non_circuit_mean" in meta
    assert "ratio" in meta
    assert "per_head" in meta
    assert meta["interpretation"] == "low entropy = focused attention (PTH-like)"


def test_run_per_head_values_positive(circuit_results):
    r = circuit_results[0]
    per_head = r.metadata["per_head"]
    assert len(per_head) >= 1
    for val in per_head.values():
        assert val > 0


def test_run_n_samples_positive(circuit_results):
    r = circuit_results[0]
    assert r.n_samples >= 1


def test_run_non_circuit_mean_positive(circuit_results):
    r = circuit_results[0]
    assert r.metadata["non_circuit_mean"] > 0
