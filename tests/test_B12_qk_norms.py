import importlib.util
import sys
from pathlib import Path

import pytest

from mechval.metrics.common import EvalResult, load_model

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechval" / "metrics"
    / "core" / "structural" / "qk_norms" / "B12_qk_norms.py"
)
_spec = importlib.util.spec_from_file_location("_b12_qk_norms", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)

compute_qk_metrics = _mod.compute_qk_metrics
run = _mod.run

TASK = "ioi"


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


@pytest.fixture(scope="module")
def circuit_results(gpt2_model, tmp_path_factory):
    out = tmp_path_factory.mktemp("b12")
    return run(model=gpt2_model, tasks=[TASK], device="cpu", save=False, resume=False,
               output_dir=str(out))


def test_compute_qk_metrics_keys(gpt2_model):
    metrics = compute_qk_metrics(gpt2_model)
    n_layers = gpt2_model.cfg.n_layers
    n_heads = gpt2_model.cfg.n_heads
    assert len(metrics) == n_layers * n_heads


def test_compute_qk_metrics_frob_norm_positive(gpt2_model):
    metrics = compute_qk_metrics(gpt2_model)
    for v in metrics.values():
        assert v["qk_frob_norm"] > 0


def test_compute_qk_metrics_sv_gap_above_one(gpt2_model):
    metrics = compute_qk_metrics(gpt2_model)
    for v in metrics.values():
        assert v["sv_gap"] >= 1.0


def test_compute_qk_metrics_subkeys(gpt2_model):
    metrics = compute_qk_metrics(gpt2_model)
    for v in metrics.values():
        for key in ["qk_frob_norm", "sv_gap", "top_sv"]:
            assert key in v


def test_run_produces_two_results_per_task(circuit_results):
    assert len(circuit_results) == 2


def test_run_metric_ids(circuit_results):
    ids = {r.metric_id for r in circuit_results}
    assert "B12.qk_frob_norm" in ids
    assert "B12.sv_gap" in ids


def test_run_frob_norm_result(circuit_results):
    r = [x for x in circuit_results if x.metric_id == "B12.qk_frob_norm"][0]
    assert isinstance(r, EvalResult)
    assert r.value > 0
    assert r.metadata["task"] == TASK
    assert "per_head" in r.metadata
    assert "ratio" in r.metadata


def test_run_sv_gap_result(circuit_results):
    r = [x for x in circuit_results if x.metric_id == "B12.sv_gap"][0]
    assert isinstance(r, EvalResult)
    assert r.value >= 1.0
    assert r.metadata["task"] == TASK


def test_run_n_samples_positive(circuit_results):
    for r in circuit_results:
        assert r.n_samples >= 1
