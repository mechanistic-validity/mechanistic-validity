import importlib.util
import sys
from pathlib import Path

import pytest

from mechval.metrics.common import EvalResult, load_model

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechval" / "metrics"
    / "core" / "structural" / "effective_rank" / "B14_k_alignment.py"
)
_spec = importlib.util.spec_from_file_location("_b14_k_alignment", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)

compute_alignment_scores = _mod.compute_alignment_scores
run = _mod.run

TASK = "ioi"


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


@pytest.fixture(scope="module")
def circuit_results(gpt2_model):
    return run(model=gpt2_model, tasks=[TASK], resume=False)


def test_compute_alignment_scores_keys(gpt2_model):
    scores = compute_alignment_scores(gpt2_model)
    n_layers = gpt2_model.cfg.n_layers
    n_heads = gpt2_model.cfg.n_heads
    assert len(scores) == n_layers * n_heads


def test_compute_alignment_scores_subkeys(gpt2_model):
    scores = compute_alignment_scores(gpt2_model)
    for v in scores.values():
        assert "qk_embed_alignment" in v
        assert "ov_unembed_alignment" in v


def test_compute_alignment_scores_qk_range(gpt2_model):
    scores = compute_alignment_scores(gpt2_model)
    for v in scores.values():
        assert 0.0 <= v["qk_embed_alignment"] <= 1.0


def test_compute_alignment_scores_ov_nonnegative(gpt2_model):
    scores = compute_alignment_scores(gpt2_model)
    for v in scores.values():
        assert v["ov_unembed_alignment"] >= 0.0


def test_run_returns_eval_result(circuit_results):
    assert len(circuit_results) >= 1
    r = circuit_results[0]
    assert isinstance(r, EvalResult)
    assert r.metric_id == "B14.k_alignment"


def test_run_value_is_alignment_ratio(circuit_results):
    r = circuit_results[0]
    assert r.value > 0


def test_run_metadata_fields(circuit_results):
    r = circuit_results[0]
    meta = r.metadata
    assert meta["task"] == TASK
    assert meta["n_circuit_heads"] >= 1
    for key in ["circuit_qk_embed_alignment", "background_qk_embed_alignment",
                "qk_alignment_ratio", "circuit_ov_unembed_alignment",
                "background_ov_unembed_alignment", "ov_alignment_ratio"]:
        assert key in meta


def test_run_n_samples_positive(circuit_results):
    r = circuit_results[0]
    assert r.n_samples >= 1
