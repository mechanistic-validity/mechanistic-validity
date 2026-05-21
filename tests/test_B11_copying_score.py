import importlib.util
import sys
from pathlib import Path

import pytest

from mechval.metrics.common import EvalResult, load_model

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechval" / "metrics"
    / "core" / "structural" / "copying_score" / "B11_copying_score.py"
)
_spec = importlib.util.spec_from_file_location("_b11_copying", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)

compute_copying_scores = _mod.compute_copying_scores
run = _mod.run

TASK = "ioi"


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


@pytest.fixture(scope="module")
def circuit_results(gpt2_model, tmp_path_factory):
    out = tmp_path_factory.mktemp("b11")
    return run(model=gpt2_model, tasks=[TASK], device="cpu", save=False, resume=False,
               output_dir=str(out))


def test_compute_copying_scores_keys(gpt2_model):
    scores = compute_copying_scores(gpt2_model)
    n_layers = gpt2_model.cfg.n_layers
    n_heads = gpt2_model.cfg.n_heads
    assert len(scores) == n_layers * n_heads
    for (L, H) in scores:
        assert 0 <= L < n_layers
        assert 0 <= H < n_heads


def test_compute_copying_scores_subkeys(gpt2_model):
    scores = compute_copying_scores(gpt2_model)
    for v in scores.values():
        assert "score" in v
        assert "top_eigenvalues" in v
        assert "trace" in v
        assert isinstance(v["top_eigenvalues"], list)
        assert len(v["top_eigenvalues"]) <= 5


def test_run_returns_eval_result(circuit_results):
    assert len(circuit_results) >= 1
    r = circuit_results[0]
    assert isinstance(r, EvalResult)
    assert r.metric_id == "B11.copying_score"


def test_run_metadata_fields(circuit_results):
    r = circuit_results[0]
    meta = r.metadata
    assert meta["task"] == TASK
    for key in ["circuit_mean", "ratio", "per_head"]:
        assert key in meta
    assert isinstance(meta["per_head"], dict)


def test_run_n_samples_positive(circuit_results):
    r = circuit_results[0]
    assert r.n_samples >= 1
