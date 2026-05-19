import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechval" / "metrics"
    / "causal" / "transportability" / "38_cross_model_invariance.py"
)
_spec = importlib.util.spec_from_file_location("cross_model_38", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["cross_model_38"] = _mod
_spec.loader.exec_module(_mod)

compute_layer_histogram = _mod.compute_layer_histogram
compute_sv_distribution = _mod.compute_sv_distribution
MODEL_PARAM_COUNTS = _mod.MODEL_PARAM_COUNTS
run_cross_model_invariance = _mod.run_cross_model_invariance

from mechval.metrics.common import EvalResult, load_model

TASK = "ioi"


def test_compute_layer_histogram_basic():
    heads = {(0, 0), (0, 1), (2, 3)}
    hist = compute_layer_histogram(heads, n_layers=4)
    assert hist.shape == (4,)
    assert hist.sum() == pytest.approx(1.0)
    assert hist[0] == pytest.approx(2.0 / 3.0)
    assert hist[2] == pytest.approx(1.0 / 3.0)
    assert hist[1] == pytest.approx(0.0)
    assert hist[3] == pytest.approx(0.0)


def test_compute_layer_histogram_empty():
    hist = compute_layer_histogram(set(), n_layers=3)
    assert hist.sum() == pytest.approx(0.0)


def test_compute_layer_histogram_single_layer():
    heads = {(1, 0), (1, 1), (1, 2)}
    hist = compute_layer_histogram(heads, n_layers=3)
    assert hist[1] == pytest.approx(1.0)
    assert hist[0] == pytest.approx(0.0)
    assert hist[2] == pytest.approx(0.0)


def test_model_param_counts():
    assert "gpt2" in MODEL_PARAM_COUNTS
    assert MODEL_PARAM_COUNTS["gpt2"] == 124_000_000
    assert MODEL_PARAM_COUNTS["gpt2-xl"] > MODEL_PARAM_COUNTS["gpt2"]


def test_compute_sv_distribution_with_model():
    model = load_model("gpt2", "cpu")
    heads = {(0, 0), (0, 1)}
    sv = compute_sv_distribution(model, heads, layer=0)
    assert len(sv) > 0
    assert sv.sum() == pytest.approx(1.0, abs=0.01)


def test_compute_sv_distribution_no_heads():
    model = load_model("gpt2", "cpu")
    sv = compute_sv_distribution(model, set(), layer=0)
    assert len(sv) == 0


def test_run_cross_model_invariance_single_model():
    results = run_cross_model_invariance(["gpt2"], tasks=[TASK], device="cpu", n_prompts=3)
    assert len(results) == 0
