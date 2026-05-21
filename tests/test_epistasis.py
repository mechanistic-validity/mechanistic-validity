import importlib.util
import math
import sys
from pathlib import Path

import pytest

from mechval.metrics.common import EvalResult, load_model

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechval" / "metrics" / "extended" / "genetics" / "GN2_epistasis.py"
)
_spec = importlib.util.spec_from_file_location("_mod", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)

run_epistasis = _mod.run_epistasis

TASK = "ioi"


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


@pytest.fixture(scope="module")
def results(gpt2_model):
    return run_epistasis(gpt2_model, [TASK], n_prompts=5)


def test_returns_results(results):
    assert len(results) >= 1


def test_is_eval_result(results):
    assert isinstance(results[0], EvalResult)


def test_metric_id(results):
    assert results[0].metric_id == "GN2.epistasis"


def test_value_is_finite(results):
    assert math.isfinite(results[0].value)


def test_metadata_has_task(results):
    assert results[0].metadata["task"] == TASK


def test_n_samples_positive(results):
    assert results[0].n_samples > 0
