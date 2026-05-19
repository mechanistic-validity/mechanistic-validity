import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechval" / "metrics"
    / "causal" / "scm_pearl" / "04_causal_scrubbing.py"
)
_spec = importlib.util.spec_from_file_location("causal_scrubbing_04", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["causal_scrubbing_04"] = _mod
_spec.loader.exec_module(_mod)

run_causal_scrubbing = _mod.run_causal_scrubbing
scrub_circuit = _mod.scrub_circuit

from mechval.metrics.common import EvalResult, load_model

TASK = "ioi"


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


@pytest.fixture(scope="module")
def circuit_results(gpt2_model):
    return run_causal_scrubbing(gpt2_model, tasks=[TASK], n_prompts=4)


def test_run_returns_eval_results(circuit_results):
    assert len(circuit_results) == 1
    assert isinstance(circuit_results[0], EvalResult)


def test_metric_id(circuit_results):
    assert circuit_results[0].metric_id == "C4.causal_scrubbing"


def test_value_is_kl_divergence(circuit_results):
    r = circuit_results[0]
    assert r.value == pytest.approx(r.metadata["kl_divergence"])


def test_kl_non_negative(circuit_results):
    assert circuit_results[0].value >= 0.0


def test_metadata_keys(circuit_results):
    meta = circuit_results[0].metadata
    expected = {"task", "kl_divergence", "logit_diff_recovery",
                "n_circuit_heads", "slow"}
    assert set(meta.keys()) == expected


def test_slow_flag(circuit_results):
    assert circuit_results[0].metadata["slow"] is True


def test_recovery_is_finite(circuit_results):
    recovery = circuit_results[0].metadata["logit_diff_recovery"]
    assert np.isfinite(recovery)


def test_n_samples_positive(circuit_results):
    assert circuit_results[0].n_samples > 0


def test_task_metadata(circuit_results):
    assert circuit_results[0].metadata["task"] == TASK
