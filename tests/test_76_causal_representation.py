import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from mechval.metrics.common import EvalResult, load_model

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "mechval"
    / "metrics"
    / "representational"
    / "linear_probe"
    / "76_causal_representation.py"
)
_spec = importlib.util.spec_from_file_location("causal_rep_76", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["causal_rep_76"] = _mod
_spec.loader.exec_module(_mod)

make_counterfactual_pairs = _mod.make_counterfactual_pairs
pick_control_layer = _mod.pick_control_layer

TASK = "ioi"


def test_make_counterfactual_pairs_different_answers():
    class FakePrompt:
        pass

    prompts = [FakePrompt() for _ in range(6)]
    correct_ids = [10, 20, 10, 30, 20, 30]
    pairs = make_counterfactual_pairs(prompts, correct_ids)
    for i, j in pairs:
        assert correct_ids[i] != correct_ids[j]


def test_make_counterfactual_pairs_no_reuse():
    class FakePrompt:
        pass

    prompts = [FakePrompt() for _ in range(10)]
    correct_ids = [1, 2, 1, 2, 3, 4, 3, 4, 5, 6]
    pairs = make_counterfactual_pairs(prompts, correct_ids)
    used = set()
    for i, j in pairs:
        assert i not in used
        assert j not in used
        used.add(i)
        used.add(j)


def test_make_counterfactual_pairs_all_same():
    class FakePrompt:
        pass

    prompts = [FakePrompt() for _ in range(5)]
    correct_ids = [42, 42, 42, 42, 42]
    pairs = make_counterfactual_pairs(prompts, correct_ids)
    assert len(pairs) == 0


def test_make_counterfactual_pairs_empty():
    pairs = make_counterfactual_pairs([], [])
    assert len(pairs) == 0


def test_pick_control_layer_not_in_circuit():
    circuit_layers = {2, 5, 7}
    control = pick_control_layer(12, circuit_layers)
    assert control not in circuit_layers
    assert 0 <= control < 12


def test_pick_control_layer_all_circuit():
    circuit_layers = {0, 1, 2, 3}
    control = pick_control_layer(4, circuit_layers)
    assert control == 0


def test_pick_control_layer_deterministic():
    a = pick_control_layer(12, {3, 5}, seed=42)
    b = pick_control_layer(12, {3, 5}, seed=42)
    assert a == b


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


def test_run_causal_representation_returns_results(gpt2_model):
    run_causal_representation = _mod.run_causal_representation
    results = run_causal_representation(gpt2_model, [TASK], n_prompts=4)
    assert isinstance(results, list)
    for r in results:
        assert isinstance(r, EvalResult)
        assert r.metric_id == "R3.causal_representation"
        assert "best_circuit_iia" in r.metadata
        assert "control_iia" in r.metadata
        assert 0.0 <= r.metadata["best_circuit_iia"] <= 1.0
        assert 0.0 <= r.metadata["control_iia"] <= 1.0
