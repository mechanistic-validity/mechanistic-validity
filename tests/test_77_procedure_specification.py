import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechanistic_validity" / "instruments"
    / "causal" / "mdc_glennan" / "77_procedure_specification.py"
)
_spec = importlib.util.spec_from_file_location("procedure_spec_77", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["procedure_spec_77"] = _mod
_spec.loader.exec_module(_mod)

compute_head_logit_attribution = _mod.compute_head_logit_attribution
compute_pathway_ordering_score = _mod.compute_pathway_ordering_score
build_pathway_chains = _mod.build_pathway_chains
run_procedure_specification = _mod.run_procedure_specification

from mechanistic_validity.instruments.common import EvalResult, load_model

TASK = "ioi"


def test_compute_pathway_ordering_score_monotonic():
    attribs_list = [
        {(0, 0): 1.0, (1, 0): 2.0, (2, 0): 3.0},
        {(0, 0): 0.5, (1, 0): 1.5, (2, 0): 2.5},
    ]
    chains = [[{(0, 0)}, {(1, 0)}, {(2, 0)}]]
    score = compute_pathway_ordering_score(attribs_list, chains)
    assert score == pytest.approx(1.0)


def test_compute_pathway_ordering_score_non_monotonic():
    attribs_list = [
        {(0, 0): 3.0, (1, 0): 1.0, (2, 0): 2.0},
    ]
    chains = [[{(0, 0)}, {(1, 0)}, {(2, 0)}]]
    score = compute_pathway_ordering_score(attribs_list, chains)
    assert score == pytest.approx(0.5)


def test_compute_pathway_ordering_score_empty_chains():
    attribs_list = [{(0, 0): 1.0}]
    chains = []
    score = compute_pathway_ordering_score(attribs_list, chains)
    assert score == pytest.approx(0.0)


def test_compute_pathway_ordering_score_single_node_chain():
    attribs_list = [{(0, 0): 1.0}]
    chains = [[{(0, 0)}]]
    score = compute_pathway_ordering_score(attribs_list, chains)
    assert score == pytest.approx(0.0)


def test_build_pathway_chains_simple():
    circuit = {
        "roles": {
            "detector": [(0, 0)],
            "integrator": [(1, 0)],
            "executor": [(2, 0)],
        },
        "pathways": [("detector", "integrator"), ("integrator", "executor")],
    }
    chains = build_pathway_chains(circuit)
    assert len(chains) >= 1
    longest = max(chains, key=len)
    assert len(longest) == 3


def test_build_pathway_chains_no_pathways():
    circuit = {
        "roles": {"only_role": [(0, 0)]},
        "pathways": [],
    }
    chains = build_pathway_chains(circuit)
    assert len(chains) == 0


def test_build_pathway_chains_branching():
    circuit = {
        "roles": {
            "source": [(0, 0)],
            "left": [(1, 0)],
            "right": [(1, 1)],
        },
        "pathways": [("source", "left"), ("source", "right")],
    }
    chains = build_pathway_chains(circuit)
    assert len(chains) == 2


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


def test_compute_head_logit_attribution_returns_dict(gpt2_model):
    from mechanistic_validity.instruments.common import generate_prompts, get_token_ids

    prompts = generate_prompts(TASK, gpt2_model.tokenizer, n_prompts=2)
    correct_ids, incorrect_ids = get_token_ids(prompts, gpt2_model.tokenizer)
    tokens = gpt2_model.to_tokens(prompts[0].text)
    attribs = compute_head_logit_attribution(
        gpt2_model, tokens, correct_ids[0], incorrect_ids[0],
    )
    assert isinstance(attribs, dict)
    n_total = gpt2_model.cfg.n_layers * gpt2_model.cfg.n_heads
    assert len(attribs) == n_total
    for (L, H), val in attribs.items():
        assert np.isfinite(val)


@pytest.fixture(scope="module")
def circuit_results(gpt2_model):
    return run_procedure_specification(gpt2_model, tasks=[TASK], n_prompts=5)


def test_run_procedure_specification_returns_results(circuit_results):
    assert len(circuit_results) >= 1
    for r in circuit_results:
        assert isinstance(r, EvalResult)


def test_metric_id_is_correct(circuit_results):
    for r in circuit_results:
        assert r.metric_id == "A1.procedure_specification"


def test_value_is_finite(circuit_results):
    for r in circuit_results:
        assert np.isfinite(r.value)
        assert 0.0 <= r.value <= 1.0


def test_metadata_has_ordering_score(circuit_results):
    for r in circuit_results:
        assert "ordering_score" in r.metadata
        assert r.metadata["ordering_score"] == pytest.approx(r.value)


def test_metadata_has_pathway_fraction(circuit_results):
    for r in circuit_results:
        assert "pathway_fraction" in r.metadata
        assert np.isfinite(r.metadata["pathway_fraction"])


def test_metadata_has_passed_flag(circuit_results):
    for r in circuit_results:
        assert "passed" in r.metadata
        assert isinstance(r.metadata["passed"], bool)


def test_metadata_task(circuit_results):
    for r in circuit_results:
        assert r.metadata["task"] == TASK
