import importlib.util
import sys
from pathlib import Path

import pytest

from mechval.metrics.common import EvalResult

_EJ_PATH = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "mechval"
    / "metrics"
    / "structural"
    / "template_distance"
    / "27_edge_jaccard.py"
)
_spec = importlib.util.spec_from_file_location("_ej27", _EJ_PATH)
_ej_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _ej_mod
_spec.loader.exec_module(_ej_mod)

jaccard_similarity = _ej_mod.jaccard_similarity
_parse_head_key = _ej_mod._parse_head_key
_extract_edges_from_eap_data = _ej_mod._extract_edges_from_eap_data
run_edge_jaccard = _ej_mod.run_edge_jaccard


def test_jaccard_similarity_identical():
    assert jaccard_similarity({1, 2, 3}, {1, 2, 3}) == pytest.approx(1.0)


def test_jaccard_similarity_disjoint():
    assert jaccard_similarity({1, 2}, {3, 4}) == pytest.approx(0.0)


def test_jaccard_similarity_partial():
    assert jaccard_similarity({1, 2, 3}, {2, 3, 4}) == pytest.approx(0.5)


def test_jaccard_similarity_empty():
    assert jaccard_similarity(set(), set()) == pytest.approx(0.0)


def test_jaccard_similarity_one_empty():
    assert jaccard_similarity({1, 2}, set()) == pytest.approx(0.0)


def test_jaccard_similarity_symmetric():
    a = {1, 2, 3}
    b = {3, 4, 5}
    assert jaccard_similarity(a, b) == pytest.approx(jaccard_similarity(b, a))


def test_parse_head_key_valid():
    assert _parse_head_key("L5H3") == (5, 3)
    assert _parse_head_key("L0H0") == (0, 0)
    assert _parse_head_key("L11H11") == (11, 11)


def test_parse_head_key_invalid():
    assert _parse_head_key("invalid") is None
    assert _parse_head_key("H5L3") is None
    assert _parse_head_key("") is None
    assert _parse_head_key("L") is None
    assert _parse_head_key("LH") is None


def test_extract_edges_from_eap_sets_format():
    data = {
        "sets": {
            "eap_top15": {
                "heads": ["L0H1", "L2H3", "L5H7"]
            }
        }
    }
    edges = _extract_edges_from_eap_data(data, "ioi")
    assert edges is not None
    assert len(edges) > 0
    for sl, sh, rl, rh in edges:
        assert sl < rl


def test_extract_edges_from_eap_head_scores_format():
    data = {
        "head_scores": {
            "L0H1": {"total": 0.5},
            "L2H3": {"total": -0.8},
            "L5H7": {"total": 0.3},
        }
    }
    edges = _extract_edges_from_eap_data(data, "ioi")
    assert edges is not None
    for sl, sh, rl, rh in edges:
        assert sl < rl


def test_extract_edges_returns_none_for_empty_data():
    assert _extract_edges_from_eap_data({}, "ioi") is None


def test_extract_edges_from_single_layer_returns_none():
    data = {
        "sets": {
            "eap_top15": {
                "heads": ["L5H0", "L5H1", "L5H2"]
            }
        }
    }
    edges = _extract_edges_from_eap_data(data, "ioi")
    assert edges is None


TASK = "ioi"


def test_run_edge_jaccard_no_eap_data_returns_empty():
    results = run_edge_jaccard([TASK])
    assert isinstance(results, list)
    for r in results:
        assert isinstance(r, EvalResult)
        assert r.metric_id == "C27.edge_jaccard"
        assert 0.0 <= r.value <= 1.0


def test_run_edge_jaccard_unknown_task():
    results = run_edge_jaccard(["nonexistent_xyz"])
    assert results == []
