import importlib.util
import sys
from pathlib import Path

import pytest

from mechanistic_validity.instruments.common import EvalResult

_WEJ_PATH = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "mechanistic_validity"
    / "instruments"
    / "structural"
    / "weight_alignment"
    / "28_weight_eap_jaccard.py"
)
_spec = importlib.util.spec_from_file_location("_wej28", _WEJ_PATH)
_wej_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _wej_mod
_spec.loader.exec_module(_wej_mod)

jaccard_similarity = _wej_mod.jaccard_similarity
_parse_head_key = _wej_mod._parse_head_key
_extract_heads_from_eap_data = _wej_mod._extract_heads_from_eap_data
run_weight_eap_jaccard = _wej_mod.run_weight_eap_jaccard


def test_jaccard_similarity_identical():
    assert jaccard_similarity({(0, 1), (2, 3)}, {(0, 1), (2, 3)}) == pytest.approx(1.0)


def test_jaccard_similarity_disjoint():
    assert jaccard_similarity({(0, 1)}, {(2, 3)}) == pytest.approx(0.0)


def test_jaccard_similarity_partial():
    a = {(0, 1), (2, 3), (4, 5)}
    b = {(2, 3), (4, 5), (6, 7)}
    assert jaccard_similarity(a, b) == pytest.approx(0.5)


def test_jaccard_similarity_empty():
    assert jaccard_similarity(set(), set()) == pytest.approx(0.0)


def test_parse_head_key_valid():
    assert _parse_head_key("L0H0") == (0, 0)
    assert _parse_head_key("L11H11") == (11, 11)
    assert _parse_head_key("L3H7") == (3, 7)


def test_parse_head_key_invalid():
    assert _parse_head_key("X0Y0") is None
    assert _parse_head_key("") is None
    assert _parse_head_key("layer0head1") is None


def test_extract_heads_sets_format():
    data = {
        "sets": {
            "eap_top15": {
                "heads": ["L0H1", "L2H3", "L5H7"]
            }
        }
    }
    heads = _extract_heads_from_eap_data(data, "ioi")
    assert heads == {(0, 1), (2, 3), (5, 7)}


def test_extract_heads_head_scores_format():
    data = {
        "head_scores": {
            "L0H1": {"total": 0.9},
            "L2H3": {"total": -0.5},
            "L5H7": {"total": 0.3},
        }
    }
    heads = _extract_heads_from_eap_data(data, "ioi")
    assert heads is not None
    assert (0, 1) in heads
    assert (2, 3) in heads
    assert (5, 7) in heads


def test_extract_heads_per_task_format():
    data = {
        "ioi": {
            "heads": [[0, 1], [2, 3]]
        }
    }
    heads = _extract_heads_from_eap_data(data, "ioi")
    assert heads == {(0, 1), (2, 3)}


def test_extract_heads_per_task_string_format():
    data = {
        "ioi": {
            "eap_heads": ["L0H1", "L2H3"]
        }
    }
    heads = _extract_heads_from_eap_data(data, "ioi")
    assert heads == {(0, 1), (2, 3)}


def test_extract_heads_wrong_task_per_task_format():
    data = {
        "sva": {
            "heads": [[0, 1]]
        }
    }
    heads = _extract_heads_from_eap_data(data, "ioi")
    assert heads is None


def test_extract_heads_empty_data():
    assert _extract_heads_from_eap_data({}, "ioi") is None


def test_extract_heads_head_scores_takes_top15():
    data = {"head_scores": {}}
    for layer in range(12):
        for head in range(2):
            data["head_scores"][f"L{layer}H{head}"] = {"total": layer + head * 0.1}
    heads = _extract_heads_from_eap_data(data, "ioi")
    assert heads is not None
    assert len(heads) == 15


TASK = "ioi"


def test_run_weight_eap_jaccard_returns_list():
    results = run_weight_eap_jaccard([TASK])
    assert isinstance(results, list)
    for r in results:
        assert isinstance(r, EvalResult)
        assert r.metric_id == "C28.weight_eap_jaccard"
        assert 0.0 <= r.value <= 1.0
        assert "task" in r.metadata
        assert "n_weight_heads" in r.metadata
        assert "n_eap_heads" in r.metadata


def test_run_weight_eap_jaccard_unknown_task():
    results = run_weight_eap_jaccard(["nonexistent_xyz"])
    assert results == []
