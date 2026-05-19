import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from mechval.metrics.common import EvalResult

_CMD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "mechval"
    / "metrics"
    / "structural"
    / "template_distance"
    / "26_cmd.py"
)
_spec = importlib.util.spec_from_file_location("_cmd26", _CMD_PATH)
_cmd_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _cmd_mod
_spec.loader.exec_module(_cmd_mod)

jaccard_distance = _cmd_mod.jaccard_distance
run_cmd = _cmd_mod.run_cmd


def test_jaccard_distance_identical_sets():
    assert jaccard_distance({1, 2, 3}, {1, 2, 3}) == pytest.approx(0.0)


def test_jaccard_distance_disjoint_sets():
    assert jaccard_distance({1, 2}, {3, 4}) == pytest.approx(1.0)


def test_jaccard_distance_partial_overlap():
    assert jaccard_distance({1, 2, 3}, {2, 3, 4}) == pytest.approx(0.5)


def test_jaccard_distance_empty_sets():
    assert jaccard_distance(set(), set()) == pytest.approx(0.0)


def test_jaccard_distance_one_empty():
    assert jaccard_distance({1, 2}, set()) == pytest.approx(1.0)


def test_jaccard_distance_subset():
    a = {1, 2, 3, 4}
    b = {2, 3}
    assert jaccard_distance(a, b) == pytest.approx(1.0 - 2 / 4)


def test_jaccard_distance_symmetric():
    a = {1, 2, 3}
    b = {3, 4, 5}
    assert jaccard_distance(a, b) == pytest.approx(jaccard_distance(b, a))


TASK = "ioi"


def test_run_cmd_returns_results_for_multiple_tasks():
    tasks = ["ioi", "sva", "induction"]
    results = run_cmd(tasks)
    assert len(results) == 1
    r = results[0]
    assert isinstance(r, EvalResult)
    assert r.metric_id == "C26.cmd"
    assert 0.0 <= r.value <= 1.0
    assert r.metadata["mean_jaccard_distance"] == pytest.approx(r.value)
    assert len(r.metadata["tasks"]) >= 2
    dm = r.metadata["distance_matrix"]
    n = len(r.metadata["tasks"])
    assert len(dm) == n
    assert len(dm[0]) == n
    for i in range(n):
        assert dm[i][i] == pytest.approx(0.0)


def test_run_cmd_distance_matrix_is_symmetric():
    tasks = ["ioi", "sva", "induction"]
    results = run_cmd(tasks)
    r = results[0]
    dm = np.array(r.metadata["distance_matrix"])
    np.testing.assert_allclose(dm, dm.T)


def test_run_cmd_pairwise_details_present():
    tasks = ["ioi", "sva"]
    results = run_cmd(tasks)
    r = results[0]
    details = r.metadata["pairwise_details"]
    assert len(details) >= 1
    for pair_key, info in details.items():
        assert "jaccard_distance" in info
        assert "n_shared" in info
        assert "n_only_a" in info
        assert "n_only_b" in info
        assert "n_union" in info
        assert info["n_shared"] + info["n_only_a"] + info["n_only_b"] == info["n_union"]


def test_run_cmd_single_task_returns_empty():
    results = run_cmd(["ioi"])
    assert results == []


def test_run_cmd_unknown_tasks_returns_empty():
    results = run_cmd(["fake_task_1", "fake_task_2"])
    assert results == []


def test_run_cmd_circuit_sizes_reported():
    results = run_cmd(["ioi", "sva", "induction"])
    r = results[0]
    sizes = r.metadata["circuit_sizes"]
    for task in r.metadata["tasks"]:
        assert task in sizes
        assert sizes[task] > 0
