import importlib.util
import json
import sys
from pathlib import Path

import pytest

import mechanistic_validity.metrics.common as common
from mechanistic_validity.metrics.common import EvalResult, set_data_dir

_MC_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechanistic_validity" / "calibrations" / "multiple_comparisons" / "93_multiple_comparisons.py"
)
_spec = importlib.util.spec_from_file_location("_mc93", _MC_PATH)
_mc_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mc_mod
_spec.loader.exec_module(_mc_mod)

benjamini_hochberg = _mc_mod.benjamini_hochberg
bonferroni = _mc_mod.bonferroni
_collect_testable_results = _mc_mod._collect_testable_results
_analyze_task = _mc_mod._analyze_task
_approximate_p_value = _mc_mod._approximate_p_value
ALPHA = _mc_mod.ALPHA


@pytest.fixture(autouse=True)
def _restore_data_dir():
    original = common.DATA_DIR
    original_mod = _mc_mod.DATA_DIR
    yield
    common.DATA_DIR = original
    _mc_mod.DATA_DIR = original_mod


def _set_dirs(path: Path):
    set_data_dir(path)
    _mc_mod.DATA_DIR = common.DATA_DIR


# -- BH correction ----------------------------------------------------------

def test_bh_known_p_values():
    p_values = [0.01, 0.04, 0.03, 0.20]
    mask = benjamini_hochberg(p_values, alpha=0.05)
    # Sorted: (0, 0.01), (2, 0.03), (1, 0.04), (3, 0.20)
    # k=1: threshold = (1/4)*0.05 = 0.0125, p=0.01 <= 0.0125 -> yes, max_k=0
    # k=2: threshold = (2/4)*0.05 = 0.025, p=0.03 > 0.025 -> no
    # k=3: threshold = (3/4)*0.05 = 0.0375, p=0.04 > 0.0375 -> no
    # k=4: threshold = (4/4)*0.05 = 0.05, p=0.20 > 0.05 -> no
    # max_k=0, so only position 0 survives -> original index 0
    assert mask[0] is True
    assert mask[1] is False
    assert mask[2] is False
    assert mask[3] is False


def test_bh_all_significant():
    p_values = [0.001, 0.005, 0.010]
    mask = benjamini_hochberg(p_values, alpha=0.05)
    # Sorted: (0, 0.001), (1, 0.005), (2, 0.010)
    # k=1: threshold = (1/3)*0.05 = 0.0167, p=0.001 <= 0.0167 -> yes
    # k=2: threshold = (2/3)*0.05 = 0.0333, p=0.005 <= 0.0333 -> yes
    # k=3: threshold = (3/3)*0.05 = 0.05, p=0.010 <= 0.05 -> yes
    # max_k=2, all 3 survive
    assert all(mask)


def test_bh_none_significant():
    p_values = [0.50, 0.60, 0.70]
    mask = benjamini_hochberg(p_values, alpha=0.05)
    assert not any(mask)


def test_bh_empty_list():
    assert benjamini_hochberg([], alpha=0.05) == []


def test_bh_single_significant():
    mask = benjamini_hochberg([0.01], alpha=0.05)
    assert mask == [True]


def test_bh_single_not_significant():
    mask = benjamini_hochberg([0.10], alpha=0.05)
    assert mask == [False]


def test_bh_preserves_original_order():
    p_values = [0.50, 0.001, 0.30, 0.002]
    mask = benjamini_hochberg(p_values, alpha=0.05)
    # Sorted: (1, 0.001), (3, 0.002), (2, 0.30), (0, 0.50)
    # k=1: threshold = (1/4)*0.05 = 0.0125, p=0.001 <= 0.0125 -> yes
    # k=2: threshold = (2/4)*0.05 = 0.025, p=0.002 <= 0.025 -> yes
    # k=3: threshold = (3/4)*0.05 = 0.0375, p=0.30 > 0.0375 -> no
    # k=4: threshold = (4/4)*0.05 = 0.05, p=0.50 > 0.05 -> no
    # max_k=1, so positions 0 and 1 in sorted order survive -> original indices 1, 3
    assert mask == [False, True, False, True]


# -- Bonferroni correction --------------------------------------------------

def test_bonferroni_known_p_values():
    p_values = [0.01, 0.04, 0.03, 0.20]
    mask = bonferroni(p_values, alpha=0.05)
    # threshold = 0.05 / 4 = 0.0125
    assert mask[0] is True   # 0.01 <= 0.0125
    assert mask[1] is False  # 0.04 > 0.0125
    assert mask[2] is False  # 0.03 > 0.0125
    assert mask[3] is False  # 0.20 > 0.0125


def test_bonferroni_all_significant():
    p_values = [0.001, 0.002, 0.003]
    mask = bonferroni(p_values, alpha=0.05)
    # threshold = 0.05 / 3 = 0.0167
    assert all(mask)


def test_bonferroni_none_significant():
    p_values = [0.10, 0.20, 0.30]
    mask = bonferroni(p_values, alpha=0.05)
    assert not any(mask)


def test_bonferroni_empty_list():
    assert bonferroni([], alpha=0.05) == []


def test_bonferroni_single_significant():
    mask = bonferroni([0.03], alpha=0.05)
    # threshold = 0.05 / 1 = 0.05, 0.03 <= 0.05
    assert mask == [True]


def test_bonferroni_more_conservative_than_bh():
    p_values = [0.005, 0.015, 0.025, 0.045]
    bh_mask = benjamini_hochberg(p_values, alpha=0.05)
    bonf_mask = bonferroni(p_values, alpha=0.05)
    # Bonferroni should never accept more than BH
    assert sum(bonf_mask) <= sum(bh_mask)


def test_bonferroni_threshold_scales_with_count():
    mask_2 = bonferroni([0.02, 0.02], alpha=0.05)
    mask_10 = bonferroni([0.02] * 10, alpha=0.05)
    # threshold for 2: 0.025, 0.02 <= 0.025 -> all True
    # threshold for 10: 0.005, 0.02 > 0.005 -> all False
    assert all(mask_2)
    assert not any(mask_10)


# -- Integration: _collect_testable_results with fake JSON -------------------

def _write_fake_result_json(data_dir: Path, filename: str, results: list[dict]):
    path = data_dir / filename
    with open(path, "w") as f:
        json.dump(results, f)


def test_collect_testable_results_reads_json(tmp_path):
    _set_dirs(tmp_path)
    fake_results = [
        {
            "metric_id": "M01.fake_metric",
            "value": 0.95,
            "baseline_random": 0.50,
            "n_samples": 100,
            "ci_low": 0.90,
            "ci_high": 1.00,
            "metadata": {"task": "ioi", "passed": True},
        },
        {
            "metric_id": "M01.other_metric",
            "value": 0.60,
            "baseline_random": 0.55,
            "n_samples": 50,
            "ci_low": None,
            "ci_high": None,
            "metadata": {"task": "ioi", "passed": True},
        },
    ]
    _write_fake_result_json(tmp_path, "01_fake_instrument.json", fake_results)

    testable = _collect_testable_results("ioi")
    assert len(testable) == 2
    assert all(t["task"] == "ioi" for t in testable)
    assert all(t["p_value"] is not None for t in testable)
    assert testable[0]["source_file"] == "01_fake_instrument.json"


def test_collect_testable_results_filters_by_task(tmp_path):
    _set_dirs(tmp_path)
    fake_results = [
        {
            "metric_id": "M01.metric",
            "value": 0.90,
            "baseline_random": 0.50,
            "n_samples": 100,
            "ci_low": 0.85,
            "ci_high": 0.95,
            "metadata": {"task": "ioi", "passed": True},
        },
        {
            "metric_id": "M01.metric_sva",
            "value": 0.80,
            "baseline_random": 0.50,
            "n_samples": 100,
            "ci_low": 0.75,
            "ci_high": 0.85,
            "metadata": {"task": "sva", "passed": True},
        },
    ]
    _write_fake_result_json(tmp_path, "01_mixed.json", fake_results)

    ioi_results = _collect_testable_results("ioi")
    sva_results = _collect_testable_results("sva")
    assert len(ioi_results) == 1
    assert len(sva_results) == 1
    assert ioi_results[0]["task"] == "ioi"
    assert sva_results[0]["task"] == "sva"


def test_collect_testable_results_no_p_values(tmp_path):
    _set_dirs(tmp_path)
    fake_results = [
        {
            "metric_id": "M01.no_data",
            "value": None,
            "baseline_random": None,
            "n_samples": 0,
            "metadata": {"task": "ioi"},
        },
    ]
    _write_fake_result_json(tmp_path, "01_empty.json", fake_results)

    testable = _collect_testable_results("ioi")
    assert len(testable) == 0


def test_collect_testable_results_uses_pass_fail_fallback(tmp_path):
    _set_dirs(tmp_path)
    fake_results = [
        {
            "metric_id": "M01.passed_only",
            "metadata": {"task": "ioi", "passed": True},
        },
        {
            "metric_id": "M01.failed_only",
            "metadata": {"task": "ioi", "passed": False},
        },
    ]
    _write_fake_result_json(tmp_path, "01_pass_fail.json", fake_results)

    testable = _collect_testable_results("ioi")
    assert len(testable) == 2
    passed_entry = [t for t in testable if t["metric_id"] == "M01.passed_only"][0]
    failed_entry = [t for t in testable if t["metric_id"] == "M01.failed_only"][0]
    assert passed_entry["p_value"] == pytest.approx(0.01)
    assert failed_entry["p_value"] == pytest.approx(0.50)


def test_collect_testable_results_ignores_non_numbered_files(tmp_path):
    _set_dirs(tmp_path)
    fake_results = [
        {
            "metric_id": "M01.metric",
            "value": 0.90,
            "baseline_random": 0.50,
            "n_samples": 100,
            "ci_low": 0.85,
            "ci_high": 0.95,
            "metadata": {"task": "ioi", "passed": True},
        },
    ]
    _write_fake_result_json(tmp_path, "summary.json", fake_results)

    testable = _collect_testable_results("ioi")
    assert len(testable) == 0


def test_collect_testable_results_empty_dir(tmp_path):
    _set_dirs(tmp_path)
    testable = _collect_testable_results("ioi")
    assert len(testable) == 0


# -- Integration: _analyze_task ----------------------------------------------

def test_analyze_task_returns_none_when_no_data(tmp_path):
    _set_dirs(tmp_path)
    result = _analyze_task("ioi")
    assert result is None


def test_analyze_task_returns_eval_result(tmp_path):
    _set_dirs(tmp_path)
    # Create results with strong and weak signals
    fake_results = [
        {
            "metric_id": "M01.strong",
            "value": 0.95,
            "baseline_random": 0.50,
            "n_samples": 100,
            "ci_low": 0.90,
            "ci_high": 1.00,
            "metadata": {"task": "ioi", "passed": True},
        },
        {
            "metric_id": "M02.weak",
            "value": 0.52,
            "baseline_random": 0.50,
            "n_samples": 100,
            "ci_low": 0.48,
            "ci_high": 0.56,
            "metadata": {"task": "ioi", "passed": False},
        },
    ]
    _write_fake_result_json(tmp_path, "01_strong.json", [fake_results[0]])
    _write_fake_result_json(tmp_path, "02_weak.json", [fake_results[1]])

    result = _analyze_task("ioi")
    assert result is not None
    assert isinstance(result, EvalResult)
    assert result.metric_id == "M93.multiple_comparisons"
    assert result.metadata["task"] == "ioi"
    assert result.metadata["n_tests"] == 2
    assert "n_survive_bh" in result.metadata
    assert "n_survive_bonferroni" in result.metadata
    assert "bh_survival_rate" in result.metadata
    assert "passed" in result.metadata
    assert "per_test" in result.metadata
    assert len(result.metadata["per_test"]) == 2


def test_analyze_task_all_pass_high_survival(tmp_path):
    _set_dirs(tmp_path)
    # All results are strongly significant -> should mostly survive correction
    fake_results = []
    for i in range(5):
        fake_results.append({
            "metric_id": f"M0{i}.metric",
            "metadata": {"task": "ioi", "passed": True},
        })
    _write_fake_result_json(tmp_path, "01_all_pass.json", fake_results)

    result = _analyze_task("ioi")
    assert result is not None
    # All get p=0.01, with 5 tests BH thresholds are 0.01, 0.02, 0.03, 0.04, 0.05
    # p=0.01 <= (1/5)*0.05 = 0.01 -> yes for first; all survive since all p equal
    assert result.metadata["n_survive_bh"] == 5
    assert result.metadata["bh_survival_rate"] == pytest.approx(1.0)
    assert result.metadata["passed"] is True


# -- _approximate_p_value ---------------------------------------------------

def test_approximate_p_value_identical_values_returns_one():
    p = _approximate_p_value(0.5, 0.5, 100, None, None)
    assert p == pytest.approx(1.0)


def test_approximate_p_value_uses_ci_for_se():
    # CI [0.80, 1.00] -> SE = 0.20 / (2 * 1.96) ~= 0.051
    p = _approximate_p_value(0.90, 0.50, 100, 0.80, 1.00)
    assert p is not None
    assert p < 0.05  # large difference relative to SE


def test_approximate_p_value_returns_none_for_single_sample():
    p = _approximate_p_value(0.90, 0.50, 1, None, None)
    assert p is None


def test_approximate_p_value_small_diff_large_se():
    # Small difference, wide CI -> high p-value
    p = _approximate_p_value(0.51, 0.50, 10, 0.01, 1.01)
    assert p is not None
    assert p > 0.05
