import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechanistic_validity" / "metrics"
    / "behavioral" / "generalization_gap" / "81_error_boundary_analysis.py"
)
_spec = importlib.util.spec_from_file_location("error_boundary_81", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["error_boundary_81"] = _mod
_spec.loader.exec_module(_mod)

DifficultyPrompt = _mod.DifficultyPrompt
generate_difficulty_prompts = _mod.generate_difficulty_prompts
compute_boundary_alignment = _mod.compute_boundary_alignment
run_error_boundary_analysis = _mod.run_error_boundary_analysis
DIFFICULTY_TEMPLATES = _mod.DIFFICULTY_TEMPLATES

from mechanistic_validity.metrics.common import EvalResult, load_model

TASK = "ioi"


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


def test_difficulty_prompt_creation():
    dp = DifficultyPrompt("text", " correct", " incorrect", "easy")
    assert dp.text == "text"
    assert dp.target_correct == " correct"
    assert dp.target_incorrect == " incorrect"
    assert dp.difficulty == "easy"


def test_generate_difficulty_prompts_ioi():
    prompts = generate_difficulty_prompts("ioi", None, n_per_level=5)
    assert len(prompts) > 0
    difficulties = {p.difficulty for p in prompts}
    assert "easy" in difficulties
    assert "medium" in difficulties
    assert "hard" in difficulties


def test_generate_difficulty_prompts_sva():
    prompts = generate_difficulty_prompts("sva", None, n_per_level=5)
    assert len(prompts) > 0
    difficulties = {p.difficulty for p in prompts}
    assert "easy" in difficulties


def test_compute_boundary_alignment_perfect_alignment():
    per_prompt = [
        {"model_correct": True, "faithfulness": 0.8},
        {"model_correct": True, "faithfulness": 0.9},
        {"model_correct": False, "faithfulness": 0.1},
        {"model_correct": False, "faithfulness": 0.0},
    ]
    alignment = compute_boundary_alignment(per_prompt, faithfulness_threshold=0.3)
    assert alignment == pytest.approx(1.0)


def test_compute_boundary_alignment_no_alignment():
    per_prompt = [
        {"model_correct": True, "faithfulness": 0.1},
        {"model_correct": True, "faithfulness": 0.0},
        {"model_correct": False, "faithfulness": 0.8},
        {"model_correct": False, "faithfulness": 0.9},
    ]
    alignment = compute_boundary_alignment(per_prompt, faithfulness_threshold=0.3)
    assert alignment == pytest.approx(0.0)


def test_compute_boundary_alignment_half():
    per_prompt = [
        {"model_correct": True, "faithfulness": 0.8},
        {"model_correct": True, "faithfulness": 0.1},
        {"model_correct": False, "faithfulness": 0.1},
        {"model_correct": False, "faithfulness": 0.8},
    ]
    alignment = compute_boundary_alignment(per_prompt, faithfulness_threshold=0.3)
    assert alignment == pytest.approx(0.5)


def test_compute_boundary_alignment_empty():
    alignment = compute_boundary_alignment([], faithfulness_threshold=0.3)
    assert alignment == pytest.approx(0.0)


def test_compute_boundary_alignment_custom_threshold():
    per_prompt = [
        {"model_correct": True, "faithfulness": 0.5},
        {"model_correct": False, "faithfulness": 0.2},
    ]
    alignment_low = compute_boundary_alignment(per_prompt, faithfulness_threshold=0.3)
    alignment_high = compute_boundary_alignment(per_prompt, faithfulness_threshold=0.6)
    assert alignment_low == pytest.approx(1.0)
    assert alignment_high == pytest.approx(0.5)


def test_run_error_boundary_analysis_returns_results(gpt2_model):
    results = run_error_boundary_analysis(gpt2_model, [TASK], n_prompts=3)
    assert len(results) >= 1
    r = results[0]
    assert isinstance(r, EvalResult)
    assert r.metric_id == "CM2.error_boundary_analysis"
    assert r.n_samples >= 1


def test_run_error_boundary_analysis_metadata_keys(gpt2_model):
    results = run_error_boundary_analysis(gpt2_model, [TASK], n_prompts=3)
    r = results[0]
    expected_keys = {
        "task", "boundary_alignment", "per_difficulty_alignment",
        "difficulty_stats", "n_circuit_heads", "faithfulness_threshold",
        "passed", "threshold",
    }
    assert expected_keys.issubset(r.metadata.keys())


def test_run_error_boundary_analysis_alignment_bounded(gpt2_model):
    results = run_error_boundary_analysis(gpt2_model, [TASK], n_prompts=3)
    r = results[0]
    assert 0.0 <= r.value <= 1.0


def test_run_error_boundary_analysis_passed_flag(gpt2_model):
    results = run_error_boundary_analysis(gpt2_model, [TASK], n_prompts=3)
    r = results[0]
    expected_passed = r.value > 0.60
    assert r.metadata["passed"] == expected_passed


def test_run_error_boundary_analysis_unknown_task_returns_empty(gpt2_model):
    results = run_error_boundary_analysis(gpt2_model, ["nonexistent_task_xyz"], n_prompts=3)
    assert results == []
