import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats as sp_stats

from mechanistic_validity.metrics.common import EvalResult, load_model

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechanistic_validity" / "calibrations" / "inter_rater" / "59_inter_rater.py"
)
_spec = importlib.util.spec_from_file_location("_ir_59", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)

_cohens_kappa = _mod._cohens_kappa
_kendalls_w = _mod._kendalls_w
rank_by_patching = _mod.rank_by_patching
rank_by_dla = _mod.rank_by_dla
rank_by_ov_norm = _mod.rank_by_ov_norm

TASK = "ioi"


def test_cohens_kappa_perfect_agreement():
    y1 = np.array([1, 1, 0, 0, 1])
    y2 = np.array([1, 1, 0, 0, 1])
    assert _cohens_kappa(y1, y2) == pytest.approx(1.0)


def test_cohens_kappa_no_agreement():
    y1 = np.array([1, 1, 0, 0])
    y2 = np.array([0, 0, 1, 1])
    kappa = _cohens_kappa(y1, y2)
    assert kappa < 0


def test_cohens_kappa_empty_arrays():
    assert _cohens_kappa(np.array([]), np.array([])) == pytest.approx(0.0)


def test_cohens_kappa_all_same_class():
    y1 = np.array([1, 1, 1, 1])
    y2 = np.array([1, 1, 1, 1])
    assert _cohens_kappa(y1, y2) == pytest.approx(1.0)


def test_cohens_kappa_known_value():
    y1 = np.array([1, 1, 1, 0, 0, 0, 1, 0, 0, 0])
    y2 = np.array([1, 1, 0, 0, 0, 0, 1, 1, 0, 0])
    n = len(y1)
    agree = np.sum(y1 == y2)
    p_o = agree / n
    p1_pos = np.mean(y1)
    p2_pos = np.mean(y2)
    p_e = p1_pos * p2_pos + (1 - p1_pos) * (1 - p2_pos)
    expected = (p_o - p_e) / (1 - p_e)
    assert _cohens_kappa(y1, y2) == pytest.approx(expected)


def test_kendalls_w_perfect_concordance():
    rankings = np.array([
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5],
    ])
    w = _kendalls_w(rankings)
    assert w == pytest.approx(1.0)


def test_kendalls_w_no_concordance():
    rankings = np.array([
        [1, 2, 3],
        [3, 2, 1],
    ])
    w = _kendalls_w(rankings)
    assert w < 1.0


def test_kendalls_w_single_rater():
    rankings = np.array([[1, 2, 3, 4]])
    w = _kendalls_w(rankings)
    assert w == pytest.approx(0.0)


def test_kendalls_w_single_item():
    rankings = np.array([[1], [1], [1]])
    w = _kendalls_w(rankings)
    assert w == pytest.approx(0.0)


def test_kendalls_w_two_raters_opposite():
    rankings = np.array([
        [1, 2, 3, 4, 5],
        [5, 4, 3, 2, 1],
    ])
    w = _kendalls_w(rankings)
    assert w < 0.5


def test_kendalls_w_bounded_zero_one():
    rng = np.random.RandomState(42)
    for _ in range(10):
        m, n = rng.randint(2, 6), rng.randint(3, 8)
        rankings = np.column_stack([
            sp_stats.rankdata(rng.randn(n)) for _ in range(m)
        ]).T
        w = _kendalls_w(rankings)
        assert 0.0 <= w <= 1.0 + 1e-10


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


def test_rank_by_patching_returns_correct_shape(gpt2_model):
    from mechanistic_validity.metrics.common import (
        generate_prompts, get_circuit_heads, get_token_ids,
    )
    prompts = generate_prompts(TASK, gpt2_model.tokenizer, 3)
    correct_ids, incorrect_ids = get_token_ids(prompts, gpt2_model.tokenizer)
    effects = rank_by_patching(gpt2_model, prompts, correct_ids, incorrect_ids)
    n_total = gpt2_model.cfg.n_layers * gpt2_model.cfg.n_heads
    assert effects.shape == (n_total,)


def test_rank_by_dla_returns_correct_shape(gpt2_model):
    from mechanistic_validity.metrics.common import (
        generate_prompts, get_token_ids,
    )
    prompts = generate_prompts(TASK, gpt2_model.tokenizer, 3)
    correct_ids, incorrect_ids = get_token_ids(prompts, gpt2_model.tokenizer)
    effects = rank_by_dla(gpt2_model, prompts, correct_ids, incorrect_ids)
    n_total = gpt2_model.cfg.n_layers * gpt2_model.cfg.n_heads
    assert effects.shape == (n_total,)


def test_rank_by_ov_norm_returns_correct_shape(gpt2_model):
    from mechanistic_validity.metrics.common import (
        generate_prompts, get_token_ids,
    )
    prompts = generate_prompts(TASK, gpt2_model.tokenizer, 3)
    correct_ids, incorrect_ids = get_token_ids(prompts, gpt2_model.tokenizer)
    effects = rank_by_ov_norm(gpt2_model, correct_ids, incorrect_ids)
    n_total = gpt2_model.cfg.n_layers * gpt2_model.cfg.n_heads
    assert effects.shape == (n_total,)


def test_rank_by_ov_norm_all_nonnegative(gpt2_model):
    from mechanistic_validity.metrics.common import (
        generate_prompts, get_token_ids,
    )
    prompts = generate_prompts(TASK, gpt2_model.tokenizer, 3)
    correct_ids, incorrect_ids = get_token_ids(prompts, gpt2_model.tokenizer)
    effects = rank_by_ov_norm(gpt2_model, correct_ids, incorrect_ids)
    assert np.all(effects >= 0)
