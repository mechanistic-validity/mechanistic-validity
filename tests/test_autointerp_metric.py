"""Tests for the AutoInterp proxy metric."""
from __future__ import annotations

import importlib

import torch
import pytest

autointerp = importlib.import_module("mechval.metrics.mechanistic_interpretability.methods.evaluation.autointerp.EX3_autointerp")


def test_activation_sparsity_all_zeros():
    acts = torch.zeros(2, 5, 32)
    assert autointerp.compute_activation_sparsity(acts) == pytest.approx(1.0)


def test_activation_sparsity_all_active():
    acts = torch.ones(2, 5, 32)
    assert autointerp.compute_activation_sparsity(acts) == pytest.approx(0.0)


def test_activation_sparsity_half_active():
    acts = torch.zeros(2, 5, 32)
    acts[:, :, :16] = 1.0
    sparsity = autointerp.compute_activation_sparsity(acts)
    assert sparsity == pytest.approx(0.5)


def test_monosemanticity_uniform():
    acts = torch.randn(100, 50, 32)
    mono = autointerp.compute_monosemanticity(acts)
    assert 0.0 < mono < 2.0


def test_kurtosis_gaussian_near_zero():
    torch.manual_seed(0)
    acts = torch.randn(1000, 50, 8)
    kurt = autointerp.compute_kurtosis(acts)
    assert kurt.shape == (8,)
    assert kurt.abs().mean() < 1.0


def test_kurtosis_sparse_positive():
    acts = torch.zeros(100, 50, 8)
    acts[0, 0, :] = 100.0
    kurt = autointerp.compute_kurtosis(acts)
    assert (kurt > 0).all()
