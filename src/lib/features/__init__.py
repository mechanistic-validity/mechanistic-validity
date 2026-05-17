"""Weight-space feature extraction for circuit discovery.

Modules:
  directions  — shared embedding-space directions (ICA + cluster-pair) and ZCA whitening
  heads       — 109 features per attention head from W_Q, W_K, W_V, W_O, W_E
  neurons     — ~160 features per MLP neuron from W_in, W_out, W_E, W_U
  composition — cross-component interaction matrices (head composition, MLP chains)
  formula     — greedy linear formula search + bootstrap stability analysis
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import torch


class ModelLike(Protocol):
    """Minimal interface for a HookedTransformer-like model.

    Anything with these attributes works — TransformerLens, our
    FactorizedHookedTransformer, or a lightweight mock for testing.
    """
    @property
    def W_Q(self) -> torch.Tensor: ...   # (n_layers, n_heads, d_head, d_model)
    @property
    def W_K(self) -> torch.Tensor: ...
    @property
    def W_V(self) -> torch.Tensor: ...
    @property
    def W_O(self) -> torch.Tensor: ...   # (n_layers, n_heads, d_head, d_model)
    @property
    def W_E(self) -> torch.Tensor: ...   # (d_vocab, d_model)
    @property
    def W_U(self) -> torch.Tensor: ...   # (d_model, d_vocab)
    @property
    def W_in(self) -> torch.Tensor: ...  # (n_layers, d_model, d_mlp)
    @property
    def W_out(self) -> torch.Tensor: ... # (n_layers, d_mlp, d_model)
    @property
    def b_in(self) -> torch.Tensor: ...  # (n_layers, d_mlp)


@dataclass
class DirectionSet:
    """Named unit vectors in d_model space + whitening transform."""
    directions: dict[str, torch.Tensor]
    whitening: torch.Tensor
    direction_names: list[str] = field(init=False)

    def __post_init__(self) -> None:
        self.direction_names = sorted(self.directions.keys())

    def whitened(self, name: str) -> torch.Tensor:
        d_w = self.whitening @ self.directions[name]
        return d_w / (d_w.norm() + 1e-10)

    def stacked(self, device: torch.device | str = "cpu") -> torch.Tensor:
        return torch.stack([self.directions[n].to(device) for n in self.direction_names])

    def stacked_whitened(self, device: torch.device | str = "cpu") -> torch.Tensor:
        return torch.stack([self.whitened(n).to(device) for n in self.direction_names])


@dataclass
class HeadFeatures:
    """Result of head feature extraction: 109 features per (layer, head) pair."""
    features: dict[tuple[int, int], dict[str, float]]
    feature_names: list[str]
    n_layers: int
    n_heads: int

    def to_matrix(self) -> np.ndarray:
        n = self.n_layers * self.n_heads
        mat = np.zeros((n, len(self.feature_names)), dtype=np.float32)
        for (l, h), fdict in self.features.items():
            idx = l * self.n_heads + h
            for fi, fname in enumerate(self.feature_names):
                mat[idx, fi] = fdict.get(fname, 0.0)
        return mat


@dataclass
class NeuronFeatures:
    """Result of neuron feature extraction: ~160 features per (layer, neuron) pair."""
    matrix: np.ndarray              # (n_neurons, n_features)
    feature_names: list[str]
    metadata: list[dict[str, Any]]  # per-neuron metadata (layer, neuron, top tokens, etc.)

    @property
    def n_neurons(self) -> int:
        return self.matrix.shape[0]

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    def feature_vector(self, layer: int, neuron: int, d_mlp: int) -> np.ndarray:
        return self.matrix[layer * d_mlp + neuron]
