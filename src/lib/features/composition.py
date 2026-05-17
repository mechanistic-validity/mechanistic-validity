"""Cross-component weight-space interaction matrices.

Head composition:
  - K-composition: ||W_O[src] @ W_K[dst]||_F reveals which heads feed attention patterns
  - Q-composition: ||W_O[src] @ W_Q[dst]||_F
  - V-composition: ||W_O[src] @ W_V[dst]||_F (OV circuit chaining)

MLP interactions:
  - chains: W_out[L] @ W_in[L+1] top connections + multi-hop paths
  - interaction_graph: W_out[L] @ W_in[L+1] community detection

All computations are weight-only — no forward pass.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


@dataclass
class CompositionMatrix:
    """Frobenius-norm composition scores between all head pairs."""
    matrix: np.ndarray     # (n_src, n_dst) where each entry is a composition score
    src_labels: list[str]  # ["L0H0", "L0H1", ...]
    dst_labels: list[str]
    composition_type: str  # "K", "Q", or "V"


@dataclass
class MLPChain:
    """A multi-hop weight path through adjacent MLP layers."""
    nodes: list[str]       # ["L3N42", "L4N100", "L5N7"]
    strength: float        # product of edge weights


@dataclass
class MLPInteractions:
    """Result of MLP interaction analysis for a layer pair or full model."""
    top_connections: list[dict[str, Any]]
    chains: list[MLPChain]


def compute_head_composition(
    model: Any,
    composition_type: str = "K",
    device: str | torch.device = "cpu",
) -> CompositionMatrix:
    """Compute pairwise head composition matrix.

    For each source head (l_s, h_s) and downstream target head (l_t, h_t) where l_t > l_s,
    compute ||W_O[src] @ W_X[dst]||_F where X is Q, K, or V.

    Args:
        model: HookedTransformer with W_O, W_Q, W_K, W_V.
        composition_type: "K", "Q", or "V".
        device: computation device.

    Returns:
        CompositionMatrix with (n_total_heads, n_total_heads) Frobenius scores.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    n_total = n_layers * n_heads

    W_target_attr = {"K": "W_K", "Q": "W_Q", "V": "W_V"}[composition_type]

    labels = [f"L{l}H{h}" for l in range(n_layers) for h in range(n_heads)]
    matrix = np.zeros((n_total, n_total), dtype=np.float32)

    with torch.no_grad():
        for l_s in range(n_layers):
            for h_s in range(n_heads):
                wo = model.W_O[l_s, h_s].float().to(device)  # (d_head, d_model)
                src_idx = l_s * n_heads + h_s
                for l_t in range(l_s + 1, n_layers):
                    for h_t in range(n_heads):
                        wt = getattr(model, W_target_attr)[l_t, h_t].float().to(device)  # (d_head, d_model)
                        score = (wo @ wt.T).norm().item()
                        dst_idx = l_t * n_heads + h_t
                        matrix[src_idx, dst_idx] = score

    return CompositionMatrix(
        matrix=matrix,
        src_labels=labels,
        dst_labels=labels,
        composition_type=composition_type,
    )


def compute_mlp_interactions(
    model: Any,
    device: str | torch.device = "cpu",
    top_k_per_layer: int = 100,
    max_chains: int = 100,
) -> MLPInteractions:
    """Compute MLP neuron interactions via W_out[L] @ W_in[L+1].

    Finds strongest direct connections between adjacent MLP layers
    and extends them into multi-hop chains.

    Args:
        model: HookedTransformer with W_out, W_in.
        device: computation device.
        top_k_per_layer: how many top connections to keep per layer pair.
        max_chains: maximum number of 3-hop chains to return.

    Returns:
        MLPInteractions with top connections and multi-hop chains.
    """
    n_layers = model.cfg.n_layers
    d_mlp = model.cfg.d_mlp

    top_connections: list[dict[str, Any]] = []
    with torch.no_grad():
        for l in range(n_layers - 1):
            M = (model.W_out[l].float().to(device) @ model.W_in[l + 1].float().to(device))  # (d_mlp, d_mlp)
            flat = M.abs().flatten()
            top_k = min(top_k_per_layer, flat.numel())
            vals, idxs = flat.topk(top_k)
            for v, idx in zip(vals, idxs):
                src = idx.item() // d_mlp
                dst = idx.item() % d_mlp
                top_connections.append({
                    "src": f"L{l}N{src}", "dst": f"L{l+1}N{dst}",
                    "strength": float(M[src, dst].item()),
                })

    top_connections.sort(key=lambda x: abs(x["strength"]), reverse=True)
    top_connections = top_connections[:500]

    edge_dict: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for c in top_connections:
        edge_dict[c["src"]].append((c["dst"], abs(c["strength"])))

    chains: list[MLPChain] = []
    for a_node in edge_dict:
        for b_node, s_ab in edge_dict[a_node]:
            if b_node in edge_dict:
                for c_node, s_bc in edge_dict[b_node]:
                    chains.append(MLPChain(
                        nodes=[a_node, b_node, c_node],
                        strength=s_ab * s_bc,
                    ))
    chains.sort(key=lambda x: x.strength, reverse=True)

    return MLPInteractions(
        top_connections=top_connections,
        chains=chains[:max_chains],
    )
