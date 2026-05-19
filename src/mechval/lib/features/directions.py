"""Shared embedding-space direction computation.

Produces 15 unit vectors in d_model space:
  - 10 ICA components from the top-k token embeddings
  - 5 cluster-pair difference vectors (K-means on embeddings, largest centroid gaps)

Also provides ZCA whitening of the embedding covariance.

Both heads.py and neurons.py use these as alignment probes.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import FastICA

from mechval.lib.features import DirectionSet


def compute_directions(
    W_E: torch.Tensor,
    n_ica: int = 10,
    n_clusters: int = 20,
    n_cluster_pairs: int = 5,
    top_k_tokens: int = 5000,
) -> dict[str, torch.Tensor]:
    """Compute embedding-space directions via ICA + cluster-pair diffs.

    Args:
        W_E: token embedding matrix (d_vocab, d_model).
        n_ica: number of ICA components to extract.
        n_clusters: K-means cluster count over embeddings.
        n_cluster_pairs: number of largest centroid-pair directions to keep.
        top_k_tokens: use only the first top_k_tokens rows of W_E.

    Returns:
        dict mapping direction name -> unit vector (d_model,) on cpu.
    """
    we = W_E[:top_k_tokens].float().cpu().numpy()
    we_norm = (we - we.mean(0)) / (we.std(0) + 1e-10)

    ica = FastICA(n_components=n_ica, random_state=42, max_iter=500)
    ica.fit(we_norm[:2000])
    directions: dict[str, torch.Tensor] = {}
    for i in range(n_ica):
        d = torch.tensor(ica.components_[i], dtype=torch.float32)
        directions[f"ica{i}"] = d / (d.norm() + 1e-10)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(we_norm)
    centroids = []
    for c in range(n_clusters):
        members = np.where(labels == c)[0]
        centroids.append(torch.tensor(we[members].mean(0), dtype=torch.float32))

    pairs: list[tuple[float, int, int, torch.Tensor]] = []
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            d = centroids[i] - centroids[j]
            pairs.append((d.norm().item(), i, j, d))
    pairs.sort(reverse=True)
    for _, i, j, d in pairs[:n_cluster_pairs]:
        directions[f"clust{i}v{j}"] = d / d.norm()

    return directions


def compute_whitening(W_E: torch.Tensor) -> torch.Tensor:
    """ZCA whitening matrix from embedding covariance.

    Returns:
        (d_model, d_model) whitening matrix on cpu.
    """
    we = W_E.float().cpu()
    we_centered = we - we.mean(0)
    cov = (we_centered.T @ we_centered) / len(we)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    eigvals = eigvals.clamp(min=1e-6)
    return eigvecs @ torch.diag(1.0 / eigvals.sqrt()) @ eigvecs.T


def compute_direction_set(
    W_E: torch.Tensor,
    n_ica: int = 10,
    n_clusters: int = 20,
    n_cluster_pairs: int = 5,
    top_k_tokens: int = 5000,
) -> DirectionSet:
    """Convenience: compute both directions and whitening, return a DirectionSet."""
    directions = compute_directions(W_E, n_ica, n_clusters, n_cluster_pairs, top_k_tokens)
    whitening = compute_whitening(W_E)
    return DirectionSet(directions=directions, whitening=whitening)
