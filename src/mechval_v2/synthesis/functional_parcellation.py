"""Functional Parcellation (S01)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Meta-protocol:  Consumes results from other protocols
Categories:     meta, parcellation
Validity layer: Construct + Internal
Establishes:    Convergent multi-signal functional grouping of circuit components
Requires:       CPU, protocol results as input
Source:         Glasser et al. 2016 (HCP brain parcellation), part10
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Groups attention heads (and optionally neurons) into functional parcels
based on convergent agreement across multiple independent signals. Each
signal produces an independent clustering; final parcels require
agreement across a configurable threshold of signals.

Signals:
  1. Weight features (norms, concentration, rank, token-level biases)
  2. Causal importance (activation patching, EAP, DAS-IIA)
  3. Structural similarity (CKA, weight-space distance)
  4. Behavioral knockout profiles (per-task knockout effect)
  5. Graph topology (PageRank, betweenness, hub scores)
  6. Cross-task generalization (how task-specific vs. shared each head is)

Adapted from: part10_brain_region_analogy_generalization/build_multimodal_parcellation.py

Usage:
    uv run python functional_parcellation.py --results-dir ./sweep_results/
    uv run python functional_parcellation.py --results-json modal_sweep_results.json
"""
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    get_circuit_heads,
    get_circuit_info,
    log,
    parse_common_args,
    save_results,
)

PROTOCOL_ID = "S01"
PROTOCOL_NAME = "Functional Parcellation (Convergent Multi-Signal)"

GPT2_HEADS = [(layer, head) for layer in range(12) for head in range(12)]
N_HEADS = len(GPT2_HEADS)


@dataclass
class Signal:
    name: str
    features: np.ndarray  # (n_heads, n_features)
    distance_metric: str = "euclidean"


def _build_weight_signal(model) -> Signal:
    features = np.zeros((N_HEADS, 6))
    for i, (layer, head) in enumerate(GPT2_HEADS):
        W_Q = model.W_Q[layer, head].detach().float().cpu().numpy()
        W_K = model.W_K[layer, head].detach().float().cpu().numpy()
        W_V = model.W_V[layer, head].detach().float().cpu().numpy()
        W_O = model.W_O[layer, head].detach().float().cpu().numpy()

        features[i, 0] = np.linalg.norm(W_Q)
        features[i, 1] = np.linalg.norm(W_K)
        features[i, 2] = np.linalg.norm(W_V)
        features[i, 3] = np.linalg.norm(W_O)

        QK = W_Q @ W_K.T
        s = np.linalg.svd(QK, compute_uv=False)
        features[i, 4] = s[0] / (s.sum() + 1e-10)

        OV = W_O @ W_V
        s_ov = np.linalg.svd(OV, compute_uv=False)
        features[i, 5] = s_ov[0] / (s_ov.sum() + 1e-10)

    mu = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True) + 1e-10
    features = (features - mu) / std

    return Signal(name="weight", features=features, distance_metric="euclidean")


def _build_causal_signal(protocol_results: list[dict]) -> Signal | None:
    causal_ids = {"activation_patching", "eap", "das_iia", "sigma_ablation"}
    features = np.zeros((N_HEADS, len(causal_ids)))
    head_to_idx = {h: i for i, h in enumerate(GPT2_HEADS)}
    found_any = False

    for col_idx, metric_id in enumerate(sorted(causal_ids)):
        for result in protocol_results:
            if result.get("status") != "success":
                continue
            for mname, evals in result.get("metrics", {}).items():
                for ev in evals:
                    if ev.get("metric_id", "").endswith(metric_id):
                        meta = ev.get("metadata", {}) if isinstance(ev, dict) else {}
                        head_scores = meta.get("head_scores", {})
                        for hkey, score in head_scores.items():
                            parsed = _parse_head_key(hkey)
                            if parsed and parsed in head_to_idx:
                                features[head_to_idx[parsed], col_idx] = abs(score) if isinstance(score, (int, float)) else 0.0
                                found_any = True

    if not found_any:
        return None
    mu = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True) + 1e-10
    features = (features - mu) / std
    return Signal(name="causal", features=features, distance_metric="cosine")


def _build_circuit_membership_signal(tasks: list[str]) -> Signal:
    features = np.zeros((N_HEADS, len(tasks)))
    head_to_idx = {h: i for i, h in enumerate(GPT2_HEADS)}
    for t_idx, task in enumerate(tasks):
        heads = get_circuit_heads(task)
        if heads:
            for h in heads:
                if h in head_to_idx:
                    features[head_to_idx[h], t_idx] = 1.0
    return Signal(name="circuit_membership", features=features, distance_metric="jaccard")


def _build_graph_topology_signal(model) -> Signal:
    features = np.zeros((N_HEADS, 3))
    head_to_idx = {h: i for i, h in enumerate(GPT2_HEADS)}

    for task in CIRCUIT_TASKS:
        _, _, edges = get_circuit_info(task)
        if not edges:
            continue
        in_degree = {}
        out_degree = {}
        for sl, sh, rl, rh in edges:
            sender = (sl, sh)
            receiver = (rl, rh)
            out_degree[sender] = out_degree.get(sender, 0) + 1
            in_degree[receiver] = in_degree.get(receiver, 0) + 1
        for h in set(list(in_degree.keys()) + list(out_degree.keys())):
            if h in head_to_idx:
                idx = head_to_idx[h]
                features[idx, 0] += in_degree.get(h, 0)
                features[idx, 1] += out_degree.get(h, 0)
                features[idx, 2] += in_degree.get(h, 0) + out_degree.get(h, 0)

    mu = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True) + 1e-10
    features = (features - mu) / std
    return Signal(name="graph_topology", features=features, distance_metric="euclidean")


def _parse_head_key(key: str) -> tuple[int, int] | None:
    try:
        if key.startswith("L") and "H" in key:
            parts = key[1:].split("H")
            return int(parts[0]), int(parts[1])
        if "." in key:
            parts = key.split(".")
            return int(parts[0]), int(parts[1])
    except (ValueError, IndexError):
        pass
    return None


def _cluster_signal(signal: Signal, n_clusters: int = 14) -> np.ndarray:
    if signal.distance_metric == "jaccard":
        dists = pdist(signal.features, metric="jaccard")
        dists = np.nan_to_num(dists, nan=1.0)
    elif signal.distance_metric == "cosine":
        dists = pdist(signal.features, metric="cosine")
        dists = np.nan_to_num(dists, nan=1.0)
    else:
        dists = pdist(signal.features, metric="euclidean")

    Z = linkage(dists, method="ward" if signal.distance_metric == "euclidean" else "average")
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")
    return labels


def _rsa_matrix(clusterings: dict[str, np.ndarray]) -> np.ndarray:
    names = sorted(clusterings.keys())
    n = len(names)
    rsa = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                rsa[i, j] = 1.0
            else:
                a = clusterings[names[i]]
                b = clusterings[names[j]]
                rdm_a = squareform(pdist(a.reshape(-1, 1), metric="hamming"))
                rdm_b = squareform(pdist(b.reshape(-1, 1), metric="hamming"))
                upper_a = rdm_a[np.triu_indices(len(rdm_a), k=1)]
                upper_b = rdm_b[np.triu_indices(len(rdm_b), k=1)]
                r, _ = spearmanr(upper_a, upper_b)
                rsa[i, j] = r
    return rsa


def _convergent_parcellation(clusterings: dict[str, np.ndarray],
                              min_agreement: int = 3) -> np.ndarray:
    n_signals = len(clusterings)
    n_heads = next(iter(clusterings.values())).shape[0]

    co_cluster = np.zeros((n_heads, n_heads))
    for labels in clusterings.values():
        for i in range(n_heads):
            for j in range(i + 1, n_heads):
                if labels[i] == labels[j]:
                    co_cluster[i, j] += 1
                    co_cluster[j, i] += 1

    agreement = co_cluster >= min_agreement
    np.fill_diagonal(agreement, True)

    parcel_labels = np.full(n_heads, -1, dtype=int)
    current_parcel = 0
    for i in range(n_heads):
        if parcel_labels[i] >= 0:
            continue
        members = np.where(agreement[i])[0]
        unassigned = members[parcel_labels[members] < 0]
        if len(unassigned) > 0:
            parcel_labels[unassigned] = current_parcel
            current_parcel += 1

    return parcel_labels


def _silhouette_score(features: np.ndarray, labels: np.ndarray) -> float:
    unique = np.unique(labels)
    if len(unique) < 2:
        return 0.0

    dists = squareform(pdist(features, metric="euclidean"))
    n = len(labels)
    sils = np.zeros(n)

    for i in range(n):
        same = labels == labels[i]
        same[i] = False
        if same.sum() == 0:
            sils[i] = 0.0
            continue
        a_i = dists[i, same].mean()

        b_i = np.inf
        for c in unique:
            if c == labels[i]:
                continue
            other = labels == c
            if other.sum() > 0:
                b_i = min(b_i, dists[i, other].mean())

        sils[i] = (b_i - a_i) / max(a_i, b_i, 1e-10)

    return float(sils.mean())


def run_parcellation(model=None, tasks: list[str] | None = None,
                     device: str = "cpu",
                     protocol_results: list[dict] | None = None,
                     n_clusters: int = 14,
                     min_agreement: int = 3) -> list[EvalResult]:
    if tasks is None:
        tasks = list(CIRCUIT_TASKS)

    log(f"Building signals for {N_HEADS} heads...")
    signals = {}

    signals["circuit_membership"] = _build_circuit_membership_signal(tasks)
    log(f"  circuit_membership: {signals['circuit_membership'].features.shape}")

    signals["graph_topology"] = _build_graph_topology_signal(model)
    log(f"  graph_topology: {signals['graph_topology'].features.shape}")

    if model is not None:
        signals["weight"] = _build_weight_signal(model)
        log(f"  weight: {signals['weight'].features.shape}")

    if protocol_results:
        causal = _build_causal_signal(protocol_results)
        if causal is not None:
            signals["causal"] = causal
            log(f"  causal: {signals['causal'].features.shape}")

    log(f"\nClustering {len(signals)} signals (k={n_clusters})...")
    clusterings = {}
    for name, signal in signals.items():
        labels = _cluster_signal(signal, n_clusters=n_clusters)
        clusterings[name] = labels
        log(f"  {name}: {len(np.unique(labels))} clusters")

    log(f"\nRSA matrix (second-order Spearman)...")
    rsa = _rsa_matrix(clusterings)
    signal_names = sorted(clusterings.keys())
    for i, name_i in enumerate(signal_names):
        row = [f"{rsa[i, j]:.2f}" for j in range(len(signal_names))]
        log(f"  {name_i:25s} {' '.join(row)}")

    log(f"\nConvergent parcellation (min_agreement={min_agreement})...")
    parcel_labels = _convergent_parcellation(clusterings, min_agreement=min_agreement)
    n_parcels = len(np.unique(parcel_labels))
    log(f"  {n_parcels} parcels found")

    parcel_sizes = {}
    for p in np.unique(parcel_labels):
        members = [GPT2_HEADS[i] for i in range(N_HEADS) if parcel_labels[i] == p]
        parcel_sizes[int(p)] = len(members)
        log(f"  Parcel {p}: {len(members)} heads — {members[:5]}{'...' if len(members) > 5 else ''}")

    all_features = np.concatenate([s.features for s in signals.values()], axis=1)
    sil = _silhouette_score(all_features, parcel_labels)
    log(f"\n  Silhouette score: {sil:.4f}")

    circuit_overlap = {}
    for task in tasks:
        task_heads = get_circuit_heads(task)
        if not task_heads:
            continue
        head_to_idx = {h: i for i, h in enumerate(GPT2_HEADS)}
        task_parcels = set()
        for h in task_heads:
            if h in head_to_idx:
                task_parcels.add(int(parcel_labels[head_to_idx[h]]))
        circuit_overlap[task] = {
            "n_heads": len(task_heads),
            "parcels": sorted(task_parcels),
            "n_parcels": len(task_parcels),
        }
        log(f"  {task}: {len(task_heads)} heads across {len(task_parcels)} parcels")

    head_assignments = {
        f"L{layer}H{head}": int(parcel_labels[i])
        for i, (layer, head) in enumerate(GPT2_HEADS)
    }

    results = [
        EvalResult(
            metric_id="S01.n_parcels",
            value=float(n_parcels),
            n_samples=N_HEADS,
            metadata={
                "n_parcels": n_parcels,
                "n_signals": len(signals),
                "signal_names": signal_names,
                "n_clusters_per_signal": n_clusters,
                "min_agreement": min_agreement,
                "silhouette": sil,
                "parcel_sizes": parcel_sizes,
                "circuit_overlap": circuit_overlap,
                "rsa_matrix": rsa.tolist(),
                "head_assignments": head_assignments,
            },
        ),
        EvalResult(
            metric_id="S01.silhouette",
            value=sil,
            n_samples=N_HEADS,
            metadata={"n_parcels": n_parcels},
        ),
        EvalResult(
            metric_id="S01.mean_rsa",
            value=float(rsa[np.triu_indices(len(rsa), k=1)].mean()),
            n_samples=len(signals) * (len(signals) - 1) // 2,
            metadata={
                "signal_names": signal_names,
                "rsa_matrix": rsa.tolist(),
            },
        ),
    ]

    return results


def run_protocol(model, tasks, n_prompts=40, device="cpu", run_cals=False,
                 protocol_results=None):
    from protocols import ProtocolResult
    t0 = time.time()
    evals = run_parcellation(
        model, tasks, device=device, protocol_results=protocol_results,
    )
    result = ProtocolResult(
        protocol_id=PROTOCOL_ID,
        protocol_name=PROTOCOL_NAME,
        tasks=tasks,
        elapsed_seconds=time.time() - t0,
    )
    result.metrics["parcellation"] = evals
    return result


def summarize(result) -> str:
    lines = [f"=== {PROTOCOL_NAME} ==="]
    for ev in result.metrics.get("parcellation", []):
        lines.append(f"  {ev.metric_id}: {ev.value:.4f}")
    return "\n".join(lines)


def main():
    parser = parse_common_args("S01: Functional Parcellation")
    parser.add_argument("--results-json", type=str, default=None)
    parser.add_argument("--n-clusters", type=int, default=14)
    parser.add_argument("--min-agreement", type=int, default=3)
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS

    protocol_results = None
    if args.results_json:
        with open(args.results_json) as f:
            protocol_results = json.load(f)

    from mechval.metrics.common import load_model
    model = load_model("gpt2", args.device)

    log("=" * 60)
    log("S01: FUNCTIONAL PARCELLATION")
    log("=" * 60)

    results = run_parcellation(
        model, tasks, device=args.device,
        protocol_results=protocol_results,
        n_clusters=args.n_clusters,
        min_agreement=args.min_agreement,
    )

    out = args.out or "meta_p1_parcellation.json"
    save_results(results, out)

    log(f"\nDone. {len(results)} results.")
    for r in results:
        log(f"  {r.metric_id}: {r.value:.4f}")


if __name__ == "__main__":
    main()
