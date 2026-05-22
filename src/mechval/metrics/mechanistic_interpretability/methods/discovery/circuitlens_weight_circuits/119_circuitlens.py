"""CircuitLens Weight-Based Circuit Recovery (C20)
Paper: Golimblevskaia et al. (2026). ICLR 2026.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     C20 — CircuitLens Weight-Based Circuit Recovery
Categories:     discovery
Validity layer: Construct
Criteria:       C2 Structural Correspondence, C5 Convergent Validity
Establishes:    Whether feature-level circuits recovered from weight
                connectivity alone (no activations) agree with circuits
                derived from activation-based methods (activation patching,
                sparse feature circuits)
Requires:       CPU or GPU, model, artifact adapter
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Implements the CircuitLens approach (Golimblevskaia et al., ICLR 2026).
For each feature at layer L:

1. Extract the feature's decoder direction from the artifact (SAE/transcoder).
2. Project that direction through the next layer's encoder weights to get
   downstream connectivity — a weight-derived circuit edge.
3. Threshold connectivity scores to produce a sparse weight-derived circuit
   (set of downstream features connected above threshold).
4. Compute an activation-derived circuit via gradient-based attribution
   (feature activation x gradient of logit diff) on task prompts.
5. Measure agreement between the two circuits as Jaccard similarity of
   the top-connected feature sets.

The key insight is that weight connectivity alone — without running any
activations — can recover meaningful circuit structure, providing
structural convergent validity for activation-based circuit discovery.

Pass condition: weight_activation_circuit_overlap > 0.2

Usage:
    uv run python 119_circuitlens.py --tasks ioi --n-prompts 50
    uv run python 119_circuitlens.py --tasks ioi sva --device cpu
"""

import numpy as np
import torch

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    InstrumentInfo,
    generate_prompts,
    get_circuit_info,
    get_token_ids,
    load_model,
    log,
    parse_common_args,
    save_incremental,
    save_results,
)

INSTRUMENT_INFO = InstrumentInfo(
    name="CircuitLens Weight-Based Circuit Recovery",
    paper_ref="Golimblevskaia et al. ICLR 2026",
    paper_cite=(
        "Golimblevskaia et al. 2026, CircuitLens: Weight-Based Circuit "
        "Recovery for Mechanistic Interpretability"
    ),
    description=(
        "Recovers feature-level circuits from weight connectivity alone "
        "(no activations). For each feature at layer L, computes downstream "
        "connectivity by projecting its decoder direction through the next "
        "layer's encoder weights. Agreement between weight-derived and "
        "activation-derived circuits provides structural convergent validity."
    ),
    category="discovery",
    tier="construct",
    origin="ICLR 2026",
)


def _get_weight_circuit(
    artifact,
    model,
    hook_name: str,
    n_features: int,
    connectivity_threshold: float,
) -> dict[int, set[int]]:
    """Derive a circuit from weight connectivity alone.

    For each of the top n_features (by decoder norm) at the artifact's
    hook layer, project its decoder direction through the next layer's
    encoder to find strongly connected downstream features.

    Returns a dict mapping source feature index to the set of downstream
    feature indices with connectivity above the threshold.
    """
    dirs = artifact.directions()
    if dirs.ndim == 3:
        dirs = dirs.mean(dim=0)  # (n_features, d_model)

    n_total = dirs.shape[0]
    n_features = min(n_features, n_total)

    # Select features by decoder norm (proxy for importance)
    norms = dirs.norm(dim=1)
    top_indices = torch.argsort(norms, descending=True)[:n_features].cpu().tolist()

    # Get the hook layer so we can look at the next layer's projection
    hook_layer = _hook_to_layer(hook_name)
    if hook_layer is None or hook_layer + 1 >= model.cfg.n_layers:
        # Fall back: use self-connectivity within the same dictionary
        connectivity = dirs @ dirs.T  # (n_total, n_total)
        conn_norms = connectivity.abs()
        # Normalize rows
        row_max = conn_norms.max(dim=1, keepdim=True).values.clamp(min=1e-8)
        conn_norms = conn_norms / row_max

        weight_circuit: dict[int, set[int]] = {}
        for src in top_indices:
            row = conn_norms[src]
            connected = (row > connectivity_threshold).nonzero(as_tuple=True)[0]
            # Exclude self-connection
            weight_circuit[src] = {
                int(c) for c in connected.cpu().tolist() if c != src
            }
        return weight_circuit

    # Project decoder directions through next layer's MLP input or attention
    # weight to get downstream connectivity
    next_layer = hook_layer + 1
    block = model.blocks[next_layer]

    # Use the MLP input weight as the "encoder" for the next layer
    # W_in: (d_model, d_mlp) — projects residual stream into MLP
    if hasattr(block, "mlp") and hasattr(block.mlp, "W_in"):
        W_next = block.mlp.W_in.detach()  # (d_model, d_mlp)
    else:
        # Fallback to self-connectivity
        W_next = dirs.T  # (d_model, n_total)

    # Downstream connectivity: decoder_dir @ W_next gives how strongly
    # each source feature connects to each downstream dimension
    # Then project back through the artifact's encoder to get feature-level
    # connectivity: (decoder @ W_next) @ encoder.T
    downstream_proj = dirs @ W_next  # (n_total, d_mlp or n_total)
    # Project back into feature space via encoder (dirs.T acts as encoder)
    connectivity = downstream_proj @ downstream_proj.T  # (n_total, n_total)
    conn_norms = connectivity.abs()
    row_max = conn_norms.max(dim=1, keepdim=True).values.clamp(min=1e-8)
    conn_norms = conn_norms / row_max

    weight_circuit: dict[int, set[int]] = {}
    for src in top_indices:
        row = conn_norms[src]
        connected = (row > connectivity_threshold).nonzero(as_tuple=True)[0]
        weight_circuit[src] = {
            int(c) for c in connected.cpu().tolist() if c != src
        }
    return weight_circuit


def _get_activation_circuit(
    model,
    artifact,
    prompts,
    correct_ids: list[int],
    incorrect_ids: list[int],
    hook_name: str,
    n_features: int,
    connectivity_threshold: float,
) -> dict[int, set[int]]:
    """Derive a circuit from activation-based attribution.

    Uses gradient-based attribution: for each prompt, compute
    feature_activation * gradient_of_logit_diff w.r.t. feature.
    Features with high attribution are connected in the activation circuit.
    """
    n = min(len(prompts), len(correct_ids), len(incorrect_ids))
    total_attr = None

    for i in range(n):
        tokens = model.to_tokens(prompts[i].text)
        model.zero_grad()

        cache_dict = {}

        def cache_hook(act, hook, _name=hook_name):
            cache_dict[_name] = act
            act.retain_grad()
            return act

        with torch.enable_grad():
            logits = model.run_with_hooks(
                tokens,
                fwd_hooks=[(hook_name, cache_hook)],
            )
            logit_diff = (
                logits[0, -1, correct_ids[i]] - logits[0, -1, incorrect_ids[i]]
            )
            logit_diff.backward()

        act = cache_dict.get(hook_name)
        if act is None or act.grad is None:
            continue

        with torch.no_grad():
            feat_acts = artifact.activations(model, tokens, hook_name)
            grad = act.grad[0, -1]  # (d_model,)
            dirs = artifact.directions()
            if dirs.ndim == 3:
                dirs = dirs.mean(dim=0)
            feat_grad = grad @ dirs.T  # (n_features,)
            feat_importance = (feat_acts[0, -1] * feat_grad).abs()

            if total_attr is None:
                total_attr = torch.zeros_like(feat_importance)
            total_attr += feat_importance

        model.zero_grad()

    if total_attr is None:
        return {}

    total_attr /= max(n, 1)

    # Select top features by attribution
    n_total = total_attr.shape[0]
    n_features = min(n_features, n_total)
    top_indices = torch.argsort(total_attr, descending=True)[:n_features].cpu().tolist()

    # Build activation circuit: features above threshold are connected
    attr_np = total_attr.cpu().numpy()
    attr_max = attr_np.max() if attr_np.max() > 0 else 1.0
    attr_normalized = attr_np / attr_max

    activation_circuit: dict[int, set[int]] = {}
    for src in top_indices:
        connected = set()
        for j in top_indices:
            if j != src and attr_normalized[j] > connectivity_threshold:
                connected.add(j)
        activation_circuit[src] = connected

    return activation_circuit


def _jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def _hook_to_layer(hook_name: str) -> int | None:
    parts = hook_name.split(".")
    for j, part in enumerate(parts):
        if part == "blocks" and j + 1 < len(parts):
            try:
                return int(parts[j + 1])
            except ValueError:
                pass
    return None


def run_circuitlens(
    model,
    tasks: list[str] | None = None,
    n_prompts: int = 50,
    artifact=None,
    hook_name: str | None = None,
    n_features: int = 50,
    connectivity_threshold: float = 0.1,
) -> list[EvalResult]:
    if artifact is None:
        log("  WARNING: no artifact adapter provided, skipping CircuitLens")
        return []

    if tasks is None:
        tasks = CIRCUIT_TASKS

    tokenizer = model.tokenizer
    results = []

    effective_hook = hook_name or artifact.manifest.hook_point
    if not effective_hook:
        effective_hook = "blocks.0.hook_resid_pre"

    for task in tasks:
        circuit, all_heads, all_edges = get_circuit_info(task)
        if circuit is None or not all_heads:
            log(f"  {task}: no circuit, skipping")
            continue

        prompts = generate_prompts(task, tokenizer, n_prompts)
        if not prompts:
            log(f"  {task}: no prompts, skipping")
            continue

        correct_ids, incorrect_ids = get_token_ids(prompts, tokenizer)
        if not correct_ids:
            continue

        log(f"  {task}: {len(all_heads)} circuit heads, {len(prompts)} prompts")

        layers_in_circuit = sorted({L for L, _ in all_heads})
        task_overlaps = []

        for layer in layers_in_circuit:
            layer_hook = effective_hook.replace("blocks.0", f"blocks.{layer}")

            # Weight-derived circuit
            weight_circ = _get_weight_circuit(
                artifact, model, layer_hook, n_features, connectivity_threshold,
            )

            # Activation-derived circuit
            activation_circ = _get_activation_circuit(
                model, artifact, prompts, correct_ids, incorrect_ids,
                layer_hook, n_features, connectivity_threshold,
            )

            if not weight_circ or not activation_circ:
                log(f"    L{layer}: empty circuit, skipping")
                continue

            # Compute per-feature Jaccard and average
            common_sources = set(weight_circ.keys()) & set(activation_circ.keys())
            if not common_sources:
                # Use all sources from both circuits for overlap
                all_weight_features = set()
                for neighbors in weight_circ.values():
                    all_weight_features.update(neighbors)
                all_weight_features.update(weight_circ.keys())

                all_activation_features = set()
                for neighbors in activation_circ.values():
                    all_activation_features.update(neighbors)
                all_activation_features.update(activation_circ.keys())

                overlap = _jaccard(all_weight_features, all_activation_features)
            else:
                jaccards = []
                for src in common_sources:
                    j = _jaccard(weight_circ[src], activation_circ[src])
                    jaccards.append(j)
                overlap = float(np.mean(jaccards))

            passed = bool(overlap > 0.2)
            log(
                f"    L{layer}: overlap={overlap:.4f} "
                f"[{'PASS' if passed else 'FAIL'}]"
            )

            task_overlaps.append({
                "layer": layer,
                "hook_name": layer_hook,
                "weight_activation_circuit_overlap": float(overlap),
                "n_weight_sources": len(weight_circ),
                "n_activation_sources": len(activation_circ),
                "n_common_sources": len(common_sources) if common_sources else 0,
                "passed": passed,
            })

        if task_overlaps:
            mean_overlap = float(
                np.mean([t["weight_activation_circuit_overlap"] for t in task_overlaps])
            )
            passed = bool(mean_overlap > 0.2)

            results.append(EvalResult(
                metric_id="C20.circuitlens_weight_circuits",
                value=mean_overlap,
                n_samples=len(prompts),
                instrument_info=INSTRUMENT_INFO,
                metadata={
                    "task": task,
                    "weight_activation_circuit_overlap": mean_overlap,
                    "passed": passed,
                    "threshold": 0.2,
                    "n_features": n_features,
                    "connectivity_threshold": connectivity_threshold,
                    "n_layers_evaluated": len(task_overlaps),
                    "per_layer": task_overlaps,
                },
            ))

    return results


def main():
    parser = parse_common_args("C20: CircuitLens Weight-Based Circuit Recovery")
    parser.add_argument(
        "--hook", default=None, help="Hook point for artifact activations",
    )
    parser.add_argument(
        "--artifact-type", default="sae",
        choices=["sae", "transcoder", "crosscoder", "factor_bank"],
        help="Artifact adapter type",
    )
    parser.add_argument(
        "--artifact-path", default=None,
        help="Path or release ID for artifact",
    )
    parser.add_argument(
        "--sae-id", default=None, help="SAE ID (for SAELens artifacts)",
    )
    parser.add_argument(
        "--n-features", type=int, default=50,
        help="Number of top features to include in circuits",
    )
    parser.add_argument(
        "--connectivity-threshold", type=float, default=0.1,
        help="Threshold for weight connectivity to count as an edge",
    )
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    artifact = None
    if args.artifact_type == "sae" and args.artifact_path:
        from mechval.lib.artifacts import SAEAdapter
        artifact = SAEAdapter.from_pretrained(
            release=args.artifact_path,
            sae_id=args.sae_id or "",
            hook_point=args.hook or "",
        )
    elif args.artifact_type == "transcoder" and args.artifact_path:
        from mechval.lib.artifacts import TranscoderAdapter
        artifact = TranscoderAdapter.from_pretrained(
            release=args.artifact_path,
            sae_id=args.sae_id or "",
            hook_point=args.hook or "",
        )

    log("=" * 60)
    log("C20: CIRCUITLENS WEIGHT-BASED CIRCUIT RECOVERY")
    log("=" * 60)

    out = args.out or "119_circuitlens.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_circuitlens(
            model, [task], artifact=artifact,
            n_prompts=args.n_prompts, hook_name=args.hook,
            n_features=args.n_features,
            connectivity_threshold=args.connectivity_threshold,
        )
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)
            p = "PASS" if r.metadata["passed"] else "FAIL"
            log(
                f"  {task}: overlap="
                f"{r.metadata['weight_activation_circuit_overlap']:.4f}  [{p}]"
            )

    save_results(results, out)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
