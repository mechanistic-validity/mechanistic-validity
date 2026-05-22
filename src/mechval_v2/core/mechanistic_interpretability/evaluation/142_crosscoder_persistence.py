"""Crosscoder Cross-Layer Persistence (Evaluation EX32)
Paper: Lindsey et al. (2024). "Crosscoders." transformer-circuits.pub
=============================================
Instrument:     EX32 --- Crosscoder Cross-Layer Persistence
Categories:     evaluation
Validity layer: Construct / Measurement
Criteria:       C2 Structural Plausibility, M2 Invariance
Establishes:    Whether crosscoder features with large decoder magnitude
                across multiple layers actually activate coherently at all
                those layers, or whether cross-layer structure is an
                artifact of the dictionary
Requires:       CPU or GPU, model
=============================================

For crosscoder features claimed to span multiple layers, the decoder
matrix has substantial magnitude at multiple layers. This metric tests
whether those features actually activate coherently across those layers
by measuring activation correlations.

Core logic:
1. For each pair of adjacent layers, collect residual stream activations
   across a corpus of prompts.
2. Extract feature directions (via PCA on concatenated activations across
   layer pairs).
3. For each multi-layer feature direction, compute the Pearson correlation
   of per-position projection magnitudes between layers.
4. Average across features and layer pairs to get cross_layer_coherence.

High coherence means the features truly activate together across layers
(consistent with cross-layer superposition). Low coherence means the
cross-layer structure may be a dictionary artifact.

Pass condition: cross_layer_coherence > 0.5

Usage:
    uv run python 142_crosscoder_persistence.py --model gpt2 --device cpu
    uv run python 142_crosscoder_persistence.py --n-prompts 40 --n-features 20
"""

import numpy as np
import torch
import torch.nn.functional as F

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    InstrumentInfo,
    generate_prompts,
    load_model,
    log,
    parse_common_args,
    save_results,
)

INSTRUMENT_INFO = InstrumentInfo(
    name="Crosscoder Cross-Layer Persistence",
    paper_ref="Lindsey et al. (2024)",
    paper_cite=(
        "Lindsey et al. 2024, "
        "Crosscoders (transformer-circuits.pub)"
    ),
    description=(
        "Tests whether crosscoder features with multi-layer decoder "
        "magnitude actually activate coherently across layers. Measures "
        "per-position activation correlation between layer pairs for "
        "shared feature directions extracted via cross-layer PCA."
    ),
    category="evaluation",
    tier="established",
    origin="external",
)

COHERENCE_THRESHOLD = 0.5


@torch.no_grad()
def _collect_layer_activations(
    model,
    prompts,
    layer: int,
) -> torch.Tensor:
    """Collect residual stream activations at a single layer.

    Returns tensor of shape (total_tokens, d_model).
    """
    hook_name = f"blocks.{layer}.hook_resid_pre"
    all_acts = []
    for p in prompts:
        tokens = model.to_tokens(p.text)
        captured = {}

        def fwd_hook(value, hook, _c=captured):
            _c["act"] = value.detach()
            return value

        model.run_with_hooks(tokens, fwd_hooks=[(hook_name, fwd_hook)])
        if "act" in captured:
            all_acts.append(captured["act"].squeeze(0))

    if not all_acts:
        return torch.zeros(0, model.cfg.d_model, device=model.cfg.device)
    return torch.cat(all_acts, dim=0)


def _extract_cross_layer_directions(
    acts_a: torch.Tensor,
    acts_b: torch.Tensor,
    n_features: int,
) -> torch.Tensor:
    """Extract feature directions that span two layers via SVD on
    concatenated activations.

    Args:
        acts_a: (n_tokens, d_model) activations at layer A.
        acts_b: (n_tokens, d_model) activations at layer B.
        n_features: number of cross-layer directions to extract.

    Returns:
        directions: (n_features, 2, d_model) --- per-layer components
        of each cross-layer feature direction.
    """
    d = acts_a.shape[1]
    # Concatenate along feature dimension: (n_tokens, 2*d_model)
    combined = torch.cat([acts_a, acts_b], dim=1)
    centered = combined - combined.mean(dim=0, keepdim=True)

    k = min(n_features, min(centered.shape) - 1)
    if k < 1:
        return torch.zeros(0, 2, d, device=acts_a.device)

    _, _, Vh = torch.linalg.svd(centered, full_matrices=False)
    # Vh: (min(n, 2d), 2d) --- take top k
    top_dirs = Vh[:k]  # (k, 2*d_model)

    # Split back into per-layer components
    dirs_a = top_dirs[:, :d]  # (k, d_model)
    dirs_b = top_dirs[:, d:]  # (k, d_model)

    # Normalize per-layer components
    dirs_a = F.normalize(dirs_a, dim=1)
    dirs_b = F.normalize(dirs_b, dim=1)

    return torch.stack([dirs_a, dirs_b], dim=1)  # (k, 2, d_model)


def _cross_layer_coherence(
    acts_a: torch.Tensor,
    acts_b: torch.Tensor,
    directions: torch.Tensor,
) -> list[float]:
    """Compute per-position activation correlation between layers for
    each cross-layer feature direction.

    Args:
        acts_a: (n_tokens, d_model) layer A activations.
        acts_b: (n_tokens, d_model) layer B activations.
        directions: (n_features, 2, d_model) cross-layer directions.

    Returns:
        List of Pearson correlations, one per feature.
    """
    correlations = []
    for i in range(directions.shape[0]):
        dir_a = directions[i, 0]  # (d_model,)
        dir_b = directions[i, 1]  # (d_model,)

        # Project activations onto per-layer directions
        proj_a = acts_a @ dir_a  # (n_tokens,)
        proj_b = acts_b @ dir_b  # (n_tokens,)

        # Pearson correlation
        proj_a_centered = proj_a - proj_a.mean()
        proj_b_centered = proj_b - proj_b.mean()

        numer = (proj_a_centered * proj_b_centered).sum()
        denom = torch.sqrt((proj_a_centered ** 2).sum() * (proj_b_centered ** 2).sum())

        if denom.item() < 1e-10:
            correlations.append(0.0)
        else:
            correlations.append((numer / denom).item())

    return correlations


def run_crosscoder_persistence(
    model,
    tasks: list[str] | None = None,
    n_prompts: int = 40,
    n_features: int = 20,
    layer_pairs: list[tuple[int, int]] | None = None,
) -> list[EvalResult]:
    """Compute crosscoder cross-layer persistence across tasks.

    For each task, collects activations at multiple layers, extracts
    cross-layer feature directions, and measures activation coherence
    between layers.

    Args:
        model: HookedTransformer instance.
        tasks: list of task names (default: CIRCUIT_TASKS).
        n_prompts: number of prompts per task.
        n_features: number of cross-layer directions to test.
        layer_pairs: explicit layer pairs to test. Default: consecutive
            pairs around the middle of the model.

    Returns:
        List of EvalResult, one per task plus an aggregate.
    """
    if tasks is None:
        tasks = CIRCUIT_TASKS

    n_layers = model.cfg.n_layers
    if layer_pairs is None:
        mid = n_layers // 2
        start = max(0, mid - 2)
        end = min(n_layers - 1, mid + 2)
        layer_pairs = [(i, i + 1) for i in range(start, end)]
    if not layer_pairs:
        layer_pairs = [(0, min(1, n_layers - 1))]

    log(f"  Crosscoder Persistence: n_features={n_features}, "
        f"n_prompts={n_prompts}, layer_pairs={layer_pairs}")

    results = []
    all_coherence = []

    for task in tasks:
        prompts = generate_prompts(task, model.tokenizer, n_prompts=n_prompts)
        if not prompts:
            log(f"    {task}: no prompts, skipping")
            continue

        task_coherences = []
        pair_details = []

        for layer_a, layer_b in layer_pairs:
            if layer_a >= n_layers or layer_b >= n_layers:
                continue

            acts_a = _collect_layer_activations(model, prompts, layer_a)
            acts_b = _collect_layer_activations(model, prompts, layer_b)

            if acts_a.shape[0] < n_features + 5 or acts_b.shape[0] < n_features + 5:
                log(f"    {task} L{layer_a}-L{layer_b}: insufficient tokens, skipping")
                continue

            # Ensure same number of tokens (should match)
            n_tokens = min(acts_a.shape[0], acts_b.shape[0])
            acts_a = acts_a[:n_tokens]
            acts_b = acts_b[:n_tokens]

            directions = _extract_cross_layer_directions(acts_a, acts_b, n_features)
            if directions.shape[0] == 0:
                continue

            corrs = _cross_layer_coherence(acts_a, acts_b, directions)
            mean_corr = float(np.mean(corrs))
            task_coherences.extend(corrs)

            pair_details.append({
                "layer_a": layer_a,
                "layer_b": layer_b,
                "mean_coherence": mean_corr,
                "n_features": len(corrs),
                "per_feature_coherence": corrs[:10],
            })

        if not task_coherences:
            log(f"    {task}: no valid layer pairs")
            continue

        mean_coherence = float(np.mean(task_coherences))
        std_coherence = float(np.std(task_coherences))
        passed = mean_coherence > COHERENCE_THRESHOLD
        all_coherence.append(mean_coherence)

        log(f"    {task}: cross_layer_coherence={mean_coherence:.4f} "
            f"+/- {std_coherence:.4f} ({'PASS' if passed else 'FAIL'})")

        results.append(EvalResult(
            metric_id="EX32.crosscoder_persistence",
            value=mean_coherence,
            n_samples=len(task_coherences),
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": task,
                "cross_layer_coherence": mean_coherence,
                "cross_layer_coherence_std": std_coherence,
                "n_layer_pairs": len(pair_details),
                "n_features_total": len(task_coherences),
                "passed": passed,
                "threshold": COHERENCE_THRESHOLD,
                "layer_pairs": pair_details,
            },
        ))

    # Aggregate
    if all_coherence:
        agg_mean = float(np.mean(all_coherence))
        agg_std = float(np.std(all_coherence))
        agg_passed = agg_mean > COHERENCE_THRESHOLD
        log(f"  Aggregate: cross_layer_coherence={agg_mean:.4f} "
            f"+/- {agg_std:.4f} ({'PASS' if agg_passed else 'FAIL'})")

        results.append(EvalResult(
            metric_id="EX32.crosscoder_persistence",
            value=agg_mean,
            n_samples=sum(r.n_samples for r in results),
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": "aggregate",
                "cross_layer_coherence": agg_mean,
                "cross_layer_coherence_std": agg_std,
                "n_tasks_evaluated": len(all_coherence),
                "per_task_coherence": {
                    r.metadata["task"]: r.metadata["cross_layer_coherence"]
                    for r in results if r.metadata.get("task") != "aggregate"
                },
                "passed": agg_passed,
                "threshold": COHERENCE_THRESHOLD,
            },
        ))

    return results


def main():
    parser = parse_common_args("EX32: Crosscoder Cross-Layer Persistence")
    parser.add_argument("--n-features", type=int, default=20,
                        help="Number of cross-layer directions to test (default: 20)")
    parser.add_argument("--layer-start", type=int, default=None,
                        help="Starting layer for pairs (default: mid-2)")
    parser.add_argument("--layer-end", type=int, default=None,
                        help="Ending layer for pairs (default: mid+2)")
    args = parser.parse_args()

    model = load_model(args.model, args.device)

    log("=" * 60)
    log("EX32: CROSSCODER CROSS-LAYER PERSISTENCE")
    log("=" * 60)

    layer_pairs = None
    if args.layer_start is not None and args.layer_end is not None:
        layer_pairs = [(i, i + 1) for i in range(args.layer_start, args.layer_end)]

    tasks = args.tasks or CIRCUIT_TASKS
    results = run_crosscoder_persistence(
        model,
        tasks=tasks,
        n_prompts=args.n_prompts,
        n_features=args.n_features,
        layer_pairs=layer_pairs,
    )

    out = args.out or "142_crosscoder_persistence.json"
    save_results(results, out)
    log(f"\nDone. {len(results)} results.")


if __name__ == "__main__":
    main()
