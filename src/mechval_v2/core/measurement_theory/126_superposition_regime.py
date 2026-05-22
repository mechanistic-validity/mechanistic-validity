"""Metric: Superposition Regime Diagnostic --- weak vs. strong superposition

Paper: Liu, Liu, Gore (2025). "Superposition Yields Robust Neural
Scaling." NeurIPS 2025 Oral, Best Paper Runner-Up. arXiv:2505.10465.

Estimates whether a model layer operates in the weak or strong
superposition regime by measuring feature packing and interference.
In the strong regime (L ~ 1/m), models pack more features than
dimensions with irreducible interference, meaning no decomposition
(SAE or otherwise) can recover "the true features" --- they are one
of many valid decompositions. In the weak regime (few features, low
interference), recovery is feasible. The regime classification
informs how to interpret all downstream validity results: in strong
superposition, SAE features need utility-based validity criteria
rather than ground-truth recovery claims.

Superposition Regime Diagnostic (Measurement Theory M13)
=============================================
Instrument:     M13 --- Superposition Regime Diagnostic
Categories:     measurement theory
Validity layer: Measurement
Criteria:       M6 Construct Coverage
Establishes:    Whether a model operates in weak or strong superposition,
                informing the interpretability of decomposition results
Requires:       CPU or GPU, model
=============================================

Core logic:
1. Run model on diverse text, capturing residual stream at each layer.
2. Compute effective rank (number of significant singular values) at
   each layer via participation ratio of singular values.
3. Packing ratio = effective_rank / d_model. Values >> 1 indicate
   strong superposition (more active directions than dimensions).
4. Interference score = mean absolute pairwise cosine similarity
   between top-k principal components (high = strong interference).
5. Classify regime: weak (packing <= 0.8, low interference),
   transition (0.8 < packing <= 1.2), strong (packing > 1.2 or
   high interference).

This is a diagnostic (not pass/fail). Reports regime classification
and quantitative indicators.

Usage:
    uv run python 126_superposition_regime.py --model gpt2 --device cpu
    uv run python 126_superposition_regime.py --n-samples 200
"""

import numpy as np
import torch

from mechval.metrics.common import (
    EvalResult,
    InstrumentInfo,
    load_model,
    log,
    parse_common_args,
    save_results,
)

INSTRUMENT_INFO = InstrumentInfo(
    name="Superposition Regime Diagnostic",
    paper_ref="Liu, Liu, Gore NeurIPS 2025",
    paper_cite=(
        "Liu, Liu, Gore 2025, "
        "Superposition Yields Robust Neural Scaling "
        "(NeurIPS 2025 Best Paper Runner-Up, arXiv:2505.10465)"
    ),
    description=(
        "Estimates whether a model operates in weak or strong "
        "superposition by measuring feature packing ratio and "
        "interference. In strong superposition, no decomposition "
        "can recover unique 'true features' --- utility-based "
        "validity criteria are required."
    ),
    category="evaluation",
    tier="established",
    origin="external",
)

# Sample texts for diverse activations
_SAMPLE_TEXTS = [
    "The capital of France is Paris, which is known for the Eiffel Tower.",
    "Machine learning models can be trained on large datasets to make predictions.",
    "The quick brown fox jumps over the lazy dog near the river bank.",
    "In quantum mechanics, particles can exist in superposition of states.",
    "Shakespeare wrote many plays including Hamlet and Romeo and Juliet.",
    "The stock market experienced significant volatility during the crisis.",
    "Photosynthesis converts sunlight into chemical energy in plant cells.",
    "The ancient Romans built roads that connected their vast empire.",
    "DNA contains the genetic instructions for the development of organisms.",
    "Climate change is caused by increased greenhouse gas emissions.",
    "The neural network has multiple layers of interconnected neurons.",
    "Democracy is a system of government where citizens vote for leaders.",
    "The speed of light in a vacuum is approximately 300 million meters per second.",
    "Mozart composed over 600 works during his short lifetime.",
    "Algorithms are step-by-step procedures for solving computational problems.",
    "The Amazon rainforest produces about 20 percent of the world's oxygen.",
    "Gravity is the force that attracts objects with mass toward each other.",
    "The invention of the printing press revolutionized the spread of information.",
    "Bacteria are single-celled organisms that can be found everywhere on Earth.",
    "The Pythagorean theorem relates the sides of a right triangle.",
]


def _participation_ratio(singular_values: torch.Tensor) -> float:
    """Compute participation ratio of singular values.

    PR = (sum s_i^2)^2 / (sum s_i^4)

    This gives the effective number of significant components.
    """
    s2 = singular_values ** 2
    s4 = singular_values ** 4
    numerator = s2.sum().item() ** 2
    denominator = s4.sum().item()
    if denominator < 1e-12:
        return 0.0
    return numerator / denominator


def _pairwise_cosine_interference(
    components: torch.Tensor, n_top: int = 20
) -> float:
    """Compute mean absolute pairwise cosine similarity between top components.

    Args:
        components: (n_components, d_model) principal components (Vt from SVD).
        n_top: number of top components to compare.

    Returns:
        Mean absolute cosine similarity between all pairs.
    """
    n = min(n_top, components.shape[0])
    if n < 2:
        return 0.0

    top = components[:n]
    # Normalize
    top_normed = top / (top.norm(dim=1, keepdim=True) + 1e-8)
    # Pairwise cosine similarity matrix
    sim_matrix = top_normed @ top_normed.T

    # Extract upper triangle (exclude diagonal)
    mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
    pairwise_sims = sim_matrix[mask].abs()

    return pairwise_sims.mean().item() if pairwise_sims.numel() > 0 else 0.0


def _classify_regime(
    packing_ratio: float, interference: float
) -> str:
    """Classify superposition regime based on packing and interference.

    Returns one of: "weak", "transition", "strong"
    """
    if packing_ratio > 1.2 or (packing_ratio > 0.8 and interference > 0.15):
        return "strong"
    if packing_ratio > 0.8 or interference > 0.1:
        return "transition"
    return "weak"


@torch.no_grad()
def run_superposition_regime(
    model,
    n_samples: int = 200,
    n_top_components: int = 20,
) -> list[EvalResult]:
    """Run superposition regime diagnostic.

    Estimates whether the model operates in weak, transition, or strong
    superposition at each layer by measuring packing ratio and interference.

    Args:
        model: HookedTransformer instance.
        n_samples: number of text samples for activation collection.
        n_top_components: number of top components for interference calculation.

    Returns:
        List of EvalResult, one per layer plus aggregate.
    """
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model

    log(f"  Superposition regime diagnostic: {n_layers} layers, d_model={d_model}")

    # Collect activations at every layer
    hook_names = [f"blocks.{i}.hook_resid_post" for i in range(n_layers)]

    # Use sample texts, cycling if needed
    texts = []
    for i in range(n_samples):
        texts.append(_SAMPLE_TEXTS[i % len(_SAMPLE_TEXTS)])

    # Collect: (n_layers, n_samples, d_model)
    all_acts = {name: [] for name in hook_names}

    for text in texts:
        tokens = model.to_tokens(text)
        _, cache = model.run_with_cache(
            tokens, names_filter=lambda n: n in hook_names
        )
        for name in hook_names:
            # Mean pool over sequence positions
            act = cache[name][0].mean(dim=0).cpu()  # (d_model,)
            all_acts[name].append(act)

    results = []
    layer_regimes = []
    layer_packings = []
    layer_interferences = []

    for layer_idx in range(n_layers):
        hook_name = hook_names[layer_idx]
        acts = torch.stack(all_acts[hook_name])  # (n_samples, d_model)

        # Center activations
        acts_centered = acts - acts.mean(dim=0, keepdim=True)

        # SVD
        U, S, Vt = torch.linalg.svd(acts_centered, full_matrices=False)

        # Participation ratio (effective rank)
        eff_rank = _participation_ratio(S)
        packing_ratio = eff_rank / d_model

        # Interference score
        interference = _pairwise_cosine_interference(Vt, n_top=n_top_components)

        # Classify regime
        regime = _classify_regime(packing_ratio, interference)

        layer_regimes.append(regime)
        layer_packings.append(packing_ratio)
        layer_interferences.append(interference)

        log(f"    Layer {layer_idx}: regime={regime}, "
            f"packing={packing_ratio:.4f}, interference={interference:.4f}, "
            f"eff_rank={eff_rank:.1f}")

        results.append(EvalResult(
            metric_id="M13.superposition_regime",
            value=packing_ratio,
            n_samples=n_samples,
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": f"layer_{layer_idx}",
                "layer": layer_idx,
                "regime": regime,
                "packing_ratio": packing_ratio,
                "effective_rank": eff_rank,
                "d_model": d_model,
                "interference_score": interference,
                "n_top_components": n_top_components,
            },
        ))

    # Aggregate: overall model regime
    regime_counts = {"weak": 0, "transition": 0, "strong": 0}
    for r in layer_regimes:
        regime_counts[r] += 1

    # Model regime = majority across layers
    model_regime = max(regime_counts, key=regime_counts.get)
    mean_packing = float(np.mean(layer_packings))
    mean_interference = float(np.mean(layer_interferences))

    log(f"  Model regime: {model_regime} "
        f"(weak={regime_counts['weak']}, "
        f"transition={regime_counts['transition']}, "
        f"strong={regime_counts['strong']})")
    log(f"  Mean packing: {mean_packing:.4f}, "
        f"mean interference: {mean_interference:.4f}")

    results.append(EvalResult(
        metric_id="M13.superposition_regime",
        value=mean_packing,
        n_samples=n_samples * n_layers,
        instrument_info=INSTRUMENT_INFO,
        metadata={
            "task": "aggregate",
            "model_regime": model_regime,
            "regime_counts": regime_counts,
            "mean_packing_ratio": mean_packing,
            "mean_interference": mean_interference,
            "per_layer_regimes": layer_regimes,
            "per_layer_packings": layer_packings,
            "per_layer_interferences": layer_interferences,
            "d_model": d_model,
            "n_layers": n_layers,
        },
    ))

    return results


def main():
    parser = parse_common_args("M13: Superposition Regime Diagnostic")
    parser.add_argument("--n-samples", type=int, default=200,
                        help="Number of text samples for activation collection")
    parser.add_argument("--n-top-components", type=int, default=20,
                        help="Top components for interference calculation")
    args = parser.parse_args()

    model = load_model(args.model, args.device)

    log("=" * 60)
    log("M13: SUPERPOSITION REGIME DIAGNOSTIC")
    log("=" * 60)

    results = run_superposition_regime(
        model,
        n_samples=args.n_samples,
        n_top_components=args.n_top_components,
    )

    out = args.out or "126_superposition_regime.json"
    save_results(results, out)
    log(f"\nDone. {len(results)} results.")


if __name__ == "__main__":
    main()
