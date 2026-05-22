"""Metric: Dual Mechanism Discriminant Validity --- intrinsic vs. prompted directions

Paper: Han, Lim, Kong, Jo (2025). "Dual Mechanisms of Value Expression:
Intrinsic vs. Prompted Values in LLMs." NeurIPS 2025 Mechanistic
Interpretability Workshop / ICML 2026. arXiv:2509.24319.

Decomposes representation directions for a behavioral construct (e.g.,
safety, values, truthfulness) into intrinsic (baseline) and prompted
(instruction-following) mechanisms. Measures discriminant validity by
testing whether the two directions are genuinely distinct (not just
different names for the same direction) and whether each steers
independently after removing the shared component. This is the most
precise empirical instantiation of C4 Discriminant Validity applied
to safety-relevant constructs.

Dual Mechanism Discriminant Validity (Evaluation EX23)
=============================================
Instrument:     EX23 --- Dual Mechanism Discriminant Validity
Categories:     evaluation
Validity layer: Construct
Criteria:       C4 Discriminant Validity
Establishes:    Whether intrinsic and prompted mechanisms for the same
                construct are genuinely distinct (not conflated)
Requires:       CPU or GPU, model
=============================================

Core logic:
1. Collect activations under baseline conditions (intrinsic direction):
   model responses to neutral prompts about the target construct.
2. Collect activations under prompted conditions (prompted direction):
   model responses to system-prompted variants of the same queries.
3. Extract mean difference directions for each mechanism.
4. Compute discriminant_separation = 1 - |cos(intrinsic, prompted)|.
5. Decompose each direction into shared and independent components.
6. Test independent steering: apply residual of each direction after
   removing shared component, measure behavioral effect.

Pass condition: discriminant_separation > 0.2;
                independent_steering_effect > 0.1

Usage:
    uv run python 125_dual_mechanism.py --model gpt2 --device cpu
    uv run python 125_dual_mechanism.py --n-samples 30
"""

import numpy as np
import torch
import torch.nn.functional as F

from mechval.metrics.common import (
    EvalResult,
    InstrumentInfo,
    load_model,
    log,
    parse_common_args,
    save_results,
)

INSTRUMENT_INFO = InstrumentInfo(
    name="Dual Mechanism Discriminant Validity",
    paper_ref="Han et al. NeurIPS 2025 Workshop / ICML 2026",
    paper_cite=(
        "Han, Lim, Kong, Jo 2025, "
        "Dual Mechanisms of Value Expression: Intrinsic vs. Prompted "
        "Values in LLMs (NeurIPS 2025 Workshop, arXiv:2509.24319)"
    ),
    description=(
        "Decomposes behavioral construct directions into intrinsic "
        "(baseline) and prompted (instruction-following) mechanisms. "
        "Measures discriminant validity by testing directional separation "
        "and independent steering effects."
    ),
    category="evaluation",
    tier="emerging",
    origin="external",
)

SEPARATION_THRESHOLD = 0.2
STEERING_THRESHOLD = 0.1

# Construct-specific stimulus sets
_CONSTRUCTS: dict[str, dict[str, list[str]]] = {
    "helpfulness": {
        "neutral": [
            "What is the capital of France?",
            "How does photosynthesis work?",
            "Explain the water cycle.",
            "What year did World War II end?",
            "Describe how a rainbow forms.",
            "What is the speed of light?",
            "How do vaccines work?",
            "What causes earthquakes?",
            "Who invented the telephone?",
            "What is the largest planet?",
        ],
        "prompted_prefix": "You are an extremely helpful assistant. Answer thoroughly and accurately. ",
        "anti_prefix": "You are unhelpful. Give minimal, vague responses. ",
    },
    "formality": {
        "neutral": [
            "Tell me about dogs.",
            "What do you think about rain?",
            "Describe your favorite color.",
            "Talk about breakfast.",
            "What is a good hobby?",
            "Tell me about the ocean.",
            "Describe a sunset.",
            "What makes music good?",
            "Talk about friendship.",
            "Describe a forest.",
        ],
        "prompted_prefix": "You are a formal academic writer. Use precise, scholarly language. ",
        "anti_prefix": "You are very casual and informal. Use slang and abbreviations. ",
    },
}


@torch.no_grad()
def _collect_representations(
    model,
    texts: list[str],
    hook_layer: int,
) -> torch.Tensor:
    """Collect residual stream representations for a list of texts.

    Returns:
        reps: (n_texts, d_model) last-position residual stream at the given layer.
    """
    hook_name = f"blocks.{hook_layer}.hook_resid_post"
    reps = []
    for text in texts:
        tokens = model.to_tokens(text)
        _, cache = model.run_with_cache(
            tokens, names_filter=lambda n: n == hook_name
        )
        reps.append(cache[hook_name][0, -1].cpu())
    return torch.stack(reps)


def _extract_direction(
    positive_reps: torch.Tensor, negative_reps: torch.Tensor
) -> torch.Tensor:
    """Extract a direction as the mean difference between two representation sets.

    Returns:
        direction: (d_model,) normalized direction vector.
    """
    diff = positive_reps.mean(dim=0) - negative_reps.mean(dim=0)
    return diff / (diff.norm() + 1e-8)


def _decompose_directions(
    dir_a: torch.Tensor, dir_b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decompose two directions into shared and independent components.

    Returns:
        shared: normalized shared component
        residual_a: dir_a with shared component removed
        residual_b: dir_b with shared component removed
    """
    # Shared component: projection of each onto the other
    cos_sim = F.cosine_similarity(dir_a.unsqueeze(0), dir_b.unsqueeze(0)).item()

    # Shared direction: normalized mean of the two
    shared = (dir_a + dir_b)
    shared = shared / (shared.norm() + 1e-8)

    # Remove shared component from each
    proj_a = (dir_a @ shared) * shared
    proj_b = (dir_b @ shared) * shared
    residual_a = dir_a - proj_a
    residual_b = dir_b - proj_b

    # Normalize residuals
    residual_a = residual_a / (residual_a.norm() + 1e-8)
    residual_b = residual_b / (residual_b.norm() + 1e-8)

    return shared, residual_a, residual_b


@torch.no_grad()
def _test_steering_effect(
    model,
    test_texts: list[str],
    direction: torch.Tensor,
    hook_layer: int,
    scale: float = 5.0,
) -> float:
    """Measure the steering effect of adding a direction at a hook point.

    Returns:
        effect: mean relative change in top-5 logit values.
    """
    hook_name = f"blocks.{hook_layer}.hook_resid_post"
    perturbation = direction * scale

    effects = []
    for text in test_texts[:5]:
        tokens = model.to_tokens(text)

        # Clean output
        clean_logits = model(tokens)
        clean_top = clean_logits[0, -1].topk(10).indices
        clean_vals = clean_logits[0, -1, clean_top]

        # Steered output
        def steer_hook(value, hook):
            value[:, -1, :] += perturbation.to(value.device)
            return value

        steered_logits = model.run_with_hooks(
            tokens, fwd_hooks=[(hook_name, steer_hook)]
        )
        steered_vals = steered_logits[0, -1, clean_top]

        denom = clean_vals.abs().mean().item()
        if denom > 1e-8:
            effect = (steered_vals - clean_vals).abs().mean().item() / denom
        else:
            effect = 0.0
        effects.append(effect)

    return float(np.mean(effects)) if effects else 0.0


def run_dual_mechanism_discriminant(
    model,
    constructs: list[str] | None = None,
    n_samples: int = 10,
    hook_layer: int | None = None,
    steering_scale: float = 5.0,
) -> list[EvalResult]:
    """Run dual mechanism discriminant validity diagnostic.

    For each behavioral construct, extracts intrinsic and prompted
    directions, measures their separation, and tests independent steering.

    Args:
        model: HookedTransformer instance.
        constructs: list of construct names to test (default: all).
        n_samples: number of samples per condition.
        hook_layer: layer for representation capture (default: middle).
        steering_scale: magnitude of steering vector for effect test.

    Returns:
        List of EvalResult, one per construct plus aggregate.
    """
    if constructs is None:
        constructs = list(_CONSTRUCTS.keys())
    if hook_layer is None:
        hook_layer = model.cfg.n_layers // 2

    log(f"  Dual mechanism discriminant validity at layer {hook_layer}")
    log(f"  Constructs: {constructs}")

    results = []
    all_separations = []
    all_steering_effects = []

    for construct_name in constructs:
        if construct_name not in _CONSTRUCTS:
            log(f"    {construct_name}: unknown construct, skipping")
            continue

        construct = _CONSTRUCTS[construct_name]
        neutral_texts = construct["neutral"][:n_samples]
        prompted_texts = [
            construct["prompted_prefix"] + t for t in neutral_texts
        ]
        anti_texts = [
            construct["anti_prefix"] + t for t in neutral_texts
        ]

        # Collect representations
        neutral_reps = _collect_representations(model, neutral_texts, hook_layer)
        prompted_reps = _collect_representations(model, prompted_texts, hook_layer)
        anti_reps = _collect_representations(model, anti_texts, hook_layer)

        # Extract directions
        intrinsic_dir = _extract_direction(neutral_reps, anti_reps)
        prompted_dir = _extract_direction(prompted_reps, anti_reps)

        # Discriminant separation
        cos_sim = F.cosine_similarity(
            intrinsic_dir.unsqueeze(0), prompted_dir.unsqueeze(0)
        ).item()
        separation = 1.0 - abs(cos_sim)

        # Decompose into shared and independent
        shared, residual_intrinsic, residual_prompted = _decompose_directions(
            intrinsic_dir, prompted_dir
        )

        # Test independent steering effects
        intrinsic_effect = _test_steering_effect(
            model, neutral_texts, residual_intrinsic, hook_layer,
            scale=steering_scale,
        )
        prompted_effect = _test_steering_effect(
            model, neutral_texts, residual_prompted, hook_layer,
            scale=steering_scale,
        )
        mean_independent_effect = (intrinsic_effect + prompted_effect) / 2

        all_separations.append(separation)
        all_steering_effects.append(mean_independent_effect)

        passed_sep = separation > SEPARATION_THRESHOLD
        passed_steer = mean_independent_effect > STEERING_THRESHOLD
        passed = passed_sep and passed_steer

        log(f"    {construct_name}: separation={separation:.4f}, "
            f"cos_sim={cos_sim:.4f}, "
            f"independent_effect={mean_independent_effect:.4f} "
            f"({'PASS' if passed else 'FAIL'})")

        results.append(EvalResult(
            metric_id="EX23.dual_mechanism_discriminant",
            value=separation,
            n_samples=len(neutral_texts),
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": construct_name,
                "construct": construct_name,
                "discriminant_separation": separation,
                "cosine_similarity": cos_sim,
                "intrinsic_independent_effect": intrinsic_effect,
                "prompted_independent_effect": prompted_effect,
                "mean_independent_effect": mean_independent_effect,
                "hook_layer": hook_layer,
                "steering_scale": steering_scale,
                "passed_separation": passed_sep,
                "passed_steering": passed_steer,
                "passed": passed,
                "threshold_separation": SEPARATION_THRESHOLD,
                "threshold_steering": STEERING_THRESHOLD,
            },
        ))

    # Aggregate
    if all_separations:
        agg_sep = float(np.mean(all_separations))
        agg_steer = float(np.mean(all_steering_effects))
        agg_passed = agg_sep > SEPARATION_THRESHOLD and agg_steer > STEERING_THRESHOLD

        log(f"  Aggregate: separation={agg_sep:.4f}, "
            f"independent_effect={agg_steer:.4f} "
            f"({'PASS' if agg_passed else 'FAIL'})")

        results.append(EvalResult(
            metric_id="EX23.dual_mechanism_discriminant",
            value=agg_sep,
            n_samples=sum(r.n_samples for r in results),
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": "aggregate",
                "mean_separation": agg_sep,
                "mean_independent_effect": agg_steer,
                "n_constructs": len(all_separations),
                "per_construct": {
                    r.metadata["task"]: {
                        "separation": r.metadata["discriminant_separation"],
                        "effect": r.metadata["mean_independent_effect"],
                    }
                    for r in results if r.metadata.get("task") != "aggregate"
                },
                "passed": agg_passed,
            },
        ))

    return results


def main():
    parser = parse_common_args("EX23: Dual Mechanism Discriminant Validity")
    parser.add_argument("--constructs", nargs="+", default=None,
                        help="Constructs to test (default: all)")
    parser.add_argument("--n-samples", type=int, default=10,
                        help="Samples per condition")
    parser.add_argument("--hook-layer", type=int, default=None,
                        help="Layer for representation capture (default: middle)")
    parser.add_argument("--steering-scale", type=float, default=5.0,
                        help="Magnitude of steering vector for effect test")
    args = parser.parse_args()

    model = load_model(args.model, args.device)

    log("=" * 60)
    log("EX23: DUAL MECHANISM DISCRIMINANT VALIDITY")
    log("=" * 60)

    results = run_dual_mechanism_discriminant(
        model,
        constructs=args.constructs,
        n_samples=args.n_samples,
        hook_layer=args.hook_layer,
        steering_scale=args.steering_scale,
    )

    out = args.out or "125_dual_mechanism.json"
    save_results(results, out)
    log(f"\nDone. {len(results)} results.")


if __name__ == "__main__":
    main()
