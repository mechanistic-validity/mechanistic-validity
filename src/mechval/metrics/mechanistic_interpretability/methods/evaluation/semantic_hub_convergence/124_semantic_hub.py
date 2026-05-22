"""Metric: Semantic Hub Convergence --- cross-lingual representation agreement

Paper: Wu, Yu, Yogatama, Lu, Kim (2025). "The Semantic Hub Hypothesis:
Language Models Share Semantic Representations Across Languages and
Modalities." ICLR 2025. arXiv:2411.04986.

Measures whether semantically equivalent inputs in different surface forms
(languages, code vs. math, formal vs. informal) converge to similar
representations at intermediate layers. The hub convergence score is the
mean cosine similarity of paired semantically-equivalent inputs at the
layer of maximum convergence. The causal cross-modal effect measures
whether intervening on the hub representation predictably changes output.
This provides the strongest possible C5 convergent validity test: two
completely different input forms producing the same internal representation.

Semantic Hub Convergence (Evaluation EX22)
=============================================
Instrument:     EX22 --- Semantic Hub Convergence
Categories:     evaluation
Validity layer: Construct
Criteria:       C5 Convergent Validity (cross-modal/cross-lingual)
Establishes:    Whether the model uses a shared semantic hub for
                processing equivalent inputs in different surface forms
Requires:       CPU or GPU, model
=============================================

Core logic:
1. Generate paired inputs: same semantic content in different surface
   forms (English/code, formal/informal, arithmetic/words).
2. Run model on each pair, capturing residual stream at every layer.
3. For each layer, compute cosine similarity between paired representations.
4. Hub layer = argmax of mean cosine similarity across pairs.
5. Hub convergence score = mean similarity at the hub layer.
6. Causal test: intervene at hub layer on form A, measure effect on
   form B's output.

Pass condition: hub_convergence_score > 0.5

Usage:
    uv run python 124_semantic_hub.py --model gpt2 --device cpu
    uv run python 124_semantic_hub.py --n-pairs 30
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
    name="Semantic Hub Convergence",
    paper_ref="Wu et al. ICLR 2025",
    paper_cite=(
        "Wu, Yu, Yogatama, Lu, Kim 2025, "
        "The Semantic Hub Hypothesis: Language Models Share Semantic "
        "Representations Across Languages and Modalities "
        "(ICLR 2025, arXiv:2411.04986)"
    ),
    description=(
        "Measures whether semantically equivalent inputs in different "
        "surface forms converge to similar representations at intermediate "
        "layers. The hub convergence score (peak cross-form cosine "
        "similarity) is the primary diagnostic."
    ),
    category="evaluation",
    tier="established",
    origin="external",
)

HUB_CONVERGENCE_THRESHOLD = 0.5

# Semantic pairs: same meaning, different surface form
_SEMANTIC_PAIRS: list[tuple[str, str]] = [
    # Arithmetic: words vs. digits
    ("two plus three equals five", "2 + 3 = 5"),
    ("seven minus four equals three", "7 - 4 = 3"),
    ("ten divided by two equals five", "10 / 2 = 5"),
    ("three times four equals twelve", "3 * 4 = 12"),
    ("eight plus one equals nine", "8 + 1 = 9"),
    # Formal vs. informal
    ("The cat is on the mat", "the cat sits on the mat"),
    ("She went to the store", "she headed to the store"),
    ("He is very happy today", "he is really happy today"),
    ("They are going home now", "they are heading home now"),
    ("The dog ran across the park", "the dog dashed across the park"),
    # Code vs. description
    ("add one to x", "x = x + 1"),
    ("check if x is greater than zero", "if x > 0"),
    ("repeat ten times", "for i in range(10)"),
    ("set y to the value of x", "y = x"),
    ("return the result", "return result"),
    # Negation pairs (should be different, control)
    ("it is raining", "it is not raining"),
    ("the answer is true", "the answer is false"),
]


def _get_semantic_pairs(n_pairs: int) -> list[tuple[str, str, bool]]:
    """Get semantic pairs with labels (True = equivalent, False = control).

    Returns list of (text_a, text_b, is_equivalent).
    """
    pairs = []
    # Equivalent pairs (first 15)
    for a, b in _SEMANTIC_PAIRS[:15]:
        pairs.append((a, b, True))
    # Control pairs (last 2 negation pairs)
    for a, b in _SEMANTIC_PAIRS[15:]:
        pairs.append((a, b, False))
    return pairs[:n_pairs]


@torch.no_grad()
def _collect_layerwise_representations(
    model, text: str
) -> list[torch.Tensor]:
    """Collect residual stream at every layer for the given text.

    Returns list of (d_model,) tensors, one per layer, at last position.
    """
    tokens = model.to_tokens(text)
    hook_names = [
        f"blocks.{i}.hook_resid_post" for i in range(model.cfg.n_layers)
    ]
    _, cache = model.run_with_cache(
        tokens, names_filter=lambda n: n in hook_names
    )
    reps = []
    for name in hook_names:
        reps.append(cache[name][0, -1].cpu())  # (d_model,)
    return reps


def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two vectors."""
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


@torch.no_grad()
def _causal_hub_test(
    model,
    text_a: str,
    text_b: str,
    hub_layer: int,
    perturbation_scale: float = 3.0,
) -> float:
    """Test whether intervening at the hub layer on text_a affects text_b.

    Perturbs text_a's representation at the hub layer in the direction of
    text_b's representation, then measures how much text_b's output changes
    when the same perturbation is applied.

    Returns:
        causal_effect: relative change in output logits for text_b when
                       perturbed at the hub layer.
    """
    hook_name = f"blocks.{hub_layer}.hook_resid_post"

    # Get text_a representation at hub layer
    tokens_a = model.to_tokens(text_a)
    _, cache_a = model.run_with_cache(
        tokens_a, names_filter=lambda n: n == hook_name
    )
    rep_a = cache_a[hook_name][0, -1]  # (d_model,)

    # Get text_b clean output
    tokens_b = model.to_tokens(text_b)
    clean_logits_b = model(tokens_b)
    clean_top = clean_logits_b[0, -1].topk(5).indices

    # Perturb text_b at hub layer in the direction of text_a
    perturbation = rep_a / (rep_a.norm() + 1e-8) * perturbation_scale

    def perturb_hook(value, hook):
        value[:, -1, :] += perturbation.to(value.device)
        return value

    perturbed_logits_b = model.run_with_hooks(
        tokens_b, fwd_hooks=[(hook_name, perturb_hook)]
    )

    # Measure change in top-5 logit values
    clean_vals = clean_logits_b[0, -1, clean_top]
    perturbed_vals = perturbed_logits_b[0, -1, clean_top]

    denom = clean_vals.abs().mean().item()
    if denom < 1e-8:
        return 0.0
    return (perturbed_vals - clean_vals).abs().mean().item() / denom


def run_semantic_hub_convergence(
    model,
    n_pairs: int = 15,
    perturbation_scale: float = 3.0,
) -> list[EvalResult]:
    """Run the semantic hub convergence diagnostic.

    Measures cross-form representation similarity at each layer and
    identifies the hub layer (peak convergence).

    Args:
        model: HookedTransformer instance.
        n_pairs: number of semantic pairs to test.
        perturbation_scale: magnitude for causal intervention test.

    Returns:
        List of EvalResult.
    """
    pairs = _get_semantic_pairs(n_pairs)
    n_layers = model.cfg.n_layers

    log(f"  Semantic hub convergence: {len(pairs)} pairs, {n_layers} layers")

    # Collect layerwise similarities
    equivalent_pairs = [(a, b) for a, b, eq in pairs if eq]
    control_pairs = [(a, b) for a, b, eq in pairs if not eq]

    # Per-layer mean cosine similarity for equivalent pairs
    layer_sims_eq = np.zeros(n_layers)
    layer_sims_ctrl = np.zeros(n_layers)

    for a_text, b_text in equivalent_pairs:
        reps_a = _collect_layerwise_representations(model, a_text)
        reps_b = _collect_layerwise_representations(model, b_text)
        for layer_idx in range(n_layers):
            sim = _cosine_sim(reps_a[layer_idx], reps_b[layer_idx])
            layer_sims_eq[layer_idx] += sim

    if equivalent_pairs:
        layer_sims_eq /= len(equivalent_pairs)

    for a_text, b_text in control_pairs:
        reps_a = _collect_layerwise_representations(model, a_text)
        reps_b = _collect_layerwise_representations(model, b_text)
        for layer_idx in range(n_layers):
            sim = _cosine_sim(reps_a[layer_idx], reps_b[layer_idx])
            layer_sims_ctrl[layer_idx] += sim

    if control_pairs:
        layer_sims_ctrl /= len(control_pairs)

    # Hub layer = peak similarity for equivalent pairs
    hub_layer = int(np.argmax(layer_sims_eq))
    hub_convergence = float(layer_sims_eq[hub_layer])

    # Discrimination: equivalent vs. control similarity at hub layer
    hub_discrimination = float(layer_sims_eq[hub_layer] - layer_sims_ctrl[hub_layer])

    log(f"  Hub layer: {hub_layer}, convergence: {hub_convergence:.4f}, "
        f"discrimination: {hub_discrimination:.4f}")

    # Causal hub test on a subset of equivalent pairs
    causal_effects = []
    for a_text, b_text in equivalent_pairs[:5]:
        effect = _causal_hub_test(
            model, a_text, b_text, hub_layer,
            perturbation_scale=perturbation_scale,
        )
        causal_effects.append(effect)

    mean_causal_effect = float(np.mean(causal_effects)) if causal_effects else 0.0

    passed = hub_convergence > HUB_CONVERGENCE_THRESHOLD

    log(f"  Causal effect: {mean_causal_effect:.4f}")
    log(f"  Result: {'PASS' if passed else 'FAIL'}")

    results = [EvalResult(
        metric_id="EX22.semantic_hub_convergence",
        value=hub_convergence,
        n_samples=len(equivalent_pairs),
        instrument_info=INSTRUMENT_INFO,
        metadata={
            "task": "semantic_hub",
            "hub_layer": hub_layer,
            "hub_convergence_score": hub_convergence,
            "hub_discrimination": hub_discrimination,
            "mean_causal_effect": mean_causal_effect,
            "n_equivalent_pairs": len(equivalent_pairs),
            "n_control_pairs": len(control_pairs),
            "layer_similarities_equivalent": layer_sims_eq.tolist(),
            "layer_similarities_control": layer_sims_ctrl.tolist(),
            "causal_effects": causal_effects,
            "passed": passed,
            "threshold": HUB_CONVERGENCE_THRESHOLD,
        },
    )]

    return results


def main():
    parser = parse_common_args("EX22: Semantic Hub Convergence")
    parser.add_argument("--n-pairs", type=int, default=15,
                        help="Number of semantic pairs to test")
    parser.add_argument("--perturbation-scale", type=float, default=3.0,
                        help="Magnitude for causal intervention test")
    args = parser.parse_args()

    model = load_model(args.model, args.device)

    log("=" * 60)
    log("EX22: SEMANTIC HUB CONVERGENCE")
    log("=" * 60)

    results = run_semantic_hub_convergence(
        model,
        n_pairs=args.n_pairs,
        perturbation_scale=args.perturbation_scale,
    )

    out = args.out or "124_semantic_hub.json"
    save_results(results, out)
    log(f"\nDone. {len(results)} results.")


if __name__ == "__main__":
    main()
