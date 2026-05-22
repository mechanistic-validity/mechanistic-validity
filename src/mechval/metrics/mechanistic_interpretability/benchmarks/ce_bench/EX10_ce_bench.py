"""CE-Bench Contrastive Evaluation for Feature Interpretability
================================================================
Metric:         EX10 -- CE-Bench Contrastive
Categories:     measurement
Validity layer: Measurement (M4 Sensitivity)
Criteria:       EX10 Contrastive Feature Selectivity
Establishes:    Whether artifact features respond selectively to specific
                semantic dimensions, measured via contrastive minimal pairs
Requires:       Model, artifact adapter, hook_name
================================================================

Implements a CE-Bench-style contrastive evaluation following Gulko et al.
(BlackboxNLP 2025). CE-Bench uses contrastive story pairs that differ in
exactly one semantic dimension. A well-behaved feature should activate
differentially on one member of the pair but not the other, demonstrating
sensitivity to a specific semantic dimension.

Key property: fully deterministic, no LLM judge needed.

Detection algorithm:
1. Construct minimal contrastive pairs differing in one semantic dimension
   (gender, sentiment, tense, location, quantity, formality, agency,
   certainty).
2. For each pair and each feature:
   - Run both stories through the model and extract feature activations.
   - Contrastive score = |mean_act_A - mean_act_B| / (std_A + std_B + eps)
     This is a Cohen's-d-like effect size measuring how distinguishable
     the feature's response is between the two stories.
   - Independence score = 1 - |corr(feature_acts, pair_labels)| across
     all OTHER pairs. A feature selective for gender should not also
     separate sentiment pairs.
3. Per-feature: best_contrastive_score across pairs, mean_independence.
4. Aggregate: mean contrastive score and mean independence across features.

Pass condition: mean_contrastive_score > 0.3

References:
    - Gulko et al. (2025) "CE-Bench: A Contrastive Evaluation Benchmark
      for Feature Interpretability", BlackboxNLP 2025.

Usage:
    mechval.run("ce_bench", artifact=adapter, hook_name="blocks.5.hook_resid_pre")
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
    name="CE-Bench Contrastive Evaluation",
    paper_ref="Gulko et al. BlackboxNLP 2025",
    paper_cite="Gulko et al. 2025, CE-Bench: Contrastive Evaluation for Feature Interpretability",
    description=(
        "Measures feature selectivity via contrastive minimal pairs "
        "that differ in exactly one semantic dimension"
    ),
    category="measurement",
    tier="cogsci",
    origin="established",
)

CONTRASTIVE_THRESHOLD = 0.3
EPS = 1e-8

DIMENSION_LABELS = [
    "agent_gender",
    "sentiment",
    "tense",
    "location",
    "quantity",
    "formality",
    "agency",
    "certainty",
]


def _contrastive_pairs() -> list[tuple[str, str, str]]:
    """Built-in contrastive story pairs. Each pair differs in one semantic dimension.

    Returns list of (story_A, story_B, dimension_label).
    """
    return [
        # Dimension: agent gender
        (
            "The man walked into the store and bought groceries for dinner.",
            "The woman walked into the store and bought groceries for dinner.",
            "agent_gender",
        ),
        # Dimension: sentiment
        (
            "The review said the movie was brilliant and moving.",
            "The review said the movie was terrible and boring.",
            "sentiment",
        ),
        # Dimension: tense
        (
            "She runs every morning before breakfast.",
            "She ran every morning before breakfast.",
            "tense",
        ),
        # Dimension: location
        (
            "The meeting was held in New York at the main office.",
            "The meeting was held in London at the main office.",
            "location",
        ),
        # Dimension: quantity
        (
            "Three cats sat on the windowsill watching birds.",
            "One cat sat on the windowsill watching birds.",
            "quantity",
        ),
        # Dimension: formality
        (
            "We respectfully request your attendance at the ceremony.",
            "Hey come to the ceremony if you want.",
            "formality",
        ),
        # Dimension: agency (active vs passive)
        (
            "The dog chased the cat across the yard.",
            "The cat was chased by the dog across the yard.",
            "agency",
        ),
        # Dimension: certainty
        (
            "The experiment will definitely succeed according to all data.",
            "The experiment might possibly succeed according to some data.",
            "certainty",
        ),
    ]


def _get_feature_activations(
    model, artifact, tokens: torch.Tensor, hook_name: str, n_features: int,
) -> torch.Tensor:
    """Extract feature activations for the given tokens.

    Returns:
        (n_positions, n_features) tensor of feature activations.
    """
    acts = artifact.activations(model, tokens, hook_name)
    # Flatten batch and seq: (batch, seq, n_features) -> (n_positions, n_features)
    acts = acts.reshape(-1, acts.shape[-1]).float()
    # Limit to requested number of features
    if acts.shape[-1] > n_features:
        acts = acts[:, :n_features]
    return acts


def _contrastive_score(acts_a: torch.Tensor, acts_b: torch.Tensor) -> np.ndarray:
    """Compute per-feature contrastive scores between two activation sets.

    Score = |mean_A - mean_B| / (std_A + std_B + eps)
    This is a Cohen's-d-like effect size.

    Args:
        acts_a: (n_positions_a, n_features)
        acts_b: (n_positions_b, n_features)

    Returns:
        (n_features,) array of contrastive scores.
    """
    mean_a = acts_a.mean(dim=0).cpu().numpy()
    mean_b = acts_b.mean(dim=0).cpu().numpy()
    std_a = acts_a.std(dim=0).cpu().numpy()
    std_b = acts_b.std(dim=0).cpu().numpy()
    return np.abs(mean_a - mean_b) / (std_a + std_b + EPS)


@torch.no_grad()
def run_ce_bench(
    model,
    artifact=None,
    hook_name: str = "blocks.5.hook_resid_pre",
    n_features: int = 50,
) -> list[EvalResult]:
    if artifact is None:
        log("  WARNING: no artifact adapter provided, skipping CE-Bench")
        return []

    pairs = _contrastive_pairs()
    n_pairs = len(pairs)
    log(f"  CE-Bench: {n_pairs} contrastive pairs, up to {n_features} features at {hook_name}")

    # Collect activations for all stories
    all_acts_a: list[torch.Tensor] = []
    all_acts_b: list[torch.Tensor] = []

    for story_a, story_b, dim_label in pairs:
        tokens_a = model.to_tokens(story_a)
        tokens_b = model.to_tokens(story_b)

        acts_a = _get_feature_activations(model, artifact, tokens_a, hook_name, n_features)
        acts_b = _get_feature_activations(model, artifact, tokens_b, hook_name, n_features)

        all_acts_a.append(acts_a)
        all_acts_b.append(acts_b)

    actual_n_features = all_acts_a[0].shape[-1]
    log(f"    Actual features scored: {actual_n_features}")

    # Per-pair contrastive scores: (n_pairs, n_features)
    pair_scores = np.zeros((n_pairs, actual_n_features))
    for i in range(n_pairs):
        pair_scores[i] = _contrastive_score(all_acts_a[i], all_acts_b[i])

    # Per-feature: best contrastive score across pairs
    best_contrastive = pair_scores.max(axis=0)  # (n_features,)
    best_pair_idx = pair_scores.argmax(axis=0)  # which pair each feature is most selective for

    # Independence: for each feature, how selective is it for its best pair
    # vs all other pairs? A feature that fires for everything is not selective.
    # Independence = 1 - (mean score on OTHER pairs / best score)
    independence_scores = np.zeros(actual_n_features)
    for f in range(actual_n_features):
        best_idx = best_pair_idx[f]
        other_scores = np.delete(pair_scores[:, f], best_idx)
        if best_contrastive[f] > EPS:
            # Ratio of other-pair activity to best-pair activity; low = independent
            independence_scores[f] = 1.0 - float(other_scores.mean()) / float(best_contrastive[f])
        else:
            # Feature doesn't respond to any pair
            independence_scores[f] = 1.0

    # Aggregate across features
    mean_contrastive = float(best_contrastive.mean())
    mean_independence = float(independence_scores.mean())

    # Per-pair aggregate scores for metadata
    per_pair_scores = {}
    for i, (_, _, dim_label) in enumerate(pairs):
        per_pair_scores[dim_label] = float(pair_scores[i].mean())

    passed = mean_contrastive > CONTRASTIVE_THRESHOLD

    log(f"    mean_contrastive={mean_contrastive:.4f}")
    log(f"    mean_independence={mean_independence:.4f}")
    log(f"    per_pair: {per_pair_scores}")
    log(f"    [{'PASS' if passed else 'FAIL'}]")

    return [EvalResult(
        metric_id="EX10.ce_bench_contrastive",
        value=mean_contrastive,
        n_samples=n_pairs * actual_n_features,
        instrument_info=INSTRUMENT_INFO,
        metadata={
            "hook_name": hook_name,
            "n_features_scored": actual_n_features,
            "n_pairs": n_pairs,
            "mean_contrastive": mean_contrastive,
            "mean_independence": mean_independence,
            "per_pair_scores": per_pair_scores,
            "per_feature_best_pair": {
                int(f): DIMENSION_LABELS[int(best_pair_idx[f])]
                for f in range(actual_n_features)
            },
            "passed": passed,
            "threshold": CONTRASTIVE_THRESHOLD,
        },
    )]


def main():
    parser = parse_common_args("EX10: CE-Bench Contrastive Evaluation")
    parser.add_argument("--hook", default="blocks.5.hook_resid_pre", help="Hook point")
    parser.add_argument("--n-features", type=int, default=50, help="Number of features to score")
    parser.add_argument("--artifact-path", default=None, help="Artifact release ID")
    parser.add_argument("--sae-id", default=None, help="SAE ID")
    args = parser.parse_args()

    model = load_model(args.model, args.device)

    artifact = None
    if args.artifact_path:
        from mechval.lib.artifacts import SAEAdapter
        artifact = SAEAdapter.from_pretrained(
            release=args.artifact_path,
            sae_id=args.sae_id or "",
            hook_point=args.hook,
        )

    log("=" * 60)
    log("EX10: CE-BENCH CONTRASTIVE EVALUATION")
    log("=" * 60)

    results = run_ce_bench(
        model, artifact=artifact, hook_name=args.hook,
        n_features=args.n_features,
    )

    out = args.out or "EX10_ce_bench.json"
    save_results(results, out)
    log(f"\nDone. {len(results)} results.")


if __name__ == "__main__":
    main()
