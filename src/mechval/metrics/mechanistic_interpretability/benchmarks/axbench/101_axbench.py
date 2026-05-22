"""AxBench Evaluation (Behavioral B20)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     B20 — AxBench Concept Detection & Steering
Categories:     behavioral
Validity layer: External
Criteria:       B20 Concept detection and steering against baselines
Establishes:    Whether artifact features detect concepts better than
                DiffMean baselines, and whether feature directions steer
                model behavior
Requires:       CPU or GPU, model, artifact adapter
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Implements the AxBench evaluation protocol (Huang et al., ICML 2025 Spotlight).
AxBench showed SAEs are not competitive with simple baselines (prompting,
finetuning, DiffMean) on concept detection or steering.

Two evaluation axes per concept:

**Concept Detection:**
  For each feature in the artifact, compute activations on positive vs negative
  concept examples. Detection quality = AUROC of best-separating feature.
  Compare against:
    - Random direction baseline (expected ~0.5)
    - DiffMean baseline (mean_positive - mean_negative, normalized)

**Steering Evaluation:**
  For top-k most detecting features, add/subtract the direction at inference
  time and measure behavioral shift via log-probability change on concept-
  related continuations.

Usage:
    # Programmatic (from metric_registry dispatch):
    run_axbench(model, artifact, hook_name="blocks.5.hook_resid_pre",
                concepts=[([pos_texts], [neg_texts])], n_prompts=50)

    # CLI:
    uv run python 101_axbench.py --model gpt2 --hook blocks.5.hook_resid_pre
"""

import math

import numpy as np
import torch

from mechval.metrics.common import (
    EvalResult,
    load_model,
    log,
    parse_common_args,
    save_results,
)


# ---------------------------------------------------------------------------
# AUROC computation (no sklearn dependency)
# ---------------------------------------------------------------------------

def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute AUROC from scores and binary labels without sklearn.

    Uses the Mann-Whitney U statistic formulation:
        AUROC = U / (n_pos * n_neg)
    """
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Count how many (pos, neg) pairs the positive scores higher
    u = 0.0
    for p in pos:
        u += np.sum(p > neg) + 0.5 * np.sum(p == neg)
    auroc = u / (n_pos * n_neg)
    # Return the better of auroc and 1-auroc (direction-agnostic)
    return max(auroc, 1.0 - auroc)


# ---------------------------------------------------------------------------
# Core: concept detection
# ---------------------------------------------------------------------------

@torch.no_grad()
def _get_activations_for_texts(
    model, artifact, hook_name: str, texts: list[str],
) -> torch.Tensor:
    """Run texts through the model and return artifact feature activations.

    Returns: (n_texts, n_features) tensor of per-text feature activations
    (mean-pooled across sequence positions).
    """
    all_acts = []
    for text in texts:
        tokens = model.to_tokens(text)
        acts = artifact.activations(model, tokens, hook_name)
        # acts shape: (1, seq_len, n_features) -> mean over seq
        mean_acts = acts[0].mean(dim=0)  # (n_features,)
        all_acts.append(mean_acts.cpu())
    return torch.stack(all_acts)  # (n_texts, n_features)


def _concept_detection(
    pos_acts: torch.Tensor,
    neg_acts: torch.Tensor,
) -> dict:
    """Compute per-feature AUROC for concept detection.

    Args:
        pos_acts: (n_pos, n_features) activations on positive examples
        neg_acts: (n_neg, n_features) activations on negative examples

    Returns dict with detection metrics.
    """
    n_pos = pos_acts.shape[0]
    n_neg = neg_acts.shape[0]
    n_features = pos_acts.shape[1]

    all_acts = torch.cat([pos_acts, neg_acts], dim=0).numpy()
    labels = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])

    per_feature_auroc = np.array([
        _auroc(all_acts[:, f], labels) for f in range(n_features)
    ])

    sorted_aurocs = np.sort(per_feature_auroc)[::-1]
    best_auroc = float(sorted_aurocs[0])
    top10 = sorted_aurocs[:min(10, len(sorted_aurocs))]
    mean_top10 = float(np.mean(top10))
    best_feature_idx = int(np.argmax(per_feature_auroc))

    return {
        "best_feature_detection_auroc": best_auroc,
        "mean_top10_detection": mean_top10,
        "best_feature_idx": best_feature_idx,
        "detection_rank": per_feature_auroc,  # raw array for downstream use
    }


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def _random_baseline_detection(
    pos_acts: torch.Tensor,
    neg_acts: torch.Tensor,
    n_random: int = 50,
) -> float:
    """Detection AUROC of random directions (expected ~0.5).

    Projects activations onto random unit vectors and computes AUROC.
    """
    d = pos_acts.shape[1]
    n_pos = pos_acts.shape[0]
    n_neg = neg_acts.shape[0]
    labels = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
    all_acts = torch.cat([pos_acts, neg_acts], dim=0)  # (n, d)

    aurocs = []
    for _ in range(n_random):
        direction = torch.randn(d)
        direction = direction / (direction.norm() + 1e-12)
        projections = (all_acts @ direction).numpy()
        aurocs.append(_auroc(projections, labels))

    return float(np.mean(aurocs))


def _diffmean_baseline_detection(
    pos_acts: torch.Tensor,
    neg_acts: torch.Tensor,
) -> dict:
    """DiffMean baseline: use (mean_pos - mean_neg) as detection direction.

    This is the simplest baseline from AxBench and is often hard to beat.
    """
    mean_pos = pos_acts.mean(dim=0)  # (d,)
    mean_neg = neg_acts.mean(dim=0)  # (d,)
    diff = mean_pos - mean_neg  # (d,)
    diff_norm = diff / (diff.norm() + 1e-12)

    all_acts = torch.cat([pos_acts, neg_acts], dim=0)
    n_pos = pos_acts.shape[0]
    n_neg = neg_acts.shape[0]
    labels = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])

    projections = (all_acts @ diff_norm).numpy()
    auroc = _auroc(projections, labels)

    return {
        "diffmean_auroc": float(auroc),
        "diffmean_direction": diff_norm,
    }


# ---------------------------------------------------------------------------
# Core: steering evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _steering_eval(
    model,
    artifact,
    hook_name: str,
    feature_indices: list[int],
    eval_texts: list[str],
    coeffs: list[float] | None = None,
) -> dict:
    """Evaluate whether artifact directions can steer model behavior.

    For each feature direction, add it at inference time with various
    coefficients and measure log-probability shift on eval texts.
    """
    if coeffs is None:
        coeffs = [1.0, 2.0, 5.0]

    dirs = artifact.directions()
    if dirs.ndim == 3:
        dirs = dirs.mean(dim=0)

    # Baseline: get log-probs on eval texts without steering
    baseline_logprobs = []
    for text in eval_texts:
        tokens = model.to_tokens(text)
        logits = model(tokens)
        # Mean log-prob of continuation tokens (all except first)
        log_probs = torch.log_softmax(logits[0], dim=-1)
        if tokens.shape[1] > 1:
            target_ids = tokens[0, 1:]
            position_logprobs = log_probs[:-1]
            selected = position_logprobs[range(len(target_ids)), target_ids]
            baseline_logprobs.append(selected.mean().item())
        else:
            baseline_logprobs.append(0.0)
    mean_baseline = np.mean(baseline_logprobs)

    per_feature = []
    successes = 0

    for feat_idx in feature_indices:
        direction = dirs[feat_idx]
        direction = direction / (direction.norm() + 1e-12)

        best_shift = 0.0
        coeff_shifts = {}

        for coeff in coeffs:
            shifted_logprobs = []
            for text in eval_texts:
                tokens = model.to_tokens(text)

                def add_direction(act, hook, _d=direction, _c=coeff):
                    act[:, :, :] = act + _c * _d.to(act.device)
                    return act

                logits = model.run_with_hooks(
                    tokens, fwd_hooks=[(hook_name, add_direction)],
                )
                log_probs = torch.log_softmax(logits[0], dim=-1)
                if tokens.shape[1] > 1:
                    target_ids = tokens[0, 1:]
                    position_logprobs = log_probs[:-1]
                    selected = position_logprobs[range(len(target_ids)), target_ids]
                    shifted_logprobs.append(selected.mean().item())
                else:
                    shifted_logprobs.append(0.0)

            mean_shifted = np.mean(shifted_logprobs)
            shift = abs(mean_shifted - mean_baseline)
            coeff_shifts[coeff] = float(shift)
            best_shift = max(best_shift, shift)

        # A feature "successfully steers" if it shifts log-prob by > 0.1 nats
        is_success = best_shift > 0.1
        if is_success:
            successes += 1

        per_feature.append({
            "feature_idx": feat_idx,
            "best_shift": float(best_shift),
            "coeff_shifts": coeff_shifts,
            "steers": is_success,
        })

    n_tested = len(feature_indices)
    success_rate = successes / n_tested if n_tested > 0 else 0.0
    mean_magnitude = float(np.mean([f["best_shift"] for f in per_feature])) if per_feature else 0.0

    return {
        "steering_success_rate": success_rate,
        "mean_steering_magnitude": mean_magnitude,
        "n_features_tested": n_tested,
        "n_successes": successes,
        "per_feature": per_feature,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_axbench(
    model,
    artifact=None,
    hook_name: str | None = None,
    n_prompts: int = 50,
    concepts: list[tuple[list[str], list[str]]] | None = None,
    n_steering_features: int = 10,
    n_random_baselines: int = 50,
) -> list[EvalResult]:
    """Run AxBench-style concept detection + steering evaluation.

    Args:
        model: HookedTransformer instance.
        artifact: ArtifactAdapter with directions() and activations() methods.
        hook_name: Hook point for activations (e.g. "blocks.5.hook_resid_pre").
        n_prompts: Max number of examples per concept side (capped by provided texts).
        concepts: List of (positive_texts, negative_texts) tuples. Each tuple
            defines a concept to detect. If None, uses a small built-in probe set.
        n_steering_features: Number of top-detecting features to test for steering.
        n_random_baselines: Number of random directions for baseline.

    Returns:
        List of EvalResult, one per concept.
    """
    if artifact is None:
        log("  WARNING: no artifact adapter provided, skipping AxBench")
        return []

    effective_hook = hook_name or artifact.manifest.hook_point
    if not effective_hook:
        effective_hook = "blocks.5.hook_resid_pre"

    if concepts is None:
        concepts = _default_concepts()

    log(f"  AxBench: {len(concepts)} concepts, hook={effective_hook}")

    results = []

    for concept_idx, (pos_texts, neg_texts) in enumerate(concepts):
        pos_texts = pos_texts[:n_prompts]
        neg_texts = neg_texts[:n_prompts]

        if len(pos_texts) < 2 or len(neg_texts) < 2:
            log(f"  concept {concept_idx}: too few examples "
                f"({len(pos_texts)} pos, {len(neg_texts)} neg), skipping")
            continue

        log(f"  concept {concept_idx}: {len(pos_texts)} pos, {len(neg_texts)} neg")

        # --- Concept Detection ---
        pos_acts = _get_activations_for_texts(model, artifact, effective_hook, pos_texts)
        neg_acts = _get_activations_for_texts(model, artifact, effective_hook, neg_texts)

        detection = _concept_detection(pos_acts, neg_acts)
        log(f"    detection: best AUROC={detection['best_feature_detection_auroc']:.4f}, "
            f"top-10 mean={detection['mean_top10_detection']:.4f}")

        # --- Random baseline ---
        random_auroc = _random_baseline_detection(pos_acts, neg_acts, n_random=n_random_baselines)
        log(f"    random baseline AUROC={random_auroc:.4f}")

        # --- DiffMean baseline ---
        diffmean = _diffmean_baseline_detection(pos_acts, neg_acts)
        log(f"    DiffMean baseline AUROC={diffmean['diffmean_auroc']:.4f}")

        beats_random = detection["best_feature_detection_auroc"] > random_auroc + 0.05
        beats_diffmean = detection["best_feature_detection_auroc"] >= diffmean["diffmean_auroc"] - 0.02

        # --- Steering ---
        detection_rank = detection["detection_rank"]
        top_feature_indices = list(np.argsort(detection_rank)[::-1][:n_steering_features])

        steering = _steering_eval(
            model, artifact, effective_hook, top_feature_indices, pos_texts,
        )
        log(f"    steering: success_rate={steering['steering_success_rate']:.2f}, "
            f"mean_magnitude={steering['mean_steering_magnitude']:.4f}")

        concept_label = f"concept_{concept_idx}"
        passed = beats_random and detection["best_feature_detection_auroc"] > 0.6

        log(f"    beats_random={beats_random}, beats_diffmean={beats_diffmean}, "
            f"[{'PASS' if passed else 'FAIL'}]")

        results.append(EvalResult(
            metric_id="B20.axbench_evaluation",
            value=detection["best_feature_detection_auroc"],
            baseline_random=random_auroc,
            n_samples=len(pos_texts) + len(neg_texts),
            metadata={
                "concept": concept_label,
                "concept_idx": concept_idx,
                "n_positive": len(pos_texts),
                "n_negative": len(neg_texts),
                # Detection
                "concept_detection_auroc": detection["best_feature_detection_auroc"],
                "mean_top10_detection": detection["mean_top10_detection"],
                "best_feature_idx": detection["best_feature_idx"],
                # Baselines
                "random_baseline_auroc": random_auroc,
                "diffmean_baseline_auroc": diffmean["diffmean_auroc"],
                "beats_random_baseline": beats_random,
                "beats_diffmean_baseline": beats_diffmean,
                # Steering
                "steering_score": steering["steering_success_rate"],
                "steering_success_rate": steering["steering_success_rate"],
                "mean_steering_magnitude": steering["mean_steering_magnitude"],
                "n_steering_features": steering["n_features_tested"],
                "steering_successes": steering["n_successes"],
                # Summary
                "passed": passed,
                "hook_name": effective_hook,
                "n_random_baselines": n_random_baselines,
            },
        ))

    # Log aggregate summary
    if results:
        mean_detection = np.mean([r.metadata["concept_detection_auroc"] for r in results])
        mean_steering = np.mean([r.metadata["steering_success_rate"] for r in results])
        n_beat_random = sum(1 for r in results if r.metadata["beats_random_baseline"])
        n_beat_diffmean = sum(1 for r in results if r.metadata["beats_diffmean_baseline"])
        log(f"  SUMMARY: mean detection AUROC={mean_detection:.4f}, "
            f"mean steering rate={mean_steering:.2f}")
        log(f"  beats random: {n_beat_random}/{len(results)}, "
            f"beats DiffMean: {n_beat_diffmean}/{len(results)}")

    return results


# ---------------------------------------------------------------------------
# Built-in concept probe set
# ---------------------------------------------------------------------------

def _default_concepts() -> list[tuple[list[str], list[str]]]:
    """Small built-in concept set for quick evaluation without external data.

    These are short, unambiguous prompts designed to activate concept-relevant
    features in typical language models.
    """
    return [
        # Concept 0: Sentiment (positive vs negative)
        (
            [
                "This movie was absolutely wonderful and I loved every minute of it.",
                "The food at this restaurant is delicious and the service is excellent.",
                "I had an amazing experience and would highly recommend this place.",
                "The weather today is beautiful and perfect for a walk in the park.",
                "She gave a brilliant performance that moved the entire audience.",
                "This is the best book I have read in years, truly outstanding.",
                "The team played magnificently and deserved their victory.",
                "What a fantastic day, everything went perfectly as planned.",
                "The garden looks gorgeous with all the flowers in bloom.",
                "I am thrilled with the results, they exceeded all expectations.",
            ],
            [
                "This movie was terrible and a complete waste of time.",
                "The food at this restaurant is awful and the service is horrible.",
                "I had a dreadful experience and would never come back.",
                "The weather today is miserable with heavy rain and cold wind.",
                "She gave a poor performance that disappointed everyone.",
                "This is the worst book I have ever read, completely boring.",
                "The team played terribly and deserved their humiliating loss.",
                "What a horrible day, everything went wrong from the start.",
                "The garden looks neglected with dead plants everywhere.",
                "I am furious with the results, they are completely unacceptable.",
            ],
        ),
        # Concept 1: Formal vs informal register
        (
            [
                "We respectfully request your presence at the annual conference.",
                "The committee has resolved to implement the proposed amendments.",
                "Pursuant to our agreement, the deliverables shall be submitted.",
                "I am writing to inform you of the changes to our policy.",
                "The board of directors convened to discuss strategic initiatives.",
                "Please find enclosed the documentation for your review.",
                "We acknowledge receipt of your correspondence dated March 15.",
                "The undersigned hereby certifies the accuracy of this report.",
                "In accordance with established protocols, we have initiated.",
                "The organization has undertaken a comprehensive review.",
            ],
            [
                "Hey dude, wanna grab some pizza tonight?",
                "Lol that was so funny I can't even deal right now.",
                "Yo check this out, it's totally insane!",
                "Nah man, I'm just gonna chill at home tonight.",
                "Omg that party was lit, we should do it again!",
                "Bruh you're not gonna believe what happened today.",
                "Yeah whatever, I don't really care about that stuff.",
                "Haha nice one, you totally got me with that joke.",
                "Dude seriously? That's the craziest thing I've heard.",
                "Gonna bounce, catch you later alright?",
            ],
        ),
        # Concept 2: Scientific/technical vs everyday language
        (
            [
                "The mitochondria generate ATP through oxidative phosphorylation.",
                "Quantum entanglement violates Bell's inequality in experiments.",
                "The catalyst lowers the activation energy of the reaction.",
                "Neuroplasticity allows the brain to reorganize synaptic connections.",
                "The algorithm achieves O(n log n) time complexity on average.",
                "Photosynthesis converts carbon dioxide and water into glucose.",
                "The p-value indicates statistical significance below the threshold.",
                "Tectonic plates move due to convection currents in the mantle.",
                "The protein folds into its tertiary structure via hydrophobic.",
                "Electromagnetic radiation propagates as transverse waves.",
            ],
            [
                "I went to the store to buy some milk and bread.",
                "The kids played in the backyard all afternoon.",
                "We had spaghetti for dinner and watched a movie.",
                "She walked the dog around the block before bedtime.",
                "He fixed the leaky faucet in the kitchen sink.",
                "They drove to the beach and built sandcastles.",
                "I called my mom to wish her a happy birthday.",
                "The cat slept on the couch all day long.",
                "We planted tomatoes and peppers in the garden.",
                "She baked cookies for the school bake sale.",
            ],
        ),
    ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = parse_common_args("B20: AxBench Concept Detection & Steering")
    parser.add_argument("--hook", default=None,
                        help="Hook point (e.g. blocks.5.hook_resid_pre)")
    parser.add_argument("--artifact-path", default=None,
                        help="SAE release ID for artifact")
    parser.add_argument("--sae-id", default=None,
                        help="SAE ID within release")
    parser.add_argument("--n-steering-features", type=int, default=10,
                        help="Top-k features to test for steering")
    args = parser.parse_args()

    model = load_model(args.model, args.device)

    artifact = None
    if args.artifact_path:
        from mechval.lib.artifacts import SAEAdapter
        artifact = SAEAdapter.from_pretrained(
            release=args.artifact_path,
            sae_id=args.sae_id or "",
            hook_point=args.hook or "",
        )

    log("=" * 60)
    log("B20: AXBENCH CONCEPT DETECTION & STEERING")
    log("=" * 60)

    results = run_axbench(
        model,
        artifact=artifact,
        hook_name=args.hook,
        n_prompts=args.n_prompts,
        n_steering_features=args.n_steering_features,
    )

    out = args.out or "101_axbench.json"
    save_results(results, out)
    log(f"\nDone. {len(results)} concepts evaluated.")


if __name__ == "__main__":
    main()
