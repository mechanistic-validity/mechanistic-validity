"""Activation Reasoning (External EX14)
Paper: Helff et al. (2026). ICLR 2026.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     EX14 — Activation Reasoning
Categories:     external
Validity layer: External (E5 Downstream Task Prediction)
Criteria:       E5 Feature-Description Downstream Reasoning
Establishes:    Whether SAE feature descriptions enable correct downstream
                reasoning when features are mapped to logical propositions
                and composed via symbolic rules
Requires:       Model, artifact adapter with directions() and activations()
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Implements the Activation Reasoning approach (Helff, Harle, Stammer,
Friedrich et al., ICLR 2026). AR maps SAE features to logical
propositions, then applies symbolic rules to compose higher-order
structures and steer behavior. This makes SAE features active
participants in reasoning rather than passive read-outs.

The metric tests the external validity criterion operationally:
feature descriptions should enable correct downstream reasoning.
AR scales robustly with reasoning complexity and generalizes across
model backbones.

Procedure:
1. Run the model on task prompts, collecting activations at the
   artifact's hook point(s).
2. Encode activations through the artifact adapter to get feature
   activations.
3. Map top-k active features to binary propositions
   (feature_active > threshold = True).
4. Apply simple logical composition rules (AND, OR, NOT) over
   propositions to form compound predicates.
5. Test whether the composed propositions predict downstream task
   behavior correctly (correct vs incorrect answer prediction).
6. Report downstream_accuracy.

Pass condition: downstream_accuracy > 0.6

References:
    Helff, Harle, Stammer, Friedrich et al. (ICLR 2026)
    "Activation Reasoning"

Usage:
    uv run python 109_activation_reasoning.py --tasks ioi --n-prompts 40
    uv run python 109_activation_reasoning.py --tasks ioi sva --device cpu
"""

import numpy as np
import torch

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    InstrumentInfo,
    generate_prompts,
    get_token_ids,
    load_model,
    log,
    parse_common_args,
    save_incremental,
    save_results,
)

INSTRUMENT_INFO = InstrumentInfo(
    name="Activation Reasoning",
    paper_ref="Helff, Harle, Stammer, Friedrich et al. ICLR 2026",
    paper_cite=(
        "Helff, Harle, Stammer, Friedrich et al. 2026, "
        "Activation Reasoning"
    ),
    description=(
        "Maps SAE features to logical propositions and applies symbolic "
        "composition rules (AND, OR, NOT) to test whether feature descriptions "
        "enable correct downstream reasoning"
    ),
    category="external",
    tier="cogsci",
    origin="established",
)

DOWNSTREAM_ACCURACY_THRESHOLD = 0.6


@torch.no_grad()
def _get_feature_propositions(
    feature_acts: torch.Tensor,
    top_k: int = 10,
    threshold: float = 0.0,
) -> torch.Tensor:
    """Map feature activations to binary propositions.

    For each position, select the top-k most active features, then
    binarize: feature_active > threshold => True.

    Args:
        feature_acts: (batch, seq, n_features) feature activations.
        top_k: number of top features to consider per position.
        threshold: activation threshold for binarization.

    Returns:
        (batch, seq, n_features) boolean tensor of propositions.
    """
    n_features = feature_acts.shape[-1]
    effective_k = min(top_k, n_features)

    # Get top-k mask
    _, top_indices = feature_acts.topk(effective_k, dim=-1)
    top_k_mask = torch.zeros_like(feature_acts, dtype=torch.bool)
    top_k_mask.scatter_(-1, top_indices, True)

    # Binarize: active above threshold AND in top-k
    propositions = (feature_acts > threshold) & top_k_mask
    return propositions


@torch.no_grad()
def _apply_logical_composition(
    props_correct: torch.Tensor,
    props_incorrect: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Apply logical composition rules over proposition pairs.

    Given propositions from correct and incorrect contexts, compute
    composed predicates that distinguish the two.

    Args:
        props_correct: (n_correct, n_features) binary propositions from
            correct-answer contexts.
        props_incorrect: (n_incorrect, n_features) binary propositions from
            incorrect-answer contexts.

    Returns:
        Dict with per-feature logical composition scores:
        - "and_selectivity": fraction of correct contexts where feature is
          active AND it is not active in the mean incorrect context
        - "or_coverage": fraction of (correct OR not-incorrect) contexts
          where the feature discriminates
        - "not_specificity": fraction of incorrect contexts where feature
          is NOT active (i.e., absence is informative)
    """
    n_correct = props_correct.shape[0]
    n_incorrect = props_incorrect.shape[0]

    # Per-feature activation rates
    correct_rate = props_correct.float().mean(dim=0)    # (n_features,)
    incorrect_rate = props_incorrect.float().mean(dim=0)

    # AND: feature active in correct AND not in incorrect
    # High value => feature is selective for correct answer
    and_selectivity = correct_rate * (1.0 - incorrect_rate)

    # OR: feature active in correct OR absent in incorrect
    or_coverage = torch.clamp(correct_rate + (1.0 - incorrect_rate), max=1.0)

    # NOT: feature absent in incorrect contexts (absence is informative)
    not_specificity = 1.0 - incorrect_rate

    return {
        "and_selectivity": and_selectivity,
        "or_coverage": or_coverage,
        "not_specificity": not_specificity,
    }


@torch.no_grad()
def _predict_from_propositions(
    model,
    artifact,
    prompts,
    correct_ids: list[int],
    incorrect_ids: list[int],
    hook_name: str,
    top_k: int = 10,
    threshold: float = 0.0,
) -> tuple[float, dict]:
    """Predict correct vs incorrect answer using logical proposition composition.

    Split prompts into a train set (to learn which composed propositions
    discriminate) and a test set (to evaluate downstream accuracy).

    Returns:
        (downstream_accuracy, detail_dict)
    """
    n = min(len(prompts), len(correct_ids), len(incorrect_ids))
    if n < 4:
        return 0.0, {"error": "too few prompts", "n": n}

    # Collect feature propositions for each prompt at last token position
    all_props = []
    all_labels = []  # 1 = model gets it right, 0 = wrong

    for i in range(n):
        tokens = model.to_tokens(prompts[i].text)
        feat_acts = artifact.activations(model, tokens, hook_name)
        # (1, seq, n_features) -> last token -> (n_features,)
        last_acts = feat_acts[0, -1]

        # Binarize
        n_features = last_acts.shape[0]
        effective_k = min(top_k, n_features)
        _, top_idx = last_acts.topk(effective_k)
        props = torch.zeros(n_features, dtype=torch.bool, device=last_acts.device)
        props[top_idx] = last_acts[top_idx] > threshold

        all_props.append(props)

        # Check if model predicts correctly
        logits = model(tokens)
        correct_logit = logits[0, -1, correct_ids[i]].item()
        incorrect_logit = logits[0, -1, incorrect_ids[i]].item()
        all_labels.append(1 if correct_logit > incorrect_logit else 0)

    props_tensor = torch.stack(all_props)  # (n, n_features)
    labels = torch.tensor(all_labels, dtype=torch.float32)

    # Split: first half train, second half test
    split = n // 2
    train_props, test_props = props_tensor[:split], props_tensor[split:]
    train_labels, test_labels = labels[:split], labels[split:]

    # On train set: compute logical composition scores for correct vs incorrect
    correct_mask = train_labels == 1
    incorrect_mask = train_labels == 0

    if correct_mask.sum() < 1 or incorrect_mask.sum() < 1:
        return 0.0, {"error": "degenerate train split", "n_correct": int(correct_mask.sum()),
                      "n_incorrect": int(incorrect_mask.sum())}

    compositions = _apply_logical_composition(
        train_props[correct_mask],
        train_props[incorrect_mask],
    )

    # Use AND selectivity as the discriminative score per feature:
    # features that are active in correct contexts and absent in incorrect
    feature_scores = compositions["and_selectivity"]

    # Select top discriminative features
    n_disc = max(1, min(top_k, n_features // 4))
    _, disc_features = feature_scores.topk(n_disc)

    # On test set: predict using majority vote over discriminative features
    # For each test prompt, count how many discriminative features are active
    test_disc_active = test_props[:, disc_features].float().sum(dim=-1)  # (n_test,)
    disc_threshold = n_disc / 2.0

    # Predict: if more than half of discriminative features are active => correct
    predictions = (test_disc_active >= disc_threshold).float()

    # Accuracy
    n_test = test_labels.shape[0]
    if n_test == 0:
        return 0.0, {"error": "empty test set"}

    downstream_accuracy = float((predictions == test_labels).float().mean())

    detail = {
        "n_train": split,
        "n_test": n_test,
        "n_discriminative_features": n_disc,
        "train_correct_rate": float(correct_mask.float().mean()),
        "test_correct_rate": float((test_labels == 1).float().mean()),
        "mean_and_selectivity": float(compositions["and_selectivity"].mean()),
        "mean_or_coverage": float(compositions["or_coverage"].mean()),
        "mean_not_specificity": float(compositions["not_specificity"].mean()),
        "max_and_selectivity": float(compositions["and_selectivity"].max()),
    }

    return downstream_accuracy, detail


def run_activation_reasoning(
    model,
    tasks: list[str],
    artifact=None,
    n_prompts: int = 40,
    hook_name: str | None = None,
    top_k: int = 10,
    threshold: float = 0.0,
) -> list[EvalResult]:
    """Run Activation Reasoning evaluation across tasks.

    Args:
        model: HookedTransformer instance.
        tasks: list of task names to evaluate.
        artifact: ArtifactAdapter with directions() and activations().
        n_prompts: number of prompts per task.
        hook_name: hook point for activation collection.
        top_k: number of top features for proposition extraction.
        threshold: activation threshold for binarization.

    Returns:
        List of EvalResult, one per task.
    """
    if artifact is None:
        log("  WARNING: no artifact adapter provided, skipping activation reasoning")
        return []

    tokenizer = model.tokenizer

    effective_hook = hook_name or getattr(
        getattr(artifact, "manifest", None), "hook_point", None
    ) or "blocks.0.hook_resid_pre"

    results = []

    for task in tasks:
        prompts = generate_prompts(task, tokenizer, n_prompts)
        if not prompts:
            log(f"  {task}: no prompts, skipping")
            continue

        correct_ids, incorrect_ids = get_token_ids(prompts, tokenizer)
        if not correct_ids:
            log(f"  {task}: no valid token ids, skipping")
            continue

        log(f"  {task}: {len(prompts)} prompts, hook={effective_hook}")

        downstream_accuracy, detail = _predict_from_propositions(
            model, artifact, prompts, correct_ids, incorrect_ids,
            effective_hook, top_k=top_k, threshold=threshold,
        )

        passed = downstream_accuracy > DOWNSTREAM_ACCURACY_THRESHOLD

        log(f"    downstream_accuracy={downstream_accuracy:.4f} "
            f"[{'PASS' if passed else 'FAIL'}]")

        results.append(EvalResult(
            metric_id="EX14.activation_reasoning",
            value=downstream_accuracy,
            n_samples=len(prompts),
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": task,
                "hook_name": effective_hook,
                "top_k": top_k,
                "threshold": threshold,
                "downstream_accuracy": downstream_accuracy,
                "passed": passed,
                "pass_threshold": DOWNSTREAM_ACCURACY_THRESHOLD,
                **detail,
            },
        ))

    return results


def main():
    parser = parse_common_args("EX14: Activation Reasoning")
    parser.add_argument("--hook", default=None, help="Hook point for artifact activations")
    parser.add_argument("--top-k", type=int, default=10, help="Top-k features for propositions")
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="Activation threshold for binarization")
    parser.add_argument("--artifact-type", default="sae",
                        choices=["sae", "transcoder", "crosscoder", "factor_bank"],
                        help="Artifact adapter type")
    parser.add_argument("--artifact-path", default=None, help="Path or release ID for artifact")
    parser.add_argument("--sae-id", default=None, help="SAE ID (for SAELens artifacts)")
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
    log("EX14: ACTIVATION REASONING")
    log("=" * 60)

    out = args.out or "109_activation_reasoning.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_activation_reasoning(
            model, [task], artifact=artifact,
            n_prompts=args.n_prompts, hook_name=args.hook,
            top_k=args.top_k, threshold=args.threshold,
        )
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)
            p = "PASS" if r.metadata["passed"] else "FAIL"
            log(f"  {task}: accuracy={r.value:.4f}  [{p}]")

    save_results(results, out)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
