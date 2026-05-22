"""SAE Architecture Duality (Measurement M7)
Paper: Lindsey et al. (2025). NeurIPS 2025.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     M07 — Architecture Duality
Categories:     measurement
Validity layer: Measurement
Criteria:       M2 Hyperparameter Sensitivity, M6 Artifact Quality
Establishes:    Whether two different SAE architectures trained on the same
                model/hook agree on what features exist (construct validity
                for the decomposition method itself)
Requires:       Two artifact adapters (different architectures, same hook point)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Based on Lindsey et al., NeurIPS 2025: each SAE architecture imposes
structural assumptions that determine what concepts can be detected.
Switching architectures reveals entirely new concepts or hides existing
ones. "An SAE does not just reveal concepts -- it determines what can be
seen at all."

This is a construct validity test: do different decomposition methods agree
on what features exist?

Method:
    1. Collect activations at a shared hook point from the model.
    2. Encode activations through both artifact adapters.
    3. Compute feature overlap: Jaccard similarity of active feature sets
       at a threshold (features with activation > threshold are "active").
    4. Compute direction agreement: mean max cosine similarity between
       encoder directions of the two artifacts.
    5. Report architecture_agreement = mean(feature_overlap, direction_agreement).

Pass condition: architecture_agreement > 0.3

Usage:
    uv run python 110_architecture_duality.py --artifact-a-path <release> --artifact-b-path <release>
    uv run python 110_architecture_duality.py --device cpu
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
    name="Architecture Duality (Lindsey et al. 2025)",
    paper_ref="Lindsey et al., NeurIPS 2025",
    paper_cite="Lindsey et al. 2025, Architecture Duality of SAEs",
    description=(
        "Tests whether two SAE architectures trained on the same model/hook "
        "agree on what features exist — a construct validity check for the "
        "decomposition method itself"
    ),
    category="measurement",
    tier="measurement_theory",
    origin="established",
)

AGREEMENT_THRESHOLD = 0.3


def compute_feature_overlap(
    acts_a: torch.Tensor,
    acts_b: torch.Tensor,
    threshold: float = 0.01,
) -> float:
    """Jaccard similarity of active feature sets across two encodings.

    For each input position, determines which features are "active"
    (activation > threshold) in each artifact, then computes Jaccard
    similarity averaged over positions.

    Args:
        acts_a: (batch, seq, n_features_a) activations from artifact A.
        acts_b: (batch, seq, n_features_b) activations from artifact B.
        threshold: activation magnitude threshold for "active".

    Returns:
        Mean Jaccard similarity over positions (0 to 1).
    """
    acts_a_flat = acts_a.reshape(-1, acts_a.shape[-1])
    acts_b_flat = acts_b.reshape(-1, acts_b.shape[-1])

    active_a = (acts_a_flat.abs() > threshold).float()
    active_b = (acts_b_flat.abs() > threshold).float()

    n_active_a = active_a.sum(dim=-1)
    n_active_b = active_b.sum(dim=-1)

    # Jaccard over the count of active features (not identity-matched):
    # Use the ratio min(n_a, n_b) / max(n_a, n_b) as a proxy for
    # how similarly sparse the two encodings are at each position.
    min_active = torch.minimum(n_active_a, n_active_b)
    max_active = torch.maximum(n_active_a, n_active_b)

    valid = max_active > 0
    if valid.sum() == 0:
        return 0.0

    jaccard = (min_active[valid] / max_active[valid]).mean().item()
    return jaccard


def compute_direction_agreement(
    dirs_a: torch.Tensor,
    dirs_b: torch.Tensor,
) -> float:
    """Mean max cosine similarity between decoder directions.

    For each direction in A, finds the most similar direction in B,
    takes the absolute cosine similarity, and averages over all
    directions in A. Then does the same B->A and averages the two.

    Args:
        dirs_a: (n_features_a, d_model) decoder directions from artifact A.
        dirs_b: (n_features_b, d_model) decoder directions from artifact B.

    Returns:
        Symmetric mean-max-cosine similarity (0 to 1).
    """
    dirs_a = dirs_a.float()
    dirs_b = dirs_b.float()

    norms_a = dirs_a.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    norms_b = dirs_b.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    dirs_a_normed = dirs_a / norms_a
    dirs_b_normed = dirs_b / norms_b

    # Chunk the computation to avoid OOM on large dictionaries
    chunk_size = 1024
    n_a = dirs_a_normed.shape[0]
    n_b = dirs_b_normed.shape[0]

    # A -> B: for each dir in A, max cosine sim to any dir in B
    max_cos_a_to_b = torch.zeros(n_a, device=dirs_a.device)
    for start in range(0, n_a, chunk_size):
        end = min(start + chunk_size, n_a)
        cos_sim = dirs_a_normed[start:end] @ dirs_b_normed.T  # (chunk, n_b)
        max_cos_a_to_b[start:end] = cos_sim.abs().max(dim=-1).values

    # B -> A: for each dir in B, max cosine sim to any dir in A
    max_cos_b_to_a = torch.zeros(n_b, device=dirs_b.device)
    for start in range(0, n_b, chunk_size):
        end = min(start + chunk_size, n_b)
        cos_sim = dirs_b_normed[start:end] @ dirs_a_normed.T  # (chunk, n_a)
        max_cos_b_to_a[start:end] = cos_sim.abs().max(dim=-1).values

    mean_a_to_b = max_cos_a_to_b.mean().item()
    mean_b_to_a = max_cos_b_to_a.mean().item()

    return (mean_a_to_b + mean_b_to_a) / 2.0


@torch.no_grad()
def run_architecture_duality(
    model,
    tasks: list[str],
    artifact_a=None,
    artifact_b=None,
    n_prompts: int = 40,
    hook_name: str | None = None,
    activation_threshold: float = 0.01,
) -> list[EvalResult]:
    """Run architecture duality analysis between two artifact adapters.

    Args:
        model: HookedTransformer model.
        tasks: List of task names to evaluate on.
        artifact_a: First artifact adapter.
        artifact_b: Second artifact adapter.
        n_prompts: Number of prompts per task.
        hook_name: Hook point override (defaults to artifact's hook point).
        activation_threshold: Threshold for "active" feature detection.

    Returns:
        List of EvalResult with architecture_agreement scores.
    """
    if artifact_a is None or artifact_b is None:
        log("  WARNING: two artifact adapters required, skipping architecture duality")
        return []

    tokenizer = model.tokenizer
    results = []

    effective_hook = hook_name or artifact_a.manifest.hook_point
    if not effective_hook:
        effective_hook = "blocks.0.hook_resid_pre"

    # Compute direction agreement once (independent of task)
    dirs_a = artifact_a.directions()
    dirs_b = artifact_b.directions()
    if dirs_a.ndim == 3:
        dirs_a = dirs_a.mean(dim=0)
    if dirs_b.ndim == 3:
        dirs_b = dirs_b.mean(dim=0)
    direction_agreement = compute_direction_agreement(dirs_a, dirs_b)
    log(f"  Direction agreement: {direction_agreement:.4f}")

    for task in tasks:
        prompts = generate_prompts(task, tokenizer, n_prompts)
        if not prompts:
            log(f"  {task}: no prompts, skipping")
            continue

        correct_ids, incorrect_ids = get_token_ids(prompts, tokenizer)
        if not correct_ids:
            log(f"  {task}: no token IDs, skipping")
            continue

        log(f"  {task}: {len(prompts)} prompts")

        # Collect feature activations from both artifacts
        overlap_scores = []
        n = min(len(prompts), len(correct_ids))

        for i in range(n):
            tokens = model.to_tokens(prompts[i].text)

            acts_a = artifact_a.activations(model, tokens, effective_hook)
            acts_b = artifact_b.activations(model, tokens, effective_hook)

            overlap = compute_feature_overlap(acts_a, acts_b, activation_threshold)
            overlap_scores.append(overlap)

            if (i + 1) % 10 == 0:
                log(f"    processed {i+1}/{n} prompts")

        if not overlap_scores:
            continue

        feature_overlap = float(np.mean(overlap_scores))
        architecture_agreement = (feature_overlap + direction_agreement) / 2.0
        passed = architecture_agreement > AGREEMENT_THRESHOLD

        log(f"    feature_overlap={feature_overlap:.4f}  "
            f"direction_agreement={direction_agreement:.4f}  "
            f"agreement={architecture_agreement:.4f}  "
            f"[{'PASS' if passed else 'FAIL'}]")

        type_a = getattr(artifact_a, "artifact_type", "unknown")
        type_b = getattr(artifact_b, "artifact_type", "unknown")

        results.append(EvalResult(
            metric_id="M7.architecture_agreement",
            value=architecture_agreement,
            n_samples=n,
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": task,
                "feature_overlap": feature_overlap,
                "direction_agreement": direction_agreement,
                "architecture_agreement": architecture_agreement,
                "passed": passed,
                "threshold": AGREEMENT_THRESHOLD,
                "activation_threshold": activation_threshold,
                "artifact_a_type": type_a,
                "artifact_b_type": type_b,
                "hook_name": effective_hook,
                "n_features_a": int(dirs_a.shape[0]),
                "n_features_b": int(dirs_b.shape[0]),
            },
        ))

    return results


def main():
    parser = parse_common_args("M7: SAE Architecture Duality")
    parser.add_argument("--hook", default=None, help="Hook point for artifact activations")
    parser.add_argument("--artifact-a-type", default="sae",
                        choices=["sae", "transcoder", "crosscoder", "factor_bank"],
                        help="Artifact adapter type for A")
    parser.add_argument("--artifact-a-path", default=None,
                        help="Path or release ID for artifact A")
    parser.add_argument("--sae-a-id", default=None,
                        help="SAE ID for artifact A (for SAELens artifacts)")
    parser.add_argument("--artifact-b-type", default="sae",
                        choices=["sae", "transcoder", "crosscoder", "factor_bank"],
                        help="Artifact adapter type for B")
    parser.add_argument("--artifact-b-path", default=None,
                        help="Path or release ID for artifact B")
    parser.add_argument("--sae-b-id", default=None,
                        help="SAE ID for artifact B (for SAELens artifacts)")
    parser.add_argument("--activation-threshold", type=float, default=0.01,
                        help="Threshold for active feature detection")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    artifact_a = _load_artifact(args.artifact_a_type, args.artifact_a_path,
                                args.sae_a_id, args.hook)
    artifact_b = _load_artifact(args.artifact_b_type, args.artifact_b_path,
                                args.sae_b_id, args.hook)

    log("=" * 60)
    log("M7: SAE ARCHITECTURE DUALITY")
    log("=" * 60)

    out = args.out or "110_architecture_duality.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_architecture_duality(
            model, [task],
            artifact_a=artifact_a,
            artifact_b=artifact_b,
            n_prompts=args.n_prompts,
            hook_name=args.hook,
            activation_threshold=args.activation_threshold,
        )
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)
            p = "PASS" if r.metadata["passed"] else "FAIL"
            log(f"  {task}: agreement={r.value:.4f}  [{p}]")

    save_results(results, out, args=args)
    log(f"\nDone. {len(results)} tasks evaluated.")


def _load_artifact(artifact_type: str, artifact_path: str | None,
                   sae_id: str | None, hook: str | None):
    if artifact_path is None:
        return None

    if artifact_type == "sae":
        from mechval.lib.artifacts import SAEAdapter
        return SAEAdapter.from_pretrained(
            release=artifact_path,
            sae_id=sae_id or "",
            hook_point=hook or "",
        )
    elif artifact_type == "transcoder":
        from mechval.lib.artifacts import TranscoderAdapter
        return TranscoderAdapter.from_pretrained(
            release=artifact_path,
            sae_id=sae_id or "",
            hook_point=hook or "",
        )
    else:
        log(f"  WARNING: unsupported artifact type {artifact_type}")
        return None


if __name__ == "__main__":
    main()
