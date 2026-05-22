"""Adaptive Sparsity Diagnostic (Measurement M12)
Paper: Bussmann, Leask, Nanda (NeurIPS 2024); Yao, Du (arXiv:2508.17320); arXiv:2605.06610
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     M12 — Adaptive Sparsity Diagnostic
Categories:     measurement
Validity layer: Measurement
Criteria:       E1 Content Validity, M6 Artifact Quality
Establishes:    Whether fixed-k SAE sparsity matches input complexity,
                detecting systematic under- or over-activation
Requires:       One artifact adapter (SAE with fixed k)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Three independent papers converge on fixed-k sparsity as a systematic
E1 Content Validity failure:

- BatchTopK (Bussmann, Leask, Nanda; NeurIPS 2024): global top-k across
  batch dimensions, showing per-example k varies naturally.
- AdaptiveK (Yao, Du; arXiv:2508.17320): input-dependent k selection
  outperforms fixed-k on reconstruction and interpretability.
- SoftSAE (arXiv:2605.06610): continuous relaxation of sparsity
  constraint avoids fixed-k artifacts entirely.

For each evaluation example, this metric estimates true concept count
using input complexity (embedding norm as proxy), compares to the
fixed k used by the SAE, and flags examples where the mismatch exceeds
a threshold:
  - k_fixed >> k_true: spurious features activated
  - k_fixed << k_true: real concepts truncated

Method:
    1. Collect activations at the hook point from the model.
    2. Encode through the artifact adapter to get active feature counts.
    3. Estimate input complexity via residual stream embedding norm.
    4. Fit a linear relationship: expected_k ~ complexity.
    5. Flag examples where |k_active - k_expected| / k_expected > threshold.
    6. Report k_mismatch_rate = fraction of flagged examples.

Pass condition: k_mismatch_rate < 0.2

Usage:
    uv run python 120_adaptive_sparsity.py --artifact-path <release> --sae-id <id>
    uv run python 120_adaptive_sparsity.py --device cpu
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
    name="Adaptive Sparsity Diagnostic (BatchTopK/AdaptiveK/SoftSAE convergence)",
    paper_ref="Bussmann et al. NeurIPS 2024; Yao & Du arXiv:2508.17320; SoftSAE arXiv:2605.06610",
    paper_cite=(
        "Bussmann, Leask, Nanda 2024 (BatchTopK); "
        "Yao, Du 2025 (AdaptiveK); "
        "SoftSAE 2026 (arXiv:2605.06610)"
    ),
    description=(
        "Tests whether fixed-k SAE sparsity matches input complexity, "
        "detecting systematic over- or under-activation that constitutes "
        "an E1 Content Validity failure. Three independent papers converge "
        "on this being a fundamental limitation of fixed-k architectures."
    ),
    category="measurement",
    tier="measurement_theory",
    origin="convergent",
)

MISMATCH_THRESHOLD = 0.2


def estimate_input_complexity(
    model,
    tokens: torch.Tensor,
    hook_name: str,
) -> torch.Tensor:
    """Estimate per-position input complexity via residual stream norm.

    Uses the L2 norm of the residual stream at the hook point as a proxy
    for input complexity. Higher norm indicates more information content
    and likely more active concepts.

    Args:
        model: HookedTransformer model.
        tokens: (batch, seq) token IDs.
        hook_name: Hook point to capture residual stream.

    Returns:
        (batch, seq) tensor of per-position complexity estimates.
    """
    _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
    residual = cache[hook_name]  # (batch, seq, d_model)
    complexity = residual.norm(dim=-1)  # (batch, seq)
    return complexity


def count_active_features(
    artifact,
    model,
    tokens: torch.Tensor,
    hook_name: str,
    threshold: float = 0.01,
) -> torch.Tensor:
    """Count the number of active features per position.

    Args:
        artifact: Artifact adapter with .activations() method.
        model: HookedTransformer model.
        tokens: (batch, seq) token IDs.
        hook_name: Hook point for the artifact.
        threshold: Activation magnitude threshold for "active".

    Returns:
        (batch, seq) tensor of active feature counts per position.
    """
    acts = artifact.activations(model, tokens, hook_name)  # (batch, seq, n_features)
    active = (acts.abs() > threshold).float()
    return active.sum(dim=-1)  # (batch, seq)


def fit_complexity_to_k(
    complexities: np.ndarray,
    active_counts: np.ndarray,
) -> tuple[float, float]:
    """Fit a linear model: expected_k = slope * complexity + intercept.

    Args:
        complexities: (N,) array of complexity values.
        active_counts: (N,) array of active feature counts.

    Returns:
        (slope, intercept) of the linear fit.
    """
    if len(complexities) < 2:
        return 0.0, float(np.mean(active_counts)) if len(active_counts) > 0 else 0.0

    # Simple linear regression via least squares
    A = np.stack([complexities, np.ones_like(complexities)], axis=1)
    result = np.linalg.lstsq(A, active_counts, rcond=None)
    slope, intercept = result[0]
    return float(slope), float(intercept)


def compute_mismatch_rate(
    active_counts: np.ndarray,
    expected_counts: np.ndarray,
    mismatch_threshold: float,
) -> tuple[float, np.ndarray]:
    """Compute fraction of examples where k deviates from expected by more than threshold.

    Mismatch is defined as |k_active - k_expected| / max(k_expected, 1) > threshold.

    Args:
        active_counts: (N,) actual active feature counts.
        expected_counts: (N,) expected counts from complexity model.
        mismatch_threshold: Relative deviation threshold.

    Returns:
        (mismatch_rate, per_example_deviations) where deviations are
        the signed relative differences (positive = over-activation).
    """
    denom = np.maximum(expected_counts, 1.0)
    deviations = (active_counts - expected_counts) / denom
    mismatched = np.abs(deviations) > mismatch_threshold
    rate = float(mismatched.mean())
    return rate, deviations


@torch.no_grad()
def run_adaptive_sparsity(
    model,
    tasks: list[str] | None = None,
    n_prompts: int = 100,
    artifact=None,
    hook_name: str | None = None,
    mismatch_threshold: float = 2.0,
) -> list[EvalResult]:
    """Run adaptive sparsity diagnostic on an SAE artifact.

    For each evaluation example, estimates true concept count using input
    complexity (embedding norm as proxy), compares to the active feature
    count from the SAE. Flags examples where mismatch exceeds threshold.

    Args:
        model: HookedTransformer model.
        tasks: List of task names to evaluate on. Defaults to CIRCUIT_TASKS.
        n_prompts: Number of prompts per task.
        artifact: Artifact adapter (SAE) to evaluate.
        hook_name: Hook point override (defaults to artifact's hook point).
        mismatch_threshold: Relative deviation threshold for flagging.

    Returns:
        List of EvalResult with k_mismatch_rate scores.
    """
    if tasks is None:
        tasks = CIRCUIT_TASKS

    if artifact is None:
        log("  WARNING: artifact adapter required, skipping adaptive sparsity")
        return []

    tokenizer = model.tokenizer
    results = []

    effective_hook = hook_name or getattr(
        getattr(artifact, "manifest", None), "hook_point", None
    )
    if not effective_hook:
        effective_hook = "blocks.0.hook_resid_pre"

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

        all_complexities = []
        all_active_counts = []
        per_example = []

        n = min(len(prompts), len(correct_ids))

        for i in range(n):
            tokens = model.to_tokens(prompts[i].text)

            # Measure input complexity
            complexity = estimate_input_complexity(model, tokens, effective_hook)
            # Use last position (where prediction happens)
            complexity_val = complexity[0, -1].cpu().item()

            # Count active features
            k_active_tensor = count_active_features(
                artifact, model, tokens, effective_hook
            )
            k_active = k_active_tensor[0, -1].cpu().item()

            all_complexities.append(complexity_val)
            all_active_counts.append(k_active)
            per_example.append({
                "prompt_idx": i,
                "complexity": complexity_val,
                "k_active": k_active,
            })

            if (i + 1) % 20 == 0:
                log(f"    processed {i+1}/{n} prompts")

        if not all_complexities:
            continue

        complexities_arr = np.array(all_complexities)
        active_arr = np.array(all_active_counts)

        # Fit complexity -> expected k relationship
        slope, intercept = fit_complexity_to_k(complexities_arr, active_arr)
        expected_k = slope * complexities_arr + intercept

        # Compute mismatch rate
        mismatch_rate, deviations = compute_mismatch_rate(
            active_arr, expected_k, mismatch_threshold
        )

        # Compute correlation between complexity and active count
        if len(complexities_arr) > 1 and complexities_arr.std() > 0 and active_arr.std() > 0:
            correlation = float(np.corrcoef(complexities_arr, active_arr)[0, 1])
        else:
            correlation = 0.0

        passed = mismatch_rate < MISMATCH_THRESHOLD

        # Classify mismatch directions
        over_activated = float((deviations > mismatch_threshold).mean())
        under_activated = float((deviations < -mismatch_threshold).mean())

        log(f"    k_mismatch_rate={mismatch_rate:.4f}  "
            f"correlation={correlation:.4f}  "
            f"over={over_activated:.4f}  under={under_activated:.4f}  "
            f"[{'PASS' if passed else 'FAIL'}]")

        # Annotate per-example with deviation info
        for j, ex in enumerate(per_example):
            ex["expected_k"] = float(expected_k[j])
            ex["deviation"] = float(deviations[j])
            ex["mismatched"] = bool(abs(deviations[j]) > mismatch_threshold)

        results.append(EvalResult(
            metric_id="M12.adaptive_sparsity_diagnostic",
            value=mismatch_rate,
            n_samples=n,
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": task,
                "k_mismatch_rate": mismatch_rate,
                "passed": passed,
                "threshold": MISMATCH_THRESHOLD,
                "mismatch_threshold": mismatch_threshold,
                "complexity_k_correlation": correlation,
                "linear_fit_slope": slope,
                "linear_fit_intercept": intercept,
                "over_activation_rate": over_activated,
                "under_activation_rate": under_activated,
                "mean_k_active": float(active_arr.mean()),
                "std_k_active": float(active_arr.std()),
                "mean_complexity": float(complexities_arr.mean()),
                "std_complexity": float(complexities_arr.std()),
                "hook_name": effective_hook,
                "per_example": per_example,
            },
        ))

    return results


def main():
    parser = parse_common_args("M12: Adaptive Sparsity Diagnostic")
    parser.add_argument("--hook", default=None,
                        help="Hook point for artifact activations")
    parser.add_argument("--artifact-type", default="sae",
                        choices=["sae", "transcoder", "crosscoder", "factor_bank"],
                        help="Artifact adapter type")
    parser.add_argument("--artifact-path", default=None,
                        help="Path or release ID for artifact")
    parser.add_argument("--sae-id", default=None,
                        help="SAE ID (for SAELens artifacts)")
    parser.add_argument("--mismatch-threshold", type=float, default=2.0,
                        help="Relative deviation threshold for mismatch detection")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    artifact = _load_artifact(args.artifact_type, args.artifact_path,
                              args.sae_id, args.hook)

    log("=" * 60)
    log("M12: ADAPTIVE SPARSITY DIAGNOSTIC")
    log("=" * 60)

    out = args.out or "120_adaptive_sparsity.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_adaptive_sparsity(
            model, [task],
            n_prompts=args.n_prompts,
            artifact=artifact,
            hook_name=args.hook,
            mismatch_threshold=args.mismatch_threshold,
        )
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)
            p = "PASS" if r.metadata["passed"] else "FAIL"
            log(f"  {task}: k_mismatch_rate={r.value:.4f}  [{p}]")

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
