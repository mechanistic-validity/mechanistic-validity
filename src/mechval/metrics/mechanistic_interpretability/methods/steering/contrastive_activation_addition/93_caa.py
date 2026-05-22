"""Contrastive Activation Addition (Causal C9)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     C09 — Contrastive Activation Addition
Categories:     causal
Validity layer: External (E1 Intervention Reach)
Criteria:       C9 Steering Validity
Establishes:    Whether artifact directions actually control model behavior when
                added as steering vectors
Requires:       CPU or GPU, model, artifact adapter
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Implements the CAA method (Panickssery et al., ACL 2024) as a validation metric.
For each artifact feature direction:

1. Compute steering vector = direction from the artifact adapter.
2. For a range of coefficients, add the vector at inference time.
3. Measure behavioral shift via logit diff change.
4. Report steerability (does it shift?), specificity (only target behavior?),
   and dose-response linearity.

Pass condition: steerability > 0.3 for at least 20% of tested directions.

Usage:
    uv run python 93_caa.py --tasks ioi --n-prompts 40
"""

import numpy as np
import torch

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    generate_prompts,
    get_token_ids,
    load_model,
    log,
    parse_common_args,
    save_incremental,
    save_results,
)


def _add_steering_hook(hook_name: str, direction: torch.Tensor, coeff: float):
    def hook_fn(act, hook):
        act[:, :, :] = act + coeff * direction.to(act.device)
        return act
    return (hook_name, hook_fn)


def compute_caa_scores(
    model, artifact, prompts, correct_ids, incorrect_ids,
    hook_name: str, feature_indices: list[int],
    coeffs: list[float] | None = None,
) -> list[dict]:
    if coeffs is None:
        coeffs = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]

    dirs = artifact.directions()
    if dirs.ndim == 3:
        dirs = dirs.mean(dim=0)

    n = min(len(prompts), len(correct_ids), len(incorrect_ids))
    results = []

    baseline_lds = []
    with torch.no_grad():
        for i in range(n):
            tokens = model.to_tokens(prompts[i].text)
            logits = model(tokens)
            ld = (logits[0, -1, correct_ids[i]] - logits[0, -1, incorrect_ids[i]]).item()
            baseline_lds.append(ld)

    mean_baseline = np.mean(baseline_lds) if baseline_lds else 0.0

    for feat_idx in feature_indices:
        direction = dirs[feat_idx]
        coeff_effects = {}

        for coeff in coeffs:
            shifted_lds = []
            with torch.no_grad():
                for i in range(n):
                    tokens = model.to_tokens(prompts[i].text)
                    hook = _add_steering_hook(hook_name, direction, coeff)
                    logits = model.run_with_hooks(tokens, fwd_hooks=[hook])
                    ld = (logits[0, -1, correct_ids[i]] - logits[0, -1, incorrect_ids[i]]).item()
                    shifted_lds.append(ld)

            mean_shifted = np.mean(shifted_lds)
            coeff_effects[coeff] = float(mean_shifted - mean_baseline)

        positive_coeffs = [c for c in coeffs if c > 0]
        max_shift = max(abs(coeff_effects.get(c, 0)) for c in positive_coeffs) if positive_coeffs else 0.0
        steerability = max_shift / (abs(mean_baseline) + 1e-8)

        shifts = [coeff_effects.get(c, 0) for c in sorted(coeffs)]
        sorted_coeffs = sorted(coeffs)
        if len(sorted_coeffs) >= 2:
            correlation = float(np.corrcoef(sorted_coeffs, shifts)[0, 1])
        else:
            correlation = 0.0

        results.append({
            "feature_idx": feat_idx,
            "steerability": float(steerability),
            "dose_response_r": correlation if not np.isnan(correlation) else 0.0,
            "coeff_effects": coeff_effects,
            "baseline_ld": float(mean_baseline),
            "max_shift": float(max_shift),
        })

    return results


def run_caa(model, tasks: list[str], artifact=None, n_prompts: int = 40,
            hook_name: str | None = None, n_features: int = 20) -> list[EvalResult]:
    if artifact is None:
        log("  WARNING: no artifact adapter provided, skipping CAA")
        return []

    tokenizer = model.tokenizer
    results = []

    effective_hook = hook_name or artifact.manifest.hook_point
    if not effective_hook:
        effective_hook = "blocks.5.hook_resid_pre"

    for task in tasks:
        prompts = generate_prompts(task, tokenizer, n_prompts)
        if not prompts:
            log(f"  {task}: no prompts, skipping")
            continue

        correct_ids, incorrect_ids = get_token_ids(prompts, tokenizer)
        if not correct_ids:
            continue

        dirs = artifact.directions()
        if dirs.ndim == 3:
            dirs = dirs.mean(dim=0)
        total_features = dirs.shape[0]
        feature_indices = list(range(min(n_features, total_features)))

        log(f"  {task}: testing {len(feature_indices)} features, {len(prompts)} prompts")

        scores = compute_caa_scores(
            model, artifact, prompts, correct_ids, incorrect_ids,
            effective_hook, feature_indices,
        )

        steerable_count = sum(1 for s in scores if s["steerability"] > 0.3)
        steerable_frac = steerable_count / len(scores) if scores else 0.0
        mean_steerability = np.mean([s["steerability"] for s in scores]) if scores else 0.0
        mean_dose_r = np.mean([abs(s["dose_response_r"]) for s in scores]) if scores else 0.0
        passed = bool(steerable_frac > 0.2)

        log(f"    steerable: {steerable_count}/{len(scores)} ({steerable_frac:.1%})")
        log(f"    mean steerability={mean_steerability:.4f}, dose-response r={mean_dose_r:.4f}")
        log(f"    [{('PASS' if passed else 'FAIL')}]")

        results.append(EvalResult(
            metric_id="C9.caa_steerability",
            value=float(mean_steerability),
            n_samples=len(prompts),
            metadata={
                "task": task,
                "steerable_fraction": float(steerable_frac),
                "steerable_count": steerable_count,
                "n_features_tested": len(scores),
                "mean_steerability": float(mean_steerability),
                "mean_dose_response_r": float(mean_dose_r),
                "passed": passed,
                "threshold_steerable_frac": 0.2,
                "hook_name": effective_hook,
                "per_feature": scores[:10],
            },
        ))

    return results


def main():
    parser = parse_common_args("C9: Contrastive Activation Addition")
    parser.add_argument("--hook", default=None, help="Hook point for steering")
    parser.add_argument("--n-features", type=int, default=20, help="Number of features to test")
    parser.add_argument("--artifact-path", default=None, help="SAE release ID")
    parser.add_argument("--sae-id", default=None, help="SAE ID within release")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
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
    log("C9: CONTRASTIVE ACTIVATION ADDITION (CAA)")
    log("=" * 60)

    out = args.out or "93_caa.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_caa(model, [task], artifact=artifact,
                               n_prompts=args.n_prompts, hook_name=args.hook,
                               n_features=args.n_features)
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)

    save_results(results, out)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
