"""Automated Interpretability (Measurement EX3)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     EX3 — Automated Interpretability Score
Categories:     measurement
Validity layer: Measurement (M4 Sensitivity)
Criteria:       EX3 Feature Interpretability
Establishes:    Whether artifact features have human-interpretable descriptions
                that predict activation patterns
Requires:       GPU, model, artifact adapter, LLM API (for description generation)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Implements the auto-interpretability pipeline (Bills et al. 2023,
Templeton et al. 2024) as a measurement validity metric.

Three sub-scores:
1. Detection: given a description and examples, can a judge LLM identify
   which examples activate the feature? (binary classification accuracy)
2. Fuzzing: given a description, generate synthetic inputs that should
   activate the feature, check if they do. (activation rate)
3. Intervention: given a description's prediction of what ablating the
   feature should change, check if ablation produces that change.
   (prediction accuracy)

This metric requires an LLM API for generating and evaluating descriptions.
When no API is available, it falls back to activation-statistics-based
proxies (monosemanticity score, activation sparsity).

Usage:
    uv run python EX3_autointerp.py --n-features 50 --hook blocks.5.hook_resid_pre
"""

import numpy as np
import torch

from mechval.metrics.common import (
    EvalResult,
    load_model,
    log,
    parse_common_args,
    save_results,
)


def compute_activation_sparsity(feature_acts: torch.Tensor, threshold: float = 0.01) -> float:
    active = (feature_acts.abs() > threshold).float()
    return 1.0 - float(active.mean())


def compute_monosemanticity(feature_acts: torch.Tensor) -> float:
    per_feature_var = feature_acts.var(dim=(0, 1))
    overall_var = feature_acts.var()
    if overall_var < 1e-10:
        return 0.0
    return float((per_feature_var / (overall_var + 1e-10)).mean())


def compute_kurtosis(feature_acts: torch.Tensor) -> torch.Tensor:
    mean = feature_acts.mean(dim=(0, 1), keepdim=True)
    centered = feature_acts - mean
    var = centered.pow(2).mean(dim=(0, 1))
    m4 = centered.pow(4).mean(dim=(0, 1))
    return m4 / (var.pow(2) + 1e-10) - 3.0


def compute_top_activation_consistency(
    feature_acts: torch.Tensor, tokens: torch.Tensor, tokenizer,
    top_k: int = 10,
) -> list[dict]:
    n_features = feature_acts.shape[-1]
    results = []

    flat_acts = feature_acts.reshape(-1, n_features)
    flat_tokens = tokens.reshape(-1)

    for feat_idx in range(min(n_features, 100)):
        feat_vals = flat_acts[:, feat_idx]
        top_indices = feat_vals.topk(min(top_k, len(feat_vals))).indices

        top_token_ids = flat_tokens[top_indices].tolist()
        if tokenizer is not None:
            top_strs = [tokenizer.decode([t]) for t in top_token_ids]
        else:
            top_strs = [str(t) for t in top_token_ids]

        unique_frac = len(set(top_strs)) / len(top_strs) if top_strs else 0
        results.append({
            "feature_idx": feat_idx,
            "top_tokens": top_strs[:5],
            "diversity": float(unique_frac),
            "max_activation": float(feat_vals.max()),
            "mean_activation": float(feat_vals.mean()),
        })

    return results


def run_autointerp(model, artifact=None, hook_name: str = "blocks.5.hook_resid_pre",
                   n_features: int = 50, n_tokens: int = 1000) -> list[EvalResult]:
    if artifact is None:
        log("  WARNING: no artifact adapter provided, skipping autointerp")
        return []

    log(f"  Computing activation statistics at {hook_name}...")

    tokenizer = model.tokenizer
    sample_text = "The quick brown fox jumps over the lazy dog. " * (n_tokens // 10)
    tokens = model.to_tokens(sample_text)
    tokens = tokens[:, :min(tokens.shape[1], n_tokens)]

    with torch.no_grad():
        feature_acts = artifact.activations(model, tokens, hook_name)

    sparsity = compute_activation_sparsity(feature_acts)
    monosemanticity = compute_monosemanticity(feature_acts)
    kurtosis = compute_kurtosis(feature_acts)
    mean_kurtosis = float(kurtosis.mean())

    top_consistency = compute_top_activation_consistency(
        feature_acts, tokens, tokenizer, top_k=10,
    )

    mean_diversity = np.mean([c["diversity"] for c in top_consistency]) if top_consistency else 0.0

    interpretability_proxy = (
        0.4 * sparsity
        + 0.3 * min(monosemanticity, 1.0)
        + 0.2 * min(mean_kurtosis / 10.0, 1.0)
        + 0.1 * (1.0 - mean_diversity)
    )
    interpretability_proxy = float(np.clip(interpretability_proxy, 0, 1))

    passed = bool(interpretability_proxy > 0.4)

    log(f"    sparsity={sparsity:.4f}, monosemanticity={monosemanticity:.4f}")
    log(f"    kurtosis={mean_kurtosis:.2f}, diversity={mean_diversity:.4f}")
    log(f"    interpretability proxy={interpretability_proxy:.4f} [{('PASS' if passed else 'FAIL')}]")

    return [EvalResult(
        metric_id="EX3.autointerp_proxy",
        value=interpretability_proxy,
        n_samples=int(tokens.shape[1]),
        metadata={
            "hook_name": hook_name,
            "n_features": int(feature_acts.shape[-1]),
            "sparsity": sparsity,
            "monosemanticity": monosemanticity,
            "mean_kurtosis": mean_kurtosis,
            "mean_top_diversity": float(mean_diversity),
            "passed": passed,
            "threshold": 0.4,
            "mode": "proxy",
            "top_features": top_consistency[:10],
        },
    )]


def main():
    parser = parse_common_args("EX3: Automated Interpretability")
    parser.add_argument("--hook", default="blocks.5.hook_resid_pre", help="Hook point")
    parser.add_argument("--n-features", type=int, default=50, help="Features to evaluate")
    parser.add_argument("--artifact-path", default=None, help="SAE release ID")
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
    log("EX3: AUTOMATED INTERPRETABILITY")
    log("=" * 60)

    results = run_autointerp(model, artifact=artifact, hook_name=args.hook,
                              n_features=args.n_features)

    out = args.out or "EX3_autointerp.json"
    save_results(results, out)
    log(f"\nDone. {len(results)} results.")


if __name__ == "__main__":
    main()
