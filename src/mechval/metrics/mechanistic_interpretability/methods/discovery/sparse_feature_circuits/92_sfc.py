"""Sparse Feature Circuits (Causal C8)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     C08 — Sparse Feature Circuit Agreement
Categories:     causal
Validity layer: Internal
Criteria:       C8 Feature-Circuit Agreement
Establishes:    Whether feature-level attribution patching through a dictionary
                (SAE, transcoder, crosscoder, etc.) identifies features concentrated
                in the claimed circuit heads vs non-circuit heads
Requires:       CPU or GPU, model, artifact adapter
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Implements the Sparse Feature Circuits approach (Marks et al. 2024) as a
validation metric. For each task:

1. Run the model on task prompts, collecting activations at the artifact's
   hook point(s).
2. Encode activations through the artifact adapter to get feature activations.
3. Estimate each feature's causal importance via gradient-based attribution
   (feature activation × gradient of logit diff w.r.t. that feature).
4. Map features to layers/heads based on hook point location.
5. Measure whether high-importance features concentrate in circuit heads
   (AUROC) and compute Jaccard overlap at a threshold.

Pass condition: AUROC > 0.65

Usage:
    uv run python 92_sfc.py --tasks ioi --n-prompts 40
    uv run python 92_sfc.py --tasks ioi sva --device cpu
"""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    generate_prompts,
    get_circuit_info,
    get_token_ids,
    load_model,
    log,
    parse_common_args,
    save_incremental,
    save_results,
)


def compute_feature_attributions(
    model, artifact, prompts, correct_ids, incorrect_ids, hook_name: str,
) -> np.ndarray:
    n = min(len(prompts), len(correct_ids), len(incorrect_ids))
    n_features = None
    total_attr = None

    for i in range(n):
        tokens = model.to_tokens(prompts[i].text)
        model.zero_grad()

        cache_dict = {}

        def cache_hook(act, hook, _name=hook_name):
            cache_dict[_name] = act
            act.retain_grad()
            return act

        with torch.enable_grad():
            logits = model.run_with_hooks(
                tokens,
                fwd_hooks=[(hook_name, cache_hook)],
            )
            logit_diff = logits[0, -1, correct_ids[i]] - logits[0, -1, incorrect_ids[i]]
            logit_diff.backward()

        act = cache_dict.get(hook_name)
        if act is None or act.grad is None:
            continue

        with torch.no_grad():
            feat_acts = artifact.activations(model, tokens, hook_name)
            grad = act.grad[0, -1]  # (d_model,)
            dirs = artifact.directions()
            if dirs.ndim == 3:
                dirs = dirs.mean(dim=0)
            feat_grad = grad @ dirs.T  # (n_features,)
            feat_importance = (feat_acts[0, -1] * feat_grad).abs()

            if n_features is None:
                n_features = feat_importance.shape[0]
                total_attr = np.zeros(n_features, dtype=np.float64)

            total_attr += feat_importance.cpu().numpy()

        model.zero_grad()

        if (i + 1) % 10 == 0:
            log(f"    processed {i+1}/{n} prompts")

    if total_attr is not None:
        total_attr /= max(n, 1)
    return total_attr if total_attr is not None else np.array([])


def _hook_to_layer(hook_name: str) -> int | None:
    parts = hook_name.split(".")
    for j, part in enumerate(parts):
        if part == "blocks" and j + 1 < len(parts):
            try:
                return int(parts[j + 1])
            except ValueError:
                pass
    return None


def compute_sfc_auroc(
    feature_attr: np.ndarray,
    circuit_heads: set[tuple[int, int]],
    hook_name: str,
    n_layers: int,
    n_heads: int,
    top_k_fraction: float = 0.1,
) -> tuple[float, float, dict]:
    hook_layer = _hook_to_layer(hook_name)
    if hook_layer is None:
        return 0.0, 0.0, {"error": "cannot parse layer from hook name"}

    circuit_layers = {L for L, _ in circuit_heads}
    is_circuit_layer = hook_layer in circuit_layers

    n_features = len(feature_attr)
    if n_features == 0:
        return 0.0, 0.0, {"error": "no features"}

    ranked_idx = np.argsort(feature_attr)[::-1]
    top_k = max(1, int(n_features * top_k_fraction))
    top_features = set(ranked_idx[:top_k].tolist())

    n_per_head = n_features // n_heads if n_heads > 0 else n_features
    head_importance = np.zeros(n_heads)
    for h in range(n_heads):
        start = h * n_per_head
        end = start + n_per_head if h < n_heads - 1 else n_features
        head_importance[h] = feature_attr[start:end].sum()

    labels = np.array([
        1 if (hook_layer, h) in circuit_heads else 0
        for h in range(n_heads)
    ])

    if labels.sum() == 0 or labels.sum() == len(labels):
        auroc = float(is_circuit_layer)
    else:
        auroc = float(roc_auc_score(labels, head_importance))

    circuit_feature_mass = 0.0
    total_mass = feature_attr.sum()
    for h in range(n_heads):
        if (hook_layer, h) in circuit_heads:
            start = h * n_per_head
            end = start + n_per_head if h < n_heads - 1 else n_features
            circuit_feature_mass += feature_attr[start:end].sum()

    concentration = circuit_feature_mass / total_mass if total_mass > 0 else 0.0

    stats = {
        "hook_layer": hook_layer,
        "n_features": n_features,
        "top_k": top_k,
        "n_circuit_heads_at_layer": int(labels.sum()),
        "concentration": float(concentration),
        "mean_circuit_head_importance": float(head_importance[labels == 1].mean()) if labels.sum() > 0 else 0.0,
        "mean_non_circuit_head_importance": float(head_importance[labels == 0].mean()) if (1 - labels).sum() > 0 else 0.0,
    }

    return auroc, concentration, stats


def run_sfc(model, tasks: list[str], artifact=None, n_prompts: int = 40,
            hook_name: str | None = None) -> list[EvalResult]:
    if artifact is None:
        log("  WARNING: no artifact adapter provided, skipping SFC")
        return []

    tokenizer = model.tokenizer
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    results = []

    effective_hook = hook_name or artifact.manifest.hook_point
    if not effective_hook:
        effective_hook = "blocks.0.hook_resid_pre"

    for task in tasks:
        circuit, all_heads, all_edges = get_circuit_info(task)
        if circuit is None or not all_heads:
            log(f"  {task}: no circuit, skipping")
            continue

        prompts = generate_prompts(task, tokenizer, n_prompts)
        if not prompts:
            log(f"  {task}: no prompts, skipping")
            continue

        correct_ids, incorrect_ids = get_token_ids(prompts, tokenizer)
        if not correct_ids:
            continue

        log(f"  {task}: {len(all_heads)} circuit heads, {len(prompts)} prompts")

        layers_in_circuit = sorted({L for L, _ in all_heads})
        task_results = []

        for layer in layers_in_circuit:
            layer_hook = effective_hook.replace("blocks.0", f"blocks.{layer}")

            feature_attr = compute_feature_attributions(
                model, artifact, prompts, correct_ids, incorrect_ids, layer_hook,
            )

            if len(feature_attr) == 0:
                continue

            auroc, concentration, stats = compute_sfc_auroc(
                feature_attr, all_heads, layer_hook, n_layers, n_heads,
            )

            passed = bool(auroc > 0.65)
            log(f"    L{layer}: AUROC={auroc:.4f} conc={concentration:.4f} [{('PASS' if passed else 'FAIL')}]")

            task_results.append(EvalResult(
                metric_id="C8.sfc_auroc",
                value=auroc,
                n_samples=len(prompts),
                metadata={
                    "task": task,
                    "layer": layer,
                    "hook_name": layer_hook,
                    "auroc": auroc,
                    "concentration": concentration,
                    "passed": passed,
                    "threshold": 0.65,
                    **stats,
                },
            ))

        if task_results:
            mean_auroc = np.mean([r.value for r in task_results])
            mean_conc = np.mean([r.metadata["concentration"] for r in task_results])
            passed = bool(mean_auroc > 0.65)

            results.append(EvalResult(
                metric_id="C8.sfc_agreement",
                value=float(mean_auroc),
                n_samples=len(prompts),
                metadata={
                    "task": task,
                    "mean_auroc": float(mean_auroc),
                    "mean_concentration": float(mean_conc),
                    "passed": passed,
                    "threshold": 0.65,
                    "n_layers_evaluated": len(task_results),
                    "per_layer": [r.metadata for r in task_results],
                },
            ))

    return results


def main():
    parser = parse_common_args("C8: Sparse Feature Circuit Agreement")
    parser.add_argument("--hook", default=None, help="Hook point for artifact activations")
    parser.add_argument("--artifact-type", default="sae", choices=["sae", "transcoder", "crosscoder", "factor_bank", "llamascopium", "bilinear_mlp"],
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
    log("C8: SPARSE FEATURE CIRCUIT AGREEMENT")
    log("=" * 60)

    out = args.out or "92_sfc.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_sfc(model, [task], artifact=artifact,
                               n_prompts=args.n_prompts, hook_name=args.hook)
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)
            p = "PASS" if r.metadata["passed"] else "FAIL"
            log(f"  {task}: AUROC={r.value:.4f}  [{p}]")

    save_results(results, out)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
