"""IIA Variant Suite (Neuron-Level, IIA@k, Cross-Layer)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     A02 — Counterfactual DAS/IIA
Categories:     causal, representational
Validity layer: Internal + Representational
Criteria:       I2 Sufficiency
Establishes:    IIA holds across neuron-level, top-k, and cross-layer intervention granularities
Requires:       GPU, model
Doc:            /instruments_v2/causal/a02-counterfactual-das
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Extends 01_das_iia with three IIA variants from the 81-metric taxonomy:

  Metric #7  — Neuron-level IIA: swap raw hook_z (no learned rotation).
  Metric #14 — IIA@k: DAS-IIA restricted to the top-k heads by
               individual IIA contribution. Sweeps k=1,2,4,8,15.
  Metric #15 — Cross-layer IIA: DAS-IIA grouped by layer.

Usage:
    uv run python 15_iia_variants.py --tasks ioi sva --n-prompts 40
    uv run python 15_iia_variants.py --device cuda --subspace-dim 2
"""
import importlib

import numpy as np
import torch

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    generate_prompts,
    get_circuit_heads,
    get_token_ids,
    load_model,
    log,
    logit_diff_from_logits,
    parse_common_args,
    save_results,
)

_das = importlib.import_module("mechval.metrics.core.causal.counterfactual_das.01_das_iia")
compute_iia_with_rotation = _das.compute_iia_with_rotation
make_counterfactual_pairs = _das.make_counterfactual_pairs
train_rotation = _das.train_rotation


# ---------------------------------------------------------------------------
# Metric #7 — Neuron-level IIA (no learned rotation)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_neuron_iia(model, prompts, correct_ids, incorrect_ids,
                       pairs, layer: int, head: int, device: str) -> float:
    """IIA by swapping the full hook_z vector at (layer, head) — no rotation."""
    n_correct = 0
    n_total = 0
    hook_name = f"blocks.{layer}.attn.hook_z"

    for base_idx, source_idx in pairs:
        base_tokens = model.to_tokens(prompts[base_idx].text).to(device)
        source_tokens = model.to_tokens(prompts[source_idx].text).to(device)

        _, source_cache = model.run_with_cache(
            source_tokens, names_filter=lambda n: n == hook_name,
        )
        source_z = source_cache[hook_name][0, -1, head].clone()

        def _hook(z, hook, _head=head, _z=source_z):
            z[0, -1, _head] = _z
            return z

        logits = model.run_with_hooks(base_tokens, fwd_hooks=[(hook_name, _hook)])
        source_correct_id = correct_ids[source_idx]
        source_incorrect_id = incorrect_ids[source_idx]
        ld = logit_diff_from_logits(logits, source_correct_id, source_incorrect_id)

        if ld > 0:
            n_correct += 1
        n_total += 1

    return n_correct / max(n_total, 1)


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def run_iia_variants(model, tasks: list[str], n_prompts: int = 40,
                     subspace_dim: int = 2,
                     iia_k_values: list[int] | None = None) -> list[EvalResult]:
    if iia_k_values is None:
        iia_k_values = [1, 2, 4, 8, 15]

    tokenizer = model.tokenizer
    device = str(model.cfg.device)
    rng = np.random.RandomState(42)
    results = []

    for task in tasks:
        circuit_heads = get_circuit_heads(task)
        if not circuit_heads:
            log(f"  {task}: no circuit, skipping")
            continue

        prompts = generate_prompts(task, tokenizer, n_prompts)
        if not prompts:
            log(f"  {task}: no prompts, skipping")
            continue

        correct_ids, incorrect_ids = get_token_ids(prompts, tokenizer)
        if not correct_ids:
            continue

        n_train = len(prompts) // 2
        train_prompts = prompts[:n_train]
        test_prompts = prompts[n_train:]
        train_correct = correct_ids[:n_train]
        train_incorrect = incorrect_ids[:n_train]
        test_correct = correct_ids[n_train:]
        test_incorrect = incorrect_ids[n_train:]

        train_pairs = make_counterfactual_pairs(
            train_prompts, train_correct, train_incorrect, rng,
        )
        test_pairs = make_counterfactual_pairs(
            test_prompts, test_correct, test_incorrect, rng,
        )

        sorted_heads = sorted(circuit_heads)
        log(f"  {task} ({len(sorted_heads)} heads, {len(prompts)} prompts)...")

        # --- Metric #7: Neuron-level IIA ---
        best_neuron_iia = 0.0
        best_neuron_head = None
        per_head_neuron = {}

        for L, H in sorted_heads:
            iia = compute_neuron_iia(
                model, test_prompts, test_correct, test_incorrect,
                test_pairs, L, H, device,
            )
            per_head_neuron[f"L{L}H{H}"] = iia
            if iia > best_neuron_iia:
                best_neuron_iia = iia
                best_neuron_head = (L, H)

        log(f"    neuron-IIA: best={best_neuron_iia:.3f} (head={best_neuron_head})")
        results.append(EvalResult(
            metric_id="C15.neuron_iia",
            value=best_neuron_iia,
            n_samples=len(test_prompts),
            metadata={
                "task": task,
                "best_head": list(best_neuron_head) if best_neuron_head else None,
                "per_head_iia": per_head_neuron,
                "n_circuit_heads": len(sorted_heads),
            },
        ))

        # --- Train DAS rotations for all heads (used by IIA@k and cross-layer) ---
        per_head_das_iia = {}
        per_head_rotation = {}

        for L, H in sorted_heads:
            R = train_rotation(
                model, train_prompts, train_correct, train_incorrect,
                train_pairs, L, H, subspace_dim, device,
            )
            iia = compute_iia_with_rotation(
                model, test_prompts, test_correct, test_incorrect,
                test_pairs, L, H, R, device,
            )
            per_head_das_iia[f"L{L}H{H}"] = iia
            per_head_rotation[(L, H)] = R

        # --- Metric #14: IIA@k ---
        ranked_heads = sorted(
            per_head_das_iia.items(), key=lambda x: x[1], reverse=True,
        )

        for k in iia_k_values:
            if k > len(ranked_heads):
                continue
            top_k_keys = [key for key, _ in ranked_heads[:k]]
            best_iia_at_k = max(per_head_das_iia[key] for key in top_k_keys)
            log(f"    IIA@{k}: {best_iia_at_k:.3f} (top heads: {top_k_keys[:3]}...)")

            results.append(EvalResult(
                metric_id=f"C15.iia_at_k{k}",
                value=best_iia_at_k,
                n_samples=len(test_prompts),
                metadata={
                    "task": task,
                    "k": k,
                    "subspace_dim": subspace_dim,
                    "top_k_heads": top_k_keys,
                    "top_k_iias": [per_head_das_iia[key] for key in top_k_keys],
                },
            ))

        # --- Metric #15: Cross-layer IIA ---
        layer_groups: dict[int, list[tuple[int, int]]] = {}
        for L, H in sorted_heads:
            layer_groups.setdefault(L, []).append((L, H))

        per_layer_iia = {}
        for layer_idx in sorted(layer_groups.keys()):
            layer_head_list = layer_groups[layer_idx]
            best_layer_iia = 0.0
            for L, H in layer_head_list:
                iia = per_head_das_iia[f"L{L}H{H}"]
                if iia > best_layer_iia:
                    best_layer_iia = iia
            per_layer_iia[layer_idx] = best_layer_iia

        best_layer = max(per_layer_iia, key=per_layer_iia.get) if per_layer_iia else -1
        best_cross_iia = per_layer_iia.get(best_layer, 0.0)

        log(f"    cross-layer: best_layer={best_layer} IIA={best_cross_iia:.3f}")
        log(f"      per-layer: {per_layer_iia}")

        results.append(EvalResult(
            metric_id="C15.cross_layer_iia",
            value=best_cross_iia,
            n_samples=len(test_prompts),
            metadata={
                "task": task,
                "subspace_dim": subspace_dim,
                "best_layer": best_layer,
                "per_layer_iia": {str(k): v for k, v in per_layer_iia.items()},
                "n_layers_with_circuit_heads": len(layer_groups),
            },
        ))

    return results


def main():
    parser = parse_common_args("C15: IIA Variant Suite")
    parser.add_argument("--subspace-dim", type=int, default=2,
                        help="DAS subspace dimension (default: 2)")
    parser.add_argument("--iia-k-values", nargs="+", type=int,
                        default=[1, 2, 4, 8, 15],
                        help="k values for IIA@k sweep")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("C15: IIA VARIANT SUITE (Neuron-Level, IIA@k, Cross-Layer)")
    log("=" * 60)

    results = run_iia_variants(
        model, tasks, args.n_prompts, args.subspace_dim, args.iia_k_values,
    )

    out = args.out or "15_iia_variants.json"
    save_results(results, out)

    log(f"\nDone. {len(results)} results across "
        f"{len(set(r.metadata['task'] for r in results))} tasks.")


if __name__ == "__main__":
    main()
