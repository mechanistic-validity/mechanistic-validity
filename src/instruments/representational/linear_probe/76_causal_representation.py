"""Causal Representation Test (Interchange Intervention)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     E02 — Linear Probe
Categories:     representational
Validity layer: Internal
Criteria:       R3 Causal Representation (proposed)
Establishes:    Whether representation is load-bearing (not just decodable)
Requires:       GPU or CPU, model
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Simplified interchange intervention accuracy (IIA) without DAS rotation.
Generates counterfactual prompt pairs, patches residual stream from source
into base at circuit layers, and checks whether model output follows the
patched source. Control: patch at a random non-circuit layer.

Pass condition: circuit_layer_IIA > 0.7 and control_layer_IIA < 0.3.

Usage:
    uv run python 76_causal_representation.py --tasks ioi sva
    uv run python 76_causal_representation.py --device cpu --n-prompts 40
"""
import sys
from pathlib import Path

import numpy as np
import torch

_INSTRUMENTS = Path(__file__).resolve().parents[2]  # up to src/instruments/
sys.path.insert(0, str(_INSTRUMENTS))
from _common import (
    CIRCUIT_TASKS,
    EvalResult,
    generate_prompts,
    get_circuit_heads,
    get_token_ids,
    heads_to_layer_dict,
    load_model,
    log,
    parse_common_args,
    save_results,
)

IIA_CIRCUIT_THRESHOLD = 0.70
IIA_CONTROL_THRESHOLD = 0.30


def make_counterfactual_pairs(prompts, correct_ids) -> list[tuple[int, int]]:
    """Generate pairs of prompt indices with different correct answers."""
    n = min(len(prompts), len(correct_ids))
    pairs = []
    used = set()

    for i in range(n):
        if i in used:
            continue
        for j in range(i + 1, n):
            if j in used:
                continue
            if correct_ids[i] != correct_ids[j]:
                pairs.append((i, j))
                used.add(i)
                used.add(j)
                break

    return pairs


@torch.no_grad()
def cache_residual_at_layer(model, tokens: torch.Tensor,
                            layer: int) -> torch.Tensor:
    """Run forward pass and cache residual stream at a specific layer (last position)."""
    hook_name = f"blocks.{layer}.hook_resid_post"
    _, cache = model.run_with_cache(tokens, names_filter=lambda n: n == hook_name)
    return cache[hook_name][0, -1].clone()  # (d_model,)


@torch.no_grad()
def run_with_patch(model, tokens: torch.Tensor, layer: int,
                   patched_activation: torch.Tensor) -> torch.Tensor:
    """Run forward pass with residual stream patched at a specific layer (last position)."""
    hook_name = f"blocks.{layer}.hook_resid_post"

    def patch_hook(activation, hook):
        activation[0, -1] = patched_activation.to(activation.device)
        return activation

    logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, patch_hook)])
    return logits


@torch.no_grad()
def compute_iia(model, prompts, correct_ids, pairs: list[tuple[int, int]],
                layer: int) -> float:
    """Compute interchange intervention accuracy at a given layer.

    For each pair (A, B):
      - Cache residual from prompt A at the layer
      - Run prompt B with patched residual from A
      - Check if output follows A's correct answer
    IIA = fraction where patched output matches source (A).
    """
    n_follow_source = 0
    n_total = 0

    for idx_a, idx_b in pairs:
        tokens_a = model.to_tokens(prompts[idx_a].text)
        tokens_b = model.to_tokens(prompts[idx_b].text)

        # Cache source activation from prompt A
        source_act = cache_residual_at_layer(model, tokens_a, layer)

        # Run prompt B with patched activation from A
        patched_logits = run_with_patch(model, tokens_b, layer, source_act)

        # Check: does the model now predict A's correct answer?
        source_correct = correct_ids[idx_a]
        base_correct = correct_ids[idx_b]

        last_logits = patched_logits[0, -1]
        source_logit = last_logits[source_correct].item()
        base_logit = last_logits[base_correct].item()

        if source_logit > base_logit:
            n_follow_source += 1
        n_total += 1

    if n_total == 0:
        return 0.0
    return n_follow_source / n_total


def pick_control_layer(n_layers: int, circuit_layers: set[int],
                       seed: int = 42) -> int:
    """Pick a random non-circuit layer for the control condition."""
    rng = np.random.RandomState(seed)
    non_circuit = [L for L in range(n_layers) if L not in circuit_layers]
    if not non_circuit:
        # Fallback: use layer 0 if all layers are circuit layers
        return 0
    return int(rng.choice(non_circuit))


def run_causal_representation(model, tasks: list[str],
                              n_prompts: int = 40) -> list[EvalResult]:
    tokenizer = model.tokenizer
    n_layers = model.cfg.n_layers
    results = []

    for task in tasks:
        circuit_heads = get_circuit_heads(task)
        if not circuit_heads:
            log(f"  {task}: no circuit, skipping")
            continue

        prompts = generate_prompts(task, tokenizer, n_prompts)
        if not prompts:
            continue

        correct_ids, incorrect_ids = get_token_ids(prompts, tokenizer)
        if not correct_ids:
            continue

        # Build counterfactual pairs
        pairs = make_counterfactual_pairs(prompts, correct_ids)
        if len(pairs) < 3:
            log(f"  {task}: too few counterfactual pairs ({len(pairs)}), skipping")
            continue

        # Find layers with circuit heads
        circuit_layers_dict = heads_to_layer_dict(circuit_heads)
        circuit_layer_set = set(circuit_layers_dict.keys())
        circuit_layers_sorted = sorted(circuit_layer_set)

        # Pick control layer
        control_layer = pick_control_layer(n_layers, circuit_layer_set)

        log(f"  {task} ({len(circuit_heads)} heads, {len(pairs)} pairs, "
            f"circuit layers={circuit_layers_sorted}, control={control_layer})...")

        # Compute IIA at each circuit layer
        per_layer_iia = {}
        best_circuit_iia = 0.0

        for layer in circuit_layers_sorted:
            iia = compute_iia(model, prompts, correct_ids, pairs, layer)
            per_layer_iia[layer] = iia
            log(f"    layer {layer} (circuit): IIA={iia:.3f}")
            if iia > best_circuit_iia:
                best_circuit_iia = iia

        # Compute IIA at control layer
        control_iia = compute_iia(model, prompts, correct_ids, pairs, control_layer)
        log(f"    layer {control_layer} (control): IIA={control_iia:.3f}")

        # Pass conditions
        pass_circuit = best_circuit_iia > IIA_CIRCUIT_THRESHOLD
        pass_control = control_iia < IIA_CONTROL_THRESHOLD
        passed = pass_circuit and pass_control

        log(f"    circuit_IIA={best_circuit_iia:.3f} "
            f"({'PASS' if pass_circuit else 'FAIL'} >{IIA_CIRCUIT_THRESHOLD}), "
            f"control_IIA={control_iia:.3f} "
            f"({'PASS' if pass_control else 'FAIL'} <{IIA_CONTROL_THRESHOLD})")

        results.append(EvalResult(
            metric_id="R3.causal_representation",
            value=best_circuit_iia,
            n_samples=len(pairs),
            metadata={
                "task": task,
                "n_circuit_heads": len(circuit_heads),
                "n_pairs": len(pairs),
                "circuit_layers": circuit_layers_sorted,
                "control_layer": control_layer,
                "per_layer_iia": per_layer_iia,
                "best_circuit_iia": best_circuit_iia,
                "control_iia": control_iia,
                "pass_circuit": pass_circuit,
                "pass_control": pass_control,
                "passed": passed,
                "iia_circuit_threshold": IIA_CIRCUIT_THRESHOLD,
                "iia_control_threshold": IIA_CONTROL_THRESHOLD,
            },
        ))

    return results


def main():
    parser = parse_common_args("R3: Causal Representation Test")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("R3: CAUSAL REPRESENTATION TEST (INTERCHANGE INTERVENTION)")
    log("=" * 60)

    results = run_causal_representation(model, tasks, args.n_prompts)

    out = args.out or "76_causal_representation.json"
    save_results(results, out)

    log(f"\nDone. {len(results)} tasks evaluated.")
    for r in results:
        t = r.metadata["task"]
        log(f"  {t}: circuit_IIA={r.metadata['best_circuit_iia']:.3f}, "
            f"control_IIA={r.metadata['control_iia']:.3f}, "
            f"passed={r.metadata['passed']}")


if __name__ == "__main__":
    main()
