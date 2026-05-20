"""Centered Kernel Alignment (CKA)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     E92 — CKA
Categories:     representational
Validity layer: Internal
Criteria:       I4 Consistency
Establishes:    Circuit subnetwork captures full model representation structure
Requires:       CPU or GPU, model
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For each task, collects activations from circuit heads and all heads on
a set of prompts, then computes linear CKA between the circuit
subnetwork's representation and the full model's representation at each
layer.

CKA measures similarity between two sets of representations -- higher
CKA means the circuit captures more of the full model's computation.

Linear CKA(X, Y) = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
where X and Y are centered activation matrices (n_samples x d).

Pass condition: mean CKA across layers > 0.60

Usage:
    uv run python 92_cka.py --tasks ioi sva --device cpu
    uv run python 92_cka.py --device cuda --n-prompts 40
"""

import numpy as np
import torch

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    generate_prompts,
    get_circuit_heads,
    load_model,
    log,
    parse_common_args,
    save_incremental,
    save_results,
)


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute linear CKA between two centered activation matrices.

    X: (n, d1), Y: (n, d2) -- both should be centered.
    Returns CKA score in [0, 1].
    """
    XtX = X.T @ X
    YtY = Y.T @ Y
    YtX = Y.T @ X

    numerator = np.linalg.norm(YtX, "fro") ** 2
    denominator = np.linalg.norm(XtX, "fro") * np.linalg.norm(YtY, "fro")

    if denominator < 1e-10:
        return 0.0
    return float(numerator / denominator)


@torch.no_grad()
def collect_head_activations(model, prompts, layer: int,
                             heads: set[int] | None = None) -> np.ndarray:
    """Collect concatenated head outputs at last token for a layer.

    If heads is None, collects all heads. Returns (n_prompts, n_heads * d_head).
    """
    hook_name = f"blocks.{layer}.attn.hook_z"
    n_heads = model.cfg.n_heads
    selected = sorted(heads) if heads is not None else list(range(n_heads))
    activations = []

    for p in prompts:
        tokens = model.to_tokens(p.text)
        _, cache = model.run_with_cache(tokens, names_filter=lambda n: n == hook_name)
        z = cache[hook_name][0, -1]  # (n_heads, d_head)
        act = z[selected].reshape(-1).cpu().float().numpy()
        activations.append(act)

    return np.stack(activations, axis=0)


@torch.no_grad()
def run_cka(model, task: str, n_prompts: int = 40) -> EvalResult | None:
    tokenizer = model.tokenizer
    n_layers = model.cfg.n_layers

    circuit_heads = get_circuit_heads(task)
    if not circuit_heads:
        log(f"  {task}: no circuit, skipping")
        return None

    prompts = generate_prompts(task, tokenizer, n_prompts)
    if not prompts or len(prompts) < 5:
        log(f"  {task}: insufficient prompts")
        return None

    heads_by_layer: dict[int, set[int]] = {}
    for L, H in circuit_heads:
        heads_by_layer.setdefault(L, set()).add(H)

    log(f"  {task}: {len(circuit_heads)} circuit heads, {len(prompts)} prompts")

    per_layer_cka = []
    for layer in range(n_layers):
        full_acts = collect_head_activations(model, prompts, layer, heads=None)
        full_centered = full_acts - full_acts.mean(axis=0, keepdims=True)

        circuit_heads_in_layer = heads_by_layer.get(layer)
        if circuit_heads_in_layer is None:
            per_layer_cka.append(0.0)
            continue

        circuit_acts = collect_head_activations(model, prompts, layer, circuit_heads_in_layer)
        circuit_centered = circuit_acts - circuit_acts.mean(axis=0, keepdims=True)

        cka = linear_cka(full_centered, circuit_centered)
        per_layer_cka.append(cka)
        log(f"    layer {layer}: CKA={cka:.3f} ({len(circuit_heads_in_layer)} circuit heads)")

    circuit_layer_ckas = [per_layer_cka[L] for L in heads_by_layer]
    mean_cka = float(np.mean(circuit_layer_ckas)) if circuit_layer_ckas else 0.0
    passed = bool(mean_cka > 0.60)

    log(f"    mean CKA (circuit layers)={mean_cka:.3f}  [{'PASS' if passed else 'FAIL'}]")

    return EvalResult(
        metric_id="E92.cka",
        value=mean_cka,
        n_samples=len(prompts),
        metadata={
            "task": task,
            "per_layer_cka": per_layer_cka,
            "circuit_layers": sorted(heads_by_layer.keys()),
            "mean_cka_circuit_layers": mean_cka,
            "passed": passed,
            "threshold": 0.60,
        },
    )


def main():
    parser = parse_common_args("E92: Centered Kernel Alignment")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("E92: CENTERED KERNEL ALIGNMENT (CKA)")
    log("=" * 60)

    out = args.out or "92_cka.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        result = run_cka(model, task, args.n_prompts)
        if result is None:
            continue
        results.append(result)
        save_incremental(result, jsonl_out)
        p = "PASS" if result.metadata["passed"] else "FAIL"
        log(f"  {task}: mean_cka={result.value:.3f}  [{p}]")

    save_results(results, out)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
