"""CKA Cross-Layer Analysis (E6b)
---
Instrument:     E6b -- CKA Cross-Layer
Categories:     representational
Validity layer: Internal
Criteria:       E6b Representation Preservation
Establishes:    Information is preserved across circuit layers (Kornblith et al., 2019)
Requires:       CPU or GPU, model
---

CKA (Centered Kernel Alignment) measures similarity between two sets
of representations. This instrument computes:
1. CKA between consecutive circuit layers (information preservation)
2. CKA between circuit layers and non-circuit layers (structural analysis)

Linear CKA(X, Y) = ||K_X K_Y||_F^2 / (||K_X||_F * ||K_Y||_F)
where K_X = X X^T is the linear kernel (Gram matrix).

Pass condition: mean CKA between consecutive circuit layers > 0.3

Usage:
    uv run python E6b_cka_cross_arch.py --tasks ioi sva --device cpu
    uv run python E6b_cka_cross_arch.py --device cuda --n-prompts 20
"""

import numpy as np
import torch

from mechanistic_validity.metrics.common import (
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
    """Compute linear CKA between two activation matrices.

    X: (n, d1), Y: (n, d2). Both are centered before computation.
    CKA(X, Y) = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
    """
    n = X.shape[0]
    if n < 2:
        return 0.0

    X_centered = X - X.mean(axis=0, keepdims=True)
    Y_centered = Y - Y.mean(axis=0, keepdims=True)

    XtX = X_centered.T @ X_centered
    YtY = Y_centered.T @ Y_centered
    YtX = Y_centered.T @ X_centered

    numerator = np.linalg.norm(YtX, "fro") ** 2
    denominator = np.linalg.norm(XtX, "fro") * np.linalg.norm(YtY, "fro")

    if denominator < 1e-10:
        return 0.0
    return float(numerator / denominator)


@torch.no_grad()
def collect_residual_stream(model, prompts, layer: int) -> np.ndarray:
    """Collect residual stream activations at last token for a layer.

    Returns (n_prompts, d_model).
    """
    hook_name = f"blocks.{layer}.hook_resid_post"
    activations = []

    for p in prompts:
        tokens = model.to_tokens(p.text)
        _, cache = model.run_with_cache(tokens, names_filter=lambda n: n == hook_name)
        act = cache[hook_name][0, -1].cpu().float().numpy()  # (d_model,)
        activations.append(act)

    return np.stack(activations, axis=0)


def run_cka_analysis(model, tasks: list[str], n_prompts: int = 10) -> list[EvalResult]:
    tokenizer = model.tokenizer
    n_layers = model.cfg.n_layers
    results = []

    for task in tasks:
        circuit_heads = get_circuit_heads(task)
        if not circuit_heads:
            log(f"  {task}: no circuit, skipping")
            continue

        prompts = generate_prompts(task, tokenizer, n_prompts)
        if not prompts or len(prompts) < 3:
            log(f"  {task}: insufficient prompts")
            continue

        circuit_layers = sorted({L for L, _ in circuit_heads})
        non_circuit_layers = sorted(set(range(n_layers)) - set(circuit_layers))

        log(f"  {task}: circuit layers={circuit_layers}, {len(prompts)} prompts")

        layer_acts = {}
        for layer in range(n_layers):
            layer_acts[layer] = collect_residual_stream(model, prompts, layer)

        # CKA matrix between all circuit layers
        n_circuit = len(circuit_layers)
        cka_matrix = np.zeros((n_circuit, n_circuit), dtype=np.float64)
        for i, li in enumerate(circuit_layers):
            for j, lj in enumerate(circuit_layers):
                cka_matrix[i, j] = linear_cka(layer_acts[li], layer_acts[lj])

        # CKA between consecutive circuit layers
        consecutive_ckas = []
        for i in range(len(circuit_layers) - 1):
            cka_val = cka_matrix[i, i + 1]
            consecutive_ckas.append(cka_val)
            log(f"    layer {circuit_layers[i]} -> {circuit_layers[i+1]}: CKA={cka_val:.3f}")

        # CKA between circuit and non-circuit layers
        circuit_vs_non = []
        for cl in circuit_layers:
            for ncl in non_circuit_layers:
                cka_val = linear_cka(layer_acts[cl], layer_acts[ncl])
                circuit_vs_non.append(cka_val)

        # CKA between first and last circuit layer
        first_last_cka = 0.0
        if len(circuit_layers) >= 2:
            first_last_cka = linear_cka(
                layer_acts[circuit_layers[0]], layer_acts[circuit_layers[-1]]
            )

        mean_consecutive = float(np.mean(consecutive_ckas)) if consecutive_ckas else 0.0
        mean_circuit_vs_non = float(np.mean(circuit_vs_non)) if circuit_vs_non else 0.0
        passed = bool(mean_consecutive > 0.3)

        log(f"    mean consecutive CKA={mean_consecutive:.3f}  "
            f"[{'PASS' if passed else 'FAIL'}]")
        log(f"    first-last CKA={first_last_cka:.3f}, "
            f"circuit-vs-non={mean_circuit_vs_non:.3f}")

        layer_pair_details = []
        for i, li in enumerate(circuit_layers):
            for j, lj in enumerate(circuit_layers):
                if i < j:
                    layer_pair_details.append({
                        "layer_a": li,
                        "layer_b": lj,
                        "cka": float(cka_matrix[i, j]),
                    })

        results.append(EvalResult(
            metric_id="E6b.cka_cross_layer",
            value=mean_consecutive,
            n_samples=len(prompts),
            metadata={
                "task": task,
                "passed": passed,
                "threshold": 0.3,
                "mean_consecutive_cka": mean_consecutive,
                "first_last_cka": first_last_cka,
                "mean_circuit_vs_non_circuit": mean_circuit_vs_non,
                "circuit_layers": circuit_layers,
                "consecutive_ckas": consecutive_ckas,
                "cka_matrix": cka_matrix.tolist(),
                "layer_pairs": layer_pair_details,
            },
        ))

    return results


def main():
    parser = parse_common_args("E6b: CKA Cross-Layer Analysis")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("E6b: CKA CROSS-LAYER ANALYSIS")
    log("=" * 60)

    out = args.out or "E6b_cka_cross_arch.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_cka_analysis(model, [task], args.n_prompts)
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)
            p = "PASS" if r.metadata["passed"] else "FAIL"
            log(f"  {task}: mean_cka={r.value:.3f}  [{p}]")

    save_results(results, out, args=args)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
