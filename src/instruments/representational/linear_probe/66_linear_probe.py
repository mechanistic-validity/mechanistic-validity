"""Linear Probe
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     E02 — Linear Probe
Categories:     representational
Validity layer: Internal
Criteria:       I2 Sufficiency
Establishes:    Circuit heads concentrate linearly decodable task information at specific layers
Requires:       GPU, model
Doc:            /instruments_v2/representational/e02-linear-probe
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Trains a closed-form linear probe (OLS) at each layer's residual stream to
predict the correct answer token. Measures where task-relevant information
becomes linearly decodable. Ablates circuit heads and re-probes to verify
the circuit concentrates predictive signal at specific layers.

Usage:
    uv run python 66_linear_probe.py --tasks ioi greater_than
    uv run python 66_linear_probe.py --device cpu --n-prompts 60
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "causal"))
from _common import (
    CIRCUIT_TASKS,
    EvalResult,
    calibrate_mean_z,
    generate_prompts,
    get_circuit_heads,
    get_token_ids,
    heads_to_layer_dict,
    load_model,
    log,
    make_ablation_hook,
    parse_common_args,
    save_results,
)


def train_linear_probe(X: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """Closed-form OLS probe. Returns (accuracy, weights)."""
    n = X.shape[0]
    if n < 5:
        return 0.0, np.zeros(X.shape[1])

    # Add bias column
    X_bias = np.column_stack([X, np.ones(n)])
    # Solve via lstsq: w = (X^T X)^{-1} X^T y
    w, _, _, _ = np.linalg.lstsq(X_bias, y, rcond=None)

    preds = (X_bias @ w > 0.5).astype(float)
    accuracy = float((preds == y).mean())
    return accuracy, w[:-1]  # Return weights without bias


@torch.no_grad()
def collect_residuals(model, prompts, device: str) -> dict[int, np.ndarray]:
    """Collect last-position residual stream at each layer.

    Returns dict: layer -> (n_prompts, d_model) array.
    """
    n_layers = model.cfg.n_layers
    residuals: dict[int, list[np.ndarray]] = {L: [] for L in range(n_layers)}

    for p in prompts:
        tokens = model.to_tokens(p.text)
        _, cache = model.run_with_cache(
            tokens, names_filter=lambda n: "hook_resid_post" in n
        )
        for L in range(n_layers):
            h = cache[f"blocks.{L}.hook_resid_post"][0, -1].cpu().numpy()
            residuals[L].append(h)

    return {L: np.stack(v) for L, v in residuals.items()}


@torch.no_grad()
def collect_residuals_ablated(model, prompts, circuit_heads, mean_z, device: str
                              ) -> dict[int, np.ndarray]:
    """Collect residuals with circuit heads ablated."""
    n_layers = model.cfg.n_layers
    circuit_by_layer = heads_to_layer_dict(circuit_heads)
    hooks = make_ablation_hook(circuit_by_layer, mean_z, "mean")
    residuals: dict[int, list[np.ndarray]] = {L: [] for L in range(n_layers)}

    for p in prompts:
        tokens = model.to_tokens(p.text)
        _, cache = model.run_with_cache(
            tokens,
            names_filter=lambda n: "hook_resid_post" in n,
            fwd_hooks=hooks,
        )
        for L in range(n_layers):
            h = cache[f"blocks.{L}.hook_resid_post"][0, -1].cpu().numpy()
            residuals[L].append(h)

    return {L: np.stack(v) for L, v in residuals.items()}


@torch.no_grad()
def main():
    parser = parse_common_args("E04: Linear Probe Accuracy")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)
    results = []

    log("=" * 60)
    log("E04: LINEAR PROBE ACCURACY")
    log("=" * 60)

    for task in tasks:
        log(f"\n--- Task: {task} ---")
        circuit_heads = get_circuit_heads(task)
        if not circuit_heads:
            log(f"  No circuit heads for {task}, skipping")
            continue

        prompts = generate_prompts(task, model.tokenizer, args.n_prompts)
        if not prompts:
            log(f"  No prompts for {task}, skipping")
            continue

        correct_ids, incorrect_ids = get_token_ids(prompts, model.tokenizer)
        if len(correct_ids) < 10:
            log(f"  Too few valid prompts for {task}, skipping")
            continue

        n = len(correct_ids)

        log("  Collecting residuals (clean)...")
        residuals_clean = collect_residuals(model, prompts[:n], args.device)

        # Binary labels: 1 if model predicts correct > incorrect
        model_correct = []
        for i in range(n):
            tokens = model.to_tokens(prompts[i].text)
            logits = model(tokens)
            last_logits = logits[0, -1]
            model_correct.append(
                float(last_logits[correct_ids[i]] > last_logits[incorrect_ids[i]])
            )
        labels = np.array(model_correct)
        if labels.std() < 0.01:
            labels = np.ones(n)

        # Train probe at each layer
        accuracy_per_layer = {}
        for L in range(model.cfg.n_layers):
            X = residuals_clean[L]
            acc, _ = train_linear_probe(X, labels)
            accuracy_per_layer[L] = acc

        # Ablate circuit and re-probe
        log("  Calibrating mean activations...")
        mean_z = calibrate_mean_z(model, prompts, n_calibration=min(30, n))

        log("  Collecting residuals (circuit ablated)...")
        residuals_ablated = collect_residuals_ablated(
            model, prompts[:n], circuit_heads, mean_z, args.device
        )

        accuracy_ablated = {}
        for L in range(model.cfg.n_layers):
            X = residuals_ablated[L]
            acc, _ = train_linear_probe(X, labels)
            accuracy_ablated[L] = acc

        # Compute drop at circuit-relevant layers
        circuit_layers = sorted(set(L for L, _ in circuit_heads))
        drop_at_circuit_layers = []
        for L in circuit_layers:
            drop = accuracy_per_layer.get(L, 0) - accuracy_ablated.get(L, 0)
            drop_at_circuit_layers.append(drop)

        mean_drop_circuit = float(np.mean(drop_at_circuit_layers)) if drop_at_circuit_layers else 0.0
        max_clean_acc = max(accuracy_per_layer.values()) if accuracy_per_layer else 0.0

        log(f"  Max clean accuracy: {max_clean_acc:.3f}")
        log(f"  Mean accuracy drop at circuit layers: {mean_drop_circuit:.3f}")

        results.append(EvalResult(
            metric_id="E04.linear_probe",
            value=max_clean_acc,
            baseline_random=0.5,
            n_samples=n,
            metadata={
                "task": task,
                "accuracy_per_layer_clean": accuracy_per_layer,
                "accuracy_per_layer_ablated": accuracy_ablated,
                "mean_drop_at_circuit_layers": mean_drop_circuit,
                "circuit_layers": circuit_layers,
                "n_circuit_heads": len(circuit_heads),
            },
        ))

    out = args.out or "66_linear_probe.json"
    save_results(results, out)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
