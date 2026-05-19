"""Probe Decodability with Selectivity Baseline
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     E02 — Linear Probe
Categories:     representational
Validity layer: Internal
Criteria:       R1 Probe Decodability (proposed)
Establishes:    Whether target variable is linearly decodable from circuit layer activations
Requires:       GPU or CPU, model
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For each layer containing circuit heads, collects residual stream activations,
labels prompts by correct/incorrect prediction, and trains a linear probe
(logistic regression via gradient descent). Reports probe accuracy minus a
selectivity baseline (Hewitt & Liang 2019) on a control task at the same layer.
Pass condition: selectivity > 0.10.

Usage:
    uv run python 75_probe_decodability.py --tasks ioi sva
    uv run python 75_probe_decodability.py --device cpu --n-prompts 60
"""

import numpy as np
import torch

from mechanistic_validity.metrics.common import (
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

PROBE_LR = 0.01
PROBE_EPOCHS = 100
SELECTIVITY_THRESHOLD = 0.10
TRAIN_FRACTION = 0.7


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    pos = x >= 0
    result = np.empty_like(x)
    result[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    result[~pos] = exp_x / (1.0 + exp_x)
    return result


def train_logistic_probe(X_train: np.ndarray, y_train: np.ndarray,
                         X_test: np.ndarray, y_test: np.ndarray,
                         lr: float = PROBE_LR,
                         epochs: int = PROBE_EPOCHS) -> float:
    """Train a logistic regression probe via gradient descent. Returns test accuracy."""
    n_features = X_train.shape[1]
    n_train = X_train.shape[0]
    if n_train < 4 or X_test.shape[0] < 2:
        return 0.5

    # Normalize features
    mu = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std < 1e-8] = 1.0
    X_train_n = (X_train - mu) / std
    X_test_n = (X_test - mu) / std

    # Initialize weights
    w = np.zeros(n_features)
    b = 0.0

    for _ in range(epochs):
        logits = X_train_n @ w + b
        preds = _sigmoid(logits)
        # Clamp for numerical stability
        preds = np.clip(preds, 1e-7, 1.0 - 1e-7)
        # Gradient of binary cross-entropy
        error = preds - y_train
        grad_w = (X_train_n.T @ error) / n_train
        grad_b = error.mean()
        w -= lr * grad_w
        b -= lr * grad_b

    # Test accuracy
    test_logits = X_test_n @ w + b
    test_preds = (test_logits > 0).astype(float)
    accuracy = float((test_preds == y_test).mean())
    return accuracy


@torch.no_grad()
def collect_residual_activations(model, prompts, layer: int) -> np.ndarray:
    """Collect residual stream activations at last position for a given layer.

    Returns array of shape (n_prompts, d_model).
    """
    hook_name = f"blocks.{layer}.hook_resid_post"
    activations = []

    for p in prompts:
        tokens = model.to_tokens(p.text)
        _, cache = model.run_with_cache(tokens, names_filter=lambda n: n == hook_name)
        act = cache[hook_name][0, -1].cpu().numpy()  # (d_model,)
        activations.append(act)

    return np.stack(activations, axis=0)


@torch.no_grad()
def label_prompts_by_prediction(model, prompts, correct_ids) -> np.ndarray:
    """Label each prompt: 1 if model's top prediction matches correct_id, 0 otherwise."""
    labels = []
    for i, p in enumerate(prompts):
        if i >= len(correct_ids):
            break
        tokens = model.to_tokens(p.text)
        logits = model(tokens)
        top_pred = logits[0, -1].argmax().item()
        labels.append(1.0 if top_pred == correct_ids[i] else 0.0)
    return np.array(labels)


def generate_control_labels(n: int, seed: int = 123) -> np.ndarray:
    """Generate random binary labels as a selectivity control task."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, 2, size=n).astype(float)


def split_train_test(X: np.ndarray, y: np.ndarray,
                     frac: float = TRAIN_FRACTION,
                     seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data into train/test."""
    rng = np.random.RandomState(seed)
    n = X.shape[0]
    idx = rng.permutation(n)
    split = int(n * frac)
    return X[idx[:split]], y[idx[:split]], X[idx[split:]], y[idx[split:]]


def run_probe_decodability(model, tasks: list[str],
                           n_prompts: int = 40) -> list[EvalResult]:
    tokenizer = model.tokenizer
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

        # Find layers containing circuit heads
        circuit_layers = sorted(heads_to_layer_dict(circuit_heads).keys())
        log(f"  {task} ({len(circuit_heads)} heads, layers {circuit_layers})...")

        # Label prompts by correctness of model prediction
        labels = label_prompts_by_prediction(model, prompts, correct_ids)
        n_pos = int(labels.sum())
        log(f"    labels: {n_pos}/{len(labels)} correct predictions")

        # Skip if labels are too unbalanced
        if n_pos < 3 or (len(labels) - n_pos) < 3:
            log(f"    skipping: too few samples in one class")
            continue

        # Control labels for selectivity baseline
        control_labels = generate_control_labels(len(labels))

        per_layer_results = {}
        best_selectivity = -float("inf")

        for layer in circuit_layers:
            X = collect_residual_activations(model, prompts[:len(labels)], layer)

            # Real task probe
            X_tr, y_tr, X_te, y_te = split_train_test(X, labels)
            task_acc = train_logistic_probe(X_tr, y_tr, X_te, y_te)

            # Control task probe (selectivity baseline)
            X_tr_c, y_tr_c, X_te_c, y_te_c = split_train_test(X, control_labels)
            control_acc = train_logistic_probe(X_tr_c, y_tr_c, X_te_c, y_te_c)

            selectivity = task_acc - control_acc
            passed = selectivity > SELECTIVITY_THRESHOLD

            log(f"    layer {layer}: task_acc={task_acc:.3f}, "
                f"control_acc={control_acc:.3f}, "
                f"selectivity={selectivity:.3f} {'PASS' if passed else 'FAIL'}")

            per_layer_results[layer] = {
                "task_accuracy": task_acc,
                "control_accuracy": control_acc,
                "selectivity": selectivity,
                "pass": passed,
            }

            if selectivity > best_selectivity:
                best_selectivity = selectivity

        # Overall pass: at least one circuit layer passes
        any_pass = any(v["pass"] for v in per_layer_results.values())

        results.append(EvalResult(
            metric_id="R1.probe_decodability",
            value=best_selectivity,
            n_samples=len(labels),
            metadata={
                "task": task,
                "n_circuit_heads": len(circuit_heads),
                "circuit_layers": circuit_layers,
                "n_correct_predictions": n_pos,
                "selectivity_threshold": SELECTIVITY_THRESHOLD,
                "per_layer": per_layer_results,
                "any_layer_passes": any_pass,
            },
        ))

    return results


def main():
    parser = parse_common_args("R1: Probe Decodability")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("R1: PROBE DECODABILITY WITH SELECTIVITY BASELINE")
    log("=" * 60)

    results = run_probe_decodability(model, tasks, args.n_prompts)

    out = args.out or "75_probe_decodability.json"
    save_results(results, out)

    log(f"\nDone. {len(results)} tasks evaluated.")
    for r in results:
        t = r.metadata["task"]
        log(f"  {t}: selectivity={r.value:.3f} "
            f"(pass={r.metadata['any_layer_passes']})")


if __name__ == "__main__":
    main()
