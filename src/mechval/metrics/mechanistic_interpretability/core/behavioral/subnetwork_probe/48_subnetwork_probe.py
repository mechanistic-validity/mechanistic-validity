"""Subnetwork Probing
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     D07 — Subnetwork Probe
Categories:     behavioral
Validity layer: Internal
Criteria:       I2 Sufficiency
Establishes:    Circuit concentrates linearly decodable task information vs random subnetworks
Requires:       GPU, model
Doc:            /instruments_v2/behavioral/d07-subnetwork-probe
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Trains a linear probe on the circuit's residual stream output to predict
the correct answer, then trains the same probe on random subnetworks of
the same size. Compares probe accuracy: if the circuit's residual stream
is more linearly decodable than random subnetworks, the circuit
concentrates task-relevant information.

Uses a simple logistic regression (manual implementation to avoid sklearn
dependency).

Framework reference: Behavioral Pillar D09 -- linear decodability as a
measure of information concentration in the circuit subnetwork.

Usage:
    uv run python 48_subnetwork_probe.py --tasks ioi sva
    uv run python 48_subnetwork_probe.py --device cuda --n-prompts 60
"""

import numpy as np
import torch
import torch.nn.functional as F

from mechval.metrics.common import (
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

N_RANDOM_BASELINES = 10


def logistic_probe_accuracy(features, labels):
    """Train a simple logistic regression probe and return accuracy.

    features: (n_samples, d) numpy array
    labels: (n_samples,) binary numpy array
    Returns: accuracy on the same data (in-sample, for comparison purposes).
    """
    n_samples, d = features.shape
    if n_samples < 4 or d == 0:
        return 0.5

    # Normalize features
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True) + 1e-8
    X = (features - mean) / std

    # Add bias column
    X = np.concatenate([X, np.ones((n_samples, 1))], axis=1)

    # Simple gradient descent logistic regression
    w = np.zeros(X.shape[1])
    lr = 0.1
    for _ in range(200):
        logits = X @ w
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
        grad = X.T @ (probs - labels) / n_samples
        w -= lr * grad

    # Compute accuracy
    predictions = (X @ w > 0).astype(float)
    accuracy = float((predictions == labels).mean())
    return accuracy


@torch.no_grad()
def extract_residual_features(model, prompts, heads_to_keep, mean_z, n_max=None):
    """Extract residual stream at final position with only specified heads active."""
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    all_heads = {(L, H) for L in range(n_layers) for H in range(n_heads)}
    complement = all_heads - heads_to_keep
    complement_hooks = make_ablation_hook(heads_to_layer_dict(complement), mean_z, "mean")

    features = []
    limit = n_max if n_max else len(prompts)
    for i, p in enumerate(prompts[:limit]):
        tokens = model.to_tokens(p.text)
        logits = model.run_with_hooks(tokens, fwd_hooks=complement_hooks)
        # Use last-position logits as features (captures circuit output)
        last_logits = logits[0, -1].cpu().numpy()
        # Reduce dimensionality: take top-100 logit values
        top_indices = np.argsort(np.abs(last_logits))[-100:]
        features.append(last_logits[top_indices])

    return np.array(features)


@torch.no_grad()
def run_subnetwork_probe(model, tasks, n_prompts, n_random=N_RANDOM_BASELINES):
    tokenizer = model.tokenizer
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    all_heads_list = [(L, H) for L in range(n_layers) for H in range(n_heads)]
    results = []
    rng = np.random.RandomState(42)

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

        n_use = min(len(correct_ids), len(prompts))
        prompts_use = prompts[:n_use]

        log(f"  {task} ({len(circuit_heads)} heads, {n_use} prompts)...")
        mean_z = calibrate_mean_z(model, prompts_use, n_calibration=min(100, n_use))

        # Labels: 1 if model gets it right (logit_diff > 0), 0 otherwise
        labels = []
        for i in range(n_use):
            tokens = model.to_tokens(prompts_use[i].text)
            logits = model(tokens)
            ld = (logits[0, -1, correct_ids[i]] - logits[0, -1, incorrect_ids[i]]).item()
            labels.append(1.0 if ld > 0 else 0.0)
        labels = np.array(labels)

        # Skip if labels are all one class
        if labels.sum() == 0 or labels.sum() == len(labels):
            log(f"    {task}: all same class, skipping")
            continue

        # Circuit probe accuracy
        circuit_features = extract_residual_features(
            model, prompts_use, circuit_heads, mean_z,
        )
        circuit_accuracy = logistic_probe_accuracy(circuit_features, labels)

        # Random subnetwork baselines (same size as circuit)
        k = len(circuit_heads)
        random_accuracies = []
        for _ in range(n_random):
            indices = rng.choice(len(all_heads_list), size=k, replace=False)
            random_heads = {all_heads_list[idx] for idx in indices}
            random_features = extract_residual_features(
                model, prompts_use, random_heads, mean_z,
            )
            acc = logistic_probe_accuracy(random_features, labels)
            random_accuracies.append(acc)

        mean_random = float(np.mean(random_accuracies))
        std_random = float(np.std(random_accuracies))
        advantage = circuit_accuracy - mean_random
        z_score = advantage / std_random if std_random > 1e-8 else 0.0

        log(f"    circuit_acc={circuit_accuracy:.3f}, random_mean={mean_random:.3f}, "
            f"advantage={advantage:.3f}, z={z_score:.2f}")

        results.append(EvalResult(
            metric_id="D09.subnetwork_probe",
            value=circuit_accuracy,
            baseline_random=mean_random,
            n_samples=n_use,
            metadata={
                "task": task,
                "circuit_probe_accuracy": circuit_accuracy,
                "random_mean_accuracy": mean_random,
                "random_std_accuracy": std_random,
                "advantage": advantage,
                "z_score": z_score,
                "n_circuit_heads": len(circuit_heads),
                "n_random_baselines": n_random,
                "n_prompts": n_use,
            },
        ))

    return results


def main():
    parser = parse_common_args("D09: Subnetwork Probing")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("D09: SUBNETWORK PROBING")
    log("=" * 60)

    n_random = args.n_random_baselines
    results = run_subnetwork_probe(model, tasks, args.n_prompts, n_random)

    out = args.out or "48_subnetwork_probe.json"
    save_results(results, out)

    log(f"\nDone. {len(results)} tasks evaluated.")
    for r in results:
        t = r.metadata["task"]
        log(f"  {t}: circuit={r.value:.3f}, random={r.baseline_random:.3f}, "
            f"z={r.metadata['z_score']:.2f}")


if __name__ == "__main__":
    main()
