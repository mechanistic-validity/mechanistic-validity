"""Representational Similarity Analysis
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     E03 — RSA
Categories:     representational
Validity layer: Internal
Criteria:       I4 Consistency
Establishes:    Circuit layers encode task-relevant similarity structure
Requires:       GPU, model
Doc:            /instruments_v2/representational/e03-rsa
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Computes RSA between model residual stream representations and a
task-defined target similarity structure.

For each task, the target RDM encodes which prompts should have similar
representations (e.g., IOI prompts with the same indirect object).
The neural RDM is built from cosine distances of residual stream
activations at each layer.

The RSA score (Spearman correlation between target and neural RDMs)
should peak at layers containing circuit heads -- indicating those
layers encode the task-relevant similarity structure.

Usage:
    uv run python 61_rsa.py --tasks ioi sva --device cpu
    uv run python 61_rsa.py --device cuda --n-prompts 40
"""

import numpy as np
import torch
from scipy import stats as sp_stats

from mechanistic_validity.instruments.common import (
    CIRCUIT_TASKS,
    EvalResult,
    generate_prompts,
    get_circuit_heads,
    load_model,
    log,
    parse_common_args,
    save_results,
)


def cosine_rdm(activations: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine distance RDM. activations: (n, d)."""
    norms = np.linalg.norm(activations, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    normed = activations / norms
    sim = normed @ normed.T
    return 1.0 - sim  # cosine distance


def build_target_rdm(prompts, task: str) -> np.ndarray:
    """Build target RDM from prompt metadata.

    Heuristic: prompts sharing the same target_correct are "similar" (distance=0),
    others are "dissimilar" (distance=1). This captures the core task structure --
    prompts requiring the same answer should be represented similarly.
    """
    n = len(prompts)
    rdm = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                rdm[i, j] = 0.0
            elif hasattr(prompts[i], 'target_correct') and hasattr(prompts[j], 'target_correct'):
                if prompts[i].target_correct == prompts[j].target_correct:
                    rdm[i, j] = 0.0
    return rdm


def rdm_to_upper_tri(rdm: np.ndarray) -> np.ndarray:
    """Extract upper triangle of RDM as a flat vector."""
    n = rdm.shape[0]
    idx = np.triu_indices(n, k=1)
    return rdm[idx]


@torch.no_grad()
def collect_residual_activations(model, prompts, layer: int) -> np.ndarray:
    """Collect last-token residual stream at a given layer. Returns (n_prompts, d_model)."""
    activations = []
    hook_name = f"blocks.{layer}.hook_resid_post"
    for p in prompts:
        tokens = model.to_tokens(p.text)
        _, cache = model.run_with_cache(tokens, names_filter=lambda n: n == hook_name)
        act = cache[hook_name][0, -1].cpu().float().numpy()
        activations.append(act)
    return np.stack(activations, axis=0)


@torch.no_grad()
def main():
    parser = parse_common_args("E61: Representational Similarity Analysis")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)
    tokenizer = model.tokenizer
    n_layers = model.cfg.n_layers

    log("=" * 60)
    log("E61: REPRESENTATIONAL SIMILARITY ANALYSIS (RSA)")
    log("=" * 60)

    results = []

    for task in tasks:
        circuit_heads = get_circuit_heads(task)
        if not circuit_heads:
            log(f"  {task}: no circuit, skipping")
            continue

        circuit_layers = {L for L, _ in circuit_heads}
        prompts = generate_prompts(task, tokenizer, args.n_prompts)
        if not prompts or len(prompts) < 5:
            log(f"  {task}: insufficient prompts")
            continue

        target_rdm = build_target_rdm(prompts, task)
        target_vec = rdm_to_upper_tri(target_rdm)

        # Skip if target RDM has no variance (all same/different)
        if np.std(target_vec) < 1e-10:
            log(f"  {task}: target RDM has no variance, skipping")
            continue

        log(f"  {task}: {len(prompts)} prompts, circuit layers={sorted(circuit_layers)}")

        per_layer_rsa = []
        circuit_rsa_scores = []
        non_circuit_rsa_scores = []

        for layer in range(n_layers):
            acts = collect_residual_activations(model, prompts, layer)
            neural_rdm = cosine_rdm(acts)
            neural_vec = rdm_to_upper_tri(neural_rdm)

            if np.std(neural_vec) < 1e-10:
                rsa_score = 0.0
            else:
                rsa_score, _ = sp_stats.spearmanr(target_vec, neural_vec)
                if np.isnan(rsa_score):
                    rsa_score = 0.0

            per_layer_rsa.append(float(rsa_score))

            if layer in circuit_layers:
                circuit_rsa_scores.append(rsa_score)
            else:
                non_circuit_rsa_scores.append(rsa_score)

        mean_circuit_rsa = float(np.mean(circuit_rsa_scores)) if circuit_rsa_scores else 0.0
        mean_non_circuit_rsa = float(np.mean(non_circuit_rsa_scores)) if non_circuit_rsa_scores else 0.0
        peak_layer = int(np.argmax(per_layer_rsa))
        peak_rsa = per_layer_rsa[peak_layer]

        log(f"    peak RSA={peak_rsa:.3f} at layer {peak_layer}")
        log(f"    circuit RSA={mean_circuit_rsa:.3f}, non-circuit={mean_non_circuit_rsa:.3f}")

        results.append(EvalResult(
            metric_id="E61.rsa_circuit_vs_non_circuit",
            value=mean_circuit_rsa - mean_non_circuit_rsa,
            n_samples=len(prompts),
            metadata={
                "task": task,
                "mean_circuit_rsa": mean_circuit_rsa,
                "mean_non_circuit_rsa": mean_non_circuit_rsa,
                "peak_layer": peak_layer,
                "peak_rsa": peak_rsa,
                "per_layer_rsa": per_layer_rsa,
                "circuit_layers": sorted(circuit_layers),
            },
        ))
        results.append(EvalResult(
            metric_id="E61.peak_rsa",
            value=peak_rsa,
            n_samples=len(prompts),
            metadata={"task": task, "peak_layer": peak_layer},
        ))

    out = args.out or "61_rsa.json"
    save_results(results, out)
    log(f"\nDone. {len(results)} metrics.")


if __name__ == "__main__":
    main()
