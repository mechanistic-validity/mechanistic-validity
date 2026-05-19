"""DAS-IIA via PCA Direction (Simplified)
---
Instrument:     DAS.iia -- Distributed Alignment Search IIA
Categories:     causal
Validity layer: Internal
Criteria:       DAS IIA
Establishes:    PCA-based causal direction in residual stream encodes task variable (Geiger et al., 2024)
Requires:       CPU or GPU, model
---

Simplified DAS that does not require pyvene. For the circuit's
highest-layer head:
1. Collect activations on base and counterfactual prompts
2. Compute PCA on (base - counterfactual) activation differences
3. Intervene by replacing top-k PCA components of base with counterfactual
4. IIA = fraction of interventions where model output flips correctly

Pass condition: IIA >= 0.7

Usage:
    uv run python DAS_iia_task.py --tasks ioi --n-prompts 10
    uv run python DAS_iia_task.py --tasks ioi sva --device cpu
"""

import numpy as np
import torch

from mechanistic_validity.instruments.common import (
    CIRCUIT_TASKS,
    EvalResult,
    generate_prompts,
    get_circuit_heads,
    get_token_ids,
    load_model,
    log,
    logit_diff_from_logits,
    parse_common_args,
    save_incremental,
    save_results,
)


def make_counterfactual_pairs(prompts, correct_ids, incorrect_ids):
    """Pair each prompt with one that has a different correct target."""
    pairs = []
    n = len(prompts)
    for i in range(n):
        candidates = [j for j in range(n) if j != i and correct_ids[j] != correct_ids[i]]
        if not candidates:
            candidates = [j for j in range(n) if j != i]
        if candidates:
            pairs.append((i, candidates[0]))
    return pairs


@torch.no_grad()
def collect_activations(model, prompts, layer: int, head: int):
    """Collect hook_z activations at last token for a specific head.

    Returns list of tensors, each shape (d_head,).
    """
    hook_name = f"blocks.{layer}.attn.hook_z"
    acts = []
    for p in prompts:
        tokens = model.to_tokens(p.text)
        _, cache = model.run_with_cache(tokens, names_filter=lambda n: n == hook_name)
        z = cache[hook_name][0, -1, head]  # (d_head,)
        acts.append(z.cpu())
    return acts


def compute_pca_directions(base_acts: list[torch.Tensor],
                           cf_acts: list[torch.Tensor],
                           pairs: list[tuple[int, int]],
                           k_components: int) -> torch.Tensor:
    """Compute top-k PCA directions from base-counterfactual differences.

    Returns rotation matrix R of shape (d_head, k_components).
    """
    diffs = []
    for base_idx, cf_idx in pairs:
        diff = base_acts[base_idx] - cf_acts[cf_idx]
        diffs.append(diff)

    diff_matrix = torch.stack(diffs).float()  # (n_pairs, d_head)
    diff_centered = diff_matrix - diff_matrix.mean(dim=0, keepdim=True)

    actual_k = min(k_components, diff_centered.shape[0], diff_centered.shape[1])
    if actual_k == 0:
        return torch.eye(diff_centered.shape[1])[:, :1]

    try:
        U, S, Vt = torch.linalg.svd(diff_centered, full_matrices=False)
        directions = Vt[:actual_k].T  # (d_head, actual_k)
    except torch._C._LinAlgError:
        directions = torch.eye(diff_centered.shape[1])[:, :actual_k]

    return directions


@torch.no_grad()
def compute_iia(model, task: str, n_prompts: int = 10,
                k_components: int = 1) -> tuple[float, dict]:
    """Compute IIA for a task using PCA-based intervention.

    Returns (iia_score, metadata_dict).
    """
    tokenizer = model.tokenizer
    device = str(model.cfg.device)

    circuit_heads = get_circuit_heads(task)
    if not circuit_heads:
        return 0.0, {"error": "no circuit"}

    prompts = generate_prompts(task, tokenizer, n_prompts)
    if not prompts:
        return 0.0, {"error": "no prompts"}

    correct_ids, incorrect_ids = get_token_ids(prompts, tokenizer)
    if not correct_ids:
        return 0.0, {"error": "no token ids"}

    # Use the highest-layer circuit head
    sorted_heads = sorted(circuit_heads, key=lambda h: (h[0], h[1]), reverse=True)
    layer, head = sorted_heads[0]

    pairs = make_counterfactual_pairs(prompts, correct_ids, incorrect_ids)
    if not pairs:
        return 0.0, {"error": "no counterfactual pairs"}

    # Split into train/test
    n_train = max(1, len(pairs) // 2)
    train_pairs = pairs[:n_train]
    test_pairs = pairs[n_train:]
    if not test_pairs:
        test_pairs = train_pairs

    # Collect activations for all prompts
    base_acts = collect_activations(model, prompts, layer, head)

    # Learn PCA directions from train pairs
    directions = compute_pca_directions(base_acts, base_acts, train_pairs, k_components)
    directions = directions.to(device)
    proj = directions @ directions.T  # (d_head, d_head)

    # Measure base accuracy (without intervention)
    hook_name = f"blocks.{layer}.attn.hook_z"
    n_base_correct = 0
    for base_idx, _ in test_pairs:
        tokens = model.to_tokens(prompts[base_idx].text)
        logits = model(tokens)
        ld = logit_diff_from_logits(logits, correct_ids[base_idx], incorrect_ids[base_idx])
        if ld > 0:
            n_base_correct += 1
    base_accuracy = n_base_correct / max(len(test_pairs), 1)

    # Measure IIA: intervene and check if output flips to source's target
    n_correct = 0
    n_total = 0
    for base_idx, cf_idx in test_pairs:
        base_z = base_acts[base_idx].to(device)
        cf_z = base_acts[cf_idx].to(device)
        intervened_z = base_z - base_z @ proj + cf_z @ proj

        def _hook(z, hook, _head=head, _z=intervened_z):
            z[0, -1, _head] = _z
            return z

        tokens = model.to_tokens(prompts[base_idx].text).to(device)
        logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, _hook)])

        # Check if the model now predicts the counterfactual's correct answer
        cf_correct_id = correct_ids[cf_idx]
        cf_incorrect_id = incorrect_ids[cf_idx]
        ld = logit_diff_from_logits(logits, cf_correct_id, cf_incorrect_id)

        if ld > 0:
            n_correct += 1
        n_total += 1

    iia = n_correct / max(n_total, 1)

    metadata = {
        "task": task,
        "layer_intervened": layer,
        "head_intervened": head,
        "n_components_used": int(directions.shape[1]),
        "base_accuracy": base_accuracy,
        "intervention_accuracy": iia,
        "n_test_pairs": n_total,
        "n_correct": n_correct,
    }

    return iia, metadata


def run_das_iia(model, tasks: list[str], n_prompts: int = 10) -> list[EvalResult]:
    results = []

    for task in tasks:
        log(f"  {task}...")
        iia, metadata = compute_iia(model, task, n_prompts=n_prompts)

        if "error" in metadata:
            log(f"    skipped: {metadata['error']}")
            continue

        passed = bool(iia >= 0.7)
        log(f"    IIA={iia:.3f}  [{'PASS' if passed else 'FAIL'}]  "
            f"(layer={metadata['layer_intervened']}, "
            f"base_acc={metadata['base_accuracy']:.3f})")

        results.append(EvalResult(
            metric_id="DAS.iia",
            value=iia,
            n_samples=metadata["n_test_pairs"],
            metadata={
                **metadata,
                "passed": passed,
                "threshold": 0.7,
            },
        ))

    return results


def main():
    parser = parse_common_args("DAS.iia: Distributed Alignment Search IIA")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("DAS.iia: DISTRIBUTED ALIGNMENT SEARCH (PCA-based)")
    log("=" * 60)

    out = args.out or "DAS_iia_task.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_das_iia(model, [task], args.n_prompts)
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)
            p = "PASS" if r.metadata["passed"] else "FAIL"
            log(f"  {task}: IIA={r.value:.3f}  [{p}]")

    save_results(results, out, args=args)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
