"""Constrained DAS-IIA (Primary Metric)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     A02 — Counterfactual DAS/IIA
Categories:     causal, representational
Validity layer: Internal + Representational
Criteria:       I2 Sufficiency
Establishes:    Learned rotation aligns residual stream with causal variable (IIA on held-out pairs)
Requires:       GPU, model
Doc:            /instruments_v2/causal/a02-counterfactual-das
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Trains a linear rotation of residual-stream subspace onto a binary causal
variable, then measures interchange intervention accuracy (IIA) on held-out
counterfactual pairs. Main comparable to MIB benchmark.

Uses full-forward-pass CE loss with SVD warm-start (matching stage 44).

Usage:
    uv run python 01_das_iia.py --tasks ioi sva --n-prompts 40
    uv run python 01_das_iia.py --subspace-dims 1 2 4 --device cuda
"""
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mechanistic_validity.instruments.common import (
    CIRCUIT_TASKS,
    LITERATURE_BASELINES,
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


def make_counterfactual_pairs(prompts, correct_ids, incorrect_ids, rng):
    """Create counterfactual pairs by matching prompts with different targets."""
    pairs = []
    n = len(prompts)
    for i in range(n):
        candidates = [j for j in range(n) if j != i and correct_ids[j] != correct_ids[i]]
        if not candidates:
            candidates = [j for j in range(n) if j != i]
        j = rng.choice(candidates)
        pairs.append((i, j))
    return pairs


@torch.no_grad()
def compute_iia_with_rotation(model, prompts, correct_ids, incorrect_ids,
                               pairs, layer: int, head: int,
                               rotation: torch.Tensor, device: str) -> float:
    """Compute IIA: does intervened output flip to source's target?"""
    n_correct = 0
    n_total = 0
    hook_name = f"blocks.{layer}.attn.hook_z"
    R = rotation.to(device)
    proj = R @ R.t()

    for base_idx, source_idx in pairs:
        base_tokens = model.to_tokens(prompts[base_idx].text).to(device)
        source_tokens = model.to_tokens(prompts[source_idx].text).to(device)

        _, base_cache = model.run_with_cache(base_tokens, names_filter=lambda n: n == hook_name)
        _, source_cache = model.run_with_cache(source_tokens, names_filter=lambda n: n == hook_name)

        base_z = base_cache[hook_name][0, -1, head]
        source_z = source_cache[hook_name][0, -1, head]
        intervened_z = base_z - base_z @ proj + source_z @ proj

        def _hook(z, hook, _head=head, _z=intervened_z):
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


def train_rotation(model, prompts, correct_ids, incorrect_ids, pairs,
                   layer: int, head: int, d_sub: int, device: str,
                   n_steps: int = 200, lr: float = 5e-3) -> torch.Tensor:
    """Train an orthogonal rotation R via full-forward-pass CE loss + SVD warm-start."""
    d_head = model.cfg.d_head
    hook_name = f"blocks.{layer}.attn.hook_z"

    with torch.no_grad():
        cached = []
        for base_idx, source_idx in pairs:
            if base_idx >= len(correct_ids) or source_idx >= len(correct_ids):
                continue
            base_tokens = model.to_tokens(prompts[base_idx].text).to(device)
            source_tokens = model.to_tokens(prompts[source_idx].text).to(device)
            _, bc = model.run_with_cache(base_tokens, names_filter=lambda n: n == hook_name)
            _, sc = model.run_with_cache(source_tokens, names_filter=lambda n: n == hook_name)
            cached.append((
                base_tokens,
                bc[hook_name][0, -1, head].clone(),
                sc[hook_name][0, -1, head].clone(),
                correct_ids[source_idx],
            ))

    diffs = torch.stack([s - b for (_, b, s, _) in cached])
    actual_k = min(d_sub, diffs.shape[0], d_head)
    try:
        _, _, Vt = torch.linalg.svd(diffs.cpu(), full_matrices=False)
        R_param = Vt[:actual_k, :].t().clone().contiguous().to(device).requires_grad_(True)
    except torch._C._LinAlgError:
        R_param = torch.randn(d_head, actual_k, device=device).requires_grad_(True)

    optimizer = torch.optim.Adam([R_param], lr=lr)
    rng = random.Random(42)

    for step in range(n_steps):
        batch = rng.sample(cached, min(8, len(cached)))
        total_loss = torch.tensor(0.0, device=device)

        for base_tokens, base_z, source_z, src_correct_id in batch:
            Q, _ = torch.linalg.qr(R_param)
            proj = Q @ Q.t()
            intervened_z = base_z - base_z @ proj + source_z @ proj

            def _hook(z, hook, _head=head, _z=intervened_z):
                z[0, -1, _head] = _z
                return z

            logits = model.run_with_hooks(base_tokens, fwd_hooks=[(hook_name, _hook)])
            log_probs = F.log_softmax(logits[0, -1, :], dim=-1)
            total_loss = total_loss - log_probs[src_correct_id]

        total_loss = total_loss / len(batch)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    with torch.no_grad():
        Q, _ = torch.linalg.qr(R_param)
    return Q.detach()


def random_rotation(d_head: int, d_sub: int, device: str) -> torch.Tensor:
    """Haar-distributed random orthogonal matrix."""
    R = torch.randn(d_head, d_sub, device=device)
    Q, _ = torch.linalg.qr(R)
    return Q


def run_untrained_baseline(trained_model, tasks: list[str],
                           n_prompts: int, subspace_dims: list[int],
                           rng) -> dict[str, dict[int, float]]:
    """Run DAS-IIA on a randomly-initialized model as a baseline.

    Returns {task: {d_sub: best_iia}} for the untrained model.
    """
    from transformer_lens import HookedTransformer

    device = str(trained_model.cfg.device)
    log("  Loading untrained model (random init)...")
    untrained = HookedTransformer(trained_model.cfg)
    untrained.tokenizer = trained_model.tokenizer
    untrained.to(device)
    untrained.eval()

    tokenizer = untrained.tokenizer
    results = {}

    for task in tasks:
        circuit_heads = get_circuit_heads(task)
        if not circuit_heads:
            continue

        prompts = generate_prompts(task, tokenizer, n_prompts)
        if not prompts:
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

        train_pairs = make_counterfactual_pairs(train_prompts, train_correct, train_incorrect, rng)
        test_pairs = make_counterfactual_pairs(test_prompts, test_correct, test_incorrect, rng)

        task_results = {}
        for d_sub in subspace_dims:
            best_iia = 0.0
            for L, H in sorted(circuit_heads):
                R = train_rotation(untrained, train_prompts, train_correct, train_incorrect,
                                   train_pairs, L, H, d_sub, device, n_steps=100)
                iia = compute_iia_with_rotation(untrained, test_prompts, test_correct,
                                                 test_incorrect, test_pairs, L, H, R, device)
                if iia > best_iia:
                    best_iia = iia
            task_results[d_sub] = best_iia
            log(f"    untrained {task} k={d_sub}: best_IIA={best_iia:.3f}")
        results[task] = task_results

    del untrained
    if device != "cpu":
        torch.cuda.empty_cache()
    return results


def run_das_iia(model, tasks: list[str], n_prompts: int = 40,
                subspace_dims: list[int] | None = None,
                n_random_baselines: int = 20,
                mode: str = "trained") -> list[EvalResult]:
    """Run DAS-IIA evaluation.

    mode: "trained" = trained model IIA only (fast, default)
          "random" = random rotation baselines only
          "untrained" = untrained-model baseline only
    """
    if subspace_dims is None:
        subspace_dims = [1, 2, 4]

    tokenizer = model.tokenizer
    device = str(model.cfg.device)
    results = []
    rng = np.random.RandomState(42)

    if mode == "untrained":
        log("Running untrained-model baseline ONLY...")
        untrained_baselines = run_untrained_baseline(
            model, tasks, n_prompts, subspace_dims, np.random.RandomState(99))
        for task, task_results in untrained_baselines.items():
            for d_sub, iia in task_results.items():
                results.append(EvalResult(
                    metric_id=f"C1.das_iia_untrained_k{d_sub}",
                    value=iia,
                    n_samples=n_prompts // 2,
                    metadata={"task": task, "subspace_dim": d_sub, "mode": "untrained"},
                ))
        return results

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

        train_pairs = make_counterfactual_pairs(train_prompts, train_correct, train_incorrect, rng)
        test_pairs = make_counterfactual_pairs(test_prompts, test_correct, test_incorrect, rng)

        log(f"  {task} ({len(circuit_heads)} heads, {len(prompts)} prompts)...")

        for d_sub in subspace_dims:
            best_iia = 0.0
            best_head = None
            per_head_iia = {}

            for L, H in sorted(circuit_heads):
                R = train_rotation(model, train_prompts, train_correct, train_incorrect,
                                   train_pairs, L, H, d_sub, device)
                iia = compute_iia_with_rotation(model, test_prompts, test_correct,
                                                 test_incorrect, test_pairs, L, H, R, device)
                per_head_iia[f"L{L}H{H}"] = iia
                if iia > best_iia:
                    best_iia = iia
                    best_head = (L, H)

            if mode == "random":
                random_iias = []
                if best_head is not None:
                    L_best, H_best = best_head
                    for _ in range(n_random_baselines):
                        R_rand = random_rotation(model.cfg.d_head, d_sub, device)
                        rand_iia = compute_iia_with_rotation(
                            model, test_prompts, test_correct, test_incorrect,
                            test_pairs, L_best, H_best, R_rand, device,
                        )
                        random_iias.append(rand_iia)
                baseline_random = float(np.mean(random_iias)) if random_iias else None
                log(f"    k={d_sub}: random_baseline={baseline_random:.3f} (head={best_head})")
                results.append(EvalResult(
                    metric_id=f"C1.das_iia_random_k{d_sub}",
                    value=baseline_random or 0.0,
                    n_samples=len(test_prompts),
                    metadata={
                        "task": task, "subspace_dim": d_sub, "mode": "random",
                        "best_head": list(best_head) if best_head else None,
                        "n_random_trials": n_random_baselines,
                        "all_random_iias": random_iias,
                    },
                ))
            else:
                lit = LITERATURE_BASELINES.get(task, {}).get("das_iia")
                log(f"    k={d_sub}: best_IIA={best_iia:.3f} (head={best_head})")
                results.append(EvalResult(
                    metric_id=f"C1.das_iia_k{d_sub}",
                    value=best_iia,
                    baseline_literature=lit,
                    n_samples=len(test_prompts),
                    metadata={
                        "task": task, "subspace_dim": d_sub, "mode": "trained",
                        "best_head": list(best_head) if best_head else None,
                        "per_head_iia": per_head_iia,
                        "n_circuit_heads": len(circuit_heads),
                    },
                ))

    return results


def main():
    parser = parse_common_args("C1: DAS-IIA")
    parser.add_argument("--subspace-dims", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--mode", choices=["trained", "random", "untrained"],
                        default="trained",
                        help="trained=DAS-IIA only, random=Haar baselines only, untrained=random-init model only")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("C1: DAS-IIA (Constrained Distributed Alignment Search)")
    log("=" * 60)

    results = run_das_iia(model, tasks, args.n_prompts, args.subspace_dims,
                          args.n_random_baselines, mode=args.mode)

    suffix = f"_{args.mode}" if args.mode != "trained" else ""
    out = args.out or f"01_das_iia{suffix}.json"
    save_results(results, out)

    log(f"\nDone. {len(results)} results across {len(set(r.metadata['task'] for r in results))} tasks.")


if __name__ == "__main__":
    main()
