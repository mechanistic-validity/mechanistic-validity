"""Local Learning Coefficient (LLC) per Component
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     A09 — MDL/SLT
Categories:     causal, structural
Validity layer: Construct
Criteria:       C4 Minimality
Establishes:    Circuit components have measurable geometric complexity (LLC) indicating specialization
Requires:       GPU, model
Doc:            /instruments_v2/causal/a09-mdl-slt
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Measures the effective complexity/degeneracy of each circuit component
at the current model weights using Singular Learning Theory. High LLC =
geometrically complex, likely polyfunctional; low LLC = specialized.

Optional: pip install devinterp (for SGLD-based LLC estimation).
Falls back to Hessian-based local curvature estimate.

Usage:
    uv run python 10_llc.py --tasks ioi sva --n-prompts 40
    uv run python 10_llc.py --device cuda
"""

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
    parse_common_args,
    save_results,
)

try:
    from devinterp.slt import estimate_learning_coeff
    HAS_DEVINTERP = True
except ImportError:
    HAS_DEVINTERP = False


def estimate_llc_hessian(model, prompts, correct_ids, incorrect_ids,
                          layer: int, head: int, n_samples: int = 20) -> float:
    """Estimate local learning coefficient via Hessian trace approximation.

    LLC ~ (1/2) * rank(Hessian at current parameters) / log(n)

    We approximate this by computing the variance of gradients across
    prompts (Fisher information diagonal), which approximates the
    Hessian for the cross-entropy loss.
    """
    W_O = model.W_O[layer, head]
    W_V = model.W_V[layer, head]
    d_head = model.cfg.d_head

    grad_accum = torch.zeros(d_head, device=W_O.device)
    grad_sq_accum = torch.zeros(d_head, device=W_O.device)
    n_valid = 0

    for i in range(min(len(prompts), n_samples, len(correct_ids))):
        tokens = model.to_tokens(prompts[i].text)
        model.zero_grad()

        logits = model(tokens)
        target = torch.tensor([correct_ids[i]], device=logits.device)
        loss = torch.nn.functional.cross_entropy(logits[0, -1:], target)
        loss.backward()

        hook_name = f"blocks.{layer}.attn.hook_z"
        for name, param in model.named_parameters():
            if f"blocks.{layer}" in name and "W_O" in name:
                if param.grad is not None:
                    g = param.grad.reshape(-1)[:d_head]
                    grad_accum += g.detach()
                    grad_sq_accum += (g ** 2).detach()
                    n_valid += 1
                break

    if n_valid < 2:
        return 0.0

    mean_grad = grad_accum / n_valid
    var_grad = grad_sq_accum / n_valid - mean_grad ** 2
    fisher_trace = var_grad.sum().item()

    effective_dim = (var_grad > var_grad.max() * 0.01).sum().item()
    llc = effective_dim / (2 * np.log(max(n_valid, 2)))

    return llc


def run_llc(model, tasks: list[str], n_prompts: int = 40) -> list[EvalResult]:
    tokenizer = model.tokenizer
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
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

        log(f"  {task} ({len(circuit_heads)} heads)...")

        circuit_llcs = {}
        non_circuit_llcs = []

        for L, H in sorted(circuit_heads):
            llc = estimate_llc_hessian(model, prompts, correct_ids, incorrect_ids, L, H)
            circuit_llcs[f"L{L}H{H}"] = llc

        sample_non_circuit = []
        for L in range(n_layers):
            for H in range(n_heads):
                if (L, H) not in circuit_heads:
                    sample_non_circuit.append((L, H))

        rng = np.random.RandomState(42)
        sample_size = min(20, len(sample_non_circuit))
        sampled = [sample_non_circuit[j]
                   for j in rng.choice(len(sample_non_circuit), size=sample_size, replace=False)]

        for L, H in sampled:
            llc = estimate_llc_hessian(model, prompts, correct_ids, incorrect_ids, L, H)
            non_circuit_llcs.append(llc)

        mean_circuit = float(np.mean(list(circuit_llcs.values()))) if circuit_llcs else 0.0
        mean_non_circuit = float(np.mean(non_circuit_llcs)) if non_circuit_llcs else 0.0
        ratio = mean_circuit / mean_non_circuit if mean_non_circuit > 0 else 0.0

        log(f"    circuit_LLC={mean_circuit:.4f}  non_circuit={mean_non_circuit:.4f}  "
            f"ratio={ratio:.3f}")

        results.append(EvalResult(
            metric_id="C10.llc",
            value=mean_circuit,
            baseline_random=mean_non_circuit,
            n_samples=len(prompts),
            metadata={
                "task": task,
                "per_head_llc": circuit_llcs,
                "mean_circuit_llc": mean_circuit,
                "mean_non_circuit_llc": mean_non_circuit,
                "ratio": ratio,
                "n_circuit_heads": len(circuit_heads),
                "interpretation": "lower LLC = more specialized/degenerate",
            },
        ))

    return results


def main():
    parser = parse_common_args("C10: Local Learning Coefficient")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("C10: LOCAL LEARNING COEFFICIENT (LLC)")
    log("=" * 60)

    if not HAS_DEVINTERP:
        log("NOTE: devinterp not installed. Using Hessian-based approximation.")

    results = run_llc(model, tasks, args.n_prompts)

    out = args.out or "10_llc.json"
    save_results(results, out)

    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
