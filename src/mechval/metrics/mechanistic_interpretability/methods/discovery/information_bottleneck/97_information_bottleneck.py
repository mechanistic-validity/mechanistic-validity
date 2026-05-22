"""Information Bottleneck Circuit Discovery (Causal C13)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     C13 — Information Bottleneck Circuit Discovery
Categories:     causal
Validity layer: Internal
Criteria:       C13 IB Circuit Overlap
Establishes:    Whether IB-discovered circuits overlap with claimed circuit edges
Requires:       CPU or GPU, model
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Implements a simplified version of IBCircuit (Bian et al., ICML 2025).
Learns binary edge masks between attention heads via relaxed Bernoulli
(Hard Concrete) sampling, optimizing:

    L = faithfulness (KL between masked/full model output)
        - beta * compression (expected number of active edges)

For each task, the method discovers a circuit by gradient descent on
the edge mask logits, then compares discovered edges to the claimed
circuit edges using Jaccard overlap. Faithfulness of the discovered
circuit is also measured.

Pass condition: Jaccard > 0.25

Usage:
    uv run python 97_information_bottleneck.py --tasks ioi --n-prompts 40
    uv run python 97_information_bottleneck.py --tasks ioi sva --device cpu
"""

import numpy as np
import torch
import torch.nn.functional as F

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    generate_prompts,
    get_circuit_info,
    get_token_ids,
    load_model,
    log,
    parse_common_args,
    save_incremental,
    save_results,
)


# ---------------------------------------------------------------------------
# Hard Concrete relaxation (Louizos et al., 2018)
# ---------------------------------------------------------------------------
# Stretched sigmoid mapping: z ~ Bernoulli relaxed onto (0,1) via
# the hard concrete distribution, which allows gradient-based
# optimization of discrete masks.

HARD_CONCRETE_GAMMA = -0.1
HARD_CONCRETE_ZETA = 1.1


def _hard_concrete_sample(log_alpha: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """Sample from the hard concrete distribution.

    Returns values in [0, 1] (clamped stretched sigmoid).
    """
    u = torch.rand_like(log_alpha).clamp(1e-6, 1.0 - 1e-6)
    s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + log_alpha) / temperature)
    s_bar = s * (HARD_CONCRETE_ZETA - HARD_CONCRETE_GAMMA) + HARD_CONCRETE_GAMMA
    return s_bar.clamp(0.0, 1.0)


def _hard_concrete_expected_l0(log_alpha: torch.Tensor) -> torch.Tensor:
    """Expected L0 norm (probability that each gate is non-zero)."""
    return torch.sigmoid(log_alpha - np.log(-HARD_CONCRETE_GAMMA / HARD_CONCRETE_ZETA))


# ---------------------------------------------------------------------------
# IB circuit discovery
# ---------------------------------------------------------------------------

def discover_circuit_ib(
    model,
    prompts,
    correct_ids: list[int],
    incorrect_ids: list[int],
    n_steps: int = 200,
    beta: float = 0.1,
    lr: float = 0.1,
    temperature: float = 0.5,
    threshold: float = 0.5,
) -> tuple[np.ndarray, set[tuple[int, int, int, int]], dict]:
    """Discover a circuit via information bottleneck edge masking.

    Returns:
        edge_probs: (n_total, n_total) array of learned gate probabilities
        discovered_edges: set of (Ls, Hs, Lr, Hr) tuples above threshold
        stats: dict with training statistics
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    n_total = n_layers * n_heads
    device = next(model.parameters()).device

    # Initialize log_alpha for each possible forward edge (sender layer < receiver layer)
    log_alpha = torch.zeros(n_total, n_total, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([log_alpha], lr=lr)

    # Build the forward edge mask (only sender layer < receiver layer)
    forward_mask = torch.zeros(n_total, n_total, device=device)
    for Ls in range(n_layers):
        for Hs in range(n_heads):
            s_idx = Ls * n_heads + Hs
            for Lr in range(Ls + 1, n_layers):
                for Hr in range(n_heads):
                    r_idx = Lr * n_heads + Hr
                    forward_mask[s_idx, r_idx] = 1.0

    n_possible = int(forward_mask.sum().item())
    n_prompts_use = min(len(prompts), len(correct_ids), len(incorrect_ids))

    loss_history = []

    for step in range(n_steps):
        # Sample binary masks via hard concrete
        gates = _hard_concrete_sample(log_alpha, temperature) * forward_mask

        # Compute faithfulness loss: KL between masked and full model output
        total_faith_loss = torch.tensor(0.0, device=device)
        prompt_idx = step % n_prompts_use

        tokens = model.to_tokens(prompts[prompt_idx].text)

        # Cache full model outputs (detached)
        with torch.no_grad():
            full_logits = model(tokens)
            full_probs = F.softmax(full_logits[0, -1], dim=-1).detach()

        # Run with edge masking: scale hook_z outputs by gate values
        # For each receiver head, the effective input is a weighted sum
        # of sender head outputs scaled by their edge gates
        cache_dict = {}

        def make_cache_hook(layer):
            def hook_fn(z, hook):
                cache_dict[layer] = z
                return z
            return hook_fn

        def make_mask_hook(recv_layer):
            def hook_fn(z, hook):
                # Scale incoming information from sender heads
                batch, seq, n_h, d_h = z.shape
                for Hr in range(n_heads):
                    r_idx = recv_layer * n_heads + Hr
                    # Aggregate gate scaling from all sender heads
                    scale = torch.tensor(0.0, device=z.device)
                    n_senders = 0
                    for Ls in range(recv_layer):
                        for Hs in range(n_heads):
                            s_idx = Ls * n_heads + Hs
                            if forward_mask[s_idx, r_idx] > 0:
                                scale = scale + gates[s_idx, r_idx]
                                n_senders += 1
                    if n_senders > 0:
                        scale = scale / n_senders
                        z[:, :, Hr, :] = z[:, :, Hr, :] * scale
                return z
            return hook_fn

        fwd_hooks = []
        for L in range(n_layers):
            fwd_hooks.append((f"blocks.{L}.attn.hook_z", make_cache_hook(L)))
            if L > 0:
                fwd_hooks.append((f"blocks.{L}.attn.hook_z", make_mask_hook(L)))

        masked_logits = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)
        masked_log_probs = F.log_softmax(masked_logits[0, -1], dim=-1)

        # KL(full || masked) — faithfulness: how well does the masked circuit approximate the full model
        total_faith_loss = F.kl_div(masked_log_probs, full_probs, reduction="sum")

        # Compression loss: expected L0 norm of the gates (only on forward edges)
        expected_l0 = (_hard_concrete_expected_l0(log_alpha) * forward_mask).sum()
        compression_loss = expected_l0 / max(n_possible, 1)

        loss = total_faith_loss + beta * compression_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(float(loss.item()))

        if (step + 1) % 50 == 0:
            log(f"      step {step+1}/{n_steps}: loss={loss.item():.4f}, "
                f"faith={total_faith_loss.item():.4f}, "
                f"L0={expected_l0.item():.1f}/{n_possible}")

    # Extract final edge probabilities
    with torch.no_grad():
        edge_probs_tensor = torch.sigmoid(log_alpha) * forward_mask
        edge_probs = edge_probs_tensor.cpu().numpy()

    # Threshold to get discovered edges
    discovered_edges = set()
    for Ls in range(n_layers):
        for Hs in range(n_heads):
            s_idx = Ls * n_heads + Hs
            for Lr in range(Ls + 1, n_layers):
                for Hr in range(n_heads):
                    r_idx = Lr * n_heads + Hr
                    if edge_probs[s_idx, r_idx] > threshold:
                        discovered_edges.add((Ls, Hs, Lr, Hr))

    stats = {
        "n_steps": n_steps,
        "beta": beta,
        "final_loss": loss_history[-1] if loss_history else 0.0,
        "n_discovered_edges": len(discovered_edges),
        "n_possible_edges": n_possible,
        "mean_gate_prob": float(edge_probs[forward_mask.cpu().numpy() > 0].mean()),
        "final_expected_l0": float(expected_l0.item()),
    }

    return edge_probs, discovered_edges, stats


def compute_jaccard(set_a: set, set_b: set) -> float:
    """Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union


def compute_discovered_faithfulness(
    model, prompts, correct_ids, incorrect_ids,
    discovered_edges: set[tuple[int, int, int, int]],
) -> float:
    """Measure logit diff faithfulness of the discovered circuit.

    Faithfulness = mean(logit_diff with only discovered edges) / mean(logit_diff full).
    Non-discovered edges are zeroed out.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    n = min(len(prompts), len(correct_ids), len(incorrect_ids), 20)

    # Build per-receiver gate dict
    receiver_gates: dict[int, dict[int, bool]] = {}
    for Ls, Hs, Lr, Hr in discovered_edges:
        r_idx = Lr * n_heads + Hr
        s_idx = Ls * n_heads + Hs
        receiver_gates.setdefault(r_idx, {})[s_idx] = True

    def make_faith_hook(recv_layer):
        def hook_fn(z, hook):
            for Hr in range(n_heads):
                r_idx = recv_layer * n_heads + Hr
                # Check if this receiver has any discovered senders
                has_sender = r_idx in receiver_gates
                if not has_sender and recv_layer > 0:
                    # No discovered edges feed into this head — zero it
                    z[:, :, Hr, :] = 0.0
            return z
        return hook_fn

    faith_num, faith_den = 0.0, 0.0
    with torch.no_grad():
        for i in range(n):
            tokens = model.to_tokens(prompts[i].text)
            full_logits = model(tokens)
            full_ld = (full_logits[0, -1, correct_ids[i]] - full_logits[0, -1, incorrect_ids[i]]).item()

            hooks = [
                (f"blocks.{L}.attn.hook_z", make_faith_hook(L))
                for L in range(1, n_layers)
            ]
            masked_logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
            masked_ld = (masked_logits[0, -1, correct_ids[i]] - masked_logits[0, -1, incorrect_ids[i]]).item()

            faith_num += masked_ld
            faith_den += full_ld

    if abs(faith_den) < 1e-8:
        return 0.0
    return faith_num / faith_den


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_information_bottleneck(
    model, tasks: list[str], n_prompts: int = 40, n_steps: int = 200,
) -> list[EvalResult]:
    tokenizer = model.tokenizer
    results = []

    for task in tasks:
        circuit, all_heads, all_edges = get_circuit_info(task)
        if circuit is None or not all_edges:
            log(f"  {task}: no circuit edges, skipping")
            continue

        prompts = generate_prompts(task, tokenizer, n_prompts)
        if not prompts:
            log(f"  {task}: no prompts, skipping")
            continue

        correct_ids, incorrect_ids = get_token_ids(prompts, tokenizer)
        if not correct_ids:
            continue

        log(f"  {task}: {len(all_edges)} circuit edges, {len(prompts)} prompts, "
            f"{n_steps} IB steps")

        edge_probs, discovered_edges, ib_stats = discover_circuit_ib(
            model, prompts, correct_ids, incorrect_ids, n_steps=n_steps,
        )

        jaccard = compute_jaccard(discovered_edges, all_edges)
        passed = bool(jaccard > 0.25)

        faithfulness = compute_discovered_faithfulness(
            model, prompts, correct_ids, incorrect_ids, discovered_edges,
        )

        # Precision / recall of edge discovery
        if discovered_edges:
            precision = len(discovered_edges & all_edges) / len(discovered_edges)
        else:
            precision = 0.0
        if all_edges:
            recall = len(discovered_edges & all_edges) / len(all_edges)
        else:
            recall = 0.0

        log(f"    Jaccard={jaccard:.4f}, Faithfulness={faithfulness:.4f}  "
            f"[{'PASS' if passed else 'FAIL'}]")
        log(f"    Discovered {len(discovered_edges)} edges "
            f"(precision={precision:.3f}, recall={recall:.3f})")

        # Top discovered edges by probability
        n_layers = model.cfg.n_layers
        n_heads = model.cfg.n_heads
        top_edges = []
        for Ls in range(n_layers):
            for Hs in range(n_heads):
                s_idx = Ls * n_heads + Hs
                for Lr in range(Ls + 1, n_layers):
                    for Hr in range(n_heads):
                        r_idx = Lr * n_heads + Hr
                        prob = float(edge_probs[s_idx, r_idx])
                        if prob > 0.01:
                            top_edges.append({
                                "edge": f"L{Ls}H{Hs}->L{Lr}H{Hr}",
                                "prob": prob,
                                "in_circuit": (Ls, Hs, Lr, Hr) in all_edges,
                            })
        top_edges.sort(key=lambda e: e["prob"], reverse=True)

        results.append(EvalResult(
            metric_id="C13.information_bottleneck_circuit",
            value=jaccard,
            n_samples=len(prompts),
            metadata={
                "task": task,
                "jaccard": jaccard,
                "faithfulness": faithfulness,
                "precision": precision,
                "recall": recall,
                "passed": passed,
                "threshold": 0.25,
                "n_circuit_edges": len(all_edges),
                "n_discovered_edges": len(discovered_edges),
                "n_overlap": len(discovered_edges & all_edges),
                "ib_stats": ib_stats,
                "top_edges": top_edges[:30],
            },
        ))

    return results


def main():
    parser = parse_common_args("C13: Information Bottleneck Circuit Discovery")
    parser.add_argument("--n-steps", type=int, default=200,
                        help="Number of IB optimization steps")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("C13: INFORMATION BOTTLENECK CIRCUIT DISCOVERY")
    log("=" * 60)

    out = args.out or "97_information_bottleneck.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_information_bottleneck(
            model, [task], args.n_prompts, args.n_steps,
        )
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)
            p = "PASS" if r.metadata["passed"] else "FAIL"
            log(f"  {task}: Jaccard={r.value:.4f}  [{p}]")

    save_results(results, out)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
