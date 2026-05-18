"""Procedure Specification (Algorithmic Step Detection)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     A05 — MDC/Glennan Mechanisms
Categories:     causal
Validity layer: Internal
Criteria:       A1 Procedure Specification (proposed)
Establishes:    Whether circuit implements a detectable step-by-step procedure
Requires:       GPU or CPU, model
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Tests whether the circuit implements an ordered procedure:

1. For each pathway in circuit (e.g., detector->integrator->executor):
   a. Measure information gain at each step: logit attribution should
      accumulate along the pathway.
2. For each circuit head, measure what fraction of its output is
   explained by its claimed input heads (from pathways).
3. Report: ordering_score (monotonic information gain),
   pathway_fraction (how much is pathway-mediated).
4. Pass: ordering_score > 0.7 (information accumulates along claimed path).

Usage:
    uv run python 77_procedure_specification.py --tasks ioi sva
    uv run python 77_procedure_specification.py --device cpu --n-prompts 60
"""
import sys
from pathlib import Path

import numpy as np
import torch

_INSTRUMENTS = Path(__file__).resolve().parents[2]  # up to src/instruments/
sys.path.insert(0, str(_INSTRUMENTS))
from _common import (
    CIRCUIT_TASKS,
    EvalResult,
    generate_prompts,
    get_circuit_info,
    get_token_ids,
    load_model,
    log,
    logit_diff_from_logits,
    parse_common_args,
    save_results,
)


@torch.no_grad()
def compute_head_logit_attribution(model, tokens, correct_id: int,
                                   incorrect_id: int) -> dict[tuple[int, int], float]:
    """Compute per-head logit attribution (direct logit contribution via W_U).

    For each head, projects hook_z through W_O and W_U to get direct
    contribution to logit diff at the last position.
    """
    _, cache = model.run_with_cache(
        tokens, names_filter=lambda n: "hook_z" in n,
    )

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    W_U = model.W_U.cpu().float()  # (d_model, d_vocab)

    attribs = {}
    for L in range(n_layers):
        z = cache[f"blocks.{L}.attn.hook_z"][0, -1].cpu().float()  # (n_heads, d_head)
        W_O = model.W_O[L].cpu().float()  # (n_heads, d_head, d_model)
        for H in range(n_heads):
            # Head contribution to residual: z[H] @ W_O[H] -> (d_model,)
            resid_contrib = z[H] @ W_O[H]  # (d_model,)
            # Project through unembedding
            logit_contrib = resid_contrib @ W_U  # (d_vocab,)
            attribs[(L, H)] = (logit_contrib[correct_id] - logit_contrib[incorrect_id]).item()

    return attribs


@torch.no_grad()
def compute_pathway_ordering_score(attribs_list: list[dict[tuple[int, int], float]],
                                   pathway_chains: list[list[set[tuple[int, int]]]]) -> float:
    """Compute how often logit attribution increases along pathway chains.

    For each chain of role-groups [role_A_heads, role_B_heads, role_C_heads, ...],
    check whether mean |attribution| increases monotonically from early to late.

    Returns fraction of chain steps where attribution increases.
    """
    total_steps = 0
    monotonic_steps = 0

    for chain in pathway_chains:
        if len(chain) < 2:
            continue

        # Average attribution per role across prompts
        role_magnitudes = []
        for role_heads in chain:
            role_vals = []
            for attribs in attribs_list:
                head_vals = [abs(attribs.get(h, 0.0)) for h in role_heads]
                if head_vals:
                    role_vals.append(float(np.mean(head_vals)))
            role_magnitudes.append(float(np.mean(role_vals)) if role_vals else 0.0)

        for i in range(len(role_magnitudes) - 1):
            total_steps += 1
            if role_magnitudes[i + 1] >= role_magnitudes[i]:
                monotonic_steps += 1

    if total_steps == 0:
        return 0.0
    return monotonic_steps / total_steps


@torch.no_grad()
def compute_pathway_fraction(model, prompts, correct_ids, incorrect_ids,
                             circuit: dict) -> float:
    """For each receiver head, measure what fraction of its output variance
    is explained by pathway senders (vs non-pathway heads in earlier layers).

    Uses correlation between sender z and receiver z across prompts.
    """
    roles = circuit["roles"]
    pathways = circuit["pathways"]

    # Build map: receiver_role -> set of sender roles
    receiver_to_senders = {}
    for sender_role, receiver_role in pathways:
        receiver_to_senders.setdefault(receiver_role, set()).add(sender_role)

    # Collect z activations across prompts
    z_by_prompt = []
    for i, p in enumerate(prompts):
        if i >= len(correct_ids):
            break
        tokens = model.to_tokens(p.text)
        _, cache = model.run_with_cache(
            tokens, names_filter=lambda n: "hook_z" in n,
        )
        z_dict = {}
        for L in range(model.cfg.n_layers):
            z = cache[f"blocks.{L}.attn.hook_z"][0, -1].cpu()  # (n_heads, d_head)
            for H in range(model.cfg.n_heads):
                z_dict[(L, H)] = z[H].float()
        z_by_prompt.append(z_dict)

    if len(z_by_prompt) < 4:
        return 0.0

    fractions = []

    for recv_role, sender_roles in receiver_to_senders.items():
        recv_heads = roles.get(recv_role, [])
        sender_heads = set()
        for sr in sender_roles:
            sender_heads.update(roles.get(sr, []))

        for rL, rH in recv_heads:
            # Collect receiver z across prompts
            recv_z = torch.stack([zp[(rL, rH)] for zp in z_by_prompt])  # (n, d_head)
            recv_norm = recv_z.norm(dim=1)  # (n,)

            if recv_norm.std() < 1e-8:
                continue

            # Correlation with sender heads
            sender_corrs = []
            for sL, sH in sender_heads:
                if sL >= rL:
                    continue
                send_z = torch.stack([zp[(sL, sH)] for zp in z_by_prompt])
                send_norm = send_z.norm(dim=1)
                if send_norm.std() < 1e-8:
                    continue
                corr = float(np.corrcoef(recv_norm.numpy(), send_norm.numpy())[0, 1])
                if not np.isnan(corr):
                    sender_corrs.append(abs(corr))

            # Correlation with non-sender heads in earlier layers
            non_sender_corrs = []
            all_circuit_heads = set()
            for rh in roles.values():
                all_circuit_heads.update(rh)
            for nL in range(rL):
                for nH in range(model.cfg.n_heads):
                    if (nL, nH) in sender_heads or (nL, nH) not in all_circuit_heads:
                        continue
                    ns_z = torch.stack([zp[(nL, nH)] for zp in z_by_prompt])
                    ns_norm = ns_z.norm(dim=1)
                    if ns_norm.std() < 1e-8:
                        continue
                    corr = float(np.corrcoef(recv_norm.numpy(), ns_norm.numpy())[0, 1])
                    if not np.isnan(corr):
                        non_sender_corrs.append(abs(corr))

            mean_sender = float(np.mean(sender_corrs)) if sender_corrs else 0.0
            mean_non_sender = float(np.mean(non_sender_corrs)) if non_sender_corrs else 0.0
            total = mean_sender + mean_non_sender
            if total > 1e-8:
                fractions.append(mean_sender / total)

    return float(np.mean(fractions)) if fractions else 0.0


def build_pathway_chains(circuit: dict) -> list[list[set[tuple[int, int]]]]:
    """Build maximal chains through the pathway graph.

    Each chain is a list of sets of heads (one set per role in the chain).
    """
    roles = circuit["roles"]
    pathways = circuit["pathways"]

    # Build adjacency: role -> set of downstream roles
    adj = {}
    in_degree = {}
    for sender, receiver in pathways:
        adj.setdefault(sender, set()).add(receiver)
        in_degree.setdefault(receiver, 0)
        in_degree[receiver] = in_degree.get(receiver, 0) + 1
        if sender not in in_degree:
            in_degree[sender] = 0

    # Find source roles (no incoming edges)
    sources = [r for r, deg in in_degree.items() if deg == 0]

    # DFS to enumerate all maximal paths from sources
    chains = []

    def dfs(role, current_chain):
        current_chain.append(set(roles.get(role, [])))
        next_roles = adj.get(role, set())
        if not next_roles:
            chains.append(list(current_chain))
        else:
            for nr in next_roles:
                dfs(nr, current_chain)
        current_chain.pop()

    for src in sources:
        dfs(src, [])

    return chains


def run_procedure_specification(model, tasks: list[str],
                                n_prompts: int = 40) -> list[EvalResult]:
    tokenizer = model.tokenizer
    results = []

    for task in tasks:
        circuit, all_heads, _ = get_circuit_info(task)
        if circuit is None or not all_heads:
            log(f"  {task}: no circuit, skipping")
            continue

        prompts = generate_prompts(task, tokenizer, n_prompts)
        if not prompts:
            log(f"  {task}: no prompts, skipping")
            continue

        correct_ids, incorrect_ids = get_token_ids(prompts, tokenizer)
        if not correct_ids:
            continue

        log(f"  {task} ({len(all_heads)} heads, {len(prompts)} prompts)...")

        # Step 1: Compute per-head logit attributions across prompts
        attribs_list = []
        for i, p in enumerate(prompts):
            if i >= len(correct_ids):
                break
            tokens = model.to_tokens(p.text)
            attribs = compute_head_logit_attribution(
                model, tokens, correct_ids[i], incorrect_ids[i],
            )
            attribs_list.append(attribs)

        # Step 2: Build pathway chains and measure ordering
        pathway_chains = build_pathway_chains(circuit)
        ordering_score = compute_pathway_ordering_score(attribs_list, pathway_chains)

        # Step 3: Measure pathway fraction
        pathway_fraction = compute_pathway_fraction(
            model, prompts, correct_ids, incorrect_ids, circuit,
        )

        n_chains = len(pathway_chains)
        chain_lengths = [len(c) for c in pathway_chains]

        log(f"    ordering_score={ordering_score:.3f}  "
            f"pathway_fraction={pathway_fraction:.3f}  "
            f"n_chains={n_chains}")

        passed = ordering_score > 0.7

        results.append(EvalResult(
            metric_id="A1.procedure_specification",
            value=ordering_score,
            n_samples=len(prompts),
            metadata={
                "task": task,
                "ordering_score": ordering_score,
                "pathway_fraction": pathway_fraction,
                "n_chains": n_chains,
                "chain_lengths": chain_lengths,
                "n_circuit_heads": len(all_heads),
                "n_pathways": len(circuit["pathways"]),
                "passed": passed,
                "threshold": 0.7,
            },
        ))

    return results


def main():
    parser = parse_common_args("A1: Procedure Specification")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("A1: PROCEDURE SPECIFICATION (Algorithmic Step Detection)")
    log("=" * 60)

    results = run_procedure_specification(model, tasks, args.n_prompts)

    out = args.out or "77_procedure_specification.json"
    save_results(results, out)

    log(f"\nDone. {len(results)} tasks evaluated.")
    for r in results:
        t = r.metadata["task"]
        p = "PASS" if r.metadata["passed"] else "FAIL"
        log(f"  {t}: ordering={r.value:.3f}  "
            f"pathway_frac={r.metadata['pathway_fraction']:.3f}  [{p}]")


if __name__ == "__main__":
    main()
