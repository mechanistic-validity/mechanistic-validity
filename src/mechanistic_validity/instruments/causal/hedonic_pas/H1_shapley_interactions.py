"""Hedonic Shapley Interactions (PAS + OCA)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     H1 — Hedonic Synergy
Categories:     causal
Validity layer: Internal
Criteria:       Pairwise interaction structure between circuit heads
Establishes:    Whether head pairs exhibit synergy (joint > sum of parts)
                or redundancy, and whether OV-space geometry predicts it
Requires:       CPU or GPU, model
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Based on Chowdhury et al. 2025 ("Hedonic Neurons", UMass, arXiv 2509.23684).

PAS (Pairwise Ablation Synergy), Section 3.2:
  phi_PAS(i,j) = -( l_{-{i,j}}(x) - l_{-i}(x) - l_{-j}(x) + l(x) )

OCA (Orthogonal Co-Activation), Section 3.2:
  phi_OCA(i,j) = (1 - |cos(W_i, W_j)|) * rho(a_i, a_j)
  where W_i is the OV weight column for head i, rho is activation correlation.

Pass condition: at least one pair has PAS > 0 (synergy detected).

Usage:
    uv run python H1_shapley_interactions.py --tasks ioi --n-prompts 40
"""

import itertools

import numpy as np
import torch

from mechanistic_validity.instruments.common import (
    CIRCUIT_TASKS,
    EvalResult,
    InstrumentInfo,
    calibrate_mean_z,
    generate_prompts,
    get_circuit_heads,
    get_token_ids,
    heads_to_layer_dict,
    load_model,
    log,
    logit_diff_from_logits,
    make_ablation_hook,
    parse_common_args,
    save_incremental,
    save_results,
)

INSTRUMENT_INFO = InstrumentInfo(
    name="Hedonic Synergy (PAS + OCA)",
    paper_ref="https://arxiv.org/abs/2509.23684",
    paper_cite="Chowdhury et al. 2025 (UMass)",
    description="Pairwise Ablation Synergy and Orthogonal Co-Activation between circuit heads",
    category="causal",
)


@torch.no_grad()
def compute_pas(model, prompts, correct_ids, incorrect_ids,
                circuit_heads: set[tuple[int, int]],
                mean_z: torch.Tensor) -> dict[tuple[tuple[int, int], tuple[int, int]], float]:
    """Compute PAS for all pairs of circuit heads.

    PAS(i,j) = -(l_{-{i,j}} - l_{-i} - l_{-j} + l)
    """
    heads = sorted(circuit_heads)
    pairs = list(itertools.combinations(heads, 2))
    if not pairs:
        return {}

    pas_values: dict[tuple[tuple[int, int], tuple[int, int]], float] = {}

    for head_i, head_j in pairs:
        pas_sum = 0.0
        count = 0

        hooks_i = make_ablation_hook(heads_to_layer_dict({head_i}), mean_z, "mean")
        hooks_j = make_ablation_hook(heads_to_layer_dict({head_j}), mean_z, "mean")
        hooks_ij = make_ablation_hook(heads_to_layer_dict({head_i, head_j}), mean_z, "mean")

        for idx, p in enumerate(prompts):
            if idx >= len(correct_ids):
                break

            tokens = model.to_tokens(p.text)
            cid = correct_ids[idx]
            iid = incorrect_ids[idx]

            clean_logits = model(tokens)
            ld_clean = logit_diff_from_logits(clean_logits, cid, iid)

            logits_i = model.run_with_hooks(tokens, fwd_hooks=hooks_i)
            ld_i = logit_diff_from_logits(logits_i, cid, iid)

            logits_j = model.run_with_hooks(tokens, fwd_hooks=hooks_j)
            ld_j = logit_diff_from_logits(logits_j, cid, iid)

            logits_ij = model.run_with_hooks(tokens, fwd_hooks=hooks_ij)
            ld_ij = logit_diff_from_logits(logits_ij, cid, iid)

            # PAS = -(l_{-{i,j}} - l_{-i} - l_{-j} + l)
            pas = -(ld_ij - ld_i - ld_j + ld_clean)
            pas_sum += pas
            count += 1

        if count > 0:
            pas_values[(head_i, head_j)] = pas_sum / count

    return pas_values


@torch.no_grad()
def compute_oca(model, prompts, circuit_heads: set[tuple[int, int]]) -> dict[tuple[tuple[int, int], tuple[int, int]], float]:
    """Compute OCA (Orthogonal Co-Activation) for all pairs.

    OCA(i,j) = (1 - |cos(W_i, W_j)|) * rho(a_i, a_j)
    """
    heads = sorted(circuit_heads)
    pairs = list(itertools.combinations(heads, 2))
    if not pairs:
        return {}

    # Extract OV weight columns (W_O @ W_V gives the OV circuit per head)
    ov_vecs = {}
    for layer, head in heads:
        W_V = model.W_V[layer, head]  # (d_model, d_head)
        W_O = model.W_O[layer, head]  # (d_head, d_model)
        ov = (W_O @ W_V).flatten()  # flatten to a single vector
        ov_vecs[(layer, head)] = ov

    # Collect activations (hook_z at last position) per head across prompts
    head_activations: dict[tuple[int, int], list[float]] = {h: [] for h in heads}
    for p in prompts:
        tokens = model.to_tokens(p.text)
        _, cache = model.run_with_cache(tokens, names_filter=lambda n: "hook_z" in n)
        for layer, head in heads:
            z = cache[f"blocks.{layer}.attn.hook_z"][0, -1, head]
            head_activations[(layer, head)].append(z.norm().item())

    oca_values: dict[tuple[tuple[int, int], tuple[int, int]], float] = {}
    for head_i, head_j in pairs:
        w_i = ov_vecs[head_i]
        w_j = ov_vecs[head_j]
        cos_sim = torch.nn.functional.cosine_similarity(w_i.unsqueeze(0), w_j.unsqueeze(0)).item()
        orthogonality = 1.0 - abs(cos_sim)

        acts_i = np.array(head_activations[head_i])
        acts_j = np.array(head_activations[head_j])
        if len(acts_i) < 2 or np.std(acts_i) < 1e-10 or np.std(acts_j) < 1e-10:
            rho = 0.0
        else:
            rho = float(np.corrcoef(acts_i, acts_j)[0, 1])

        oca_values[(head_i, head_j)] = orthogonality * rho

    return oca_values


@torch.no_grad()
def run_hedonic_synergy(model, tasks: list[str],
                        n_prompts: int = 10) -> list[EvalResult]:
    tokenizer = model.tokenizer
    results = []

    for task in tasks:
        circuit_heads = get_circuit_heads(task)
        if not circuit_heads or len(circuit_heads) < 2:
            log(f"  {task}: <2 circuit heads, skipping")
            continue

        prompts = generate_prompts(task, tokenizer, n_prompts)
        if not prompts:
            log(f"  {task}: no prompts, skipping")
            continue

        correct_ids, incorrect_ids = get_token_ids(prompts, tokenizer)
        if not correct_ids:
            continue

        log(f"  {task}: {len(circuit_heads)} heads, {len(prompts)} prompts")

        mean_z = calibrate_mean_z(model, prompts, n_calibration=min(100, len(prompts)))

        pas_values = compute_pas(model, prompts, correct_ids, incorrect_ids,
                                 circuit_heads, mean_z)
        oca_values = compute_oca(model, prompts, circuit_heads)

        if not pas_values:
            log(f"  {task}: no pairs computed, skipping")
            continue

        mean_pas = float(np.mean(list(pas_values.values())))
        synergy_pairs = []
        redundancy_pairs = []

        per_pair_pas = {}
        per_pair_oca = {}
        for (hi, hj), v in sorted(pas_values.items()):
            key = f"L{hi[0]}H{hi[1]}-L{hj[0]}H{hj[1]}"
            per_pair_pas[key] = float(v)
            if v > 0:
                synergy_pairs.append(key)
            else:
                redundancy_pairs.append(key)

        for (hi, hj), v in sorted(oca_values.items()):
            key = f"L{hi[0]}H{hi[1]}-L{hj[0]}H{hj[1]}"
            per_pair_oca[key] = float(v)

        passed = len(synergy_pairs) > 0

        log(f"    mean_PAS={mean_pas:.4f}  syn={len(synergy_pairs)} red={len(redundancy_pairs)}  "
            f"[{'PASS' if passed else 'FAIL'}]")

        results.append(EvalResult(
            metric_id="H1.hedonic_synergy",
            value=mean_pas,
            n_samples=len(prompts),
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": task,
                "n_heads": len(circuit_heads),
                "n_pairs": len(pas_values),
                "mean_pas": mean_pas,
                "per_pair_pas": per_pair_pas,
                "per_pair_oca": per_pair_oca,
                "synergy_pairs": synergy_pairs,
                "redundancy_pairs": redundancy_pairs,
                "passed": passed,
            },
        ))

    return results


def main():
    parser = parse_common_args("H1: Hedonic Synergy (PAS + OCA)")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("H1: HEDONIC SYNERGY")
    log("=" * 60)

    out = args.out or "H1_shapley_interactions.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_hedonic_synergy(model, [task], args.n_prompts)
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)
            p = "PASS" if r.metadata["passed"] else "FAIL"
            log(f"  {task}: mean_PAS={r.value:.4f}  [{p}]")

    save_results(results, out, args=args)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
