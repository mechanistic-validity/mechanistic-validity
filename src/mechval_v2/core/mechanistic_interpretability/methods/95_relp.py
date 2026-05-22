"""Relevance Patching (Causal C11) — LRP-based attribution
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     C11 — Relevance Patching
Categories:     causal
Validity layer: Internal
Criteria:       C11 Relevance Patching Agreement
Establishes:    Whether LRP-based relevance scores discriminate circuit edges from non-circuit edges
Requires:       CPU or GPU, model
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Implements a simplified Layer-wise Relevance Propagation (LRP) approach
for computing edge-level attribution scores between attention heads,
inspired by Relevance Patching (arXiv 2508.21258). For each edge
(sender -> receiver), relevance is computed by decomposing the output
logit-diff backward through attention layers: at each receiver head,
the relevance assigned to it is distributed to sender heads in
proportion to (attention_weight * value_norm), approximating LRP's
conservation principle.

Circuit edges are treated as positives, all other forward edges as
negatives. AUROC and Pearson correlation measure agreement with the
claimed circuit.

Pass condition: AUROC > 0.70

Usage:
    uv run python 95_relp.py --tasks ioi --n-prompts 40
    uv run python 95_relp.py --tasks ioi sva --device cpu
"""

import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

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


def compute_relp_scores(model, prompts, correct_ids, incorrect_ids) -> np.ndarray:
    """Compute LRP-based relevance scores between all pairs of attention heads.

    For each prompt:
    1. Run forward pass, caching hook_z (head outputs) and attention patterns.
    2. Compute initial relevance at each head in the last layer from the
       gradient of logit_diff w.r.t. hook_z (this seeds the backward pass).
    3. Propagate relevance backward layer-by-layer: each receiver head
       distributes its relevance to sender heads proportional to
       attention_weight * ||value_contribution||, approximating LRP's
       z-rule decomposition.

    Returns an (n_total, n_total) matrix of mean relevance flow scores,
    where n_total = n_layers * n_heads.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    n_total = n_layers * n_heads
    n = min(len(prompts), len(correct_ids), len(incorrect_ids))
    edge_scores = np.zeros((n_total, n_total), dtype=np.float64)

    for i in range(n):
        tokens = model.to_tokens(prompts[i].text)
        model.zero_grad()

        z_cache = {}
        pattern_cache = {}

        def make_z_hook(layer):
            def hook_fn(z, hook):
                z_cache[layer] = z
                z.retain_grad()
                return z
            return hook_fn

        def make_pattern_hook(layer):
            def hook_fn(pattern, hook):
                pattern_cache[layer] = pattern.detach()
                return pattern
            return hook_fn

        fwd_hooks = []
        for L in range(n_layers):
            fwd_hooks.append((f"blocks.{L}.attn.hook_z", make_z_hook(L)))
            fwd_hooks.append((f"blocks.{L}.attn.hook_pattern", make_pattern_hook(L)))

        with torch.enable_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)
            logit_diff = logits[0, -1, correct_ids[i]] - logits[0, -1, incorrect_ids[i]]
            logit_diff.backward()

        # Seed relevance from gradient magnitude at each head (last token position).
        # R[layer, head] = ||grad of logit_diff w.r.t. z at (layer, head, last_pos)||
        head_relevance = np.zeros((n_layers, n_heads), dtype=np.float64)
        for L in range(n_layers):
            z = z_cache[L]
            if z.grad is not None:
                # (batch, seq, n_heads, d_head) -> take last token
                g = z.grad[0, -1].detach()  # (n_heads, d_head)
                for H in range(n_heads):
                    head_relevance[L, H] = g[H].norm().item()

        # Backward LRP-style propagation: from later layers to earlier layers.
        # For each receiver layer Lr, distribute its relevance to sender layers Ls < Lr
        # proportional to attention_weight * ||z_sender||.
        for Lr in range(n_layers - 1, 0, -1):
            # Attention pattern at receiver layer: (batch, n_heads, seq_q, seq_k)
            pattern = pattern_cache.get(Lr)
            if pattern is None:
                continue
            # Use last-token query attention: (n_heads, seq_k)
            attn_weights = pattern[0, :, -1, :].cpu().numpy()  # (n_heads, seq_k)

            for Hr in range(n_heads):
                r_relevance = head_relevance[Lr, Hr]
                if r_relevance < 1e-12:
                    continue

                # Collect contributions from all sender heads at layers < Lr
                contributions = []
                for Ls in range(Lr):
                    z_s = z_cache[Ls][0].detach()  # (seq, n_heads, d_head)
                    for Hs in range(n_heads):
                        # Value contribution ~ attention * ||z_sender||
                        # Sum attention over key positions, weighted by sender norm at each position
                        z_norms = z_s[:, Hs, :].norm(dim=-1).cpu().numpy()  # (seq,)
                        score = float(np.dot(attn_weights[Hr, :z_norms.shape[0]], z_norms))
                        contributions.append((Ls, Hs, score))

                # Normalize contributions and distribute relevance (LRP conservation)
                total_contrib = sum(c[2] for c in contributions)
                if total_contrib < 1e-12:
                    continue
                for Ls, Hs, contrib in contributions:
                    fraction = contrib / total_contrib
                    flow = r_relevance * fraction
                    s_idx = Ls * n_heads + Hs
                    r_idx = Lr * n_heads + Hr
                    edge_scores[s_idx, r_idx] += flow
                    # Also propagate relevance backward to the sender head
                    head_relevance[Ls, Hs] += flow

        model.zero_grad()

        if (i + 1) % 10 == 0:
            log(f"    processed {i+1}/{n} prompts")

    edge_scores /= max(n, 1)
    return edge_scores


def compute_auroc_and_correlation(
    edge_scores: np.ndarray,
    circuit_edges: set[tuple[int, int, int, int]],
    n_layers: int,
    n_heads: int,
) -> tuple[float, float, dict]:
    """Compute AUROC and Pearson correlation between relevance scores and circuit labels."""
    labels = []
    scores = []
    for Ls in range(n_layers):
        for Hs in range(n_heads):
            s_idx = Ls * n_heads + Hs
            for Lr in range(Ls + 1, n_layers):
                for Hr in range(n_heads):
                    r_idx = Lr * n_heads + Hr
                    is_circuit = (Ls, Hs, Lr, Hr) in circuit_edges
                    labels.append(1 if is_circuit else 0)
                    scores.append(abs(edge_scores[s_idx, r_idx]))

    labels = np.array(labels)
    scores = np.array(scores)

    if labels.sum() == 0 or labels.sum() == len(labels):
        return 0.0, 0.0, {"n_circuit": int(labels.sum()), "n_total": len(labels)}

    auroc = float(roc_auc_score(labels, scores))

    # Pearson correlation between binary circuit membership and relevance magnitude
    if scores.std() > 1e-12:
        corr, _ = pearsonr(labels, scores)
        corr = float(corr)
    else:
        corr = 0.0

    circuit_scores = scores[labels == 1]
    non_circuit_scores = scores[labels == 0]

    return auroc, corr, {
        "n_circuit_edges": int(labels.sum()),
        "n_non_circuit_edges": int((1 - labels).sum()),
        "mean_circuit_score": float(circuit_scores.mean()),
        "mean_non_circuit_score": float(non_circuit_scores.mean()),
        "median_circuit_score": float(np.median(circuit_scores)),
        "median_non_circuit_score": float(np.median(non_circuit_scores)),
    }


def run_relevance_patching(model, tasks: list[str], n_prompts: int = 40) -> list[EvalResult]:
    tokenizer = model.tokenizer
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
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

        log(f"  {task}: {len(all_edges)} circuit edges, {len(prompts)} prompts")

        edge_scores = compute_relp_scores(model, prompts, correct_ids, incorrect_ids)
        auroc, corr, stats = compute_auroc_and_correlation(
            edge_scores, all_edges, n_layers, n_heads,
        )
        passed = bool(auroc > 0.70)

        log(f"    AUROC={auroc:.4f}  Pearson={corr:.4f}  [{('PASS' if passed else 'FAIL')}]")
        log(f"    circuit edges: {stats.get('n_circuit_edges', 0)}, "
            f"mean|score|={stats.get('mean_circuit_score', 0):.4f}")
        log(f"    non-circuit:   {stats.get('n_non_circuit_edges', 0)}, "
            f"mean|score|={stats.get('mean_non_circuit_score', 0):.4f}")

        top_edges = []
        for Ls in range(n_layers):
            for Hs in range(n_heads):
                s_idx = Ls * n_heads + Hs
                for Lr in range(Ls + 1, n_layers):
                    for Hr in range(n_heads):
                        r_idx = Lr * n_heads + Hr
                        score = float(edge_scores[s_idx, r_idx])
                        if abs(score) > 1e-6:
                            top_edges.append({
                                "edge": f"L{Ls}H{Hs}->L{Lr}H{Hr}",
                                "score": score,
                                "in_circuit": (Ls, Hs, Lr, Hr) in all_edges,
                            })
        top_edges.sort(key=lambda e: abs(e["score"]), reverse=True)

        results.append(EvalResult(
            metric_id="C11.relp_agreement",
            value=auroc,
            n_samples=len(prompts),
            metadata={
                "task": task,
                "auroc": auroc,
                "pearson_correlation": corr,
                "passed": passed,
                "threshold": 0.70,
                "n_circuit_edges": stats.get("n_circuit_edges", 0),
                "n_non_circuit_edges": stats.get("n_non_circuit_edges", 0),
                "mean_circuit_score": stats.get("mean_circuit_score", 0),
                "mean_non_circuit_score": stats.get("mean_non_circuit_score", 0),
                "median_circuit_score": stats.get("median_circuit_score", 0),
                "median_non_circuit_score": stats.get("median_non_circuit_score", 0),
                "top_edges": top_edges[:30],
            },
        ))

    return results


def main():
    parser = parse_common_args("C11: Relevance Patching (LRP-based attribution)")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("C11: RELEVANCE PATCHING (LRP-BASED ATTRIBUTION)")
    log("=" * 60)

    out = args.out or "95_relp.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_relevance_patching(model, [task], args.n_prompts)
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)
            p = "PASS" if r.metadata["passed"] else "FAIL"
            log(f"  {task}: AUROC={r.value:.4f}  [{p}]")

    save_results(results, out)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
