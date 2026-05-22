"""Contextual Decomposition for Transformers (Causal C12)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     C12 — Contextual Decomposition
Categories:     causal
Validity layer: Internal
Criteria:       C12 Contextual Decomposition Head Discrimination
Establishes:    Whether closed-form decomposition of per-head contributions discriminates circuit heads from non-circuit heads
Requires:       CPU or GPU, model
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Implements contextual decomposition for transformers (CD-T), following
Hsu & Yu et al. (ICLR 2025). For each attention head, the method
decomposes the head's output into a "relevant" contribution to the
logit difference by measuring how much of the head's hook_z output
(projected through W_O and the unembedding) contributes to the
correct-vs-incorrect logit difference. This is a closed-form
computation requiring only a single forward pass per prompt (no
gradient computation, no training).

Heads are ranked by their mean absolute relevant contribution across
prompts. Circuit heads are treated as positives, all other heads as
negatives. AUROC measures discrimination quality.

Pass condition: AUROC > 0.75 (higher than EAP since CD-T is more
accurate per Hsu & Yu et al.)

Usage:
    uv run python 96_contextual_decomposition.py --tasks ioi --n-prompts 40
    uv run python 96_contextual_decomposition.py --tasks ioi sva --device cpu
"""

import numpy as np
import torch
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


@torch.no_grad()
def compute_cd_head_scores(model, prompts, correct_ids, incorrect_ids) -> np.ndarray:
    """Compute per-head relevant contribution to the logit difference.

    For each head (L, H), extracts hook_z at the last token position,
    projects through W_O and W_U to get the head's contribution to
    logits, then measures the signed contribution to (correct - incorrect)
    logit difference. This is the closed-form contextual decomposition:
    no gradients needed.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    n = min(len(prompts), len(correct_ids), len(incorrect_ids))
    head_scores = np.zeros((n_layers, n_heads), dtype=np.float64)

    for i in range(n):
        tokens = model.to_tokens(prompts[i].text)

        hook_names = [f"blocks.{L}.attn.hook_z" for L in range(n_layers)]
        _, cache = model.run_with_cache(tokens, names_filter=lambda n: n in hook_names)

        for L in range(n_layers):
            z = cache[f"blocks.{L}.attn.hook_z"]  # (batch, seq, n_heads, d_head)
            z_last = z[0, -1]  # (n_heads, d_head)

            # Project through W_O: each head's output in residual stream space
            # W_O shape: (n_heads, d_head, d_model)
            W_O = model.blocks[L].attn.W_O  # (n_heads, d_head, d_model)
            b_O = model.blocks[L].attn.b_O  # (d_model,)

            # Per-head residual stream contribution: z_h @ W_O_h
            # z_last[h] is (d_head,), W_O[h] is (d_head, d_model)
            head_residual = torch.einsum("hd,hdm->hm", z_last, W_O)  # (n_heads, d_model)

            # Project through unembedding to get logit contribution per head
            # W_U shape: (d_model, d_vocab)
            W_U = model.unembed.W_U  # (d_model, d_vocab)

            # Logit contribution per head: head_residual @ W_U -> (n_heads, d_vocab)
            head_logits = head_residual @ W_U  # (n_heads, d_vocab)

            # Relevant contribution: difference between correct and incorrect token logits
            for h in range(n_heads):
                contribution = (head_logits[h, correct_ids[i]] - head_logits[h, incorrect_ids[i]]).item()
                head_scores[L, h] += contribution

        if (i + 1) % 10 == 0:
            log(f"    processed {i+1}/{n} prompts")

    head_scores /= max(n, 1)
    return head_scores


def compute_head_auroc(head_scores: np.ndarray, circuit_heads: set[tuple[int, int]],
                       n_layers: int, n_heads: int) -> tuple[float, dict]:
    """Compute AUROC for head-level discrimination."""
    labels = []
    scores = []
    for L in range(n_layers):
        for H in range(n_heads):
            is_circuit = (L, H) in circuit_heads
            labels.append(1 if is_circuit else 0)
            scores.append(abs(head_scores[L, H]))

    labels = np.array(labels)
    scores = np.array(scores)

    if labels.sum() == 0 or labels.sum() == len(labels):
        return 0.0, {"n_circuit": int(labels.sum()), "n_total": len(labels)}

    auroc = float(roc_auc_score(labels, scores))

    circuit_scores = scores[labels == 1]
    non_circuit_scores = scores[labels == 0]

    return auroc, {
        "n_circuit_heads": int(labels.sum()),
        "n_non_circuit_heads": int((1 - labels).sum()),
        "mean_circuit_score": float(circuit_scores.mean()),
        "mean_non_circuit_score": float(non_circuit_scores.mean()),
        "median_circuit_score": float(np.median(circuit_scores)),
        "median_non_circuit_score": float(np.median(non_circuit_scores)),
    }


def run_contextual_decomposition(model, tasks: list[str], n_prompts: int = 40) -> list[EvalResult]:
    tokenizer = model.tokenizer
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    results = []

    for task in tasks:
        circuit, all_heads, all_edges = get_circuit_info(task)
        if circuit is None or not all_heads:
            log(f"  {task}: no circuit heads, skipping")
            continue

        prompts = generate_prompts(task, tokenizer, n_prompts)
        if not prompts:
            log(f"  {task}: no prompts, skipping")
            continue

        correct_ids, incorrect_ids = get_token_ids(prompts, tokenizer)
        if not correct_ids:
            continue

        log(f"  {task}: {len(all_heads)} circuit heads, {len(prompts)} prompts")

        head_scores = compute_cd_head_scores(model, prompts, correct_ids, incorrect_ids)
        auroc, stats = compute_head_auroc(head_scores, all_heads, n_layers, n_heads)
        passed = bool(auroc > 0.75)

        log(f"    AUROC={auroc:.4f}  [{('PASS' if passed else 'FAIL')}]")
        log(f"    circuit heads: {stats.get('n_circuit_heads', 0)}, "
            f"mean|score|={stats.get('mean_circuit_score', 0):.4f}")
        log(f"    non-circuit:   {stats.get('n_non_circuit_heads', 0)}, "
            f"mean|score|={stats.get('mean_non_circuit_score', 0):.4f}")

        # Build ranked head list for metadata
        ranked_heads = []
        for L in range(n_layers):
            for H in range(n_heads):
                score = float(head_scores[L, H])
                if abs(score) > 1e-8:
                    ranked_heads.append({
                        "head": f"L{L}H{H}",
                        "score": score,
                        "abs_score": abs(score),
                        "in_circuit": (L, H) in all_heads,
                    })
        ranked_heads.sort(key=lambda h: h["abs_score"], reverse=True)

        results.append(EvalResult(
            metric_id="C12.contextual_decomposition",
            value=auroc,
            n_samples=len(prompts),
            metadata={
                "task": task,
                "auroc": auroc,
                "passed": passed,
                "threshold": 0.75,
                "n_circuit_heads": stats.get("n_circuit_heads", 0),
                "n_non_circuit_heads": stats.get("n_non_circuit_heads", 0),
                "mean_circuit_score": stats.get("mean_circuit_score", 0),
                "mean_non_circuit_score": stats.get("mean_non_circuit_score", 0),
                "median_circuit_score": stats.get("median_circuit_score", 0),
                "median_non_circuit_score": stats.get("median_non_circuit_score", 0),
                "top_heads": ranked_heads[:30],
            },
        ))

    return results


def main():
    parser = parse_common_args("C12: Contextual Decomposition for Transformers")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("C12: CONTEXTUAL DECOMPOSITION FOR TRANSFORMERS (CD-T)")
    log("=" * 60)

    out = args.out or "96_contextual_decomposition.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_contextual_decomposition(model, [task], args.n_prompts)
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)
            p = "PASS" if r.metadata["passed"] else "FAIL"
            log(f"  {task}: AUROC={r.value:.4f}  [{p}]")

    save_results(results, out)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
