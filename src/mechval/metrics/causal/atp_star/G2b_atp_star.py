"""AtP* Attribution Patching (G2b)
---
Instrument:     G2b -- AtP* Attribution Patching
Categories:     causal
Validity layer: Internal
Criteria:       G2b Edge Attribution with Cancellation Correction
Establishes:    Circuit edges carry significant attribution under AtP* (Kramar et al., arXiv 2403.00745)
Requires:       CPU or GPU, model
---

Standard attribution patching (AtP) computes:
    attr(edge) = grad * (clean_act - corrupt_act)
which can suffer from cancellation when positive and negative gradient
components cancel out. AtP* corrects this by splitting gradients:
    pos_attr = max(grad, 0) * (clean - corrupt)
    neg_attr = min(grad, 0) * (clean - corrupt)
    atp_star = sum(|pos_attr|) + sum(|neg_attr|)

For each edge (src_head -> dst_head), we compute both standard AtP and
AtP* scores. The cancellation ratio quantifies how much information
standard AtP loses.

Pass condition: >= 80% of circuit edges have AtP* attribution above threshold.

Usage:
    uv run python G2b_atp_star.py --tasks ioi --n-prompts 10
    uv run python G2b_atp_star.py --tasks ioi sva --device cpu
"""

import numpy as np
import torch

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    InstrumentInfo,
    generate_prompts,
    get_circuit_info,
    get_token_ids,
    load_model,
    log,
    parse_common_args,
    save_incremental,
    save_results,
)

INSTRUMENT_INFO = InstrumentInfo(
    name="AtP* Attribution Patching",
    paper_ref="https://arxiv.org/abs/2403.00745",
    paper_cite="Kramar et al. 2024 (DeepMind)",
    description="Cancellation-corrected attribution patching for circuit edge discovery",
    category="causal",
)


def compute_atp_star_scores(
    model, prompts, correct_ids, incorrect_ids
) -> tuple[np.ndarray, np.ndarray]:
    """Compute both standard AtP and AtP* edge attribution scores.

    Returns (atp_scores, atp_star_scores), each shape (n_total, n_total)
    where n_total = n_layers * n_heads.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    n = min(len(prompts), len(correct_ids), len(incorrect_ids))
    n_total = n_layers * n_heads
    atp_scores = np.zeros((n_total, n_total), dtype=np.float64)
    atp_star_scores = np.zeros((n_total, n_total), dtype=np.float64)

    for i in range(n):
        tokens = model.to_tokens(prompts[i].text)
        model.zero_grad()

        clean_cache_dict = {}

        def make_clean_hook(layer):
            def hook_fn(z, hook):
                clean_cache_dict[layer] = z
                z.retain_grad()
                return z
            return hook_fn

        fwd_hooks = [
            (f"blocks.{L}.attn.hook_z", make_clean_hook(L))
            for L in range(n_layers)
        ]

        with torch.enable_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)
            logit_diff = logits[0, -1, correct_ids[i]] - logits[0, -1, incorrect_ids[i]]
            logit_diff.backward()

        with torch.no_grad():
            corrupt_cache = {}
            for L in range(n_layers):
                # Mean across sequence as the "corrupted" baseline
                z_clean = clean_cache_dict[L][0].detach()  # (seq, n_heads, d_head)
                corrupt_cache[L] = z_clean.mean(dim=0, keepdim=True).expand_as(z_clean)

            for Ls in range(n_layers):
                z_clean_s = clean_cache_dict[Ls][0, -1].detach()  # (n_heads, d_head)
                z_corrupt_s = corrupt_cache[Ls][-1]  # (n_heads, d_head)
                delta_s = z_clean_s - z_corrupt_s  # (n_heads, d_head)

                for Lr in range(Ls + 1, n_layers):
                    z_r = clean_cache_dict[Lr]
                    if z_r.grad is None:
                        continue
                    grad_r = z_r.grad[0, -1].detach()  # (n_heads, d_head)

                    for Hs in range(n_heads):
                        s_idx = Ls * n_heads + Hs
                        d_s = delta_s[Hs]  # (d_head,)

                        for Hr in range(n_heads):
                            r_idx = Lr * n_heads + Hr
                            g_r = grad_r[Hr]  # (d_head,)

                            # Standard AtP: sum of elementwise grad * delta
                            standard = torch.sum(g_r * d_s).item()
                            atp_scores[s_idx, r_idx] += standard

                            # AtP*: split by gradient sign
                            pos_grad = torch.clamp(g_r, min=0)
                            neg_grad = torch.clamp(g_r, max=0)
                            pos_attr = torch.sum(torch.abs(pos_grad * d_s)).item()
                            neg_attr = torch.sum(torch.abs(neg_grad * d_s)).item()
                            atp_star_scores[s_idx, r_idx] += pos_attr + neg_attr

        model.zero_grad()

        if (i + 1) % 10 == 0:
            log(f"    processed {i+1}/{n} prompts")

    atp_scores /= max(n, 1)
    atp_star_scores /= max(n, 1)
    return atp_scores, atp_star_scores


def score_circuit_edges(
    atp_star_scores: np.ndarray,
    circuit_edges: set[tuple[int, int, int, int]],
    n_layers: int,
    n_heads: int,
    threshold_percentile: float = 50.0,
) -> tuple[float, dict]:
    """Compute fraction of circuit edges above a threshold.

    The threshold is set at the given percentile of all forward-edge
    AtP* scores (including non-circuit edges).
    """
    all_scores = []
    circuit_scores = []
    non_circuit_scores = []

    for Ls in range(n_layers):
        for Hs in range(n_heads):
            s_idx = Ls * n_heads + Hs
            for Lr in range(Ls + 1, n_layers):
                for Hr in range(n_heads):
                    r_idx = Lr * n_heads + Hr
                    score = atp_star_scores[s_idx, r_idx]
                    all_scores.append(score)
                    if (Ls, Hs, Lr, Hr) in circuit_edges:
                        circuit_scores.append(score)
                    else:
                        non_circuit_scores.append(score)

    if not all_scores or not circuit_scores:
        return 0.0, {"n_circuit": len(circuit_scores), "n_total": len(all_scores)}

    threshold = float(np.percentile(all_scores, threshold_percentile))
    fraction_above = float(np.mean([s > threshold for s in circuit_scores]))

    return fraction_above, {
        "n_circuit_edges": len(circuit_scores),
        "n_non_circuit_edges": len(non_circuit_scores),
        "threshold": threshold,
        "mean_circuit_score": float(np.mean(circuit_scores)),
        "mean_non_circuit_score": float(np.mean(non_circuit_scores)) if non_circuit_scores else 0.0,
        "median_circuit_score": float(np.median(circuit_scores)),
        "median_non_circuit_score": float(np.median(non_circuit_scores)) if non_circuit_scores else 0.0,
    }


def run_atp_star(model, tasks: list[str], n_prompts: int = 5) -> list[EvalResult]:
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

        atp_scores, atp_star_scores = compute_atp_star_scores(
            model, prompts, correct_ids, incorrect_ids
        )
        fraction_above, stats = score_circuit_edges(
            atp_star_scores, all_edges, n_layers, n_heads
        )
        passed = bool(fraction_above >= 0.80)

        # Cancellation ratio: how much larger AtP* is compared to |AtP|
        per_edge_details = []
        cancellation_ratios = []
        for Ls, Hs, Lr, Hr in sorted(all_edges):
            s_idx = Ls * n_heads + Hs
            r_idx = Lr * n_heads + Hr
            std_score = float(atp_scores[s_idx, r_idx])
            star_score = float(atp_star_scores[s_idx, r_idx])
            if star_score > 1e-10:
                ratio = star_score / max(abs(std_score), 1e-10)
                cancellation_ratios.append(ratio)
            per_edge_details.append({
                "edge": f"L{Ls}H{Hs}->L{Lr}H{Hr}",
                "atp": std_score,
                "atp_star": star_score,
            })

        mean_cancellation = float(np.mean(cancellation_ratios)) if cancellation_ratios else 1.0

        log(f"    fraction_above_threshold={fraction_above:.3f}  "
            f"[{'PASS' if passed else 'FAIL'}]  "
            f"cancellation_ratio={mean_cancellation:.2f}")

        results.append(EvalResult(
            metric_id="G2b.atp_star",
            value=fraction_above,
            n_samples=len(prompts),
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": task,
                "fraction_above_threshold": fraction_above,
                "passed": passed,
                "pass_threshold": 0.80,
                "cancellation_ratio": mean_cancellation,
                "per_edge": per_edge_details,
                **stats,
            },
        ))

    return results


def main():
    parser = parse_common_args("G2b: AtP* Attribution Patching")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("G2b: AtP* ATTRIBUTION PATCHING")
    log("=" * 60)

    out = args.out or "G2b_atp_star.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_atp_star(model, [task], args.n_prompts)
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)
            p = "PASS" if r.metadata["passed"] else "FAIL"
            log(f"  {task}: fraction={r.value:.3f}  [{p}]")

    save_results(results, out, args=args)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
