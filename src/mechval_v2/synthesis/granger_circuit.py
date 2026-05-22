"""Granger Causality Circuit Graph (S08)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Meta-protocol:  Standalone (runs on model directly)
Categories:     meta, causal
Validity layer: Internal
Establishes:    Directed circuit graph via conditional independence tests
Requires:       GPU (forward passes), protocol results optional
Source:         Granger 1969 (Nobel Prize 2003), ICLR 2025
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Treats layer depth as "time" and component activations as multivariate
time series. Tests whether head A (layer ℓ_A) Granger-causes head B
(layer ℓ_B > ℓ_A) — i.e., whether A's activation improves prediction
of B beyond B's own history.

Produces a directed graph with F-test p-values on each edge. No
interventions needed — purely observational.

Usage:
    uv run python granger_circuit.py --tasks ioi --n-prompts 40 --device cuda
"""
import time

import numpy as np
from scipy import stats

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    get_circuit_heads,
    get_circuit_info,
    log,
    parse_common_args,
    save_results,
)

PROTOCOL_ID = "S08"
PROTOCOL_NAME = "Granger Causality Circuit Graph"

GPT2_HEADS = [(layer, head) for layer in range(12) for head in range(12)]
N_HEADS = len(GPT2_HEADS)


def _collect_activations(model, tokens, device: str = "cpu") -> np.ndarray:
    import torch

    model.eval()
    acts = np.zeros((tokens.shape[0], tokens.shape[1], N_HEADS))

    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens.to(device),
            names_filter=lambda n: "hook_z" in n,
        )

    for i, (layer, head) in enumerate(GPT2_HEADS):
        hook_name = f"blocks.{layer}.attn.hook_z"
        if hook_name in cache:
            z = cache[hook_name][:, :, head, :]  # (batch, seq, d_head)
            acts[:, :, i] = z.float().cpu().numpy().mean(axis=-1)

    return acts


def _granger_f_test(x: np.ndarray, y: np.ndarray, max_lag: int = 3) -> tuple[float, float]:
    n = len(y)
    if n <= max_lag + 2:
        return 0.0, 1.0

    Y_target = y[max_lag:]
    n_obs = len(Y_target)

    X_restricted = np.column_stack([
        y[max_lag - lag: n - lag] for lag in range(1, max_lag + 1)
    ])

    X_unrestricted = np.column_stack([
        X_restricted,
        *[x[max_lag - lag: n - lag].reshape(-1, 1) for lag in range(1, max_lag + 1)]
    ])

    X_r = np.column_stack([np.ones(n_obs), X_restricted])
    X_u = np.column_stack([np.ones(n_obs), X_unrestricted])

    try:
        beta_r = np.linalg.lstsq(X_r, Y_target, rcond=None)[0]
        resid_r = Y_target - X_r @ beta_r
        rss_r = (resid_r ** 2).sum()

        beta_u = np.linalg.lstsq(X_u, Y_target, rcond=None)[0]
        resid_u = Y_target - X_u @ beta_u
        rss_u = (resid_u ** 2).sum()

        p = max_lag
        df1 = p
        df2 = n_obs - 2 * max_lag - 1
        if df2 <= 0 or rss_u <= 0:
            return 0.0, 1.0

        f_stat = ((rss_r - rss_u) / df1) / (rss_u / df2)
        p_value = 1.0 - stats.f.cdf(f_stat, df1, df2)
        return float(f_stat), float(p_value)
    except (np.linalg.LinAlgError, ValueError):
        return 0.0, 1.0


def run_granger(model=None, tasks: list[str] | None = None,
                device: str = "cpu", n_prompts: int = 40) -> list[EvalResult]:
    if tasks is None:
        tasks = list(CIRCUIT_TASKS)
    if model is None:
        log("  Model required for Granger causality")
        return []

    import torch
    from mechval.metrics.common import generate_prompts

    results = []
    for task in tasks:
        prompts = generate_prompts(task, model.tokenizer, n_prompts=n_prompts)
        if not prompts:
            log(f"  {task}: no prompts, skipping")
            continue

        texts = [p.text if hasattr(p, "text") else str(p) for p in prompts]
        tokens = model.to_tokens(texts)
        log(f"  {task}: collecting activations ({tokens.shape[0]} prompts)...")
        acts = _collect_activations(model, tokens, device)

        mean_acts = acts.mean(axis=(0, 1))  # (N_HEADS,)

        log(f"  {task}: computing Granger tests...")
        edges = []
        for i, (l_a, h_a) in enumerate(GPT2_HEADS):
            for j, (l_b, h_b) in enumerate(GPT2_HEADS):
                if l_b <= l_a:
                    continue

                x_series = acts[:, :, i].flatten()
                y_series = acts[:, :, j].flatten()

                f_stat, p_val = _granger_f_test(x_series, y_series, max_lag=2)

                if p_val < 0.05:
                    edges.append({
                        "sender": f"L{l_a}H{h_a}",
                        "receiver": f"L{l_b}H{h_b}",
                        "f_stat": f_stat,
                        "p_value": p_val,
                        "sender_idx": i,
                        "receiver_idx": j,
                    })

        bonferroni_edges = [e for e in edges if e["p_value"] * len(edges) < 0.05]
        log(f"    {len(edges)} raw edges (p<0.05), {len(bonferroni_edges)} after Bonferroni")

        gt_heads = get_circuit_heads(task)
        _, _, gt_edges = get_circuit_info(task)
        head_to_idx = {h: i for i, h in enumerate(GPT2_HEADS)}

        if gt_edges:
            gt_edge_set = {(f"L{sl}H{sh}", f"L{rl}H{rh}") for sl, sh, rl, rh in gt_edges}
            granger_edge_set = {(e["sender"], e["receiver"]) for e in bonferroni_edges}
            intersection = gt_edge_set & granger_edge_set
            union = gt_edge_set | granger_edge_set
            edge_jaccard = len(intersection) / len(union) if union else 0.0
        else:
            edge_jaccard = 0.0

        in_degree = {}
        out_degree = {}
        for e in bonferroni_edges:
            out_degree[e["sender"]] = out_degree.get(e["sender"], 0) + 1
            in_degree[e["receiver"]] = in_degree.get(e["receiver"], 0) + 1

        hub_heads = sorted(out_degree.items(), key=lambda x: x[1], reverse=True)[:10]
        log(f"    Hub heads (out-degree): {hub_heads[:5]}")
        log(f"    Edge Jaccard with GT: {edge_jaccard:.4f}")

        results.append(EvalResult(
            metric_id="S08.n_granger_edges",
            value=float(len(bonferroni_edges)),
            n_samples=N_HEADS * (N_HEADS - 1) // 2,
            metadata={
                "task": task,
                "n_raw_edges": len(edges),
                "n_bonferroni_edges": len(bonferroni_edges),
                "edge_jaccard_gt": edge_jaccard,
                "top_hub_heads": [{"head": h, "out_degree": d} for h, d in hub_heads],
                "top_edges": sorted(bonferroni_edges,
                                    key=lambda e: e["f_stat"], reverse=True)[:20],
            },
        ))

    return results


def run_protocol(model, tasks, n_prompts=40, device="cpu", run_cals=False,
                 protocol_results=None):
    from protocols import ProtocolResult
    t0 = time.time()
    evals = run_granger(model, tasks, device=device, n_prompts=n_prompts)
    result = ProtocolResult(
        protocol_id=PROTOCOL_ID,
        protocol_name=PROTOCOL_NAME,
        tasks=tasks,
        elapsed_seconds=time.time() - t0,
    )
    result.metrics["granger"] = evals
    return result


def summarize(result) -> str:
    lines = [f"=== {PROTOCOL_NAME} ==="]
    for ev in result.metrics.get("granger", []):
        lines.append(f"  {ev.metric_id}: {ev.value:.0f} edges")
    return "\n".join(lines)


def main():
    parser = parse_common_args("S08: Granger Causality Circuit Graph")
    args = parser.parse_args()
    tasks = args.tasks or CIRCUIT_TASKS

    from mechval.metrics.common import load_model
    model = load_model("gpt2", args.device)

    log("=" * 60)
    log("S08: GRANGER CAUSALITY CIRCUIT GRAPH")
    log("=" * 60)

    results = run_granger(model, tasks, device=args.device, n_prompts=args.n_prompts)
    out = args.out or "meta_p8_granger.json"
    save_results(results, out)
    log(f"\nDone. {len(results)} results.")


if __name__ == "__main__":
    main()
