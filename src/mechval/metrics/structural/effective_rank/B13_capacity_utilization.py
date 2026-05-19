"""Capacity Utilization Gap
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Categories:     structural
Validity layer: Construct
Criteria:       C2 Structural plausibility
Establishes:    Circuit heads use a larger fraction of their weight-space
                capacity than non-circuit heads
Requires:       CPU, model weights only
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For each head, compute:
  - W_QK effective rank (how many independent QK patterns the head *could* use)
  - W_OV effective rank (how many independent OV read/write directions)
  - Top singular value concentration: sigma_1 / sum(sigma_i)
    High = head capacity is dominated by one direction (tight/specialized)
    Low  = capacity spread across many directions (diffuse/general)

Circuit heads that are genuinely specialized for a task should show
higher concentration (more capacity used by fewer directions) than
background heads.

Usage:
    uv run python B13_capacity_utilization.py --tasks ioi sva
"""

import numpy as np
import torch

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    get_circuit_heads,
    load_completed_tasks,
    load_model,
    log,
    parse_common_args,
    save_incremental,
    save_results,
)

JSONL_FILE = "B13_capacity_utilization.jsonl"


def effective_rank(singular_values: torch.Tensor) -> float:
    sv = singular_values[singular_values > 1e-10]
    if len(sv) == 0:
        return 0.0
    p = sv / sv.sum()
    entropy = -(p * p.log()).sum().item()
    return float(np.exp(entropy))


def sv_concentration(singular_values: torch.Tensor) -> float:
    sv = singular_values[singular_values > 1e-10]
    if len(sv) == 0:
        return 0.0
    return float(sv[0] / sv.sum())


@torch.no_grad()
def compute_head_capacity(model) -> dict[tuple[int, int], dict]:
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    results = {}

    for L in range(n_layers):
        W_Q = model.W_Q[L]
        W_K = model.W_K[L]
        W_V = model.W_V[L]
        W_O = model.W_O[L]
        for H in range(n_heads):
            W_QK = W_Q[H] @ W_K[H].T
            W_OV = W_V[H] @ W_O[H]

            sv_qk = torch.linalg.svdvals(W_QK.float())
            sv_ov = torch.linalg.svdvals(W_OV.float())

            results[(L, H)] = {
                "qk_effective_rank": effective_rank(sv_qk),
                "ov_effective_rank": effective_rank(sv_ov),
                "qk_concentration": sv_concentration(sv_qk),
                "ov_concentration": sv_concentration(sv_ov),
                "qk_spectral_norm": float(sv_qk[0]),
                "ov_spectral_norm": float(sv_ov[0]),
            }

    return results


def run(model=None, tasks: list[str] | None = None, resume: bool = True) -> list[EvalResult]:
    if tasks is None:
        tasks = CIRCUIT_TASKS

    completed, prior = load_completed_tasks(JSONL_FILE) if resume else (set(), [])
    results = list(prior)

    head_capacity = compute_head_capacity(model)
    all_heads = set(head_capacity.keys())

    for task in tasks:
        if task in completed:
            continue

        circuit_heads = get_circuit_heads(task)
        if not circuit_heads:
            log(f"  {task}: no circuit, skipping")
            continue

        circuit_set = set(circuit_heads)
        non_circuit = all_heads - circuit_set

        circuit_stats = [head_capacity[h] for h in circuit_set if h in head_capacity]
        non_circuit_stats = [head_capacity[h] for h in non_circuit if h in head_capacity]

        if not circuit_stats or not non_circuit_stats:
            continue

        def mean_field(stats, field):
            return float(np.mean([s[field] for s in stats]))

        circ_qk_rank = mean_field(circuit_stats, "qk_effective_rank")
        circ_ov_rank = mean_field(circuit_stats, "ov_effective_rank")
        circ_qk_conc = mean_field(circuit_stats, "qk_concentration")
        circ_ov_conc = mean_field(circuit_stats, "ov_concentration")
        bg_qk_rank = mean_field(non_circuit_stats, "qk_effective_rank")
        bg_ov_rank = mean_field(non_circuit_stats, "ov_effective_rank")
        bg_qk_conc = mean_field(non_circuit_stats, "qk_concentration")
        bg_ov_conc = mean_field(non_circuit_stats, "ov_concentration")

        log(f"  {task}: {len(circuit_heads)} circuit heads")
        log(f"    QK eff_rank: circuit={circ_qk_rank:.2f}  bg={bg_qk_rank:.2f}  ratio={circ_qk_rank/bg_qk_rank:.3f}")
        log(f"    OV eff_rank: circuit={circ_ov_rank:.2f}  bg={bg_ov_rank:.2f}  ratio={circ_ov_rank/bg_ov_rank:.3f}")
        log(f"    QK concentration: circuit={circ_qk_conc:.4f}  bg={bg_qk_conc:.4f}")
        log(f"    OV concentration: circuit={circ_ov_conc:.4f}  bg={bg_ov_conc:.4f}")

        r = EvalResult(
            metric_id="B13.capacity_utilization",
            value=circ_qk_conc / bg_qk_conc if bg_qk_conc > 0 else 0.0,
            n_samples=len(circuit_heads),
            metadata={
                "task": task,
                "n_circuit_heads": len(circuit_heads),
                "circuit_qk_effective_rank": circ_qk_rank,
                "circuit_ov_effective_rank": circ_ov_rank,
                "circuit_qk_concentration": circ_qk_conc,
                "circuit_ov_concentration": circ_ov_conc,
                "background_qk_effective_rank": bg_qk_rank,
                "background_ov_effective_rank": bg_ov_rank,
                "background_qk_concentration": bg_qk_conc,
                "background_ov_concentration": bg_ov_conc,
                "qk_rank_ratio": circ_qk_rank / bg_qk_rank if bg_qk_rank > 0 else 0.0,
                "ov_rank_ratio": circ_ov_rank / bg_ov_rank if bg_ov_rank > 0 else 0.0,
                "qk_concentration_ratio": circ_qk_conc / bg_qk_conc if bg_qk_conc > 0 else 0.0,
                "ov_concentration_ratio": circ_ov_conc / bg_ov_conc if bg_ov_conc > 0 else 0.0,
            },
        )
        results.append(r)
        save_incremental(r, JSONL_FILE)

    return results


def main():
    parser = parse_common_args("B13: Capacity Utilization Gap")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("B13: CAPACITY UTILIZATION GAP")
    log("=" * 60)

    results = run(model=model, tasks=tasks)
    save_results(results, args.out or "B13_capacity_utilization.json", args=args)
    log(f"\nDone. {len(results)} results.")


if __name__ == "__main__":
    main()
