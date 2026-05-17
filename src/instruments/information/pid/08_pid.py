"""Partial Information Decomposition (PID)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     C04 — Partial Information Decomposition
Categories:     information, causal
Validity layer: Internal
Criteria:       I3 Specificity
Establishes:    Decomposes head-pair information into redundant, unique, and synergistic components
Requires:       GPU, model
Doc:            /instruments_v2/information/c04-pid
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Decomposes mutual information between pairs of circuit heads and the
output into: redundant, unique (A only, B only), and synergistic.

Key application: tests whether "than" factors (e.g., factors 121 vs 219)
carry unique information about different contexts, or are redundant.

Requires: pip install dit (for PID_BROJA). Falls back to binned
approximation if unavailable.

Usage:
    uv run python 08_pid.py --tasks ioi sva --n-prompts 60
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    CIRCUIT_TASKS,
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

try:
    import dit
    from dit.pid import PID_BROJA
    HAS_DIT = True
except ImportError:
    HAS_DIT = False

N_BINS = 5


def quantile_bin(values: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    """Bin continuous values into quantile-based discrete bins."""
    if len(values) < n_bins:
        return np.zeros(len(values), dtype=int)
    percentiles = np.linspace(0, 100, n_bins + 1)[1:-1]
    thresholds = np.percentile(values, percentiles)
    return np.digitize(values, thresholds)


def compute_pid_manual(x_binned: np.ndarray, y_binned: np.ndarray,
                       z_binned: np.ndarray) -> dict:
    """Compute approximate PID using binned mutual information estimates.

    x, y = two source variables (head activations)
    z = target variable (logit diff bin)

    Returns dict with redundancy, unique_x, unique_y, synergy.
    """
    def mi(a, b):
        joint = np.zeros((a.max() + 1, b.max() + 1))
        for i in range(len(a)):
            joint[a[i], b[i]] += 1
        joint /= joint.sum()
        pa = joint.sum(axis=1)
        pb = joint.sum(axis=0)
        mi_val = 0.0
        for i in range(joint.shape[0]):
            for j in range(joint.shape[1]):
                if joint[i, j] > 0 and pa[i] > 0 and pb[j] > 0:
                    mi_val += joint[i, j] * np.log2(joint[i, j] / (pa[i] * pb[j]))
        return max(mi_val, 0.0)

    def cmi(a, b, c):
        mi_abc = mi_3way_total(a, b, c)
        return mi_abc

    def mi_3way_total(a, b, c):
        joint = {}
        n = len(a)
        for i in range(n):
            key = (a[i], b[i], c[i])
            joint[key] = joint.get(key, 0) + 1
        for k in joint:
            joint[k] /= n

        margin_ac = {}
        margin_bc = {}
        margin_c = {}
        for (ai, bi, ci), p in joint.items():
            margin_ac[(ai, ci)] = margin_ac.get((ai, ci), 0) + p
            margin_bc[(bi, ci)] = margin_bc.get((bi, ci), 0) + p
            margin_c[ci] = margin_c.get(ci, 0) + p

        cmi_val = 0.0
        for (ai, bi, ci), p_abc in joint.items():
            p_ac = margin_ac.get((ai, ci), 0)
            p_bc = margin_bc.get((bi, ci), 0)
            p_c = margin_c.get(ci, 0)
            if p_abc > 0 and p_ac > 0 and p_bc > 0 and p_c > 0:
                cmi_val += p_abc * np.log2(p_abc * p_c / (p_ac * p_bc))
        return max(cmi_val, 0.0)

    i_xz = mi(x_binned, z_binned)
    i_yz = mi(y_binned, z_binned)
    i_xyz = mi_3way_total(x_binned, y_binned, z_binned)

    redundancy = min(i_xz, i_yz)
    unique_x = max(i_xz - redundancy, 0.0)
    unique_y = max(i_yz - redundancy, 0.0)
    total_joint = i_xz + i_yz - redundancy + i_xyz
    synergy = max(total_joint - i_xz - i_yz + redundancy, 0.0)

    return {
        "redundancy": redundancy,
        "unique_x": unique_x,
        "unique_y": unique_y,
        "synergy": synergy,
        "i_xz": i_xz,
        "i_yz": i_yz,
    }


def compute_pid_dit(x_binned: np.ndarray, y_binned: np.ndarray,
                    z_binned: np.ndarray) -> dict:
    """Compute PID using the dit library's BROJA implementation."""
    outcomes = []
    for i in range(len(x_binned)):
        outcomes.append((str(x_binned[i]), str(y_binned[i]), str(z_binned[i])))

    counts = {}
    for o in outcomes:
        counts[o] = counts.get(o, 0) + 1
    total = sum(counts.values())
    pmf = {k: v / total for k, v in counts.items()}

    d = dit.Distribution(list(pmf.keys()), list(pmf.values()))
    pid = PID_BROJA(d, [[0], [1]], [2])

    return {
        "redundancy": float(pid.get_partial((frozenset({0}), frozenset({1})))),
        "unique_x": float(pid.get_partial((frozenset({0}),))),
        "unique_y": float(pid.get_partial((frozenset({1}),))),
        "synergy": float(pid.get_partial((frozenset({0, 1}),))),
    }


@torch.no_grad()
def collect_activations_for_pid(model, prompts, correct_ids, incorrect_ids,
                                 circuit_heads):
    """Collect head activation norms for PID analysis."""
    head_list = sorted(circuit_heads)
    n_valid = min(len(prompts), len(correct_ids))
    activations = {}
    logit_diffs = np.zeros(n_valid)

    for L, H in head_list:
        activations[(L, H)] = np.zeros(n_valid)

    for i in range(n_valid):
        tokens = model.to_tokens(prompts[i].text)
        logits, cache = model.run_with_cache(tokens, names_filter=lambda n: "hook_z" in n)
        logit_diffs[i] = logit_diff_from_logits(logits, correct_ids[i], incorrect_ids[i])

        for L, H in head_list:
            z = cache[f"blocks.{L}.attn.hook_z"]
            activations[(L, H)][i] = z[0, -1, H].norm().item()

    return activations, logit_diffs


def run_pid(model, tasks: list[str], n_prompts: int = 60,
            max_pairs: int = 20) -> list[EvalResult]:
    tokenizer = model.tokenizer
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

        log(f"  {task} ({len(circuit_heads)} heads, {len(prompts)} prompts)...")

        activations, logit_diffs = collect_activations_for_pid(
            model, prompts, correct_ids, incorrect_ids, circuit_heads)

        z_binned = quantile_bin(logit_diffs)
        head_list = sorted(circuit_heads)
        pairs = list(itertools.combinations(head_list, 2))[:max_pairs]

        pair_results = []
        total_redundancy = 0.0
        total_synergy = 0.0

        for (L1, H1), (L2, H2) in pairs:
            x_binned = quantile_bin(activations[(L1, H1)])
            y_binned = quantile_bin(activations[(L2, H2)])

            if HAS_DIT:
                pid = compute_pid_dit(x_binned, y_binned, z_binned)
            else:
                pid = compute_pid_manual(x_binned, y_binned, z_binned)

            pair_results.append({
                "head_a": f"L{L1}H{H1}",
                "head_b": f"L{L2}H{H2}",
                **pid,
            })
            total_redundancy += pid["redundancy"]
            total_synergy += pid["synergy"]

        n_pairs = max(len(pairs), 1)
        mean_redundancy = total_redundancy / n_pairs
        mean_synergy = total_synergy / n_pairs

        log(f"    {len(pairs)} pairs: mean_redundancy={mean_redundancy:.4f} "
            f"mean_synergy={mean_synergy:.4f}")

        results.append(EvalResult(
            metric_id="C8.pid",
            value=mean_synergy,
            n_samples=len(prompts),
            metadata={
                "task": task,
                "mean_redundancy": mean_redundancy,
                "mean_synergy": mean_synergy,
                "pair_decompositions": pair_results,
                "n_pairs": len(pairs),
                "n_circuit_heads": len(circuit_heads),
                "used_dit": HAS_DIT,
            },
        ))

    return results


def main():
    parser = parse_common_args("C8: Partial Information Decomposition")
    parser.add_argument("--max-pairs", type=int, default=20)
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("C8: PARTIAL INFORMATION DECOMPOSITION (PID)")
    log("=" * 60)

    if not HAS_DIT:
        log("NOTE: dit not installed. Using approximate binned PID.")

    results = run_pid(model, tasks, args.n_prompts, args.max_pairs)

    out = args.out or "08_pid.json"
    save_results(results, out)

    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
