"""Protocol M9 --- Observability
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Lens:         Control Theory
Validity Type: Measurement
Framework:    Representational (output-decodability of internal state)
Family:       Representational (Observability)
Validity:     Construct --- C5 Observability; Internal --- I4 Decoding

References:
    Kalman (1960) "A New Approach to Linear Filtering and Prediction
        Problems" --- observability in control theory
    Alain & Bengio (2017) "Understanding Intermediate Layers Using
        Linear Classifier Probes"
    Belinkov (2022) "Probing Classifiers: Promises, Shortcomings, and
        Advances"

Question:
    Is the circuit's internal state decodable from model outputs? If so,
    the circuit is "observable" in the control-theoretic sense: its
    hidden state can be reconstructed from the output trajectory. We
    test this by training linear probes from final logits to circuit head
    activations. High R^2 means the output contains enough information
    to reconstruct what the circuit computed internally.

Metrics:
    mean_probe_r2            --- mean R^2 of linear probes from final logits
                                to individual circuit head activations
    observability_rank_ratio --- rank of the joint [logits, activations]
                                matrix divided by the number of circuit heads
                                (captures whether the mapping is full-rank)

Calibrations:
    STRUCTURAL_CALIBRATIONS

Usage:
    uv run python e09_observability.py                       # all tasks, CPU
    uv run python e09_observability.py --device cuda          # GPU
    uv run python e09_observability.py --tasks ioi induction  # specific tasks

    # As a callable module:
    from protocols.representational.e09_observability import run_protocol
    result = run_protocol(model, tasks=["ioi"], n_prompts=40)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    generate_prompts,
    get_circuit_heads,
    get_token_ids,
    load_model,
    log,
)

from protocols import ProtocolResult
from protocols.calibration_runner import STRUCTURAL_CALIBRATIONS, run_calibrations, summarize_calibrations

PROTOCOL_ID = "M9"
PROTOCOL_NAME = "Observability"
METRICS = ["mean_probe_r2", "observability_rank_ratio"]
CALIBRATIONS = STRUCTURAL_CALIBRATIONS
OUTPUT_DIR = Path(__file__).parent / "results" / "e09_observability"

THRESHOLDS = {
    "mean_probe_r2": 0.3,
    "observability_rank_ratio": 0.8,
}


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_observability(model, tasks: list[str], n_prompts: int = 40,
                      device: str = "cpu") -> list[EvalResult]:
    """Measure observability of circuit internal state from model outputs.

    For each task:
    1. Collect circuit head activations (hook_z, mean over d_head) and
       final logits for each prompt.
    2. For each circuit head, train a linear regression from final logits
       to the head's mean activation. R^2 measures decodability.
    3. Construct the joint matrix [logits | head_activations] across
       prompts and compute its rank. The rank ratio = rank / n_circuit_heads
       indicates whether the logit-activation mapping is full-rank.
    """
    results = []
    n_layers = model.cfg.n_layers
    d_vocab = model.cfg.d_vocab

    for task in tasks:
        try:
            heads = get_circuit_heads(task)
            if not heads:
                log(f"  [M9] {task}: no circuit heads, skipping")
                continue

            prompts = generate_prompts(task, model.tokenizer, n_prompts=n_prompts)
            if len(prompts) < 5:
                log(f"  [M9] {task}: too few prompts, skipping")
                continue

            n_valid = min(len(prompts), n_prompts)
            sorted_heads = sorted(heads)
            n_circuit = len(sorted_heads)

            hook_names = [f"blocks.{L}.attn.hook_z" for L in range(n_layers)]

            # Collect data: logits and head activations per prompt
            all_logits = np.zeros((n_valid, d_vocab))
            all_head_acts = np.zeros((n_valid, n_circuit))

            for i in range(n_valid):
                tokens = model.to_tokens(prompts[i].text)
                logits, cache = model.run_with_cache(tokens, names_filter=hook_names)

                # Final token logits
                final_logits = logits[0, -1, :].cpu().numpy()
                all_logits[i, :] = final_logits

                # Head activations (mean over d_head dimension)
                for j, (L, H) in enumerate(sorted_heads):
                    z = cache[f"blocks.{L}.attn.hook_z"][0, -1, H, :]  # (d_head,)
                    all_head_acts[i, j] = z.mean().item()

            # For linear probes, use top-k logits to avoid overfitting on d_vocab
            # Use top-100 most variable logit positions as features
            logit_var = np.var(all_logits, axis=0)
            top_logit_indices = np.argsort(logit_var)[-min(100, d_vocab):]
            X_logits = all_logits[:, top_logit_indices]

            # Add bias column
            X = np.column_stack([X_logits, np.ones(n_valid)])

            # Per-head linear probe: logits -> head activation
            per_head_r2 = []
            for j in range(n_circuit):
                y = all_head_acts[:, j]
                y_var = np.var(y)
                if y_var < 1e-12:
                    per_head_r2.append(0.0)
                    continue

                try:
                    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                    y_pred = X @ beta
                    ss_res = np.sum((y - y_pred) ** 2)
                    ss_tot = np.sum((y - np.mean(y)) ** 2)
                    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
                    r2 = max(0.0, r2)
                except np.linalg.LinAlgError:
                    r2 = 0.0
                per_head_r2.append(r2)

            mean_r2 = float(np.mean(per_head_r2)) if per_head_r2 else 0.0

            # Observability Gramian rank: joint [logits, activations] matrix
            joint_matrix = np.column_stack([X_logits, all_head_acts])  # (n_valid, n_logits + n_circuit)
            # Numerical rank with tolerance
            singular_values = np.linalg.svd(joint_matrix, compute_uv=False)
            tol = max(joint_matrix.shape) * singular_values[0] * np.finfo(float).eps
            numerical_rank = int(np.sum(singular_values > tol))
            rank_ratio = numerical_rank / n_circuit if n_circuit > 0 else 0.0

            per_head_detail = {
                f"L{L}H{H}": r2_val
                for (L, H), r2_val in zip(sorted_heads, per_head_r2)
            }

            results.append(EvalResult(
                metric_id="M9.mean_probe_r2",
                value=mean_r2,
                n_samples=n_valid,
                metadata={
                    "task": task,
                    "n_circuit_heads": n_circuit,
                    "per_head_r2": per_head_detail,
                    "passed": mean_r2 > THRESHOLDS["mean_probe_r2"],
                },
            ))
            results.append(EvalResult(
                metric_id="M9.observability_rank_ratio",
                value=rank_ratio,
                n_samples=n_valid,
                metadata={
                    "task": task,
                    "numerical_rank": numerical_rank,
                    "n_circuit_heads": n_circuit,
                    "passed": rank_ratio > THRESHOLDS["observability_rank_ratio"],
                },
            ))

            log(f"  [M9] {task}: mean_probe_r2={mean_r2:.4f}, "
                f"rank_ratio={rank_ratio:.4f} (rank={numerical_rank}/{n_circuit})")

        except Exception as e:
            log(f"  [M9] {task} FAILED: {e}")

    return results


# ---------------------------------------------------------------------------
# Protocol runner
# ---------------------------------------------------------------------------

def run_protocol(model, tasks: list[str] | None = None, n_prompts: int = 40,
                 device: str = "cpu", run_cals: bool = True,
                 protocol_results: dict | None = None) -> ProtocolResult:
    """Run M9 observability analysis. Returns a ProtocolResult."""
    tasks = tasks or CIRCUIT_TASKS
    t0 = time.time()
    result = ProtocolResult(
        protocol_id=PROTOCOL_ID,
        protocol_name=PROTOCOL_NAME,
        tasks=tasks,
    )

    print(f"\n{'─' * 60}")
    print(f"  Observability — {len(tasks)} tasks, {n_prompts} prompts")
    print(f"{'─' * 60}")

    mt0 = time.time()
    try:
        all_results = run_observability(model, tasks, n_prompts=n_prompts,
                                        device=device)
        result.metrics["mean_probe_r2"] = [
            r for r in all_results if r.metric_id == "M9.mean_probe_r2"
        ]
        result.metrics["observability_rank_ratio"] = [
            r for r in all_results if r.metric_id == "M9.observability_rank_ratio"
        ]
        for r in all_results:
            task = r.metadata.get("task", "?")
            passed = r.metadata.get("passed", None)
            tag = " PASS" if passed else (" FAIL" if passed is not None else "")
            print(f"    {task:20s}  {r.metric_id:35s}  {r.value:+.4f}{tag}")
        print(f"  {len(all_results)} results in {time.time() - mt0:.1f}s")
    except Exception as e:
        print(f"  [observability] FAILED: {e}")
        result.metrics["mean_probe_r2"] = []
        result.metrics["observability_rank_ratio"] = []

    if run_cals:
        print(f"\n{'=' * 60}")
        print(f"  Calibrations ({len(CALIBRATIONS)})")
        print(f"{'=' * 60}")
        cal_tasks = tasks[:2]
        result.calibrations = run_calibrations(
            model, cal_tasks, CALIBRATIONS, n_prompts=n_prompts)

    result.elapsed_seconds = time.time() - t0
    return result


# ---------------------------------------------------------------------------
# Analysis and display
# ---------------------------------------------------------------------------

def observability_analysis(result: ProtocolResult) -> list[str]:
    """Analyze results through the observability lens."""
    lines = ["\n  Observability Analysis:", "  ──────────────────────"]

    for task in result.tasks:
        lines.append(f"\n    {task}:")

        probe = _find(result.metrics.get("mean_probe_r2", []), task)
        rank = _find(result.metrics.get("observability_rank_ratio", []), task)

        if probe:
            r2 = probe.value
            if r2 > 0.7:
                label = "highly observable (internal state readable from output)"
            elif r2 > 0.3:
                label = "moderately observable"
            else:
                label = "weakly observable (internal state is hidden)"
            lines.append(f"      Mean probe R^2:      {r2:.4f} — {label}")

            # Report top and bottom heads
            per_head = probe.metadata.get("per_head_r2", {})
            if per_head:
                sorted_heads = sorted(per_head.items(), key=lambda x: x[1], reverse=True)
                top = sorted_heads[:3]
                bottom = sorted_heads[-3:]
                lines.append(f"        Most observable:  {', '.join(f'{h}={v:.3f}' for h, v in top)}")
                lines.append(f"        Least observable: {', '.join(f'{h}={v:.3f}' for h, v in bottom)}")

        if rank:
            rr = rank.value
            if rr > 1.5:
                label = "over-determined (redundant output information)"
            elif rr > 0.8:
                label = "full-rank (all heads independently decodable)"
            else:
                label = "rank-deficient (some heads share output signatures)"
            lines.append(f"      Rank ratio:          {rr:.4f} — {label}")

        if probe and rank:
            observable = probe.value > 0.3
            full_rank = rank.value > 0.8

            if observable and full_rank:
                verdict = "FULLY OBSERVABLE — internal state recoverable from output"
            elif observable and not full_rank:
                verdict = "PARTIALLY OBSERVABLE — some heads share output channels"
            elif not observable and full_rank:
                verdict = "STRUCTURALLY OBSERVABLE but WEAK — rank OK but low R^2"
            else:
                verdict = "UNOBSERVABLE — internal state hidden from output"
            lines.append(f"      VERDICT: {verdict}")

    return lines


def _find(results: list[EvalResult], task: str) -> EvalResult | None:
    return next((r for r in results if r.metadata.get("task") == task), None)


def summarize(result: ProtocolResult) -> str:
    lines = []
    lines.append(f"\n{'=' * 70}")
    lines.append(f"  PROTOCOL {result.protocol_id}: {result.protocol_name}")
    lines.append(f"{'=' * 70}\n")

    header = f"{'Task':20s}" + "".join(f"  {m:>28s}" for m in METRICS)
    lines.append(header)
    lines.append("-" * len(header))

    for task in result.tasks:
        row = f"{task:20s}"
        for m in METRICS:
            match = _find(result.metrics.get(m, []), task)
            if match:
                v = match.value
                p = match.metadata.get("passed", None)
                tag = " PASS" if p else (" FAIL" if p is not None else " ---")
                row += f"  {v:>24.4f}{tag}"
            else:
                row += f"  {'---':>28s}"
        lines.append(row)

    lines.append("")

    for m in METRICS:
        rs = result.metrics.get(m, [])
        if not rs:
            continue
        vals = [r.value for r in rs]
        n_pass = sum(1 for r in rs if r.metadata.get("passed", False))
        lines.append(f"  {m}: mean={np.mean(vals):.4f}  std={np.std(vals):.4f}  "
                     f"range=[{min(vals):.4f}, {max(vals):.4f}]  "
                     f"passed={n_pass}/{len(rs)}")

    lines.extend(observability_analysis(result))

    if result.calibrations:
        lines.append("")
        lines.append(summarize_calibrations(result.calibrations))

    lines.append(f"\n  Elapsed: {result.elapsed_seconds:.1f}s")

    text = "\n".join(lines)
    print(text)
    return text


def save_protocol_results(result: ProtocolResult, output_dir: Path | None = None):
    output_dir = output_dir or OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, rs in result.metrics.items():
        if not rs:
            continue
        with open(output_dir / f"{name}.jsonl", "w") as f:
            for r in rs:
                f.write(json.dumps(r.to_dict(), default=str) + "\n")

    for name, rs in result.calibrations.items():
        if not rs:
            continue
        with open(output_dir / f"cal_{name}.jsonl", "w") as f:
            for r in rs:
                f.write(json.dumps(r.to_dict(), default=str) + "\n")

    summary = {
        "protocol": result.protocol_id,
        "name": result.protocol_name,
        "tasks": result.tasks,
        "elapsed_seconds": result.elapsed_seconds,
        "metrics": {
            name: {
                "n_tasks": len(rs),
                "mean": float(np.mean([r.value for r in rs])),
                "n_passed": sum(1 for r in rs if r.metadata.get("passed", False)),
                "per_task": {r.metadata.get("task", "?"): r.value for r in rs},
            }
            for name, rs in result.metrics.items() if rs
        },
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=f"Protocol {PROTOCOL_ID}: {PROTOCOL_NAME}")
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--n-prompts", type=int, default=40)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--no-calibrations", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR

    print(f"{'=' * 70}")
    print(f"  Protocol {PROTOCOL_ID}: {PROTOCOL_NAME}")
    print(f"  Model: {args.model}  Device: {args.device}  Prompts: {args.n_prompts}")
    print(f"  Tasks: {', '.join(tasks)}")
    print(f"{'=' * 70}")

    model = load_model(args.model, args.device)
    for task in tasks:
        print(f"  {task}: {len(get_circuit_heads(task))} circuit heads")

    result = run_protocol(model, tasks, n_prompts=args.n_prompts,
                          device=args.device,
                          run_cals=not args.no_calibrations)
    summarize(result)

    if not args.no_save:
        save_protocol_results(result, output_dir)

    n = sum(len(r) for r in result.metrics.values())
    nc = sum(len(r) for r in result.calibrations.values())
    print(f"\nTotal: {n} metric + {nc} calibration results in {result.elapsed_seconds:.1f}s")


if __name__ == "__main__":
    main()
