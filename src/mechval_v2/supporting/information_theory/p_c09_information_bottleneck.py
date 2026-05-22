"""Protocol C9 --- Information Bottleneck
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Lens:         Information Theory
Validity Type: Construct
Framework:    Information-theoretic (bottleneck analysis)
Family:       Information (sufficiency and compression)
Validity:     Internal --- I2 Sufficiency; Construct --- C3 Information

References:
    Tishby, Pereira & Bialek (2000) "The Information Bottleneck Method"
    Shwartz-Ziv & Tishby (2017) "Opening the Black Box of Deep Neural
        Networks via Information"
    Geiger et al. (2021) "Causal Abstractions of Neural Networks"

Question:
    How much of the model's input-output mutual information is captured
    by the circuit's internal activations? If the circuit is a faithful
    bottleneck, then I(circuit_acts; output) should be close to
    I(input; output). We approximate the sufficiency ratio via R^2 of a
    linear model from circuit activations to output logits (a practical
    lower bound). We also compute the compression ratio:
    dim(circuit activations) / dim(residual stream).

Metrics:
    sufficiency_ratio  --- R^2 of linear prediction from circuit
                          head activations to output logit diffs
    compression_ratio  --- fraction of total residual stream
                          dimensions used by the circuit

Calibrations:
    STRUCTURAL_CALIBRATIONS

Usage:
    uv run python c09_information_bottleneck.py                       # all tasks, CPU
    uv run python c09_information_bottleneck.py --device cuda          # GPU
    uv run python c09_information_bottleneck.py --tasks ioi induction  # specific tasks

    # As a callable module:
    from protocols.information.c09_information_bottleneck import run_protocol
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
    logit_diff_from_logits,
    save_results,
)

from protocols import ProtocolResult
from protocols.calibration_runner import STRUCTURAL_CALIBRATIONS, run_calibrations, summarize_calibrations

PROTOCOL_ID = "C9"
PROTOCOL_NAME = "Information Bottleneck"
METRICS = ["sufficiency_ratio", "compression_ratio"]
CALIBRATIONS = STRUCTURAL_CALIBRATIONS
OUTPUT_DIR = Path(__file__).parent / "results" / "c09_information_bottleneck"

THRESHOLDS = {
    "sufficiency_ratio": 0.5,
    "compression_ratio": 0.5,
}


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_information_bottleneck(model, tasks: list[str], n_prompts: int = 40,
                               device: str = "cpu") -> list[EvalResult]:
    """Measure information bottleneck properties of the circuit.

    For each task:
    1. Collect circuit head activations (mean over d_head) and output logit diffs.
    2. Fit a linear model from activations to logit diffs; R^2 is the
       sufficiency ratio (how much output variance the circuit explains).
    3. Compute compression ratio: n_circuit_heads / (n_layers * n_heads).
    """
    results = []
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    d_head = model.cfg.d_head

    for task in tasks:
        try:
            heads = get_circuit_heads(task)
            if not heads:
                log(f"  [C9] {task}: no circuit heads, skipping")
                continue

            prompts = generate_prompts(task, model.tokenizer, n_prompts=n_prompts)
            correct_ids, incorrect_ids = get_token_ids(prompts, model.tokenizer)
            if len(correct_ids) < 5:
                log(f"  [C9] {task}: too few valid prompts ({len(correct_ids)}), skipping")
                continue

            n_valid = min(len(prompts), len(correct_ids))
            sorted_heads = sorted(heads)
            n_circuit = len(sorted_heads)

            # Collect activations and logit diffs
            hook_names = [f"blocks.{L}.attn.hook_z" for L in range(n_layers)]
            activations = np.zeros((n_valid, n_circuit))
            logit_diffs = np.zeros(n_valid)

            for i in range(n_valid):
                tokens = model.to_tokens(prompts[i].text)
                logits, cache = model.run_with_cache(tokens, names_filter=hook_names)

                logit_diffs[i] = logit_diff_from_logits(logits, correct_ids[i], incorrect_ids[i])

                for j, (L, H) in enumerate(sorted_heads):
                    z = cache[f"blocks.{L}.attn.hook_z"][0, -1, H, :]  # (d_head,)
                    activations[i, j] = z.mean().item()

            # Fit linear model: activations -> logit_diffs, compute R^2
            # Add bias column
            X = np.column_stack([activations, np.ones(n_valid)])
            y = logit_diffs

            # Least-squares solve
            try:
                beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                y_pred = X @ beta
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
                r_squared = max(0.0, r_squared)  # Clamp negative R^2
            except np.linalg.LinAlgError:
                r_squared = 0.0

            # Compression ratio
            total_heads = n_layers * n_heads
            compression = n_circuit / total_heads

            results.append(EvalResult(
                metric_id="C9.sufficiency_ratio",
                value=r_squared,
                n_samples=n_valid,
                metadata={
                    "task": task,
                    "n_circuit_heads": n_circuit,
                    "n_prompts_used": n_valid,
                    "passed": r_squared > THRESHOLDS["sufficiency_ratio"],
                },
            ))
            results.append(EvalResult(
                metric_id="C9.compression_ratio",
                value=compression,
                n_samples=n_valid,
                metadata={
                    "task": task,
                    "n_circuit_heads": n_circuit,
                    "total_heads": total_heads,
                    "passed": compression < THRESHOLDS["compression_ratio"],
                },
            ))

            log(f"  [C9] {task}: R^2={r_squared:.4f}, compression={compression:.4f} "
                f"({n_circuit}/{total_heads} heads)")

        except Exception as e:
            log(f"  [C9] {task} FAILED: {e}")

    return results


# ---------------------------------------------------------------------------
# Protocol runner
# ---------------------------------------------------------------------------

def run_protocol(model, tasks: list[str] | None = None, n_prompts: int = 40,
                 device: str = "cpu", run_cals: bool = True,
                 protocol_results: dict | None = None) -> ProtocolResult:
    """Run C9 information bottleneck analysis. Returns a ProtocolResult."""
    tasks = tasks or CIRCUIT_TASKS
    t0 = time.time()
    result = ProtocolResult(
        protocol_id=PROTOCOL_ID,
        protocol_name=PROTOCOL_NAME,
        tasks=tasks,
    )

    print(f"\n{'─' * 60}")
    print(f"  Information Bottleneck — {len(tasks)} tasks, {n_prompts} prompts")
    print(f"{'─' * 60}")

    mt0 = time.time()
    try:
        all_results = run_information_bottleneck(model, tasks, n_prompts=n_prompts,
                                                 device=device)
        result.metrics["sufficiency_ratio"] = [
            r for r in all_results if r.metric_id == "C9.sufficiency_ratio"
        ]
        result.metrics["compression_ratio"] = [
            r for r in all_results if r.metric_id == "C9.compression_ratio"
        ]
        for r in all_results:
            task = r.metadata.get("task", "?")
            passed = r.metadata.get("passed", None)
            tag = " PASS" if passed else (" FAIL" if passed is not None else "")
            print(f"    {task:20s}  {r.metric_id:30s}  {r.value:+.4f}{tag}")
        print(f"  {len(all_results)} results in {time.time() - mt0:.1f}s")
    except Exception as e:
        print(f"  [information_bottleneck] FAILED: {e}")
        result.metrics["sufficiency_ratio"] = []
        result.metrics["compression_ratio"] = []

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

def bottleneck_analysis(result: ProtocolResult) -> list[str]:
    """Analyze results through the information bottleneck lens."""
    lines = ["\n  Information Bottleneck Analysis:", "  ────────────────────────────────"]

    for task in result.tasks:
        lines.append(f"\n    {task}:")

        suff = _find(result.metrics.get("sufficiency_ratio", []), task)
        comp = _find(result.metrics.get("compression_ratio", []), task)

        if suff:
            if suff.value > 0.8:
                label = "high sufficiency (circuit captures most output variance)"
            elif suff.value > 0.5:
                label = "moderate sufficiency"
            else:
                label = "low sufficiency (circuit misses output information)"
            lines.append(f"      Sufficiency (R^2): {suff.value:.4f} — {label}")

        if comp:
            if comp.value < 0.1:
                label = "strong compression (small circuit)"
            elif comp.value < 0.3:
                label = "moderate compression"
            else:
                label = "weak compression (large circuit)"
            lines.append(f"      Compression ratio: {comp.value:.4f} — {label}")

        if suff and comp:
            if suff.value > 0.7 and comp.value < 0.2:
                verdict = "EFFICIENT BOTTLENECK — high sufficiency with strong compression"
            elif suff.value > 0.7:
                verdict = "SUFFICIENT but DIFFUSE — high sufficiency, weak compression"
            elif comp.value < 0.2:
                verdict = "COMPRESSED but INSUFFICIENT — small circuit misses information"
            else:
                verdict = "POOR BOTTLENECK — neither sufficient nor compressed"
            lines.append(f"      VERDICT: {verdict}")

    return lines


def _find(results: list[EvalResult], task: str) -> EvalResult | None:
    return next((r for r in results if r.metadata.get("task") == task), None)


def summarize(result: ProtocolResult) -> str:
    lines = []
    lines.append(f"\n{'=' * 70}")
    lines.append(f"  PROTOCOL {result.protocol_id}: {result.protocol_name}")
    lines.append(f"{'=' * 70}\n")

    header = f"{'Task':20s}" + "".join(f"  {m:>20s}" for m in METRICS)
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
                row += f"  {v:>16.4f}{tag}"
            else:
                row += f"  {'---':>20s}"
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

    lines.extend(bottleneck_analysis(result))

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
