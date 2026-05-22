"""Protocol E8 --- Stability Margin
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Lens:         Control Theory
Validity Type: External
Framework:    Behavioral (robustness under scaled perturbation)
Family:       Behavioral (Gain Margin / Stability)
Validity:     External --- E2 Robustness; Internal --- I5 Stability

References:
    Ogata (2010) "Modern Control Engineering" --- gain margin in feedback systems
    Cohen & Saphra (2024) "Evaluating the Faithfulness of Circuit Hypotheses"
    Wang et al. (2022) "Interpretability in the Wild" --- IOI ablation studies

Question:
    How robust is the circuit's behavioral contribution to scaled
    perturbation? The gain margin is the largest multiplier on the
    ablation magnitude at which the model still retains >50% of clean
    performance. A large gain margin means the circuit's contribution
    is robust (tolerates aggressive perturbation). A small gain margin
    means the circuit is operating near instability.

    We sweep perturbation magnitudes from 0.5x to 5x and measure the
    fraction of clean logit diff retained at each level.

Metrics:
    gain_margin       --- largest magnitude where performance > 50% of clean
    performance_at_2x --- fraction of clean logit diff retained at 2x
                         perturbation strength

Calibrations:
    BEHAVIORAL_CALIBRATIONS

Usage:
    uv run python d08_stability_margin.py                       # all tasks, CPU
    uv run python d08_stability_margin.py --device cuda          # GPU
    uv run python d08_stability_margin.py --tasks ioi induction  # specific tasks

    # As a callable module:
    from protocols.behavioral.d08_stability_margin import run_protocol
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
from protocols.calibration_runner import BEHAVIORAL_CALIBRATIONS, run_calibrations, summarize_calibrations

PROTOCOL_ID = "E8"
PROTOCOL_NAME = "Stability Margin"
METRICS = ["gain_margin", "performance_at_2x"]
CALIBRATIONS = BEHAVIORAL_CALIBRATIONS
OUTPUT_DIR = Path(__file__).parent / "results" / "d08_stability_margin"

THRESHOLDS = {
    "gain_margin": 2.0,
    "performance_at_2x": 0.5,
}

MAGNITUDES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
PERFORMANCE_THRESHOLD = 0.5  # fraction of clean performance for gain margin


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_stability_margin(model, tasks: list[str], n_prompts: int = 40,
                         device: str = "cpu") -> list[EvalResult]:
    """Sweep perturbation magnitudes to find the gain margin.

    For each task:
    1. Compute clean logit diffs as baseline.
    2. For each magnitude in MAGNITUDES:
       a. For each prompt, subtract magnitude * head_contribution from
          the residual stream at each circuit head.
       b. Compute logit diff under perturbation.
       c. Measure fraction of clean performance retained.
    3. Gain margin = largest magnitude where mean performance > 50% of clean.
    4. Also report performance at 2x specifically.
    """
    results = []
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    for task in tasks:
        try:
            heads = get_circuit_heads(task)
            if not heads:
                log(f"  [E8] {task}: no circuit heads, skipping")
                continue

            prompts = generate_prompts(task, model.tokenizer, n_prompts=n_prompts)
            correct_ids, incorrect_ids = get_token_ids(prompts, model.tokenizer)
            if len(correct_ids) < 3:
                log(f"  [E8] {task}: too few valid prompts ({len(correct_ids)}), skipping")
                continue

            n_valid = min(len(prompts), len(correct_ids))

            # Group heads by layer
            heads_by_layer: dict[int, list[int]] = {}
            for L, H in heads:
                heads_by_layer.setdefault(L, []).append(H)

            # First, collect clean head contributions at hook_z for each prompt
            hook_z_names = [f"blocks.{L}.attn.hook_z" for L in range(n_layers)]

            # Compute clean logit diffs
            clean_diffs = np.zeros(n_valid)
            for i in range(n_valid):
                tokens = model.to_tokens(prompts[i].text)
                logits = model(tokens)
                clean_diffs[i] = logit_diff_from_logits(logits, correct_ids[i], incorrect_ids[i])

            mean_clean = float(np.mean(clean_diffs))
            if abs(mean_clean) < 1e-8:
                log(f"  [E8] {task}: zero clean logit diff, skipping")
                continue

            # Sweep magnitudes
            perf_curve = {}

            for magnitude in MAGNITUDES:
                perturbed_diffs = np.zeros(n_valid)

                for i in range(n_valid):
                    tokens = model.to_tokens(prompts[i].text)

                    # Build perturbation hooks: subtract magnitude * head_contribution
                    def _make_hook(target_layer, target_heads, mag):
                        def hook_fn(z, hook):
                            for H in target_heads:
                                z[0, :, H, :] -= mag * z[0, :, H, :].clone()
                            return z
                        return (f"blocks.{target_layer}.attn.hook_z", hook_fn)

                    fwd_hooks = [
                        _make_hook(L, H_list, magnitude)
                        for L, H_list in heads_by_layer.items()
                    ]

                    logits = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)
                    perturbed_diffs[i] = logit_diff_from_logits(
                        logits, correct_ids[i], incorrect_ids[i]
                    )

                mean_perturbed = float(np.mean(perturbed_diffs))
                frac_retained = mean_perturbed / mean_clean if abs(mean_clean) > 1e-8 else 0.0
                perf_curve[magnitude] = frac_retained

            # Find gain margin: largest magnitude where performance > threshold
            gain_margin = 0.0
            for mag in MAGNITUDES:
                if perf_curve[mag] > PERFORMANCE_THRESHOLD:
                    gain_margin = mag

            # Performance at 2x
            perf_at_2x = perf_curve.get(2.0, 0.0)

            results.append(EvalResult(
                metric_id="E8.gain_margin",
                value=gain_margin,
                n_samples=n_valid,
                metadata={
                    "task": task,
                    "performance_curve": {str(k): v for k, v in perf_curve.items()},
                    "magnitudes_tested": MAGNITUDES,
                    "performance_threshold": PERFORMANCE_THRESHOLD,
                    "passed": gain_margin >= THRESHOLDS["gain_margin"],
                },
            ))
            results.append(EvalResult(
                metric_id="E8.performance_at_2x",
                value=perf_at_2x,
                n_samples=n_valid,
                metadata={
                    "task": task,
                    "clean_logit_diff": mean_clean,
                    "passed": perf_at_2x > THRESHOLDS["performance_at_2x"],
                },
            ))

            log(f"  [E8] {task}: gain_margin={gain_margin:.1f}x, "
                f"perf@2x={perf_at_2x:.4f}")

        except Exception as e:
            log(f"  [E8] {task} FAILED: {e}")

    return results


# ---------------------------------------------------------------------------
# Protocol runner
# ---------------------------------------------------------------------------

def run_protocol(model, tasks: list[str] | None = None, n_prompts: int = 40,
                 device: str = "cpu", run_cals: bool = True,
                 protocol_results: dict | None = None) -> ProtocolResult:
    """Run E8 stability margin analysis. Returns a ProtocolResult."""
    tasks = tasks or CIRCUIT_TASKS
    t0 = time.time()
    result = ProtocolResult(
        protocol_id=PROTOCOL_ID,
        protocol_name=PROTOCOL_NAME,
        tasks=tasks,
    )

    print(f"\n{'─' * 60}")
    print(f"  Stability Margin — {len(tasks)} tasks, {n_prompts} prompts")
    print(f"{'─' * 60}")

    mt0 = time.time()
    try:
        all_results = run_stability_margin(model, tasks, n_prompts=n_prompts,
                                           device=device)
        result.metrics["gain_margin"] = [
            r for r in all_results if r.metric_id == "E8.gain_margin"
        ]
        result.metrics["performance_at_2x"] = [
            r for r in all_results if r.metric_id == "E8.performance_at_2x"
        ]
        for r in all_results:
            task = r.metadata.get("task", "?")
            passed = r.metadata.get("passed", None)
            tag = " PASS" if passed else (" FAIL" if passed is not None else "")
            print(f"    {task:20s}  {r.metric_id:25s}  {r.value:+.4f}{tag}")
        print(f"  {len(all_results)} results in {time.time() - mt0:.1f}s")
    except Exception as e:
        print(f"  [stability_margin] FAILED: {e}")
        result.metrics["gain_margin"] = []
        result.metrics["performance_at_2x"] = []

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

def stability_analysis(result: ProtocolResult) -> list[str]:
    """Analyze results through the stability margin lens."""
    lines = ["\n  Stability Margin Analysis:", "  ─────────────────────────"]

    for task in result.tasks:
        lines.append(f"\n    {task}:")

        gm = _find(result.metrics.get("gain_margin", []), task)
        p2x = _find(result.metrics.get("performance_at_2x", []), task)

        if gm:
            margin = gm.value
            if margin >= 4.0:
                label = "very robust (tolerates 4x+ perturbation)"
            elif margin >= 2.0:
                label = "robust (tolerates 2x perturbation)"
            elif margin >= 1.0:
                label = "marginally stable (at ablation boundary)"
            else:
                label = "unstable (fails even at partial ablation)"
            lines.append(f"      Gain margin: {margin:.1f}x — {label}")

            # Show performance curve if available
            curve = gm.metadata.get("performance_curve", {})
            if curve:
                curve_str = ", ".join(f"{k}x:{v:.2f}" for k, v in sorted(
                    ((float(k), v) for k, v in curve.items())))
                lines.append(f"        Curve: {curve_str}")

        if p2x:
            perf = p2x.value
            if perf > 0.8:
                label = "minimal degradation at 2x"
            elif perf > 0.5:
                label = "moderate degradation at 2x"
            else:
                label = "severe degradation at 2x"
            lines.append(f"      Performance at 2x: {perf:.4f} — {label}")

        if gm and p2x:
            robust = gm.value >= 2.0
            retains = p2x.value > 0.5

            if robust and retains:
                verdict = "STABLE — circuit contribution is robust to scaled perturbation"
            elif robust and not retains:
                verdict = "CLIFF EDGE — stable at 1x but drops sharply at 2x"
            elif not robust and retains:
                verdict = "PARADOXICAL — low gain margin but retains performance"
            else:
                verdict = "FRAGILE — circuit operates near instability boundary"
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

    lines.extend(stability_analysis(result))

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
