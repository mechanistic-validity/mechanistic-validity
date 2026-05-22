"""Protocol MB_QE — Quasi-Experimental Designs
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Lens:         Genetics
Validity Type: Internal
Framework:    Molecular Biology — Quasi-Experimental Causal Inference
Family:       Molecular Biology (Quasi-Experimental Designs)
Validity:     Internal — causal identification via natural discontinuities

References:
    Thistlethwaite & Campbell (1960) "Regression-discontinuity analysis:
        an alternative to the ex post facto experiment"
    Wagner et al. (2002) "Segmented regression analysis of interrupted
        time series studies in medication use research"

Question:
    Can we identify causal structure through natural discontinuities in
    circuit behavior? RDD (Regression Discontinuity Design) tests whether
    IIA scores show a sharp jump at a layer boundary, indicating the
    capability "emerges" at that layer. ITS (Interrupted Time Series) tests
    whether sigma_ablation reveals a phase transition at a particular noise
    level, distinguishing fragile thresholds from graceful degradation.

    J1 — RDD: If activation_patching effects show a discontinuous increase
    at layer L, the RDD estimate tau_RDD = mean(effects at L+) -
    mean(effects at L-) gives the causal contribution of crossing that
    layer boundary.

    J2 — ITS: Sigma_ablation across noise levels forms a "time series."
    A sudden drop indicates a fragile threshold; a gradual decline
    indicates graceful degradation.

Metrics:
    activation_patching — Causal effect per layer for RDD analysis
    das_iia             — Distributed alignment search interchange accuracy
    effect_size         — Standardized causal effect magnitude
    sigma_ablation      — Noise-level robustness for ITS analysis

Calibrations:
    STRUCTURAL_CALIBRATIONS

Usage:
    uv run python quasi_experimental.py                       # all tasks, CPU
    uv run python quasi_experimental.py --device cuda          # GPU
    uv run python quasi_experimental.py --tasks ioi induction  # specific tasks

    # As a callable module:
    from protocols.molecular_biology.quasi_experimental import run_protocol
    result = run_protocol(model, tasks=["ioi"], n_prompts=40)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from mechval.metrics.common import CIRCUIT_TASKS, EvalResult, get_circuit_heads, load_model

from protocols import ProtocolResult, import_metric_runner
from protocols.calibration_runner import STRUCTURAL_CALIBRATIONS, run_calibrations, summarize_calibrations

PROTOCOL_ID = "MB_QE"
PROTOCOL_NAME = "Quasi-Experimental Designs"
METRICS = ["activation_patching", "effect_size", "sigma_ablation", "das_iia"]
CALIBRATIONS = STRUCTURAL_CALIBRATIONS
OUTPUT_DIR = Path(__file__).parent / "results" / "mb_qe_quasi_experimental"

THRESHOLDS = {
    "activation_patching": 0.5,
    "das_iia": 0.6,
    "effect_size": 0.8,
    "sigma_ablation": 0.5,
}


def run_protocol(model, tasks: list[str] | None = None, n_prompts: int = 40,
                 device: str = "cpu", run_cals: bool = True) -> ProtocolResult:
    """Run all MB_QE metrics + calibrations. Returns a ProtocolResult."""
    tasks = tasks or CIRCUIT_TASKS
    t0 = time.time()
    result = ProtocolResult(
        protocol_id=PROTOCOL_ID,
        protocol_name=PROTOCOL_NAME,
        tasks=tasks,
    )

    for metric_name in METRICS:
        runner = import_metric_runner(metric_name)
        if runner is None:
            print(f"  [{metric_name}] not in registry, skipping")
            continue

        print(f"\n{'─' * 60}")
        print(f"  {metric_name} — {len(tasks)} tasks, {n_prompts} prompts")
        print(f"{'─' * 60}")

        mt0 = time.time()
        try:
            results = runner(model, tasks, n_prompts=n_prompts, device=device)
        except Exception as e:
            print(f"  [{metric_name}] FAILED: {e}")
            result.metrics[metric_name] = []
            continue

        result.metrics[metric_name] = results
        for r in results:
            task = r.metadata.get("task", "?")
            passed = r.metadata.get("passed", None)
            tag = " PASS" if passed else (" FAIL" if passed is not None else "")
            print(f"    {task:20s}  {r.value:+.4f}{tag}")
        print(f"  {len(results)} results in {time.time() - mt0:.1f}s")

    if run_cals:
        print(f"\n{'=' * 60}")
        print(f"  Calibrations ({len(CALIBRATIONS)})")
        print(f"{'=' * 60}")
        cal_tasks = tasks[:2]
        result.calibrations = run_calibrations(
            model, cal_tasks, CALIBRATIONS, n_prompts=n_prompts)

    result.elapsed_seconds = time.time() - t0
    return result


def quasi_experimental_analysis(result: ProtocolResult) -> list[str]:
    """Analyze results through Quasi-Experimental Design lens.

    Biology analogy: Quasi-experimental designs exploit natural
    discontinuities to identify causal effects without full randomization.

    J1 — RDD (Regression Discontinuity Design): Groups activation_patching
    results by layer and checks for discontinuous jumps at layer boundaries.
    tau_RDD = mean(effects at layer L+) - mean(effects at layer L-).
    A significant jump suggests the capability "emerges" at that layer,
    analogous to a treatment threshold in epidemiology (Thistlethwaite &
    Campbell, 1960).

    J2 — ITS (Interrupted Time Series): Uses sigma_ablation across noise
    levels as a time series and checks for level/slope changes. A sudden
    drop indicates a fragile threshold (the circuit has a critical noise
    level). A gradual decline indicates graceful degradation (Wagner et al.,
    2002).
    """
    lines = ["\n  Quasi-Experimental Analysis:", "  ----------------------------"]

    for task in result.tasks:
        ap = _find(result.metrics.get("activation_patching", []), task)
        di = _find(result.metrics.get("das_iia", []), task)
        es = _find(result.metrics.get("effect_size", []), task)
        sa = _find(result.metrics.get("sigma_ablation", []), task)

        lines.append(f"\n    {task}:")

        # --- J1: RDD — layer discontinuity analysis ---
        rdd_verdict = None
        if ap:
            layer_effects = ap.metadata.get("layer_effects", {})
            if layer_effects:
                layers = sorted(layer_effects.keys(), key=lambda x: int(x))
                vals = [layer_effects[l] for l in layers]
                max_jump = 0.0
                jump_layer = None
                for i in range(len(vals) - 1):
                    jump = vals[i + 1] - vals[i]
                    if jump > max_jump:
                        max_jump = jump
                        jump_layer = layers[i + 1]
                if jump_layer is not None:
                    tau_rdd = np.mean(vals[layers.index(jump_layer):]) - np.mean(vals[:layers.index(jump_layer)])
                    lines.append(f"      RDD (layer discontinuity):             tau_RDD={tau_rdd:.4f} at layer {jump_layer}")
                    if max_jump > THRESHOLDS["activation_patching"] and tau_rdd > THRESHOLDS["activation_patching"]:
                        rdd_verdict = "SHARP EMERGENCE"
                    elif tau_rdd > 0.1:
                        rdd_verdict = "GRADUAL DEVELOPMENT"
                    else:
                        rdd_verdict = "FLAT"
                else:
                    lines.append(f"      RDD (layer discontinuity):             no layer structure detected")
                    rdd_verdict = "FLAT"
            else:
                label = "strong" if ap.value > THRESHOLDS["activation_patching"] else "weak"
                lines.append(f"      Activation patching (aggregate):       {ap.value:.4f} — {label}")

        if di:
            label = "aligned" if di.value > THRESHOLDS["das_iia"] else ("partial" if di.value > 0.3 else "misaligned")
            lines.append(f"      DAS IIA:                               {di.value:.4f} — {label}")

        if es:
            label = "large" if es.value > THRESHOLDS["effect_size"] else ("medium" if es.value > 0.5 else "small")
            lines.append(f"      Effect size:                           {es.value:.4f} — {label}")

        # --- J2: ITS — noise threshold analysis ---
        its_verdict = None
        if sa:
            noise_effects = sa.metadata.get("noise_effects", {})
            if noise_effects:
                levels = sorted(noise_effects.keys(), key=lambda x: float(x))
                vals = [noise_effects[l] for l in levels]
                max_drop = 0.0
                for i in range(len(vals) - 1):
                    drop = vals[i] - vals[i + 1]
                    if drop > max_drop:
                        max_drop = drop
                if max_drop > THRESHOLDS["sigma_ablation"]:
                    its_verdict = "THRESHOLD FRAGILITY"
                elif vals[0] - vals[-1] > 0.1:
                    its_verdict = "GRACEFUL DEGRADATION"
                else:
                    its_verdict = "FLAT"
                lines.append(f"      ITS (noise threshold):                 max_drop={max_drop:.4f}")
            else:
                label = "robust" if sa.value > THRESHOLDS["sigma_ablation"] else "fragile"
                lines.append(f"      Sigma ablation (aggregate):            {sa.value:.4f} — {label}")

        # --- Combined verdict ---
        if rdd_verdict and its_verdict:
            verdict = f"{rdd_verdict} + {its_verdict}"
        elif rdd_verdict:
            verdict = rdd_verdict
        elif its_verdict:
            verdict = its_verdict
        else:
            evidence = []
            if ap and ap.value > THRESHOLDS["activation_patching"]:
                evidence.append("patching")
            if di and di.value > THRESHOLDS["das_iia"]:
                evidence.append("iia")
            if es and es.value > THRESHOLDS["effect_size"]:
                evidence.append("effect-size")
            if sa and sa.value > THRESHOLDS["sigma_ablation"]:
                evidence.append("sigma")
            if len(evidence) >= 3:
                verdict = "GRADUAL DEVELOPMENT + GRACEFUL DEGRADATION"
            elif evidence:
                verdict = f"PARTIAL — {', '.join(evidence)} supported"
            else:
                verdict = "FLAT"

        lines.append(f"      VERDICT: {verdict}")

    return lines


def _find(results: list[EvalResult], task: str) -> EvalResult | None:
    return next((r for r in results if r.metadata.get("task") == task), None)


def summarize(result: ProtocolResult) -> str:
    lines = []
    lines.append(f"\n{'=' * 70}")
    lines.append(f"  PROTOCOL {result.protocol_id}: {result.protocol_name}")
    lines.append(f"{'=' * 70}\n")

    header = f"{'Task':20s}" + "".join(f"  {m:>24s}" for m in METRICS)
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
                row += f"  {v:>20.4f}{tag}"
            else:
                row += f"  {'---':>24s}"
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

    lines.extend(quasi_experimental_analysis(result))

    if result.calibrations:
        lines.append("")
        lines.append(summarize_calibrations(result.calibrations))

    lines.append(f"\n  Elapsed: {result.elapsed_seconds:.1f}s")

    text = "\n".join(lines)
    print(text)
    return text


def save_results(result: ProtocolResult, output_dir: Path | None = None):
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
                          run_cals=not args.no_calibrations)
    summarize(result)

    if not args.no_save:
        save_results(result, output_dir)

    n = sum(len(r) for r in result.metrics.values())
    nc = sum(len(r) for r in result.calibrations.values())
    print(f"\nTotal: {n} metric + {nc} calibration results in {result.elapsed_seconds:.1f}s")


if __name__ == "__main__":
    main()
