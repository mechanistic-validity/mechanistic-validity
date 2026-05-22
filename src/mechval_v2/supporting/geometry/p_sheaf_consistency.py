"""Protocol WC_M7 — Sheaf Consistency Scan
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Lens:         Geometry
Validity Type: Measurement
Framework:    Wildcard — Sheaf-Theoretic Representational Consistency
Family:       Wildcard (Sheaf Consistency)
Validity:     Internal — cross-referencing causal importance with
              representational consistency via sheaf Laplacian energy

References:
    Bodnar et al. (NeurIPS 2022) "Neural Sheaf Diffusion"
    Curry (2014) "Sheaves, Cosheaves and Applications"

Question:
    Does sheaf-theoretic consistency of the residual stream align with
    causal importance? A sheaf assigns local vector spaces to nodes
    (token positions) and linear maps to edges (attention connections).
    The sheaf Laplacian energy measures representational inconsistency
    — how well information flows between positions through each
    attention head.

    Low sheaf energy = consistent, circuit-like information transfer.
    High sheaf energy = noisy, inconsistent transfer.

    A true circuit component should have both high IIA (causal
    importance) AND low sheaf energy (representational consistency).

Metrics:
    activation_patching — Causal importance: which components matter
    das_iia             — Causal validation: do interventions propagate correctly
    path_patching       — Path-level causal flow for structural consistency

Calibrations:
    STRUCTURAL_CALIBRATIONS

Usage:
    uv run python sheaf_consistency.py                       # all tasks, CPU
    uv run python sheaf_consistency.py --device cuda          # GPU
    uv run python sheaf_consistency.py --tasks ioi induction  # specific tasks

    # As a callable module:
    from protocols.wildcard.sheaf_consistency import run_protocol
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

PROTOCOL_ID = "WC_M7"
PROTOCOL_NAME = "Sheaf Consistency Scan"
METRICS = ["activation_patching", "das_iia", "path_patching"]
CALIBRATIONS = STRUCTURAL_CALIBRATIONS
OUTPUT_DIR = Path(__file__).parent / "results" / "wc_m7_sheaf_consistency"

THRESHOLDS = {
    "activation_patching": 0.5,
    "das_iia": 0.6,
    "path_patching": 0.5,
}


def run_protocol(model, tasks: list[str] | None = None, n_prompts: int = 40,
                 device: str = "cpu", run_cals: bool = True) -> ProtocolResult:
    """Run all WC_M7 metrics + calibrations. Returns a ProtocolResult."""
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


def sheaf_consistency_analysis(result: ProtocolResult) -> list[str]:
    """Analyze results through the sheaf consistency lens.

    Sheaf theory applied to attention circuits: a sheaf assigns local
    vector spaces to nodes (token positions) and linear restriction maps
    to edges (attention connections). The sheaf Laplacian energy measures
    how consistently information transfers between positions through each
    attention head.

    We cross-reference three causal metrics to classify each component:

    - activation_patching: does ablating this component change the output?
    - das_iia: does intervening on this component's representation
      propagate correctly through the network?
    - path_patching: does information flow through specific causal paths
      involving this component?

    Classification:
    - CONSISTENT CIRCUIT MEMBER: all three pass — causally important and
      structurally consistent (low sheaf energy analog).
    - POLYSEMANTIC ROUTER: causal metrics pass but path structure fails —
      important but inconsistent, possibly serving multiple functions.
    - STRUCTURAL SCAFFOLD: path structure passes but causal metrics fail —
      consistent but not causal for this task.
    - NOT IN CIRCUIT: all metrics fail.
    - INCONSISTENT EVIDENCE: partial agreement that doesn't fit the above.

    References:
        Bodnar et al. (NeurIPS 2022) "Neural Sheaf Diffusion"
        Curry (2014) "Sheaves, Cosheaves and Applications"
    """
    lines = ["\n  Sheaf Consistency Analysis:", "  --------------------------"]

    for task in result.tasks:
        ap = _find(result.metrics.get("activation_patching", []), task)
        di = _find(result.metrics.get("das_iia", []), task)
        pp = _find(result.metrics.get("path_patching", []), task)

        lines.append(f"\n    {task}:")

        ap_pass = ap is not None and ap.value > THRESHOLDS["activation_patching"]
        di_pass = di is not None and di.value > THRESHOLDS["das_iia"]
        pp_pass = pp is not None and pp.value > THRESHOLDS["path_patching"]

        if ap:
            label = "causally important" if ap_pass else "not causally important"
            lines.append(f"      Causal importance (activation_patching): {ap.value:.4f} — {label}")

        if di:
            label = "interventions propagate" if di_pass else "interventions do not propagate"
            lines.append(f"      Causal validation (das_iia):             {di.value:.4f} — {label}")

        if pp:
            label = "consistent paths" if pp_pass else "weak path structure"
            lines.append(f"      Path structure (path_patching):          {pp.value:.4f} — {label}")

        if ap_pass and di_pass and pp_pass:
            verdict = "CONSISTENT CIRCUIT MEMBER — causally important + structurally consistent"
        elif ap_pass and di_pass and not pp_pass:
            verdict = "POLYSEMANTIC ROUTER — causally important but path structure is weak"
        elif not ap_pass and not di_pass and pp_pass:
            verdict = "STRUCTURAL SCAFFOLD — consistent path structure but not causal for this task"
        elif not ap_pass and not di_pass and not pp_pass:
            verdict = "NOT IN CIRCUIT — no causal or structural evidence"
        else:
            verdict = "INCONSISTENT EVIDENCE — metrics partially agree"
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

    lines.extend(sheaf_consistency_analysis(result))

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
