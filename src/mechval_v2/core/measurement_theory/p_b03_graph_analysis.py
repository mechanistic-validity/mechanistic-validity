"""Protocol B03 — Graph & Edge Analysis
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Lens:         Measurement Theory
Validity Type: Measurement
Framework:    https://mechanistic-validity.github.io/mechanistic-validity/framework/metrics/structural/b06-template-distance/
Family:       Structural (graph topology, edge-level analysis)
Validity:     Internal — I1 Necessity, I2 Sufficiency; Construct — C1 Convergent

References:
    Conmy et al. (2023) "Towards Automated Circuit Discovery" — ACDC edge graphs
    Wang et al. (2022) "Interpretability in the Wild" — IOI circuit graph
    Goldowsky-Dill et al. (2023) "Localizing Model Behavior with Path Patching"
    Hanna et al. (2023) "How does GPT-2 compute greater-than?"

Question:
    Is the circuit graph minimal? Are all edges necessary? Are there
    alternative paths that bypass the circuit? Does the graph structure
    match known reference circuits?

Metrics:
    path_identification         — Can paths from input to output be identified?
    edge_necessity              — Are individual edges necessary for the output?
    path_specificity            — Do paths carry task-specific information?
    compositional_sufficiency   — Is the subgraph sufficient to reproduce behavior?
    graph_minimality            — Is the graph minimal (no redundant edges)?
    edge_jaccard                — Jaccard similarity to reference circuit edges
    cmd                         — Circuit-model distance (template distance)

Calibrations:
    measurement_invariance — Are graph metrics stable across different prompts?
    convergent_validity    — Do graph metrics agree with causal metrics?
    discriminant_validity  — Do graph metrics distinguish circuits from baselines?

Usage:
    uv run python b03_graph_analysis.py                       # all tasks, CPU
    uv run python b03_graph_analysis.py --device cuda          # GPU
    uv run python b03_graph_analysis.py --tasks ioi induction  # specific tasks

    # As a callable module:
    from protocols.structural.b03_graph_analysis import run_protocol
    result = run_protocol(model, tasks=["ioi"], n_prompts=40)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from mechval.metrics.common import CIRCUIT_TASKS, EvalResult, get_circuit_heads, load_model

from protocols import ProtocolResult, import_metric_runner
from protocols.calibration_runner import STRUCTURAL_CALIBRATIONS, run_calibrations, summarize_calibrations

PROTOCOL_ID = "B03"
PROTOCOL_NAME = "Graph & Edge Analysis"
METRICS = [
    "path_identification", "edge_necessity", "path_specificity",
    "compositional_sufficiency", "graph_minimality", "edge_jaccard", "cmd",
]
CALIBRATIONS = STRUCTURAL_CALIBRATIONS
OUTPUT_DIR = Path(__file__).parent / "results" / "b03_graph_analysis"

THRESHOLDS = {
    "edge_necessity": 0.5,
    "compositional_sufficiency": 0.7,
    "graph_minimality": 0.6,
    "edge_jaccard": 0.3,
}




def run_protocol(model, tasks: list[str] | None = None, n_prompts: int = 40,
                 run_cals: bool = True) -> ProtocolResult:
    """Run all B03 metrics + calibrations. Returns a ProtocolResult."""
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
            results = runner(model, tasks, n_prompts=n_prompts)
        except Exception as e:
            print(f"  [{metric_name}] FAILED: {e}")
            result.metrics[metric_name] = []
            continue

        result.metrics[metric_name] = results
        for r in results:
            task = r.metadata.get("task", "?")
            passed = r.metadata.get("passed", None)
            tag = " ✓" if passed else (" ✗" if passed is not None else "")
            print(f"    {task:20s}  {r.value:+.4f}{tag}")
        print(f"  {len(results)} results in {time.time() - mt0:.1f}s")

    if run_cals:
        print(f"\n{'═' * 60}")
        print(f"  Calibrations ({len(CALIBRATIONS)})")
        print(f"{'═' * 60}")
        cal_tasks = tasks[:2]
        result.calibrations = run_calibrations(
            model, cal_tasks, CALIBRATIONS, n_prompts=n_prompts)

    result.elapsed_seconds = time.time() - t0
    return result


def graph_analysis(result: ProtocolResult) -> list[str]:
    """Analyze results through graph topology lens."""
    lines = ["\n  Graph Topology Analysis:", "  ────────────────────────"]

    for task in result.tasks:
        lines.append(f"\n    {task}:")

        path_id = _find(result.metrics.get("path_identification", []), task)
        edge_nec = _find(result.metrics.get("edge_necessity", []), task)
        path_spec = _find(result.metrics.get("path_specificity", []), task)
        comp_suf = _find(result.metrics.get("compositional_sufficiency", []), task)
        graph_min = _find(result.metrics.get("graph_minimality", []), task)
        ej = _find(result.metrics.get("edge_jaccard", []), task)
        cmd_r = _find(result.metrics.get("cmd", []), task)

        properties = []

        if path_id:
            lines.append(f"      Path identification: {path_id.value:.4f}")

        if edge_nec:
            lines.append(f"      Edge necessity: {edge_nec.value:.4f}")
            if edge_nec.value > THRESHOLDS["edge_necessity"]:
                properties.append("necessary edges")

        if path_spec:
            lines.append(f"      Path specificity: {path_spec.value:.4f}")

        if comp_suf:
            lines.append(f"      Compositional sufficiency: {comp_suf.value:.4f}")
            if comp_suf.value > THRESHOLDS["compositional_sufficiency"]:
                properties.append("sufficient subgraph")

        if graph_min:
            lines.append(f"      Graph minimality: {graph_min.value:.4f}")
            if graph_min.value > THRESHOLDS["graph_minimality"]:
                properties.append("minimal")

        if ej:
            lines.append(f"      Edge Jaccard (vs reference): {ej.value:.4f}")
            if ej.value > THRESHOLDS["edge_jaccard"]:
                properties.append("reference-aligned")

        if cmd_r:
            lines.append(f"      Circuit-model distance: {cmd_r.value:.4f}")

        if properties:
            verdict = "Valid graph structure: " + ", ".join(properties)
        else:
            verdict = "Weak graph structure"
        lines.append(f"      VERDICT: {verdict}")

    return lines


def _find(results: list[EvalResult], task: str) -> EvalResult | None:
    return next((r for r in results if r.metadata.get("task") == task), None)


def summarize(result: ProtocolResult) -> str:
    lines = []
    lines.append(f"\n{'═' * 70}")
    lines.append(f"  PROTOCOL {result.protocol_id}: {result.protocol_name}")
    lines.append(f"{'═' * 70}\n")

    header = f"{'Task':20s}" + "".join(f"  {m:>20s}" for m in METRICS)
    lines.append(header)
    lines.append("─" * len(header))

    for task in result.tasks:
        row = f"{task:20s}"
        for m in METRICS:
            match = _find(result.metrics.get(m, []), task)
            if match:
                v = match.value
                p = match.metadata.get("passed", None)
                tag = " ✓" if p else (" ✗" if p is not None else " —")
                row += f"  {v:>18.4f}{tag}"
            else:
                row += f"  {'—':>20s}"
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

    lines.extend(graph_analysis(result))

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

    print(f"{'═' * 70}")
    print(f"  Protocol {PROTOCOL_ID}: {PROTOCOL_NAME}")
    print(f"  Model: {args.model}  Device: {args.device}  Prompts: {args.n_prompts}")
    print(f"  Tasks: {', '.join(tasks)}")
    print(f"{'═' * 70}")

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
