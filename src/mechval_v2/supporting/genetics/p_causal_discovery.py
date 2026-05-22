"""Protocol MB_CD — Causal Discovery (Structure Learning)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Lens:         Genetics
Validity Type: Internal
Framework:    Molecular Biology — Causal Graph Recovery
Family:       Molecular Biology (Causal Discovery)
Validity:     Structural — data-driven circuit topology inference

References:
    Spirtes, Glymour & Scheines (2000) "Causation, Prediction, and Search"
    Zheng et al. (2018) "DAGs with NO TEARS: Continuous Optimization
        for Structure Learning"
    Jansma (2025) for PID causal decomposition

Question:
    Can we recover a circuit's causal graph from observational data
    alone? NOTEARS learns a DAG via continuous optimization. Granger
    causality tests pairwise directed information flow. PID
    decomposition reveals redundancy/synergy structure, enabling
    d-separation tests. Together these provide data-driven circuit
    structure discovery that can be compared against known circuits.

    Parts K1-K2 from the Bio-Causal spec:
      K1 — PC Algorithm analogue: NOTEARS recovers the DAG skeleton;
           Granger causality identifies directed information flow
           between components. Compare recovered graph to known circuit.
      K2 — D-Separation Tests: PID decomposition checks conditional
           independence. High redundancy between A and B given C implies
           C mediates A-B (d-connected). Low mutual information implies
           A and B are d-separated by C.

Metrics:
    notears              — DAG structure recovery via continuous optimization
    granger_causality    — Pairwise directed information flow testing
    pid                  — Partial Information Decomposition (redundancy/synergy)

Calibrations:
    STRUCTURAL_CALIBRATIONS

Usage:
    uv run python causal_discovery.py                       # all tasks, CPU
    uv run python causal_discovery.py --device cuda          # GPU
    uv run python causal_discovery.py --tasks ioi induction  # specific tasks

    # As a callable module:
    from protocols.molecular_biology.causal_discovery import run_protocol
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

PROTOCOL_ID = "MB_CD"
PROTOCOL_NAME = "Causal Discovery (Structure Learning)"
METRICS = ["notears", "granger_causality", "pid"]
CALIBRATIONS = STRUCTURAL_CALIBRATIONS
OUTPUT_DIR = Path(__file__).parent / "results" / "mb_cd_causal_discovery"

THRESHOLDS = {
    "notears": 0.5,
    "granger_causality": 0.3,
    "pid": 0.3,
}


def run_protocol(model, tasks: list[str] | None = None, n_prompts: int = 40,
                 device: str = "cpu", run_cals: bool = True) -> ProtocolResult:
    """Run all MB_CD metrics + calibrations. Returns a ProtocolResult."""
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


def causal_discovery_analysis(result: ProtocolResult) -> list[str]:
    """Analyze results through Causal Discovery lens.

    Biology analogy: causal discovery algorithms (PC, FCI, NOTEARS)
    recover a causal graph from observational data. The PC algorithm
    (Spirtes et al., 2000) uses conditional independence tests to learn
    the DAG skeleton. NOTEARS (Zheng et al., 2018) formulates structure
    learning as a continuous optimization problem. Granger causality
    tests whether past values of one variable improve prediction of
    another — a directed information flow test. PID (Jansma, 2025)
    decomposes mutual information into redundancy, unique, and synergy
    terms, enabling d-separation tests: if C mediates A-B, then
    redundancy(A;B|C) is high; if A and B are d-separated by C, their
    conditional mutual information is low.

    K1: NOTEARS recovers the DAG; Granger identifies directed edges.
        Together they reconstruct the circuit's causal graph.
    K2: PID redundancy/synergy structure tests d-separation predictions
        from the recovered (or known) graph.
    """
    lines = ["\n  Causal Discovery Analysis:", "  --------------------------"]

    for task in result.tasks:
        nt = _find(result.metrics.get("notears", []), task)
        gc = _find(result.metrics.get("granger_causality", []), task)
        pid = _find(result.metrics.get("pid", []), task)

        lines.append(f"\n    {task}:")

        # --- K1: Graph Recovery ---
        lines.append("      K1 — PC Algorithm Analogue (Graph Recovery):")
        recovered_edges = []

        if nt:
            if nt.value > THRESHOLDS["notears"]:
                label = "strong DAG recovery"
            elif nt.value > 0.2:
                label = "partial DAG recovery"
            else:
                label = "weak recovery"
            lines.append(f"        NOTEARS (DAG structure):              {nt.value:.4f} — {label}")
            n_edges = nt.metadata.get("n_edges", None)
            if n_edges is not None:
                lines.append(f"          skeleton edges recovered: {n_edges}")
                recovered_edges.append(("notears", n_edges))

        if gc:
            if gc.value > THRESHOLDS["granger_causality"]:
                label = "significant directed flow"
            elif gc.value > 0.1:
                label = "weak directed flow"
            else:
                label = "no significant flow"
            lines.append(f"        Granger causality (directed flow):    {gc.value:.4f} — {label}")
            n_directed = gc.metadata.get("n_directed_edges", None)
            if n_directed is not None:
                lines.append(f"          estimated directed edges: {n_directed}")
                recovered_edges.append(("granger", n_directed))

        # --- K2: D-Separation Tests ---
        lines.append("      K2 — D-Separation Tests (Conditional Independence):")

        if pid:
            if pid.value > THRESHOLDS["pid"]:
                label = "structured decomposition"
            elif pid.value > 0.1:
                label = "partial structure"
            else:
                label = "unstructured"
            lines.append(f"        PID (redundancy/synergy):             {pid.value:.4f} — {label}")
            redundancy = pid.metadata.get("redundancy", None)
            synergy = pid.metadata.get("synergy", None)
            if redundancy is not None:
                lines.append(f"          redundancy: {redundancy:.4f} (high = mediation / d-connected)")
            if synergy is not None:
                lines.append(f"          synergy:    {synergy:.4f} (high = emergent interaction)")

        # --- Graph Recovery Accuracy ---
        precision = _graph_precision(nt, gc, task)
        if precision is not None:
            lines.append(f"      Graph recovery precision: {precision:.4f}")

        # --- Verdict ---
        evidence = []

        if nt and nt.value > THRESHOLDS["notears"]:
            evidence.append("notears")
        if gc and gc.value > THRESHOLDS["granger_causality"]:
            evidence.append("granger")
        if pid and pid.value > THRESHOLDS["pid"]:
            evidence.append("pid")

        if precision is not None and precision > 0.7:
            verdict = "CONSISTENT GRAPH — recovered structure matches known circuit"
        elif precision is not None and precision > 0.3:
            verdict = "PARTIAL RECOVERY — some edges match known circuit"
            novel = _novel_edges(nt, gc, task)
            if novel:
                verdict += f"; {novel} novel edges detected"
        elif precision is not None and precision <= 0.3 and len(evidence) > 0:
            verdict = "INCONSISTENT — recovered graph contradicts known circuit"
        elif len(evidence) >= 2 and precision is None:
            verdict = "NOVEL STRUCTURE — discovered edges not in known circuit (potential new findings)"
        elif len(evidence) == 1:
            verdict = f"WEAK EVIDENCE — only {evidence[0]} informative"
        else:
            verdict = "NO STRUCTURE — insufficient signal for graph recovery"
        lines.append(f"      VERDICT: {verdict}")

    return lines


def _find(results: list[EvalResult], task: str) -> EvalResult | None:
    return next((r for r in results if r.metadata.get("task") == task), None)


def _graph_precision(nt: EvalResult | None, gc: EvalResult | None,
                     task: str) -> float | None:
    """Compute precision of recovered edges against known circuit, if available."""
    known_heads = get_circuit_heads(task)
    if not known_heads:
        return None

    n_known = len(known_heads)
    if n_known == 0:
        return None

    recovered = 0
    total_recovered = 0

    if nt and nt.metadata.get("recovered_heads"):
        for head in nt.metadata["recovered_heads"]:
            total_recovered += 1
            if tuple(head) in {(h[0], h[1]) for h in known_heads}:
                recovered += 1

    if gc and gc.metadata.get("recovered_heads"):
        for head in gc.metadata["recovered_heads"]:
            total_recovered += 1
            if tuple(head) in {(h[0], h[1]) for h in known_heads}:
                recovered += 1

    if total_recovered == 0:
        return None

    return recovered / total_recovered


def _novel_edges(nt: EvalResult | None, gc: EvalResult | None,
                 task: str) -> int:
    """Count recovered edges not in the known circuit."""
    known_heads = get_circuit_heads(task)
    known_set = {(h[0], h[1]) for h in known_heads} if known_heads else set()

    novel = 0
    for source in [nt, gc]:
        if source and source.metadata.get("recovered_heads"):
            for head in source.metadata["recovered_heads"]:
                if tuple(head) not in known_set:
                    novel += 1
    return novel


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

    lines.extend(causal_discovery_analysis(result))

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
