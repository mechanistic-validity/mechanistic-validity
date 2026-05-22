"""Protocol E01 — Representation Space Analysis
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Lens:         Mechanistic Interpretability
Validity Type: Construct
Framework:    https://mechanistic-validity.github.io/mechanistic-validity/framework/metrics/representational/
Family:       Representational
Validity:     Construct — C1 Representation adequacy

References:
    Kornblith et al. (2019) "Similarity of Neural Network Representations Revisited" — CKA
    Raghu et al. (2017) "SVCCA: Singular Vector CCA" — representational similarity
    Alain & Bengio (2017) "Understanding intermediate layers using linear classifier probes"
    Kriegeskorte et al. (2008) "Representational similarity analysis" — RSA

Question:
    What do the circuit's internal representations look like? Are they
    linearly decodable? Do they align across layers? How does the
    circuit's representation geometry differ from non-circuit components?

Metrics:
    attention_entropy       — How focused are circuit heads' attention patterns?
    cka                     — Centered Kernel Alignment between circuit layers
    cka_cross_arch          — CKA across model architectures/sizes
    probe_decodability      — Can a linear probe decode task-relevant info from circuit activations?
    causal_representation   — Does patching representations transfer task behavior?

Calibrations:
    measurement_invariance  — Are representation metrics stable across prompts?
    convergent_validity     — Do different representation metrics agree?
    discriminant_validity   — Do circuit and non-circuit representations differ?

Usage:
    uv run python e01_representation_space.py
    uv run python e01_representation_space.py --tasks ioi induction --device cuda

    from protocols.representational.e01_representation_space import run_protocol
    result = run_protocol(model, tasks=["ioi"])
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

PROTOCOL_ID = "E01"
PROTOCOL_NAME = "Representation Space Analysis"
METRICS = ["attention_entropy", "cka", "cka_cross_arch", "probe_decodability", "causal_representation"]
CALIBRATIONS = ["measurement_invariance", "convergent_validity", "discriminant_validity"]
OUTPUT_DIR = Path(__file__).parent / "results" / "e01_representation_space"




def run_protocol(model, tasks: list[str] | None = None, n_prompts: int = 40,
                 run_cals: bool = True) -> ProtocolResult:
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
        result.calibrations = run_calibrations(
            model, tasks[:2], CALIBRATIONS, n_prompts=n_prompts)

    result.elapsed_seconds = time.time() - t0
    return result


def representation_analysis(result: ProtocolResult) -> list[str]:
    lines = ["\n  Representation Space Analysis:", "  ──────────────────────────────"]

    for task in result.tasks:
        lines.append(f"\n    {task}:")

        ent = _find(result.metrics.get("attention_entropy", []), task)
        cka = _find(result.metrics.get("cka", []), task)
        probe = _find(result.metrics.get("probe_decodability", []), task)
        causal_rep = _find(result.metrics.get("causal_representation", []), task)

        if ent:
            ratio = ent.metadata.get("ratio", 0)
            lines.append(f"      Attention entropy: {ent.value:.4f} "
                         f"(circuit/bg ratio: {ratio:.3f})")
            if ratio < 0.8:
                lines.append(f"        → Circuit heads are more focused than background")
            else:
                lines.append(f"        → Circuit heads have similar entropy to background")

        if cka:
            lines.append(f"      CKA similarity: {cka.value:.4f}")

        if probe:
            lines.append(f"      Probe decodability: {probe.value:.4f}")
            if probe.value > 0.8:
                lines.append(f"        → Task info is linearly decodable from circuit")
            elif probe.value > 0.5:
                lines.append(f"        → Moderate linear decodability")
            else:
                lines.append(f"        → Task info NOT linearly accessible in circuit representations")

        if causal_rep:
            lines.append(f"      Causal representation: {causal_rep.value:.4f}")

    return lines


def _find(results: list[EvalResult], task: str) -> EvalResult | None:
    return next((r for r in results if r.metadata.get("task") == task), None)


def summarize(result: ProtocolResult) -> str:
    lines = []
    lines.append(f"\n{'═' * 70}")
    lines.append(f"  PROTOCOL {result.protocol_id}: {result.protocol_name}")
    lines.append(f"{'═' * 70}\n")

    header = f"{'Task':20s}" + "".join(f"  {m:>22s}" for m in METRICS)
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
                row += f"  {v:>20.4f}{tag}"
            else:
                row += f"  {'—':>22s}"
        lines.append(row)

    lines.append("")
    for m in METRICS:
        rs = result.metrics.get(m, [])
        if not rs:
            continue
        vals = [r.value for r in rs]
        lines.append(f"  {m}: mean={np.mean(vals):.4f}  std={np.std(vals):.4f}  "
                     f"range=[{min(vals):.4f}, {max(vals):.4f}]")

    lines.extend(representation_analysis(result))

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
        if rs:
            with open(output_dir / f"{name}.jsonl", "w") as f:
                for r in rs:
                    f.write(json.dumps(r.to_dict(), default=str) + "\n")

    for name, rs in result.calibrations.items():
        if rs:
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
