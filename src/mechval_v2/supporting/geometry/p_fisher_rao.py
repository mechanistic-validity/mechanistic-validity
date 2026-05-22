"""Protocol WC_M6 --- Information Geometry (Fisher-Rao Distance)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Lens:         Geometry
Validity Type: Construct
Framework:    Wildcard --- Information Geometry
Family:       Fisher-Rao Geodesic Distance
Validity:     Distributional --- reparameterization-invariant similarity

References:
    Fisher (1925) "Theory of statistical estimation" — Fisher
        Information Matrix as the natural metric on parameter space
    Rao (1945) "Information and accuracy attainable in the estimation
        of statistical parameters" — Riemannian metric on statistical
        manifolds (Fisher-Rao metric)
    Poincare half-plane model of hyperbolic geometry — the space of
        univariate Gaussians N(mu, sigma^2) with sigma > 0 forms the
        upper half-plane with the Fisher-Rao metric

Question:
    Are circuit components distributionally distinct or redundant?
    Weight-space cosine similarity is not reparameterization-invariant:
    two components with different weights but similar activation
    distributions can appear dissimilar. Fisher-Rao geodesic distance
    on the statistical manifold of component activation distributions
    is invariant to smooth reparameterizations.

    For two Gaussians on the Poincare half-plane:
        d = arccosh(1 + ((mu1-mu2)^2 + (sigma1-sigma2)^2) / (2*sigma1*sigma2))

Metrics:
    cka                    --- Representational similarity (standard CKA)
    effect_size            --- Component importance (ablation effect)
    attention_clustering   --- Functional grouping via attention patterns

Calibrations:
    STRUCTURAL_CALIBRATIONS

Usage:
    uv run python fisher_rao.py                       # all tasks, CPU
    uv run python fisher_rao.py --device cuda          # GPU
    uv run python fisher_rao.py --tasks ioi induction  # specific tasks

    # As a callable module:
    from protocols.wildcard.fisher_rao import run_protocol
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

PROTOCOL_ID = "WC_M6"
PROTOCOL_NAME = "Information Geometry (Fisher-Rao Distance)"
METRICS = ["cka", "effect_size", "attention_clustering"]
CALIBRATIONS = STRUCTURAL_CALIBRATIONS
OUTPUT_DIR = Path(__file__).parent / "results" / "wc_m6_fisher_rao"

THRESHOLDS = {
    "cka": 0.5,
    "effect_size": 0.8,
    "attention_clustering": 0.5,
}


def run_protocol(model, tasks: list[str] | None = None, n_prompts: int = 40,
                 device: str = "cpu", run_cals: bool = True) -> ProtocolResult:
    """Run all WC_M6 metrics + calibrations. Returns a ProtocolResult."""
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


def fisher_rao_analysis(result: ProtocolResult) -> list[str]:
    """Analyze results through information geometry (Fisher-Rao) lens.

    Fisher-Rao distance is the geodesic distance on the statistical
    manifold equipped with the Fisher information metric. For Gaussian
    activation distributions, the manifold is the Poincare half-plane
    and the distance is:
        d = arccosh(1 + ((mu1-mu2)^2 + (sigma1-sigma2)^2) / (2*sigma1*sigma2))

    Interpretation:
    - CKA measures representational similarity in activation space.
    - attention_clustering finds functional groupings.
    - effect_size assesses component importance via ablation.

    When CKA is high and attention_clustering shows clear groups, the
    Fisher-Rao landscape is well-clustered: components form clean
    functional groups on the statistical manifold. When CKA is low but
    clustering exists, we have distributional divergence: components
    differ in representation but converge in function. Heterogeneous
    effect_size indicates Fisher-Rao distances spanning orders of
    magnitude.
    """
    lines = ["\n  Fisher-Rao / Information Geometry Analysis:", "  --------------------------------------------"]

    for task in result.tasks:
        cka = _find(result.metrics.get("cka", []), task)
        es = _find(result.metrics.get("effect_size", []), task)
        ac = _find(result.metrics.get("attention_clustering", []), task)

        lines.append(f"\n    {task}:")

        cka_val = cka.value if cka else None
        es_val = es.value if es else None
        ac_val = ac.value if ac else None

        if cka:
            label = "high similarity" if cka_val > THRESHOLDS["cka"] else "low similarity"
            lines.append(f"      CKA (representational similarity):     {cka_val:.4f} — {label}")

        if es:
            label = "large effect" if es_val > THRESHOLDS["effect_size"] else ("moderate" if es_val > 0.4 else "small effect")
            lines.append(f"      Effect size (component importance):    {es_val:.4f} — {label}")

        if ac:
            label = "clear groups" if ac_val > THRESHOLDS["attention_clustering"] else "diffuse"
            lines.append(f"      Attention clustering (functional):     {ac_val:.4f} — {label}")

        # Determine Fisher-Rao landscape characterization
        cka_high = cka_val is not None and cka_val > THRESHOLDS["cka"]
        ac_high = ac_val is not None and ac_val > THRESHOLDS["attention_clustering"]
        es_high = es_val is not None and es_val > THRESHOLDS["effect_size"]

        # Collect all available metric values to check uniformity
        available = [v for v in [cka_val, es_val, ac_val] if v is not None]
        uniform = len(available) >= 2 and (max(available) - min(available)) < 0.2

        if cka_high and ac_high:
            landscape = "WELL-CLUSTERED — Fisher-Rao finds clean functional groups"
        elif not cka_high and ac_high:
            landscape = "DISTRIBUTIONAL DIVERGENCE — components differ by CKA but cluster by Fisher-Rao"
        elif es_high and not uniform:
            landscape = "HETEROGENEOUS LANDSCAPE — Fisher-Rao distances span orders of magnitude"
        elif uniform:
            landscape = "HOMOGENEOUS — all components have similar distributions"
        else:
            landscape = "MIXED — no dominant distributional pattern"

        lines.append(f"      Landscape: {landscape}")

        # Determine verdict
        if not cka_high and es_high:
            verdict = "DISTRIBUTIONALLY DISTINCT — components show large Fisher-Rao distances (functionally specialized)"
        elif cka_high and not es_high:
            verdict = "DISTRIBUTIONALLY REDUNDANT — many components have small Fisher-Rao distances (functional overlap)"
        elif ac_high and cka_val is not None and THRESHOLDS["cka"] * 0.6 < cka_val <= THRESHOLDS["cka"] * 1.4:
            verdict = "WELL-SEPARATED CLUSTERS — attention_clustering high with moderate CKA"
        elif not ac_high and not cka_high and not es_high:
            verdict = "AMORPHOUS — no clear distributional structure"
        elif cka_high and ac_high:
            verdict = "WELL-SEPARATED CLUSTERS — attention_clustering high with moderate CKA"
        elif es_high and ac_high:
            verdict = "DISTRIBUTIONALLY DISTINCT — components show large Fisher-Rao distances (functionally specialized)"
        else:
            verdict = "AMORPHOUS — no clear distributional structure"

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

    lines.extend(fisher_rao_analysis(result))

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
