"""Protocol MB_EA --- Factor Set Enrichment Analysis
Lens:         Genetics
Validity Type: Construct
====================================================================
Framework:    Molecular Biology --- Gene Set Enrichment Analysis (GSEA)
Family:       Molecular Biology (Enrichment Analysis)
Validity:     External --- enrichment of known circuit components among
              top-ranked causal components

References:
    Subramanian et al. (2005) "Gene set enrichment analysis: a knowledge-
        based approach for interpreting genome-wide expression profiles"
    Mootha et al. (2003) "PGC-1alpha-responsive genes involved in
        oxidative phosphorylation are coordinately downregulated in
        human diabetes"

Question:
    GSEA ranks all genes by phenotype association, then tests whether a
    predefined gene set is enriched at the top or bottom of the ranked
    list. Adapted for interpretability:

    - "Genes" = all circuit components (attention heads)
    - "Phenotype correlation" = IIA score or causal effect per component
    - "Gene sets" = predefined component sets (e.g., "heads in layer 8",
      "heads identified by EAP", "circuit heads from Wang et al.")
    - NES > 0: a component set is enriched among the most causally
      relevant components

    The protocol ranks components by their metric values, checks if the
    known circuit heads are enriched at the top of the ranking, computes
    enrichment scores (running sum), and derives a permutation-based
    p-value by comparing observed ES to random gene sets.

Metrics:
    eap                 -- Edge Attribution Patching score per component
    activation_patching -- Activation patching causal effect per component
    das_iia             -- Distributed Alignment Search IIA per component

Calibrations:
    STRUCTURAL_CALIBRATIONS

Usage:
    uv run python enrichment_analysis.py                       # all tasks, CPU
    uv run python enrichment_analysis.py --device cuda          # GPU
    uv run python enrichment_analysis.py --tasks ioi induction  # specific tasks

    # As a callable module:
    from protocols.molecular_biology.enrichment_analysis import run_protocol
    result = run_protocol(model, tasks=["ioi"], n_prompts=40)
====================================================================
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

PROTOCOL_ID = "MB_EA"
PROTOCOL_NAME = "Factor Set Enrichment Analysis"
METRICS = ["eap", "activation_patching", "das_iia"]
CALIBRATIONS = STRUCTURAL_CALIBRATIONS
OUTPUT_DIR = Path(__file__).parent / "results" / "mb_ea_enrichment"

THRESHOLDS = {
    "eap": 0.3,
    "activation_patching": 0.5,
    "das_iia": 0.6,
}

N_PERMUTATIONS = 1000


def _compute_enrichment_score(ranked_components: list[tuple[int, int]],
                              gene_set: set[tuple[int, int]],
                              n_total: int) -> tuple[float, list[float]]:
    """Compute GSEA-style enrichment score for a gene set in a ranked list.

    Walks down the ranked list. When hitting a gene-set member the running
    sum increases by 1/|gene_set|; otherwise it decreases by 1/(N - |gene_set|).
    The enrichment score is the maximum deviation from zero.

    Returns (ES, running_sum_trace).
    """
    n_hit = len(gene_set)
    n_miss = n_total - n_hit
    if n_hit == 0 or n_miss == 0:
        return 0.0, []

    hit_increment = 1.0 / n_hit
    miss_decrement = 1.0 / n_miss

    running_sum = 0.0
    trace = []
    max_dev = 0.0
    max_dev_signed = 0.0

    for component in ranked_components:
        if component in gene_set:
            running_sum += hit_increment
        else:
            running_sum -= miss_decrement
        trace.append(running_sum)
        if abs(running_sum) > abs(max_dev):
            max_dev = abs(running_sum)
            max_dev_signed = running_sum

    return max_dev_signed, trace


def _permutation_nes(ranked_components: list[tuple[int, int]],
                     gene_set: set[tuple[int, int]],
                     n_total: int,
                     n_perms: int,
                     rng: np.random.Generator) -> tuple[float, float]:
    """Compute normalized enrichment score and permutation p-value.

    Generates n_perms random gene sets of the same size, computes their ES,
    and normalizes the observed ES by the mean of the permutation distribution.
    P-value is the fraction of permuted ES values >= observed ES.

    Returns (NES, p_value).
    """
    observed_es, _ = _compute_enrichment_score(ranked_components, gene_set, n_total)
    if observed_es == 0.0:
        return 0.0, 1.0

    perm_scores = np.empty(n_perms)
    set_size = len(gene_set)
    all_components = list(ranked_components)

    for i in range(n_perms):
        perm_indices = rng.choice(n_total, size=set_size, replace=False)
        perm_set = set(all_components[j] for j in perm_indices)
        perm_es, _ = _compute_enrichment_score(ranked_components, perm_set, n_total)
        perm_scores[i] = perm_es

    if observed_es > 0:
        pos_perms = perm_scores[perm_scores > 0]
        mean_pos = np.mean(pos_perms) if len(pos_perms) > 0 else 1.0
        nes = observed_es / mean_pos if mean_pos > 0 else 0.0
        p_value = float(np.mean(perm_scores >= observed_es))
    else:
        neg_perms = perm_scores[perm_scores < 0]
        mean_neg = np.abs(np.mean(neg_perms)) if len(neg_perms) > 0 else 1.0
        nes = observed_es / mean_neg if mean_neg > 0 else 0.0
        p_value = float(np.mean(perm_scores <= observed_es))

    return nes, p_value


def run_protocol(model, tasks: list[str] | None = None, n_prompts: int = 40,
                 device: str = "cpu", run_cals: bool = True) -> ProtocolResult:
    """Run all MB_EA metrics + calibrations. Returns a ProtocolResult."""
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


def enrichment_biology_analysis(result: ProtocolResult) -> list[str]:
    """Analyze results through Gene Set Enrichment Analysis lens.

    Biology analogy: GSEA (Subramanian et al. 2005; Mootha et al. 2003)
    ranks all genes by their association with a phenotype, then tests
    whether a predefined gene set is enriched at the top or bottom of
    the ranked list. The enrichment score is the maximum deviation of a
    running sum that increases when hitting a gene-set member and decreases
    otherwise. NES (normalized enrichment score) accounts for gene-set size
    by comparing to a permutation null.

    In the interpretability adaptation:
    - Each attention head is a "gene"
    - The causal effect (EAP, activation patching, DAS-IIA) is the
      "phenotype correlation"
    - Known circuit heads (from get_circuit_heads) are the "gene set"
    - NES > 0 means circuit heads are enriched among top causal components
    """
    lines = ["\n  Factor Set Enrichment Analysis:", "  --------------------------------"]
    rng = np.random.default_rng(42)

    for task in result.tasks:
        lines.append(f"\n    {task}:")
        circuit_heads = get_circuit_heads(task)
        circuit_set = set(circuit_heads)
        lines.append(f"      Gene set size: {len(circuit_set)} circuit heads")

        verdicts = []

        for metric_name in METRICS:
            metric_results = result.metrics.get(metric_name, [])
            task_results = [r for r in metric_results if r.metadata.get("task") == task]

            if not task_results:
                lines.append(f"      {metric_name:30s}  --- (no data)")
                continue

            # Build ranked list of components: (layer, head) sorted by value descending
            components_with_scores = []
            for r in task_results:
                layer = r.metadata.get("layer", r.metadata.get("L", None))
                head = r.metadata.get("head", r.metadata.get("H", None))
                if layer is not None and head is not None:
                    components_with_scores.append(((int(layer), int(head)), r.value))

            # If per-component results are not available, use aggregate score
            if not components_with_scores:
                val = task_results[0].value
                threshold = THRESHOLDS.get(metric_name, 0.5)
                passed = val >= threshold
                tag = "PASS" if passed else "FAIL"
                lines.append(f"      {metric_name:30s}  {val:+.4f}  ({tag})")
                if passed:
                    verdicts.append("MODERATELY ENRICHED")
                continue

            # Sort by score descending (highest causal effect first)
            components_with_scores.sort(key=lambda x: x[1], reverse=True)
            ranked_components = [c for c, _ in components_with_scores]
            n_total = len(ranked_components)

            # Compute enrichment score
            gene_set_in_ranking = circuit_set & set(ranked_components)
            if not gene_set_in_ranking:
                lines.append(f"      {metric_name:30s}  --- (no circuit heads in ranking)")
                continue

            es, trace = _compute_enrichment_score(ranked_components, gene_set_in_ranking, n_total)
            nes, p_value = _permutation_nes(ranked_components, gene_set_in_ranking,
                                            n_total, N_PERMUTATIONS, rng)

            # Determine verdict
            if nes > 1.5 and p_value < 0.05:
                verdict = "STRONGLY ENRICHED"
            elif nes > 0.5 and p_value < 0.25:
                verdict = "MODERATELY ENRICHED"
            elif nes < -0.5 and p_value < 0.25:
                verdict = "DEPLETED"
            else:
                verdict = "NOT ENRICHED"
            verdicts.append(verdict)

            lines.append(f"      {metric_name:30s}  ES={es:+.4f}  NES={nes:+.4f}  "
                         f"p={p_value:.4f}  {verdict}")

        # Overall verdict for this task
        if not verdicts:
            overall = "NO DATA"
        elif "STRONGLY ENRICHED" in verdicts:
            overall = "STRONGLY ENRICHED"
        elif "DEPLETED" in verdicts:
            overall = "DEPLETED"
        elif "MODERATELY ENRICHED" in verdicts:
            overall = "MODERATELY ENRICHED"
        else:
            overall = "NOT ENRICHED"
        lines.append(f"      VERDICT: {overall}")

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

    lines.extend(enrichment_biology_analysis(result))

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
