"""Protocol C12 --- Arbitrage Search
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Lens:         Economics
Validity Type: Construct
Framework:    Construct Validity --- Uniqueness / Substitutability
Family:       Arbitrage Search (Alternative Circuit Discovery)
Validity:     Construct --- is the circuit uniquely necessary, or can
              an alternative component set substitute for it?

References:
    Wang et al. (2022) "Interpretability in the Wild" --- IOI circuit
        identification with complement ablation (faithfulness)
    Conmy et al. (2023) "Towards Automated Circuit Discovery" --- ACDC
        greedy search over component subsets
    Hanna et al. (2023) "How does GPT-2 compute greater-than?" ---
        circuit uniqueness via systematic search

Question:
    Can an alternative set of heads (same size as the circuit) achieve
    comparable performance? If yes, the circuit is not uniquely necessary
    --- there exists "arbitrage" (an alternative computational path).
    If no substitute achieves >= 90% of circuit performance, the circuit
    is non-fungible.

    This tests construct validity: if many alternative sets work equally
    well, the "circuit" concept may not carve computation at its joints.

Metrics:
    activation_patching --- Component-level causal importance
    effect_size         --- Overall circuit importance
    cka                 --- Representational similarity

Calibrations:
    STRUCTURAL_CALIBRATIONS

Usage:
    uv run python arbitrage_search.py                       # all tasks, CPU
    uv run python arbitrage_search.py --device cuda          # GPU
    uv run python arbitrage_search.py --tasks ioi induction  # specific tasks

    # As a callable module:
    from protocols.arbitrage_search import run_protocol
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    calibrate_mean_z,
    compute_faithfulness,
    generate_prompts,
    get_circuit_heads,
    get_token_ids,
    heads_to_layer_dict,
    load_model,
    make_ablation_hook,
)

from protocols import ProtocolResult, import_metric_runner
from protocols.calibration_runner import STRUCTURAL_CALIBRATIONS, run_calibrations, summarize_calibrations

PROTOCOL_ID = "C12"
PROTOCOL_NAME = "Arbitrage Search"
METRICS = ["activation_patching", "effect_size", "cka"]
CALIBRATIONS = STRUCTURAL_CALIBRATIONS
OUTPUT_DIR = Path(__file__).parent / "results" / "c12_arbitrage_search"

THRESHOLDS = {
    "activation_patching": 0.5,
    "effect_size": 0.8,
    "cka": 0.5,
}


# ---------------------------------------------------------------------------
# Novel analysis: Arbitrage search for substitute circuits
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_head_set_faithfulness(
    model,
    head_set: set[tuple[int, int]],
    prompts,
    correct_ids: list[int],
    incorrect_ids: list[int],
    mean_z: torch.Tensor,
) -> float:
    """Evaluate faithfulness of a head set (complement ablation).

    Ablate all heads NOT in the set, keep head_set intact.
    Returns faithfulness = logit_diff(set_only) / logit_diff(full).
    """
    return compute_faithfulness(
        model, prompts, correct_ids, incorrect_ids, head_set, mean_z)


def greedy_substitute_search(
    model,
    candidate_heads: list[tuple[int, int]],
    k: int,
    prompts,
    correct_ids: list[int],
    incorrect_ids: list[int],
    mean_z: torch.Tensor,
) -> tuple[set[tuple[int, int]], float]:
    """Greedy search: build a k-head set one head at a time.

    At each step, add the candidate that most improves faithfulness.
    Returns (best_set, best_faithfulness).
    """
    selected: set[tuple[int, int]] = set()

    for _ in range(k):
        best_head = None
        best_faith = -float("inf")

        for head in candidate_heads:
            if head in selected:
                continue

            trial_set = selected | {head}
            faith = evaluate_head_set_faithfulness(
                model, trial_set, prompts, correct_ids, incorrect_ids, mean_z)

            if faith > best_faith:
                best_faith = faith
                best_head = head

        if best_head is None:
            break
        selected.add(best_head)

    final_faith = evaluate_head_set_faithfulness(
        model, selected, prompts, correct_ids, incorrect_ids, mean_z)
    return selected, final_faith


def random_substitute_search(
    model,
    candidate_heads: list[tuple[int, int]],
    k: int,
    n_random: int,
    prompts,
    correct_ids: list[int],
    incorrect_ids: list[int],
    mean_z: torch.Tensor,
) -> tuple[set[tuple[int, int]], float, list[float]]:
    """Try n_random random subsets of size k.

    Returns (best_set, best_faithfulness, all_faithfulness_scores).
    """
    rng = np.random.default_rng(42)
    best_set: set[tuple[int, int]] = set()
    best_faith = -float("inf")
    all_scores = []

    for _ in range(n_random):
        if len(candidate_heads) < k:
            break
        indices = rng.choice(len(candidate_heads), size=k, replace=False)
        trial_set = {candidate_heads[i] for i in indices}

        faith = evaluate_head_set_faithfulness(
            model, trial_set, prompts, correct_ids, incorrect_ids, mean_z)
        all_scores.append(faith)

        if faith > best_faith:
            best_faith = faith
            best_set = trial_set

    return best_set, best_faith, all_scores


def run_novel_analysis(
    model,
    tasks: list[str],
    n_prompts: int = 40,
    n_random_subsets: int = 50,
) -> list[EvalResult]:
    """Run arbitrage search for each task."""
    results = []
    tokenizer = model.tokenizer
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    all_heads = {(L, H) for L in range(n_layers) for H in range(n_heads)}

    for task in tasks:
        try:
            circuit_heads = get_circuit_heads(task)
            if not circuit_heads:
                continue

            k = len(circuit_heads)
            candidate_heads = sorted(all_heads - circuit_heads)

            if len(candidate_heads) < k:
                continue

            prompts = generate_prompts(task, tokenizer, n_prompts=n_prompts)
            if len(prompts) < 4:
                continue

            correct_ids, incorrect_ids = get_token_ids(prompts, tokenizer)
            if len(correct_ids) < 4:
                continue

            # Calibrate mean activations
            mean_z = calibrate_mean_z(model, prompts, n_calibration=min(20, len(prompts)))

            # Evaluate the actual circuit
            circuit_faith = evaluate_head_set_faithfulness(
                model, circuit_heads, prompts[:20], correct_ids[:20],
                incorrect_ids[:20], mean_z)

            print(f"    {task}: circuit faithfulness = {circuit_faith:.4f} ({k} heads)")

            # Greedy substitute search
            greedy_set, greedy_faith = greedy_substitute_search(
                model, candidate_heads, k, prompts[:20], correct_ids[:20],
                incorrect_ids[:20], mean_z)

            print(f"    {task}: greedy substitute faithfulness = {greedy_faith:.4f}")

            # Random substitute search
            _, random_best_faith, random_scores = random_substitute_search(
                model, candidate_heads, k, n_random_subsets, prompts[:20],
                correct_ids[:20], incorrect_ids[:20], mean_z)

            print(f"    {task}: best random substitute faithfulness = {random_best_faith:.4f}")

            # Best substitute fraction
            best_substitute_faith = max(greedy_faith, random_best_faith)
            if abs(circuit_faith) > 1e-8:
                best_fraction = best_substitute_faith / circuit_faith
            else:
                best_fraction = 0.0

            # Count near-substitutes (>= 90% of circuit performance)
            threshold_90 = 0.9 * circuit_faith
            n_near = sum(1 for s in random_scores if s >= threshold_90)
            if greedy_faith >= threshold_90:
                n_near += 1

            results.append(EvalResult(
                metric_id="C12.best_substitute_fraction",
                value=float(best_fraction),
                n_samples=n_random_subsets + 1,
                metadata={
                    "task": task,
                    "circuit_faithfulness": float(circuit_faith),
                    "greedy_faithfulness": float(greedy_faith),
                    "random_best_faithfulness": float(random_best_faith),
                    "circuit_size": k,
                    "n_candidates": len(candidate_heads),
                    "greedy_heads": sorted(greedy_set),
                    "arbitrage_exists": best_fraction >= 0.9,
                },
            ))

            results.append(EvalResult(
                metric_id="C12.n_near_substitutes",
                value=float(n_near),
                n_samples=n_random_subsets + 1,
                metadata={
                    "task": task,
                    "threshold": 0.9,
                    "n_random_tried": n_random_subsets,
                    "mean_random_faithfulness": float(np.mean(random_scores)) if random_scores else 0.0,
                    "std_random_faithfulness": float(np.std(random_scores)) if random_scores else 0.0,
                },
            ))

        except Exception as e:
            print(f"  [C12 novel analysis] {task} FAILED: {e}")

    return results


# ---------------------------------------------------------------------------
# Protocol runner
# ---------------------------------------------------------------------------

def run_protocol(model, tasks: list[str] | None = None, n_prompts: int = 40,
                 device: str = "cpu", run_cals: bool = True) -> ProtocolResult:
    """Run all C12 metrics + novel arbitrage search. Returns a ProtocolResult."""
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

    # Novel arbitrage search
    print(f"\n{'─' * 60}")
    print(f"  Arbitrage search novel analysis — {len(tasks)} tasks, {n_prompts} prompts")
    print(f"{'─' * 60}")
    mt0 = time.time()
    try:
        novel_results = run_novel_analysis(model, tasks, n_prompts=n_prompts)
        result.metrics["arbitrage"] = novel_results
        for r in novel_results:
            task = r.metadata.get("task", "?")
            print(f"    {task:20s}  {r.metric_id:40s}  {r.value:+.4f}")
        print(f"  {len(novel_results)} results in {time.time() - mt0:.1f}s")
    except Exception as e:
        print(f"  [arbitrage novel analysis] FAILED: {e}")
        result.metrics["arbitrage"] = []

    if run_cals:
        print(f"\n{'=' * 60}")
        print(f"  Calibrations ({len(CALIBRATIONS)})")
        print(f"{'=' * 60}")
        cal_tasks = tasks[:2]
        result.calibrations = run_calibrations(
            model, cal_tasks, CALIBRATIONS, n_prompts=n_prompts)

    result.elapsed_seconds = time.time() - t0
    return result


def arbitrage_analysis(result: ProtocolResult) -> list[str]:
    """Analyze results through the arbitrage (substitutability) lens.

    If an alternative set of k heads (same size as the circuit) achieves
    >= 90% of the circuit's faithfulness, then "arbitrage" exists: the
    circuit is not uniquely necessary. This challenges the construct
    validity of the circuit --- if many alternative sets work, the
    specific heads identified may not be the "real" circuit.

    Conversely, if no substitute comes close, the circuit is non-fungible:
    it is a privileged computational structure.

    Interpretation:
    - best_substitute_fraction < 0.5: STRONGLY NON-FUNGIBLE
    - 0.5 <= fraction < 0.9: MODERATELY NON-FUNGIBLE
    - 0.9 <= fraction < 1.1: ARBITRAGE EXISTS (substitutes available)
    - fraction > 1.1: CIRCUIT SUBOPTIMAL (substitute is better!)
    """
    lines = ["\n  Arbitrage Search Analysis:",
             "  --------------------------"]

    novel = result.metrics.get("arbitrage", [])

    for task in result.tasks:
        lines.append(f"\n    {task}:")

        ap_r = _find(result.metrics.get("activation_patching", []), task)
        frac_r = _find_by_metric_id(novel, "C12.best_substitute_fraction", task)
        nsub_r = _find_by_metric_id(novel, "C12.n_near_substitutes", task)

        if frac_r:
            if frac_r.value > 1.1:
                label = "circuit SUBOPTIMAL (substitute is better)"
            elif frac_r.value >= 0.9:
                label = "ARBITRAGE EXISTS"
            elif frac_r.value >= 0.5:
                label = "moderately non-fungible"
            else:
                label = "strongly non-fungible"
            lines.append(f"      Best substitute fraction: {frac_r.value:.4f} — {label}")
            meta = frac_r.metadata
            lines.append(f"        circuit faithfulness:   {meta.get('circuit_faithfulness', '?'):.4f}")
            lines.append(f"        greedy substitute:     {meta.get('greedy_faithfulness', '?'):.4f}")
            lines.append(f"        random best:           {meta.get('random_best_faithfulness', '?'):.4f}")
            lines.append(f"        circuit size:           {meta.get('circuit_size', '?')} heads")

        if nsub_r:
            lines.append(f"      Near-substitutes (>=90%): {int(nsub_r.value)} / "
                         f"{nsub_r.metadata.get('n_random_tried', '?') + 1}")
            lines.append(f"        mean random faith:     {nsub_r.metadata.get('mean_random_faithfulness', '?'):.4f}")

        if ap_r:
            label = "high" if ap_r.value > THRESHOLDS["activation_patching"] else "low"
            lines.append(f"      Activation patching:      {ap_r.value:.4f} — {label}")

        # Verdict
        if frac_r is None:
            verdict = "INSUFFICIENT DATA"
        elif frac_r.value > 1.1:
            verdict = "CIRCUIT SUBOPTIMAL — a better substitute set exists"
        elif frac_r.value >= 0.9:
            verdict = "FUNGIBLE — circuit has near-substitutes (construct validity concern)"
        elif frac_r.value >= 0.5:
            verdict = "MODERATELY UNIQUE — partial substitution possible"
        else:
            verdict = "NON-FUNGIBLE — no substitute achieves comparable performance"
        lines.append(f"      VERDICT: {verdict}")

    return lines


def _find(results: list[EvalResult], task: str) -> EvalResult | None:
    return next((r for r in results if r.metadata.get("task") == task), None)


def _find_by_metric_id(results: list[EvalResult], metric_id: str, task: str) -> EvalResult | None:
    return next((r for r in results
                 if r.metric_id == metric_id and r.metadata.get("task") == task), None)


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

    lines.extend(arbitrage_analysis(result))

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
    parser.add_argument("--n-random-subsets", type=int, default=50)
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
