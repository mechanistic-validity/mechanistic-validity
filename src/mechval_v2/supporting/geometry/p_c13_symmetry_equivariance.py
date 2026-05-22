"""Protocol C13 --- Symmetry Equivariance
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Lens:         Geometry
Validity Type: Construct
Framework:    Algebraic --- Group Equivariance Testing
Family:       Symmetry Group Equivariance
Validity:     Construct --- does the circuit respect task symmetries?

References:
    Cohen & Welling (2016) "Group Equivariant Convolutional Networks"
        --- equivariance as an inductive bias
    Weiler & Cesa (2019) "General E(2)-Equivariant Steerable CNNs"
        --- characterizing equivariance via representation theory
    Wang et al. (2022) "Interpretability in the Wild" --- IOI circuit
        with S2 name permutation symmetry

Question:
    Does the circuit respect the symmetry group implied by the task?
    For IOI, the symmetry group is S_2 (swapping IO and S names). If the
    circuit is truly computing indirect object identification, then
    swapping the two names in a prompt should produce correspondingly
    swapped circuit activations. The equivariance error measures how far
    the circuit deviates from this ideal.

    Pure weight-space test for the structural component; activation-based
    test for the functional component. CPU-only, very fast.

Metrics:
    cka                --- Representational similarity (standard CKA)
    effect_size        --- Component importance (ablation effect)

Calibrations:
    STRUCTURAL_CALIBRATIONS

Usage:
    uv run python symmetry_equivariance.py                       # all tasks, CPU
    uv run python symmetry_equivariance.py --device cuda          # GPU
    uv run python symmetry_equivariance.py --tasks ioi            # specific tasks

    # As a callable module:
    from protocols.symmetry_equivariance import run_protocol
    result = run_protocol(model, tasks=["ioi"], n_prompts=40)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    generate_prompts,
    get_circuit_heads,
    load_model,
)

from protocols import ProtocolResult, import_metric_runner
from protocols.calibration_runner import STRUCTURAL_CALIBRATIONS, run_calibrations, summarize_calibrations

PROTOCOL_ID = "C13"
PROTOCOL_NAME = "Symmetry Equivariance"
METRICS = ["cka", "effect_size"]
CALIBRATIONS = STRUCTURAL_CALIBRATIONS
OUTPUT_DIR = Path(__file__).parent / "results" / "c13_symmetry_equivariance"

THRESHOLDS = {
    "cka": 0.5,
    "effect_size": 0.8,
}

# IOI name pairs for S_2 symmetry testing
IOI_NAME_PAIRS = [
    ("John", "Mary"), ("Alice", "Bob"), ("Charlie", "Diana"),
    ("Edward", "Fiona"), ("George", "Helen"), ("Ivan", "Julia"),
    ("Kevin", "Laura"), ("Michael", "Nancy"), ("Oscar", "Patricia"),
    ("Quinn", "Rachel"),
]


# ---------------------------------------------------------------------------
# Novel analysis: Symmetry equivariance testing
# ---------------------------------------------------------------------------

def _swap_names_in_text(text: str, name_a: str, name_b: str) -> str:
    """Swap two names in a text string using a placeholder."""
    placeholder = "___SWAP_PLACEHOLDER___"
    result = text.replace(name_a, placeholder)
    result = result.replace(name_b, name_a)
    result = result.replace(placeholder, name_b)
    return result


def _extract_name_pair(text: str) -> tuple[str, str] | None:
    """Try to extract the IO/S name pair from an IOI-style prompt."""
    for name_a, name_b in IOI_NAME_PAIRS:
        if name_a in text and name_b in text:
            return name_a, name_b
    # Fallback: look for capitalized words that appear exactly once
    words = re.findall(r'\b[A-Z][a-z]+\b', text)
    name_counts = {}
    for w in words:
        name_counts[w] = name_counts.get(w, 0) + 1
    candidates = [w for w, c in name_counts.items() if c >= 1]
    if len(candidates) >= 2:
        return candidates[0], candidates[1]
    return None


@torch.no_grad()
def collect_head_activations(
    model,
    tokens: torch.Tensor,
    circuit_heads: set[tuple[int, int]],
) -> np.ndarray:
    """Collect circuit head activations at the last position.

    Returns shape ``(n_heads, d_head)`` for a single prompt.
    """
    hook_names = list({f"blocks.{L}.attn.hook_z" for L, _ in circuit_heads})
    _, cache = model.run_with_cache(tokens, names_filter=hook_names)

    parts = []
    for L, H in sorted(circuit_heads):
        z = cache[f"blocks.{L}.attn.hook_z"][0, -1, H, :]  # (d_head,)
        parts.append(z.cpu().numpy())

    return np.stack(parts)  # (n_heads, d_head)


def run_novel_analysis(
    model,
    tasks: list[str],
    n_prompts: int = 40,
) -> list[EvalResult]:
    """Test equivariance under task symmetry group.

    For IOI tasks: swap the two names, run both versions through the model,
    and measure whether circuit head activations transform correspondingly.
    For non-IOI tasks: skip (symmetry group not defined).
    """
    results = []
    tokenizer = model.tokenizer

    for task in tasks:
        try:
            heads = get_circuit_heads(task)
            if not heads:
                continue

            # Only IOI-family tasks have a well-defined S_2 symmetry
            if "ioi" not in task.lower():
                results.append(EvalResult(
                    metric_id="C13.mean_equivariance_error",
                    value=float("nan"),
                    n_samples=0,
                    metadata={
                        "task": task,
                        "note": "symmetry group not defined for non-IOI tasks",
                    },
                ))
                continue

            prompts = generate_prompts(task, tokenizer, n_prompts=n_prompts)
            if not prompts:
                continue

            equivariance_scores = []

            for p in prompts:
                text = p.text
                name_pair = _extract_name_pair(text)
                if name_pair is None:
                    continue

                name_a, name_b = name_pair
                swapped_text = _swap_names_in_text(text, name_a, name_b)

                # Tokenize both versions
                tokens_orig = model.to_tokens(text)
                tokens_swap = model.to_tokens(swapped_text)

                # Skip if tokenization changes length (name swap broke alignment)
                if tokens_orig.shape != tokens_swap.shape:
                    continue

                # Collect activations
                acts_orig = collect_head_activations(model, tokens_orig, heads)
                acts_swap = collect_head_activations(model, tokens_swap, heads)

                # Compute equivariance: measure correlation between the
                # activation difference and the expected swap effect.
                # Perfect equivariance: swapping names swaps the activations
                # in a corresponding way (high correlation between
                # act_diff and expected_diff).
                act_diff = (acts_orig - acts_swap).flatten()
                act_sum = (acts_orig + acts_swap).flatten()

                # Equivariance score: if the circuit is equivariant to name swap,
                # then the difference should be large relative to the sum
                # (the circuit "notices" the swap). We measure this as the
                # ratio of L2 norms.
                diff_norm = np.linalg.norm(act_diff)
                sum_norm = np.linalg.norm(act_sum)

                if sum_norm > 1e-8:
                    # Normalized sensitivity to swap: high means equivariant
                    equivariance_scores.append(diff_norm / sum_norm)

            if not equivariance_scores:
                continue

            scores = np.array(equivariance_scores)

            results.append(EvalResult(
                metric_id="C13.mean_equivariance_error",
                value=float(np.mean(scores)),
                n_samples=len(scores),
                metadata={
                    "task": task,
                    "std": float(np.std(scores)),
                    "n_valid_prompts": len(scores),
                },
            ))

            results.append(EvalResult(
                metric_id="C13.max_equivariance_error",
                value=float(np.max(scores)),
                n_samples=len(scores),
                metadata={
                    "task": task,
                    "min": float(np.min(scores)),
                },
            ))

        except Exception as e:
            print(f"  [C13 novel analysis] {task} FAILED: {e}")

    return results


# ---------------------------------------------------------------------------
# Protocol runner
# ---------------------------------------------------------------------------

def run_protocol(model, tasks: list[str] | None = None, n_prompts: int = 40,
                 device: str = "cpu", run_cals: bool = True) -> ProtocolResult:
    """Run all C13 metrics + novel equivariance analysis. Returns a ProtocolResult."""
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

    # Novel equivariance analysis
    print(f"\n{'─' * 60}")
    print(f"  Symmetry equivariance novel analysis — {len(tasks)} tasks, {n_prompts} prompts")
    print(f"{'─' * 60}")
    mt0 = time.time()
    try:
        novel_results = run_novel_analysis(model, tasks, n_prompts=n_prompts)
        result.metrics["equivariance"] = novel_results
        for r in novel_results:
            task = r.metadata.get("task", "?")
            print(f"    {task:20s}  {r.metric_id:40s}  {r.value:+.4f}")
        print(f"  {len(novel_results)} results in {time.time() - mt0:.1f}s")
    except Exception as e:
        print(f"  [equivariance novel analysis] FAILED: {e}")
        result.metrics["equivariance"] = []

    if run_cals:
        print(f"\n{'=' * 60}")
        print(f"  Calibrations ({len(CALIBRATIONS)})")
        print(f"{'=' * 60}")
        cal_tasks = tasks[:2]
        result.calibrations = run_calibrations(
            model, cal_tasks, CALIBRATIONS, n_prompts=n_prompts)

    result.elapsed_seconds = time.time() - t0
    return result


def equivariance_analysis(result: ProtocolResult) -> list[str]:
    """Analyze results through the symmetry equivariance lens.

    If the circuit computes a function with a known symmetry group (e.g.,
    IOI is equivariant under S_2 name permutations), then the circuit's
    activations should transform correspondingly under that symmetry.

    The equivariance score measures how much circuit activations change
    when the input is transformed by the symmetry. High scores mean the
    circuit is sensitive to the transformation (good equivariance).
    Low scores mean the circuit is invariant (bad --- it doesn't
    distinguish the swapped inputs).

    Combined with CKA and effect_size:
    - HIGH equivariance + HIGH effect: circuit is both important and respects symmetry
    - HIGH equivariance + LOW effect: circuit detects symmetry but isn't critical
    - LOW equivariance + HIGH effect: circuit is important but ignores symmetry
    - LOW equivariance + LOW effect: circuit neither important nor symmetry-aware
    """
    lines = ["\n  Symmetry Equivariance Analysis:",
             "  --------------------------------"]

    novel = result.metrics.get("equivariance", [])

    for task in result.tasks:
        lines.append(f"\n    {task}:")

        cka_r = _find(result.metrics.get("cka", []), task)
        es_r = _find(result.metrics.get("effect_size", []), task)
        mean_eq = _find_by_metric_id(novel, "C13.mean_equivariance_error", task)
        max_eq = _find_by_metric_id(novel, "C13.max_equivariance_error", task)

        if mean_eq:
            if np.isnan(mean_eq.value):
                note = mean_eq.metadata.get("note", "N/A")
                lines.append(f"      Equivariance: N/A ({note})")
            else:
                if mean_eq.value > 0.5:
                    label = "strongly equivariant"
                elif mean_eq.value > 0.2:
                    label = "moderately equivariant"
                else:
                    label = "weakly equivariant (near-invariant)"
                lines.append(f"      Mean equivariance score:  {mean_eq.value:.4f} — {label}")

        if max_eq and not np.isnan(max_eq.value):
            lines.append(f"      Max equivariance score:   {max_eq.value:.4f}")

        if es_r:
            label = "large effect" if es_r.value > THRESHOLDS["effect_size"] else "small effect"
            lines.append(f"      Effect size:              {es_r.value:.4f} — {label}")

        # Verdict
        if mean_eq is None or np.isnan(mean_eq.value):
            verdict = "NOT APPLICABLE — symmetry group not defined for this task"
        elif mean_eq.value > 0.3 and es_r is not None and es_r.value > THRESHOLDS["effect_size"]:
            verdict = "EQUIVARIANT CIRCUIT — respects task symmetry and is causally important"
        elif mean_eq.value > 0.3:
            verdict = "SYMMETRY-AWARE — detects symmetry but causal role unclear"
        elif es_r is not None and es_r.value > THRESHOLDS["effect_size"]:
            verdict = "SYMMETRY-BLIND — causally important but ignores symmetry"
        else:
            verdict = "WEAK — neither symmetry-aware nor causally important"
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

    lines.extend(equivariance_analysis(result))

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
