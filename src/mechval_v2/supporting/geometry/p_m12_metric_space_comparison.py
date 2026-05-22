"""Protocol M12 --- Metric Space Comparison (Gromov-Hausdorff Distance)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Lens:         Geometry
Validity Type: Measurement
Framework:    Metric Geometry --- Cross-Model Activation Geometry
Family:       Gromov-Hausdorff Distance
Validity:     Construct --- reparameterization-invariant geometric comparison

References:
    Gromov (1981) "Groups of polynomial growth and expanding maps" ---
        Gromov-Hausdorff distance between metric spaces
    Memoli (2007) "On the use of Gromov-Hausdorff distances for shape
        matching" --- algorithmic approximations of GH distance
    Burago, Burago, Ivanov (2001) "A Course in Metric Geometry" ---
        textbook treatment of GH distance and its properties

Question:
    How similar is the activation geometry of circuit heads across two
    model sizes (gpt2 vs gpt2-medium)? The Gromov-Hausdorff distance
    compares the intrinsic geometry of two metric spaces without
    requiring them to live in the same ambient space. If the normalized
    GH distance is small, the circuit's activation geometry is preserved
    across model scales --- evidence of a scale-invariant computational
    structure.

Metrics:
    cka                --- Representational similarity (standard CKA)
    effect_size        --- Component importance (ablation effect)
    activation_patching --- Component-level causal importance

Calibrations:
    STRUCTURAL_CALIBRATIONS

Usage:
    uv run python metric_space_comparison.py                       # all tasks, CPU
    uv run python metric_space_comparison.py --device cuda          # GPU
    uv run python metric_space_comparison.py --tasks ioi induction  # specific tasks

    # As a callable module:
    from protocols.metric_space_comparison import run_protocol
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

from mechval.metrics.common import CIRCUIT_TASKS, EvalResult, get_circuit_heads, load_model

from protocols import ProtocolResult, import_metric_runner
from protocols.calibration_runner import STRUCTURAL_CALIBRATIONS, run_calibrations, summarize_calibrations

PROTOCOL_ID = "M12"
PROTOCOL_NAME = "Metric Space Comparison"
METRICS = ["cka", "effect_size", "activation_patching"]
CALIBRATIONS = STRUCTURAL_CALIBRATIONS
OUTPUT_DIR = Path(__file__).parent / "results" / "m12_metric_space_comparison"

THRESHOLDS = {
    "cka": 0.5,
    "effect_size": 0.8,
    "activation_patching": 0.5,
}


# ---------------------------------------------------------------------------
# Novel analysis: Gromov-Hausdorff distance between model activation spaces
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_circuit_activations(
    model,
    tokens: torch.Tensor,
    circuit_heads: set[tuple[int, int]],
) -> np.ndarray:
    """Collect circuit head activations for a batch of prompts.

    Returns shape ``(n_prompts, n_heads * d_head)``.
    """
    hook_names = [f"blocks.{L}.attn.hook_z" for L, _ in sorted(circuit_heads)]
    hook_names = list(dict.fromkeys(hook_names))  # deduplicate, preserve order

    _, cache = model.run_with_cache(tokens, names_filter=hook_names)

    parts = []
    for L, H in sorted(circuit_heads):
        z = cache[f"blocks.{L}.attn.hook_z"][:, -1, H, :]  # (N, d_head)
        parts.append(z.cpu().numpy())

    return np.concatenate(parts, axis=1)  # (N, n_heads * d_head)


def pairwise_l2_distances(X: np.ndarray) -> np.ndarray:
    """Compute pairwise L2 distance matrix for rows of X."""
    diff = X[:, None, :] - X[None, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=2))


def approximate_gromov_hausdorff(
    D1: np.ndarray,
    D2: np.ndarray,
    n_correspondences: int = 100,
) -> float:
    """Approximate Gromov-Hausdorff distance via greedy correspondence matching.

    Tries ``n_correspondences`` random correspondences and returns the minimum
    distortion. For equal-sized distance matrices of size n, a correspondence
    is a permutation mapping rows of D1 to rows of D2.

    GH distance ~ 0.5 * min over correspondences of max distortion.
    """
    n = min(D1.shape[0], D2.shape[0])
    if n == 0:
        return 0.0

    D1 = D1[:n, :n]
    D2 = D2[:n, :n]

    best_distortion = float("inf")

    # Try identity first
    distortion = np.max(np.abs(D1 - D2))
    best_distortion = min(best_distortion, distortion)

    rng = np.random.default_rng(42)
    for _ in range(n_correspondences):
        perm = rng.permutation(n)
        D2_perm = D2[perm][:, perm]
        distortion = np.max(np.abs(D1 - D2_perm))
        best_distortion = min(best_distortion, distortion)

    return 0.5 * best_distortion


def run_novel_analysis(
    model,
    tasks: list[str],
    n_prompts: int = 40,
    device: str = "cpu",
) -> list[EvalResult]:
    """Compare circuit activation geometry between gpt2 and gpt2-medium."""
    results = []

    # Load the second model (gpt2-medium)
    try:
        model2 = load_model("gpt2-medium", device)
    except Exception as e:
        print(f"  [M12 novel analysis] Failed to load gpt2-medium: {e}")
        for task in tasks:
            results.append(EvalResult(
                metric_id="M12.gromov_hausdorff_normalized",
                value=float("nan"),
                n_samples=0,
                metadata={"task": task, "error": str(e)},
            ))
        return results

    for task in tasks:
        try:
            heads_small = get_circuit_heads(task)
            heads_large = get_circuit_heads(task)

            if not heads_small or not heads_large:
                continue

            # Generate random tokens for activation collection
            vocab_size = model.cfg.d_vocab
            rng = np.random.default_rng(42)
            tokens_np = rng.integers(0, vocab_size, size=(n_prompts, 16))
            tokens = torch.tensor(tokens_np, device=device)

            # Collect activations from both models
            acts_small = collect_circuit_activations(model, tokens, heads_small)
            acts_large = collect_circuit_activations(model2, tokens, heads_large)

            # Compute pairwise L2 distance matrices within each model
            D_small = pairwise_l2_distances(acts_small)
            D_large = pairwise_l2_distances(acts_large)

            # Normalize both distance matrices by their respective diameters
            diam_small = np.max(D_small) if D_small.size > 0 else 1.0
            diam_large = np.max(D_large) if D_large.size > 0 else 1.0
            diam_max = max(diam_small, diam_large, 1e-8)

            D_small_norm = D_small / diam_max
            D_large_norm = D_large / diam_max

            # Approximate Gromov-Hausdorff distance
            gh_dist = approximate_gromov_hausdorff(D_small_norm, D_large_norm,
                                                   n_correspondences=100)

            results.append(EvalResult(
                metric_id="M12.gromov_hausdorff_normalized",
                value=float(gh_dist),
                n_samples=n_prompts,
                metadata={
                    "task": task,
                    "model_small": "gpt2",
                    "model_large": "gpt2-medium",
                    "n_circuit_heads_small": len(heads_small),
                    "n_circuit_heads_large": len(heads_large),
                    "diameter_small": float(diam_small),
                    "diameter_large": float(diam_large),
                },
            ))

            # Diameter ratio
            diameter_ratio = diam_small / max(diam_large, 1e-8)
            results.append(EvalResult(
                metric_id="M12.diameter_ratio",
                value=float(diameter_ratio),
                n_samples=n_prompts,
                metadata={
                    "task": task,
                    "diameter_small": float(diam_small),
                    "diameter_large": float(diam_large),
                },
            ))

        except Exception as e:
            print(f"  [M12 novel analysis] {task} FAILED: {e}")

    return results


# ---------------------------------------------------------------------------
# Protocol runner
# ---------------------------------------------------------------------------

def run_protocol(model, tasks: list[str] | None = None, n_prompts: int = 40,
                 device: str = "cpu", run_cals: bool = True) -> ProtocolResult:
    """Run all M12 metrics + novel GH analysis. Returns a ProtocolResult."""
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

    # Novel Gromov-Hausdorff analysis
    print(f"\n{'─' * 60}")
    print(f"  Gromov-Hausdorff novel analysis — {len(tasks)} tasks, {n_prompts} prompts")
    print(f"{'─' * 60}")
    mt0 = time.time()
    try:
        novel_results = run_novel_analysis(model, tasks, n_prompts=n_prompts, device=device)
        result.metrics["gromov_hausdorff"] = novel_results
        for r in novel_results:
            task = r.metadata.get("task", "?")
            print(f"    {task:20s}  {r.metric_id:40s}  {r.value:+.4f}")
        print(f"  {len(novel_results)} results in {time.time() - mt0:.1f}s")
    except Exception as e:
        print(f"  [gromov_hausdorff novel analysis] FAILED: {e}")
        result.metrics["gromov_hausdorff"] = []

    if run_cals:
        print(f"\n{'=' * 60}")
        print(f"  Calibrations ({len(CALIBRATIONS)})")
        print(f"{'=' * 60}")
        cal_tasks = tasks[:2]
        result.calibrations = run_calibrations(
            model, cal_tasks, CALIBRATIONS, n_prompts=n_prompts)

    result.elapsed_seconds = time.time() - t0
    return result


def metric_space_analysis(result: ProtocolResult) -> list[str]:
    """Analyze results through the metric space comparison lens.

    Gromov-Hausdorff distance measures how different two metric spaces are,
    without requiring an embedding into a common ambient space. Normalized
    by the larger diameter, values close to 0 mean the activation geometry
    is preserved across model scales; values close to 1 mean the geometry
    is completely different.

    Combined with CKA and activation_patching:
    - LOW GH + HIGH CKA: geometry and representations are preserved
    - LOW GH + LOW CKA: intrinsic distances match but representations differ
    - HIGH GH + HIGH CKA: representations similar despite different geometry
    - HIGH GH + LOW CKA: fundamentally different computations
    """
    lines = ["\n  Metric Space Comparison (Gromov-Hausdorff) Analysis:",
             "  ----------------------------------------------------"]

    novel = result.metrics.get("gromov_hausdorff", [])

    for task in result.tasks:
        lines.append(f"\n    {task}:")

        cka_r = _find(result.metrics.get("cka", []), task)
        ap_r = _find(result.metrics.get("activation_patching", []), task)
        gh_r = _find_by_metric_id(novel, "M12.gromov_hausdorff_normalized", task)
        dr_r = _find_by_metric_id(novel, "M12.diameter_ratio", task)

        if gh_r:
            if np.isnan(gh_r.value):
                lines.append(f"      GH distance: FAILED (could not load second model)")
            elif gh_r.value < 0.1:
                label = "geometry preserved across scales"
            elif gh_r.value < 0.3:
                label = "moderate geometric distortion"
            else:
                label = "substantial geometric difference"
            if not np.isnan(gh_r.value):
                lines.append(f"      GH distance (normalized):   {gh_r.value:.4f} — {label}")

        if dr_r and not np.isnan(dr_r.value):
            if dr_r.value > 1.5:
                label = "small model activations more spread"
            elif dr_r.value < 0.67:
                label = "large model activations more spread"
            else:
                label = "similar activation spread"
            lines.append(f"      Diameter ratio (sm/lg):      {dr_r.value:.4f} — {label}")

        if cka_r:
            label = "high similarity" if cka_r.value > THRESHOLDS["cka"] else "low similarity"
            lines.append(f"      CKA:                        {cka_r.value:.4f} — {label}")

        if ap_r:
            label = "high causal importance" if ap_r.value > THRESHOLDS["activation_patching"] else "low causal importance"
            lines.append(f"      Activation patching:        {ap_r.value:.4f} — {label}")

        # Verdict
        gh_low = gh_r is not None and not np.isnan(gh_r.value) and gh_r.value < 0.2
        cka_high = cka_r is not None and cka_r.value > THRESHOLDS["cka"]

        if gh_r is None or np.isnan(gh_r.value if gh_r else float("nan")):
            verdict = "INSUFFICIENT DATA — second model unavailable"
        elif gh_low and cka_high:
            verdict = "SCALE-INVARIANT CIRCUIT — geometry and representations preserved"
        elif gh_low and not cka_high:
            verdict = "INTRINSIC GEOMETRY PRESERVED — distances match despite representation change"
        elif not gh_low and cka_high:
            verdict = "REPRESENTATION PRESERVED — similar CKA despite geometric distortion"
        else:
            verdict = "SCALE-DEPENDENT — fundamentally different geometry at different scales"
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

    lines.extend(metric_space_analysis(result))

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
