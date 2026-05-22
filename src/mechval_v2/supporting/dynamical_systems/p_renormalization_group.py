"""Protocol WC_M10 --- Renormalization Group (Multi-Scale Coarse-Graining)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Lens:         Dynamical Systems
Validity Type: Construct
Framework:    Wildcard --- Statistical Physics (Wilson RG)
Family:       Renormalization Group Flow
Validity:     Structural --- scale-invariant directions and beta functions

References:
    Wilson (1971) "Renormalization group and critical phenomena"
        --- the RG framework (Nobel Prize 1982)
    Kadanoff (1966) "Scaling laws for Ising models near T_c"
        --- block-spin renormalization that inspired Wilson
    Mehta & Schwab (2014) "An exact mapping between the variational
        renormalization group and deep learning" --- RG/DL connection

Question:
    Does the residual stream exhibit scale-invariant structure across
    layers? The RG procedure coarse-grains adjacent layers and asks
    what effective dynamics emerge at each scale. Fixed-point directions
    (eigenvalue ~ 1 at all scales) are scale-invariant features of the
    circuit. Beta functions (log(eigenvalue)/log(2)) classify directions
    as relevant (growing), irrelevant (decaying), or marginal (fixed).

    If two different tasks share fixed-point directions, they share a
    common computational substrate --- a "universality class" in RG terms.

Metrics:
    cka                       --- Representational similarity between scales
    cross_task_generalization  --- Do scale-invariant features transfer?
    effect_size               --- Overall circuit importance

Calibrations:
    STRUCTURAL_CALIBRATIONS

Usage:
    uv run python renormalization_group.py                       # all tasks, CPU
    uv run python renormalization_group.py --device cuda          # GPU
    uv run python renormalization_group.py --tasks ioi induction  # specific tasks

    # As a callable module:
    from protocols.wildcard.renormalization_group import run_protocol
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

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from mechval.metrics.common import CIRCUIT_TASKS, EvalResult, get_circuit_heads, load_model

from protocols import ProtocolResult, import_metric_runner
from protocols.calibration_runner import STRUCTURAL_CALIBRATIONS, run_calibrations, summarize_calibrations

PROTOCOL_ID = "WC_M10"
PROTOCOL_NAME = "Renormalization Group (Multi-Scale Coarse-Graining)"
METRICS = ["cka", "cross_task_generalization", "effect_size"]
CALIBRATIONS = STRUCTURAL_CALIBRATIONS
OUTPUT_DIR = Path(__file__).parent / "results" / "wc_m10_renormalization_group"

THRESHOLDS = {
    "cka": 0.5,
    "cross_task_generalization": 0.5,
    "effect_size": 0.8,
}


# ---------------------------------------------------------------------------
# Novel analysis: RG flow of residual stream across layer scales
# ---------------------------------------------------------------------------

def collect_residual_stream_trajectories(
    model,
    tokens: torch.Tensor,
    target_position: int = -1,
) -> np.ndarray:
    """Collect residual stream at *target_position* across all layers.

    Returns shape ``(N_prompts, n_layers, d_model)``.
    """
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    N = tokens.shape[0]

    hook_names = [f"blocks.{i}.hook_resid_post" for i in range(n_layers)]
    trajectories = np.zeros((N, n_layers, d_model))

    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=hook_names)

    for layer_idx, hook_name in enumerate(hook_names):
        resid = cache[hook_name][:, target_position, :]
        trajectories[:, layer_idx, :] = resid.cpu().numpy()

    return trajectories


def rg_flow_analysis(
    trajectories: np.ndarray,
    n_rg_scales: int = 4,
    n_directions: int = 20,
) -> dict:
    """Renormalization Group analysis of the residual stream.

    Coarse-grains the layer sequence by successively averaging adjacent
    layers. At each scale, finds the principal directions of change via
    least-squares on the transition operator, then computes eigenvalues
    and beta functions.

    Scale 0: all L layers (finest).
    Scale 1: average pairs -> L/2 super-layers.
    Scale 2: average pairs of super-layers -> L/4.
    ...

    Beta function: log(|eigenvalue|) / log(2).
    - beta > 0: relevant direction (growing under coarse-graining).
    - beta ~ 0: marginal / fixed-point direction (scale-invariant).
    - beta < 0: irrelevant direction (decaying).
    """
    N, L, d = trajectories.shape

    def coarse_grain(traj):
        """Average adjacent layer pairs."""
        n = traj.shape[1]
        if n <= 1:
            return traj
        pairs = n // 2
        coarsened = (traj[:, ::2, :][:, :pairs, :] + traj[:, 1::2, :][:, :pairs, :]) / 2.0
        return coarsened

    results_per_scale = {}
    current = trajectories.copy()

    for scale in range(n_rg_scales):
        n_current = current.shape[1]
        if n_current < 2:
            break

        X = current[:, :-1, :].reshape(-1, d)
        Xp = current[:, 1:, :].reshape(-1, d)

        # Least-squares transition operator: Xp ~ X @ A^T
        A, _, _, _ = np.linalg.lstsq(X, Xp, rcond=None)  # (d, d)

        eigenvalues, eigenvectors = np.linalg.eig(A.T)

        n_dir = min(n_directions, len(eigenvalues))
        growth_rates = np.real(eigenvalues[:n_dir])

        fp_closeness = np.abs(eigenvalues - 1.0)
        sorted_idx = np.argsort(fp_closeness)

        beta_fns = np.log(np.abs(eigenvalues) + 1e-8) / np.log(2.0)

        results_per_scale[scale] = {
            "n_layers_at_this_scale": n_current,
            "eigenvalues": eigenvalues[:n_dir],
            "eigenvectors": np.real(eigenvectors[:, :n_dir]),
            "growth_rates": growth_rates,
            "beta_functions": beta_fns[:n_dir],
            "most_fixed_point_like": [
                {
                    "direction_idx": int(sorted_idx[i]),
                    "eigenvalue": complex(eigenvalues[sorted_idx[i]]),
                    "beta_fn": float(beta_fns[sorted_idx[i]]),
                    "type": (
                        "fixed_point" if abs(eigenvalues[sorted_idx[i]] - 1.0) < 0.05
                        else "relevant" if abs(eigenvalues[sorted_idx[i]]) > 1.05
                        else "irrelevant"
                    ),
                }
                for i in range(min(10, len(sorted_idx)))
            ],
        }

        current = coarse_grain(current)

    # Identify directions that are fixed points at ALL scales
    fixed_point_directions = []
    n_scales = len(results_per_scale)
    if n_scales > 1:
        n_check = min(n_directions, len(results_per_scale[0]["beta_functions"]))
        for i in range(n_check):
            beta_values = []
            for s in range(n_scales):
                betas = results_per_scale[s]["beta_functions"]
                if i < len(betas):
                    beta_values.append(float(betas[i]))
            if beta_values and all(abs(b) < 0.1 for b in beta_values):
                fixed_point_directions.append({
                    "direction_idx": i,
                    "beta_values_per_scale": beta_values,
                })

    return {
        "per_scale": results_per_scale,
        "fixed_point_directions": fixed_point_directions,
        "n_fixed_points": len(fixed_point_directions),
        "n_scales_computed": n_scales,
    }


def run_novel_analysis(model, tasks: list[str], n_prompts: int = 40) -> list[EvalResult]:
    """Run RG flow novel analysis, returning EvalResults per task."""
    results = []

    for task in tasks:
        try:
            vocab_size = model.cfg.d_vocab
            tokens = torch.randint(0, vocab_size, (n_prompts, 16),
                                   device=next(model.parameters()).device)

            traj = collect_residual_stream_trajectories(model, tokens, target_position=-1)
            rg = rg_flow_analysis(traj, n_rg_scales=4, n_directions=20)

            # Number of fixed-point directions
            results.append(EvalResult(
                metric_id="WC_M10.n_fixed_point_directions",
                value=float(rg["n_fixed_points"]),
                n_samples=n_prompts,
                metadata={"task": task, "n_scales": rg["n_scales_computed"]},
            ))

            # Mean absolute beta function at finest scale (how non-trivial the RG flow is)
            if 0 in rg["per_scale"]:
                betas = rg["per_scale"][0]["beta_functions"]
                mean_abs_beta = float(np.mean(np.abs(betas)))
                results.append(EvalResult(
                    metric_id="WC_M10.mean_abs_beta_function",
                    value=mean_abs_beta,
                    n_samples=n_prompts,
                    metadata={"task": task},
                ))

                # Fraction of relevant directions (beta > 0 => growing)
                frac_relevant = float(np.mean(np.real(betas) > 0))
                results.append(EvalResult(
                    metric_id="WC_M10.fraction_relevant_directions",
                    value=frac_relevant,
                    n_samples=n_prompts,
                    metadata={"task": task},
                ))

        except Exception as e:
            print(f"  [WC_M10 novel analysis] {task} FAILED: {e}")

    return results


# ---------------------------------------------------------------------------
# Protocol runner
# ---------------------------------------------------------------------------

def run_protocol(model, tasks: list[str] | None = None, n_prompts: int = 40,
                 device: str = "cpu", run_cals: bool = True) -> ProtocolResult:
    """Run all WC_M10 metrics + novel RG analysis. Returns a ProtocolResult."""
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

    # Novel RG analysis
    print(f"\n{'─' * 60}")
    print(f"  RG flow novel analysis — {len(tasks)} tasks, {n_prompts} prompts")
    print(f"{'─' * 60}")
    mt0 = time.time()
    try:
        novel_results = run_novel_analysis(model, tasks, n_prompts=n_prompts)
        result.metrics["renormalization_group"] = novel_results
        for r in novel_results:
            task = r.metadata.get("task", "?")
            print(f"    {task:20s}  {r.metric_id:40s}  {r.value:+.4f}")
        print(f"  {len(novel_results)} results in {time.time() - mt0:.1f}s")
    except Exception as e:
        print(f"  [renormalization_group novel analysis] FAILED: {e}")
        result.metrics["renormalization_group"] = []

    if run_cals:
        print(f"\n{'=' * 60}")
        print(f"  Calibrations ({len(CALIBRATIONS)})")
        print(f"{'=' * 60}")
        cal_tasks = tasks[:2]
        result.calibrations = run_calibrations(
            model, cal_tasks, CALIBRATIONS, n_prompts=n_prompts)

    result.elapsed_seconds = time.time() - t0
    return result


def renormalization_group_analysis(result: ProtocolResult) -> list[str]:
    """Analyze results through the Renormalization Group lens.

    The RG procedure coarse-grains adjacent layers and tracks how
    the effective dynamics change across scales. At each scale, the
    transition operator's eigenvalues reveal:

    - Fixed-point directions (|lambda| ~ 1 at all scales): scale-invariant
      features that the model computes identically at every level of
      abstraction. If two tasks share fixed-point directions, they share
      a common computational substrate (same "universality class").
    - Relevant directions (beta > 0): actively written by the circuit,
      growing under coarse-graining.
    - Irrelevant directions (beta < 0): suppressed, not used at coarser
      scales.

    Combined with CKA and cross-task generalization:
    - UNIVERSAL: many fixed points + high cross-task generalization
      = shared computational substrate across tasks
    - TASK-SPECIFIC: few fixed points + low generalization
      = computation is specific to this task at every scale
    - SCALE-INVARIANT: many fixed points + low generalization
      = scale-invariant but task-specific geometry
    - SCALE-DEPENDENT: few fixed points + high generalization
      = task-general but scale-dependent
    """
    lines = ["\n  Renormalization Group Analysis:", "  ------------------------------"]

    novel = result.metrics.get("renormalization_group", [])

    for task in result.tasks:
        lines.append(f"\n    {task}:")

        cka_r = _find(result.metrics.get("cka", []), task)
        ctg_r = _find(result.metrics.get("cross_task_generalization", []), task)
        es_r = _find(result.metrics.get("effect_size", []), task)

        n_fp = _find_by_metric_id(novel, "WC_M10.n_fixed_point_directions", task)
        mean_beta = _find_by_metric_id(novel, "WC_M10.mean_abs_beta_function", task)
        frac_rel = _find_by_metric_id(novel, "WC_M10.fraction_relevant_directions", task)

        if n_fp:
            if n_fp.value > 5:
                label = "many scale-invariant features"
            elif n_fp.value > 0:
                label = "some scale-invariant features"
            else:
                label = "no scale-invariant features"
            lines.append(f"      Fixed-point directions:   {n_fp.value:.0f} — {label}")

        if mean_beta:
            if mean_beta.value < 0.05:
                label = "nearly scale-invariant flow"
            elif mean_beta.value < 0.2:
                label = "moderate RG flow"
            else:
                label = "strong RG flow (large scale-dependence)"
            lines.append(f"      Mean |beta function|:     {mean_beta.value:.4f} — {label}")

        if frac_rel:
            lines.append(f"      Fraction relevant dirs:   {frac_rel.value:.4f}")

        if ctg_r:
            label = "transfers across tasks" if ctg_r.value > THRESHOLDS["cross_task_generalization"] else "task-specific"
            lines.append(f"      Cross-task generalization: {ctg_r.value:.4f} — {label}")

        # Verdict
        many_fp = n_fp is not None and n_fp.value > 3
        generalizes = ctg_r is not None and ctg_r.value > THRESHOLDS["cross_task_generalization"]

        if many_fp and generalizes:
            verdict = "UNIVERSAL — shared scale-invariant computational substrate"
        elif many_fp and not generalizes:
            verdict = "SCALE-INVARIANT — stable geometry but task-specific"
        elif not many_fp and generalizes:
            verdict = "SCALE-DEPENDENT — task-general but varies across scales"
        elif n_fp is None or ctg_r is None:
            verdict = "INSUFFICIENT DATA"
        else:
            verdict = "TASK-SPECIFIC — computation is unique at every scale"
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

    lines.extend(renormalization_group_analysis(result))

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
