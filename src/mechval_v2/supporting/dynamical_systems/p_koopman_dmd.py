"""Protocol WC_M8 --- Koopman Operator / Dynamic Mode Decomposition
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Lens:         Dynamical Systems
Validity Type: Construct
Framework:    Wildcard --- Dynamical Systems (Koopman Theory)
Family:       Koopman Operator / DMD
Validity:     Structural --- eigenvalue spectrum of residual stream dynamics

References:
    Koopman (1931) "Hamiltonian systems and transformation in Hilbert
        space" --- the original Koopman operator on observables
    Schmid (2010) "Dynamic mode decomposition of numerical and
        experimental data" --- the DMD algorithm used here
    NeurIPS 2025 "Replacing nonlinear MLP layers with Koopman
        operators" --- finite-dimensional Koopman for neural networks

Question:
    What are the natural dynamical modes of the residual stream as it
    evolves through layers? A transformer's forward pass is a discrete-
    time dynamical system: h_{l+1} = F(h_l). The Koopman operator
    linearizes this by acting on observable functions rather than states.
    Dynamic Mode Decomposition (DMD) approximates the Koopman operator
    from data.

    Eigenvalues near |lambda|=1 are persistent modes (information
    written once, passed through). Eigenvalues with |lambda|<1 are
    decaying modes (transient computations). Comparing eigenvalue
    spectra between clean and corrupted conditions reveals which
    dynamical modes carry task-relevant information.

Metrics:
    cka             --- Representational similarity across conditions
    effect_size     --- Overall circuit importance
    activation_patching --- Component-level causal importance

Calibrations:
    STRUCTURAL_CALIBRATIONS

Usage:
    uv run python koopman_dmd.py                       # all tasks, CPU
    uv run python koopman_dmd.py --device cuda          # GPU
    uv run python koopman_dmd.py --tasks ioi induction  # specific tasks

    # As a callable module:
    from protocols.wildcard.koopman_dmd import run_protocol
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

PROTOCOL_ID = "WC_M8"
PROTOCOL_NAME = "Koopman Operator / Dynamic Mode Decomposition"
METRICS = ["cka", "effect_size", "activation_patching"]
CALIBRATIONS = STRUCTURAL_CALIBRATIONS
OUTPUT_DIR = Path(__file__).parent / "results" / "wc_m8_koopman_dmd"

THRESHOLDS = {
    "cka": 0.5,
    "effect_size": 0.8,
    "activation_patching": 0.5,
}


# ---------------------------------------------------------------------------
# Novel analysis: Koopman / DMD of residual stream trajectories
# ---------------------------------------------------------------------------

def collect_residual_stream_trajectories(
    model,
    tokens: torch.Tensor,
    target_position: int = -1,
    hook_prefix: str = "blocks.{}.hook_resid_post",
) -> np.ndarray:
    """Collect residual stream vectors at *target_position* across all layers.

    Returns shape ``(N_prompts, n_layers, d_model)`` -- one trajectory per prompt.
    """
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    N = tokens.shape[0]

    hook_names = [hook_prefix.format(i) for i in range(n_layers)]

    trajectories = np.zeros((N, n_layers, d_model))

    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=hook_names)

    for layer_idx, hook_name in enumerate(hook_names):
        resid = cache[hook_name][:, target_position, :]  # (N, d_model)
        trajectories[:, layer_idx, :] = resid.cpu().numpy()

    return trajectories


def koopman_dmd(
    trajectories: np.ndarray,
    n_modes: int = 50,
) -> dict:
    """Dynamic Mode Decomposition on residual stream trajectories.

    Uses the Schmid (2010) SVD-based algorithm:
    1. Build snapshot matrices X (layers 0..L-2) and X' (layers 1..L-1).
    2. Thin SVD of X, truncate to *n_modes*.
    3. Project into reduced space, solve eigenproblem.
    4. Lift eigenvectors back, compute amplitudes and reconstruction error.

    Returns dict with eigenvalues, modes, amplitudes, reconstruction_error,
    and a per-mode interpretation.
    """
    N, L, d = trajectories.shape

    # Mean-over-prompts snapshot matrices
    X = trajectories[:, :-1, :].reshape(N, L - 1, -1)
    Xp = trajectories[:, 1:, :].reshape(N, L - 1, -1)

    X_mean = X.mean(0).T   # (d, L-1)
    Xp_mean = Xp.mean(0).T  # (d, L-1)

    # Thin SVD of X_mean
    U, S, Vh = np.linalg.svd(X_mean, full_matrices=False)
    V = Vh.T

    r = min(n_modes, len(S))
    U_r = U[:, :r]
    S_r = S[:r]
    V_r = V[:, :r]

    # Reduced operator
    A_tilde = U_r.T @ Xp_mean @ V_r @ np.diag(1.0 / S_r)

    eigenvalues, W = np.linalg.eig(A_tilde)

    # Lift eigenvectors (DMD modes)
    Phi = Xp_mean @ V_r @ np.diag(1.0 / S_r) @ W  # (d, r)

    # Amplitudes per prompt via projection
    x0_all = trajectories[:, 0, :]  # (N, d)
    phi_norm = np.sum(Phi * np.conj(Phi), axis=0)
    phi_norm = np.where(np.abs(phi_norm) < 1e-12, 1.0, phi_norm)
    amplitudes = (x0_all @ np.conj(Phi)) / phi_norm  # (N, r)

    # Reconstruction error
    X_recon = Phi @ np.diag(eigenvalues) @ np.linalg.pinv(Phi) @ X_mean
    error = float(np.linalg.norm(Xp_mean - X_recon) / (np.linalg.norm(Xp_mean) + 1e-8))

    # Classify each mode
    interpretation = {}
    for i, ev in enumerate(eigenvalues):
        mag = abs(ev)
        angle = np.angle(ev)
        if mag > 0.99 and abs(angle) < 0.05:
            kind = "persistent_real"
        elif mag > 0.99:
            kind = "persistent_oscillating"
        elif mag < 0.5:
            kind = "fast_decay"
        elif 0.5 <= mag < 0.9:
            kind = "slow_decay"
        else:
            kind = "intermediate"
        interpretation[i] = {
            "eigenvalue": complex(ev),
            "magnitude": float(mag),
            "phase_deg": float(np.degrees(angle)),
            "type": kind,
        }

    return {
        "eigenvalues": eigenvalues,
        "modes": Phi,
        "amplitudes": amplitudes,
        "reconstruction_error": error,
        "interpretation": interpretation,
        "n_modes": r,
    }


def koopman_circuit_scan(
    model,
    clean_tokens: torch.Tensor,
    corrupted_tokens: torch.Tensor,
    target_position: int = -1,
    n_modes: int = 30,
) -> dict:
    """Compare Koopman spectra between clean and corrupted conditions.

    Modes whose eigenvalues shift between conditions carry task-relevant
    information.  Modes with stable eigenvalues are task-irrelevant.
    """
    clean_traj = collect_residual_stream_trajectories(model, clean_tokens, target_position)
    corr_traj = collect_residual_stream_trajectories(model, corrupted_tokens, target_position)

    clean_dmd = koopman_dmd(clean_traj, n_modes=n_modes)
    corr_dmd = koopman_dmd(corr_traj, n_modes=n_modes)

    clean_evs = clean_dmd["eigenvalues"]
    corr_evs = corr_dmd["eigenvalues"]

    # Sort by magnitude for comparison
    clean_sorted = sorted(enumerate(clean_evs), key=lambda x: abs(x[1]), reverse=True)
    corr_sorted = sorted(enumerate(corr_evs), key=lambda x: abs(x[1]), reverse=True)

    n_compare = min(len(clean_sorted), len(corr_sorted))
    mode_shifts = []
    for idx in range(n_compare):
        ci, c_ev = clean_sorted[idx]
        ri, r_ev = corr_sorted[idx]
        shift = abs(c_ev - r_ev)
        mode_shifts.append({
            "clean_eigenvalue": complex(c_ev),
            "corrupted_eigenvalue": complex(r_ev),
            "eigenvalue_shift": float(shift),
            "magnitude_clean": float(abs(c_ev)),
            "is_circuit_relevant": shift > 0.05,
        })

    disrupted = sorted(mode_shifts, key=lambda x: -x["eigenvalue_shift"])

    return {
        "clean_dmd": clean_dmd,
        "corrupted_dmd": corr_dmd,
        "mode_shifts": mode_shifts,
        "most_disrupted_modes": disrupted[:10],
        "n_circuit_relevant_modes": sum(1 for m in mode_shifts if m["is_circuit_relevant"]),
    }


def run_novel_analysis(model, tasks: list[str], n_prompts: int = 40) -> list[EvalResult]:
    """Run DMD novel analysis, returning EvalResult objects for each task."""
    results = []
    for task in tasks:
        try:
            heads = get_circuit_heads(task)
            n_layers = model.cfg.n_layers

            # Generate simple prompts as random tokens for trajectory collection
            vocab_size = model.cfg.d_vocab
            clean_tokens = torch.randint(0, vocab_size, (n_prompts, 16),
                                         device=next(model.parameters()).device)
            traj = collect_residual_stream_trajectories(model, clean_tokens, target_position=-1)
            dmd_result = koopman_dmd(traj, n_modes=min(30, n_layers - 1))

            # Reconstruction error
            results.append(EvalResult(
                metric_id="WC_M8.koopman_reconstruction_error",
                value=dmd_result["reconstruction_error"],
                n_samples=n_prompts,
                metadata={"task": task, "n_modes": dmd_result["n_modes"]},
            ))

            # Count persistent modes (|lambda| > 0.99)
            evs = dmd_result["eigenvalues"]
            n_persistent = sum(1 for ev in evs if abs(ev) > 0.99)
            frac_persistent = n_persistent / max(len(evs), 1)
            results.append(EvalResult(
                metric_id="WC_M8.fraction_persistent_modes",
                value=frac_persistent,
                n_samples=n_prompts,
                metadata={
                    "task": task,
                    "n_persistent": n_persistent,
                    "n_total_modes": len(evs),
                },
            ))

            # Mean eigenvalue magnitude (spectral radius proxy)
            mean_mag = float(np.mean([abs(ev) for ev in evs]))
            results.append(EvalResult(
                metric_id="WC_M8.mean_eigenvalue_magnitude",
                value=mean_mag,
                n_samples=n_prompts,
                metadata={"task": task},
            ))

        except Exception as e:
            print(f"  [WC_M8 novel analysis] {task} FAILED: {e}")

    return results


# ---------------------------------------------------------------------------
# Protocol runner
# ---------------------------------------------------------------------------

def run_protocol(model, tasks: list[str] | None = None, n_prompts: int = 40,
                 device: str = "cpu", run_cals: bool = True) -> ProtocolResult:
    """Run all WC_M8 metrics + novel Koopman/DMD analysis. Returns a ProtocolResult."""
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

    # Novel Koopman/DMD analysis
    print(f"\n{'─' * 60}")
    print(f"  Koopman/DMD novel analysis — {len(tasks)} tasks, {n_prompts} prompts")
    print(f"{'─' * 60}")
    mt0 = time.time()
    try:
        novel_results = run_novel_analysis(model, tasks, n_prompts=n_prompts)
        result.metrics["koopman_dmd"] = novel_results
        for r in novel_results:
            task = r.metadata.get("task", "?")
            print(f"    {task:20s}  {r.metric_id:40s}  {r.value:+.4f}")
        print(f"  {len(novel_results)} results in {time.time() - mt0:.1f}s")
    except Exception as e:
        print(f"  [koopman_dmd novel analysis] FAILED: {e}")
        result.metrics["koopman_dmd"] = []

    if run_cals:
        print(f"\n{'=' * 60}")
        print(f"  Calibrations ({len(CALIBRATIONS)})")
        print(f"{'=' * 60}")
        cal_tasks = tasks[:2]
        result.calibrations = run_calibrations(
            model, cal_tasks, CALIBRATIONS, n_prompts=n_prompts)

    result.elapsed_seconds = time.time() - t0
    return result


def koopman_dmd_analysis(result: ProtocolResult) -> list[str]:
    """Analyze results through the Koopman / DMD lens.

    The Koopman operator linearizes the nonlinear residual stream dynamics
    by acting on observable functions. DMD approximates this operator from
    trajectory data. The eigenvalue spectrum classifies dynamical modes:

    - Persistent modes (|lambda| ~ 1): directions written once, passed
      through layers unchanged. These are "registers" in the residual stream.
    - Decaying modes (|lambda| < 1): transient computational intermediates
      that are actively consumed and overwritten.
    - Growing modes (|lambda| > 1): amplifying instabilities (rare in
      well-trained models).

    Reconstruction error measures how well the linear DMD approximation
    captures the full nonlinear dynamics. Low error means the layer-to-layer
    computation is nearly linear in the residual stream.

    Combined with CKA (representational similarity) and activation_patching
    (causal importance), we classify:
    - LINEAR DYNAMICS + HIGH CAUSAL IMPORTANCE: clean spectral circuit
    - NONLINEAR DYNAMICS + HIGH CAUSAL IMPORTANCE: complex computation
    - LINEAR DYNAMICS + LOW CAUSAL IMPORTANCE: passthrough / background
    - NONLINEAR DYNAMICS + LOW CAUSAL IMPORTANCE: noise
    """
    lines = ["\n  Koopman / DMD Analysis:", "  ----------------------"]

    novel = result.metrics.get("koopman_dmd", [])

    for task in result.tasks:
        lines.append(f"\n    {task}:")

        cka_r = _find(result.metrics.get("cka", []), task)
        es_r = _find(result.metrics.get("effect_size", []), task)
        ap_r = _find(result.metrics.get("activation_patching", []), task)

        recon = _find_by_metric_id(novel, "WC_M8.koopman_reconstruction_error", task)
        frac_persist = _find_by_metric_id(novel, "WC_M8.fraction_persistent_modes", task)
        mean_mag = _find_by_metric_id(novel, "WC_M8.mean_eigenvalue_magnitude", task)

        if recon:
            if recon.value < 0.1:
                label = "nearly linear dynamics"
            elif recon.value < 0.3:
                label = "moderately nonlinear"
            else:
                label = "strongly nonlinear"
            lines.append(f"      Reconstruction error:     {recon.value:.4f} — {label}")

        if frac_persist:
            lines.append(f"      Persistent modes:         {frac_persist.value:.4f} "
                         f"({frac_persist.metadata.get('n_persistent', '?')}/"
                         f"{frac_persist.metadata.get('n_total_modes', '?')})")

        if mean_mag:
            lines.append(f"      Mean eigenvalue |lambda|: {mean_mag.value:.4f}")

        if ap_r:
            label = "high causal importance" if ap_r.value > THRESHOLDS["activation_patching"] else "low causal importance"
            lines.append(f"      Activation patching:      {ap_r.value:.4f} — {label}")

        # Verdict
        linear = recon is not None and recon.value < 0.2
        causal = ap_r is not None and ap_r.value > THRESHOLDS["activation_patching"]

        if linear and causal:
            verdict = "CLEAN SPECTRAL CIRCUIT — linear dynamics with high causal importance"
        elif not linear and causal:
            verdict = "COMPLEX COMPUTATION — nonlinear dynamics, high causal importance"
        elif linear and not causal:
            verdict = "PASSTHROUGH — linear dynamics, low causal importance"
        elif recon is None or ap_r is None:
            verdict = "INSUFFICIENT DATA"
        else:
            verdict = "NOISE — nonlinear dynamics, low causal importance"
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

    lines.extend(koopman_dmd_analysis(result))

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
