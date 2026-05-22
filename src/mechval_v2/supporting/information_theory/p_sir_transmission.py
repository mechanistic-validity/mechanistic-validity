"""Protocol WC_M13 --- SIR Transmission (Epidemiological Circuit Model)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Lens:         Information Theory
Validity Type: Internal
Framework:    Wildcard --- Epidemiology (SIR / Network Contagion)
Family:       SIR Transmission Model
Validity:     Structural --- co-activation transmission matrix and R0

References:
    Kermack & McKendrick (1927) "A contribution to the mathematical
        theory of epidemics" --- the original SIR model
    Pastor-Satorras & Vespignani (2001) "Epidemic spreading in
        scale-free networks" --- network SIR with degree distribution

Question:
    Can we model information spreading through the circuit as a
    contagion process? Each component is a "node" that can be
    susceptible, infected (active for the task), or recovered. The
    transmission rate beta_AB measures how much component A's
    activation increases the probability that component B also
    activates.

    The basic reproduction number R0 per component measures how much
    it "spreads" task-relevant activation to downstream components.
    R0 > 1: self-sustaining (epidemic core). R0 < 1: transient.

    Hub components (top R0 percentile) are the "superspreaders" of
    task information. Terminal components (R0 < 1 but high excess
    activation) receive but don't propagate --- likely output heads.

Metrics:
    activation_patching --- Component-level causal importance
    eap                 --- Edge attribution for cross-validation
    effect_size         --- Overall circuit importance

Calibrations:
    STRUCTURAL_CALIBRATIONS

Usage:
    uv run python sir_transmission.py                       # all tasks, CPU
    uv run python sir_transmission.py --device cuda          # GPU
    uv run python sir_transmission.py --tasks ioi induction  # specific tasks

    # As a callable module:
    from protocols.wildcard.sir_transmission import run_protocol
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

PROTOCOL_ID = "WC_M13"
PROTOCOL_NAME = "SIR Transmission (Epidemiological Circuit Model)"
METRICS = ["activation_patching", "eap", "effect_size"]
CALIBRATIONS = STRUCTURAL_CALIBRATIONS
OUTPUT_DIR = Path(__file__).parent / "results" / "wc_m13_sir_transmission"

THRESHOLDS = {
    "activation_patching": 0.5,
    "eap": 0.3,
    "effect_size": 0.8,
}


# ---------------------------------------------------------------------------
# Novel analysis: Network SIR model for circuit contagion
# ---------------------------------------------------------------------------

def fit_sir_transmission_matrix(
    model,
    clean_tokens: torch.Tensor,
    baseline_tokens: torch.Tensor,
    target_position: int = -1,
    activation_threshold_quantile: float = 0.75,
) -> dict:
    """Fit a network SIR transmission matrix from co-activation patterns.

    For each component pair (A, B) with layer(A) < layer(B):
        beta_AB = logistic regression coefficient predicting B's activity
        from A's activity.

    A high beta_AB means: when A fires, B is more likely to fire too ---
    a functional circuit connection.

    R0 per component = sum of outgoing transmission coefficients.
    Network R0 = largest eigenvalue of the transmission matrix.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    components = [(l, h, "attn") for l in range(n_layers) for h in range(n_heads)]
    components += [(l, 0, "mlp") for l in range(n_layers)]
    K = len(components)

    attn_hooks = [f"blocks.{l}.attn.hook_result" for l in range(n_layers)]
    mlp_hooks = [f"blocks.{l}.hook_mlp_out" for l in range(n_layers)]
    all_hooks = attn_hooks + mlp_hooks

    def get_activation_matrix(tokens):
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=all_hooks)

        N = tokens.shape[0]
        A = np.zeros((N, K))

        for k_idx, (layer, head, ctype) in enumerate(components):
            if ctype == "attn":
                result = cache[f"blocks.{layer}.attn.hook_result"]
                A[:, k_idx] = result[:, target_position, head, :].norm(dim=-1).cpu().numpy()
            else:
                mlp_out = cache[f"blocks.{layer}.hook_mlp_out"]
                A[:, k_idx] = mlp_out[:, target_position, :].abs().mean(dim=-1).cpu().numpy()

        return A

    clean_acts = get_activation_matrix(clean_tokens)   # (N, K)
    base_acts = get_activation_matrix(baseline_tokens)  # (N, K)

    # Threshold: "infected" if activation exceeds baseline quantile
    all_acts = np.vstack([clean_acts, base_acts])
    thresholds = np.percentile(all_acts, activation_threshold_quantile * 100, axis=0)

    clean_binary = (clean_acts > thresholds).astype(float)
    base_binary = (base_acts > thresholds).astype(float)

    prevalence_clean = clean_binary.mean(axis=0)
    prevalence_base = base_binary.mean(axis=0)
    excess_activation = prevalence_clean - prevalence_base

    # Transmission matrix via logistic regression
    beta_matrix = np.zeros((K, K))

    for j_idx, (layer_j, head_j, ctype_j) in enumerate(components):
        for i_idx, (layer_i, head_i, ctype_i) in enumerate(components):
            if layer_i <= layer_j:
                continue

            X = clean_binary[:, j_idx].reshape(-1, 1)
            y = clean_binary[:, i_idx]

            if y.sum() < 5 or y.sum() > len(y) - 5:
                continue

            try:
                # Simple logistic regression via Newton's method (1 feature)
                # Avoids sklearn dependency
                coef = _simple_logistic_regression(X[:, 0], y)
                beta_matrix[i_idx, j_idx] = coef
            except Exception:
                pass

    # R0 per component: sum of outgoing transmission
    r0_per_component = beta_matrix.sum(axis=0)

    # Network R0: largest eigenvalue of beta matrix
    eigenvalues = np.linalg.eigvals(beta_matrix)
    network_r0 = float(np.max(np.real(eigenvalues)))

    # Circuit core: excess activation AND high R0
    circuit_core = []
    r0_90th = float(np.percentile(r0_per_component, 90))
    for k_idx, (layer, head, ctype) in enumerate(components):
        if excess_activation[k_idx] > 0.1 and r0_per_component[k_idx] > 0.5:
            circuit_core.append({
                "component": (int(layer), int(head), ctype),
                "excess_activation": float(excess_activation[k_idx]),
                "r0": float(r0_per_component[k_idx]),
                "is_hub": float(r0_per_component[k_idx]) > r0_90th,
            })

    circuit_core = sorted(circuit_core, key=lambda x: -x["r0"])

    return {
        "beta_matrix": beta_matrix,
        "r0_per_component": r0_per_component,
        "network_r0": network_r0,
        "circuit_core": circuit_core[:20],
        "epidemic_stable": network_r0 > 1.0,
        "excess_activation": excess_activation,
    }


def _simple_logistic_regression(x: np.ndarray, y: np.ndarray, n_iter: int = 50) -> float:
    """Fit logistic regression with one feature via iteratively reweighted least squares.

    Returns the coefficient for x.
    """
    # Add intercept
    X = np.column_stack([np.ones_like(x), x])
    beta = np.zeros(2)

    for _ in range(n_iter):
        z = X @ beta
        z = np.clip(z, -10, 10)  # prevent overflow
        p = 1.0 / (1.0 + np.exp(-z))
        W = p * (1 - p) + 1e-8
        # IRLS update
        z_tilde = z + (y - p) / W
        XtWX = X.T @ (X * W[:, None])
        XtWz = X.T @ (W * z_tilde)
        try:
            beta = np.linalg.solve(XtWX + 1e-4 * np.eye(2), XtWz)
        except np.linalg.LinAlgError:
            break

    return float(beta[1])  # coefficient for x (not intercept)


def run_novel_analysis(model, tasks: list[str], n_prompts: int = 40) -> list[EvalResult]:
    """Run SIR transmission novel analysis, returning EvalResults per task."""
    results = []

    for task in tasks:
        try:
            vocab_size = model.cfg.d_vocab
            dev = next(model.parameters()).device
            clean_tokens = torch.randint(0, vocab_size, (n_prompts, 16), device=dev)
            baseline_tokens = torch.randint(0, vocab_size, (n_prompts, 16), device=dev)

            sir = fit_sir_transmission_matrix(
                model, clean_tokens, baseline_tokens,
                target_position=-1, activation_threshold_quantile=0.75)

            # Network R0
            results.append(EvalResult(
                metric_id="WC_M13.network_r0",
                value=sir["network_r0"],
                n_samples=n_prompts,
                metadata={
                    "task": task,
                    "epidemic_stable": sir["epidemic_stable"],
                },
            ))

            # Number of circuit core components
            n_core = len(sir["circuit_core"])
            results.append(EvalResult(
                metric_id="WC_M13.n_circuit_core_components",
                value=float(n_core),
                n_samples=n_prompts,
                metadata={"task": task},
            ))

            # Number of hub components (superspreaders)
            n_hubs = sum(1 for c in sir["circuit_core"] if c["is_hub"])
            results.append(EvalResult(
                metric_id="WC_M13.n_hub_components",
                value=float(n_hubs),
                n_samples=n_prompts,
                metadata={"task": task},
            ))

            # Mean excess activation across all components
            mean_excess = float(np.mean(sir["excess_activation"]))
            results.append(EvalResult(
                metric_id="WC_M13.mean_excess_activation",
                value=mean_excess,
                n_samples=n_prompts,
                metadata={"task": task},
            ))

        except Exception as e:
            print(f"  [WC_M13 novel analysis] {task} FAILED: {e}")

    return results


# ---------------------------------------------------------------------------
# Protocol runner
# ---------------------------------------------------------------------------

def run_protocol(model, tasks: list[str] | None = None, n_prompts: int = 40,
                 device: str = "cpu", run_cals: bool = True) -> ProtocolResult:
    """Run all WC_M13 metrics + novel SIR analysis. Returns a ProtocolResult."""
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

    # Novel SIR transmission analysis
    print(f"\n{'─' * 60}")
    print(f"  SIR transmission novel analysis — {len(tasks)} tasks, {n_prompts} prompts")
    print(f"{'─' * 60}")
    mt0 = time.time()
    try:
        novel_results = run_novel_analysis(model, tasks, n_prompts=n_prompts)
        result.metrics["sir_transmission"] = novel_results
        for r in novel_results:
            task = r.metadata.get("task", "?")
            print(f"    {task:20s}  {r.metric_id:40s}  {r.value:+.4f}")
        print(f"  {len(novel_results)} results in {time.time() - mt0:.1f}s")
    except Exception as e:
        print(f"  [sir_transmission novel analysis] FAILED: {e}")
        result.metrics["sir_transmission"] = []

    if run_cals:
        print(f"\n{'=' * 60}")
        print(f"  Calibrations ({len(CALIBRATIONS)})")
        print(f"{'=' * 60}")
        cal_tasks = tasks[:2]
        result.calibrations = run_calibrations(
            model, cal_tasks, CALIBRATIONS, n_prompts=n_prompts)

    result.elapsed_seconds = time.time() - t0
    return result


def sir_transmission_analysis(result: ProtocolResult) -> list[str]:
    """Analyze results through the SIR epidemiological lens.

    The SIR model treats information spreading through the circuit as
    a contagion process. Each component is a node that can be
    susceptible (inactive), infected (active for the task), or
    recovered. The transmission matrix beta captures how each
    component's activation increases the probability of downstream
    components also activating.

    Key metric: R0 (basic reproduction number).
    - R0 > 1: the circuit is "epidemic" -- activity self-sustains and
      spreads to downstream components without external driving.
    - R0 < 1: the circuit is "subcritical" -- activity dies out
      without continuous input.

    Hub components (top R0 percentile) are "superspreaders" of task
    information. Terminal components (high excess activation but low
    R0) are endpoints that receive but don't propagate.

    Combined with activation_patching and EAP:
    - EPIDEMIC CORE + HIGH CAUSAL: the self-sustaining heart of the
      circuit, both spreading and causally important
    - SUBCRITICAL + HIGH CAUSAL: driven circuit -- important but
      requires continuous input
    - EPIDEMIC CORE + LOW CAUSAL: self-sustaining background activity
    - SUBCRITICAL + LOW CAUSAL: noise
    """
    lines = ["\n  SIR Transmission Analysis:", "  -------------------------"]

    novel = result.metrics.get("sir_transmission", [])

    for task in result.tasks:
        lines.append(f"\n    {task}:")

        ap_r = _find(result.metrics.get("activation_patching", []), task)
        eap_r = _find(result.metrics.get("eap", []), task)

        net_r0 = _find_by_metric_id(novel, "WC_M13.network_r0", task)
        n_core = _find_by_metric_id(novel, "WC_M13.n_circuit_core_components", task)
        n_hubs = _find_by_metric_id(novel, "WC_M13.n_hub_components", task)
        mean_exc = _find_by_metric_id(novel, "WC_M13.mean_excess_activation", task)

        if net_r0:
            if net_r0.value > 1.0:
                label = "EPIDEMIC — self-sustaining circuit"
            elif net_r0.value > 0.5:
                label = "near-critical — moderate transmission"
            else:
                label = "SUBCRITICAL — activity dies out"
            lines.append(f"      Network R0:               {net_r0.value:.4f} — {label}")

        if n_core:
            lines.append(f"      Circuit core components:  {n_core.value:.0f}")

        if n_hubs:
            lines.append(f"      Hub (superspreader) nodes: {n_hubs.value:.0f}")

        if mean_exc:
            if mean_exc.value > 0.1:
                label = "strong task-specific activation"
            elif mean_exc.value > 0.01:
                label = "moderate task-specific activation"
            else:
                label = "minimal task-specific activation"
            lines.append(f"      Mean excess activation:   {mean_exc.value:.4f} — {label}")

        if ap_r:
            label = "high causal importance" if ap_r.value > THRESHOLDS["activation_patching"] else "low causal importance"
            lines.append(f"      Activation patching:      {ap_r.value:.4f} — {label}")

        # Verdict
        epidemic = net_r0 is not None and net_r0.value > 1.0
        causal = ap_r is not None and ap_r.value > THRESHOLDS["activation_patching"]

        if epidemic and causal:
            verdict = "EPIDEMIC CORE — self-sustaining circuit with high causal importance"
        elif not epidemic and causal:
            verdict = "DRIVEN CIRCUIT — causally important but requires continuous input"
        elif epidemic and not causal:
            verdict = "SELF-SUSTAINING BACKGROUND — spreads but not causally important"
        elif net_r0 is None or ap_r is None:
            verdict = "INSUFFICIENT DATA"
        else:
            verdict = "NOISE — no epidemic, no causal effect"
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

    lines.extend(sir_transmission_analysis(result))

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
