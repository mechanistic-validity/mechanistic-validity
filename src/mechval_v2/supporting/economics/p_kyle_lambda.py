"""Protocol WC_M11 --- Kyle Lambda (Market Microstructure Price Impact)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Lens:         Economics
Validity Type: External
Framework:    Wildcard --- Financial Economics (Kyle 1985)
Family:       Kyle Lambda / Price Impact
Validity:     Causal --- per-unit-activation causal effect on logits

References:
    Kyle (1985) "Continuous auctions and insider trading" --- the
        original Kyle lambda model of price impact
    Kyle lambda = sigma_v / (2 * sigma_u): signal-to-noise ratio
        of information in order flow

Question:
    What is the "price impact" of each circuit component --- how much
    does a unit increase in activation magnitude change the task-relevant
    logit difference?

    High lambda: precision instrument --- fires rarely but strongly
    moves the output (high information density per activation unit).
    Low lambda: noisy follower --- fires often but barely changes output.
    Negative lambda: inhibitory component --- activation reduces the
    target logit.

    Circuit depth (1/|lambda|) tells you how much activation is needed
    to produce a unit change in behavior. Components with low circuit
    depth are easy to steer; high circuit depth means robust to
    perturbation.

Metrics:
    activation_patching --- Component-level causal importance
    effect_size         --- Overall circuit importance
    eap                 --- Edge attribution for cross-validation

Calibrations:
    STRUCTURAL_CALIBRATIONS

Usage:
    uv run python kyle_lambda.py                       # all tasks, CPU
    uv run python kyle_lambda.py --device cuda          # GPU
    uv run python kyle_lambda.py --tasks ioi induction  # specific tasks

    # As a callable module:
    from protocols.wildcard.kyle_lambda import run_protocol
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

PROTOCOL_ID = "WC_M11"
PROTOCOL_NAME = "Kyle Lambda (Market Microstructure Price Impact)"
METRICS = ["activation_patching", "effect_size", "eap"]
CALIBRATIONS = STRUCTURAL_CALIBRATIONS
OUTPUT_DIR = Path(__file__).parent / "results" / "wc_m11_kyle_lambda"

THRESHOLDS = {
    "activation_patching": 0.5,
    "effect_size": 0.8,
    "eap": 0.3,
}


# ---------------------------------------------------------------------------
# Novel analysis: Kyle Lambda per-component price impact regression
# ---------------------------------------------------------------------------

def compute_kyle_lambda(
    model,
    tokens: torch.Tensor,
    target_position: int = -1,
    regularization: float = 1e-3,
) -> dict:
    """Compute Kyle's Lambda for each attention head and MLP.

    For each component, runs Ridge regression:
        delta_logit_i = lambda_k * |activation_k_i| + epsilon_i

    where delta_logit is the difference between the top-1 logit and the
    second-ranked logit at *target_position* (a task-agnostic proxy for
    prediction confidence).

    Returns per-component lambda, circuit_depth (1/|lambda|),
    signal-to-noise ratio, and R-squared.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    attn_hooks = [f"blocks.{l}.attn.hook_result" for l in range(n_layers)]
    mlp_hooks = [f"blocks.{l}.hook_mlp_out" for l in range(n_layers)]
    all_hooks = attn_hooks + mlp_hooks

    with torch.no_grad():
        logits, cache = model.run_with_cache(tokens, names_filter=all_hooks)

    # Task-agnostic logit difference: top1 - top2 at target_position
    logits_at_pos = logits[:, target_position, :]  # (N, vocab)
    top2 = torch.topk(logits_at_pos, k=2, dim=-1).values  # (N, 2)
    delta_logit = (top2[:, 0] - top2[:, 1]).cpu().numpy()  # (N,)

    component_lambdas = {}

    for layer in range(n_layers):
        # Per-head lambdas
        attn_result = cache[f"blocks.{layer}.attn.hook_result"]  # (N, seq, n_heads, d_head)
        for head in range(n_heads):
            act_vec = attn_result[:, target_position, head, :].norm(dim=-1).cpu().numpy()  # (N,)
            lam, r2, depth, snr = _ridge_regression(act_vec, delta_logit, regularization)

            component_lambdas[(layer, head, "attn")] = {
                "lambda": lam,
                "circuit_depth": depth,
                "signal_to_noise_ratio": snr,
                "r_squared": r2,
                "direction": "excitatory" if lam > 0 else "inhibitory",
                "is_significant": r2 > 0.05 and abs(lam) > 0.1,
            }

        # MLP lambda
        mlp_out = cache[f"blocks.{layer}.hook_mlp_out"]  # (N, seq, d_model)
        mlp_act = mlp_out[:, target_position, :].norm(dim=-1).cpu().numpy()  # (N,)
        lam, r2, depth, snr = _ridge_regression(mlp_act, delta_logit, regularization)

        component_lambdas[(layer, 0, "mlp")] = {
            "lambda": lam,
            "circuit_depth": depth,
            "signal_to_noise_ratio": snr,
            "r_squared": r2,
            "direction": "excitatory" if lam > 0 else "inhibitory",
            "is_significant": r2 > 0.05 and abs(lam) > 0.1,
        }

    # Sort by absolute lambda
    top_components = sorted(
        [(k, v) for k, v in component_lambdas.items()],
        key=lambda x: -abs(x[1]["lambda"])
    )

    # Joint R-squared: all component activations together
    all_activations = []
    for layer in range(n_layers):
        for head in range(n_heads):
            act = cache[f"blocks.{layer}.attn.hook_result"][:, target_position, head, :].norm(dim=-1).cpu().numpy()
            all_activations.append(act)
        mlp_act = cache[f"blocks.{layer}.hook_mlp_out"][:, target_position, :].norm(dim=-1).cpu().numpy()
        all_activations.append(mlp_act)

    X_all = np.stack(all_activations, axis=1)  # (N, n_components)
    _, joint_r2, _, _ = _ridge_regression_multi(X_all, delta_logit, regularization)

    return {
        "component_lambdas": component_lambdas,
        "top_components": top_components[:15],
        "joint_r_squared": joint_r2,
    }


def _ridge_regression(x: np.ndarray, y: np.ndarray, alpha: float = 1e-3):
    """Simple 1D Ridge regression: y ~ lambda * x.

    Returns (lambda, r_squared, circuit_depth, signal_to_noise).
    """
    X = x.reshape(-1, 1)
    # (X^T X + alpha I)^{-1} X^T y
    xtx = X.T @ X + alpha * np.eye(1)
    xty = X.T @ y
    coef = np.linalg.solve(xtx, xty)
    lam = float(coef[0])

    y_pred = X @ coef
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / (ss_tot + 1e-8)

    depth = 1.0 / (abs(lam) + 1e-8)
    residual_std = float(np.std(y - y_pred)) + 1e-8
    snr = abs(lam) / residual_std

    return lam, r2, depth, snr


def _ridge_regression_multi(X: np.ndarray, y: np.ndarray, alpha: float = 1e-3):
    """Multi-feature Ridge regression for joint R-squared."""
    n_features = X.shape[1]
    xtx = X.T @ X + alpha * np.eye(n_features)
    xty = X.T @ y
    coef = np.linalg.solve(xtx, xty)

    y_pred = X @ coef
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / (ss_tot + 1e-8)

    return coef, r2, None, None


def run_novel_analysis(model, tasks: list[str], n_prompts: int = 40) -> list[EvalResult]:
    """Run Kyle Lambda novel analysis, returning EvalResults per task."""
    results = []

    for task in tasks:
        try:
            vocab_size = model.cfg.d_vocab
            tokens = torch.randint(0, vocab_size, (n_prompts, 16),
                                   device=next(model.parameters()).device)

            kyle = compute_kyle_lambda(model, tokens, target_position=-1)

            # Joint R-squared
            results.append(EvalResult(
                metric_id="WC_M11.joint_r_squared",
                value=kyle["joint_r_squared"],
                n_samples=n_prompts,
                metadata={"task": task},
            ))

            # Max absolute lambda (highest price impact)
            all_lambdas = [v["lambda"] for _, v in kyle["component_lambdas"].items()]
            max_abs_lambda = float(max(abs(l) for l in all_lambdas)) if all_lambdas else 0.0
            results.append(EvalResult(
                metric_id="WC_M11.max_abs_lambda",
                value=max_abs_lambda,
                n_samples=n_prompts,
                metadata={
                    "task": task,
                    "top_component": str(kyle["top_components"][0][0]) if kyle["top_components"] else "none",
                },
            ))

            # Fraction of significant components
            n_sig = sum(1 for _, v in kyle["component_lambdas"].items() if v["is_significant"])
            frac_sig = n_sig / max(len(kyle["component_lambdas"]), 1)
            results.append(EvalResult(
                metric_id="WC_M11.fraction_significant_components",
                value=frac_sig,
                n_samples=n_prompts,
                metadata={"task": task, "n_significant": n_sig},
            ))

            # Mean signal-to-noise ratio across significant components
            sig_snrs = [v["signal_to_noise_ratio"]
                        for _, v in kyle["component_lambdas"].items()
                        if v["is_significant"]]
            mean_snr = float(np.mean(sig_snrs)) if sig_snrs else 0.0
            results.append(EvalResult(
                metric_id="WC_M11.mean_signal_to_noise",
                value=mean_snr,
                n_samples=n_prompts,
                metadata={"task": task},
            ))

        except Exception as e:
            print(f"  [WC_M11 novel analysis] {task} FAILED: {e}")

    return results


# ---------------------------------------------------------------------------
# Protocol runner
# ---------------------------------------------------------------------------

def run_protocol(model, tasks: list[str] | None = None, n_prompts: int = 40,
                 device: str = "cpu", run_cals: bool = True) -> ProtocolResult:
    """Run all WC_M11 metrics + novel Kyle Lambda analysis. Returns a ProtocolResult."""
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

    # Novel Kyle Lambda analysis
    print(f"\n{'─' * 60}")
    print(f"  Kyle Lambda novel analysis — {len(tasks)} tasks, {n_prompts} prompts")
    print(f"{'─' * 60}")
    mt0 = time.time()
    try:
        novel_results = run_novel_analysis(model, tasks, n_prompts=n_prompts)
        result.metrics["kyle_lambda"] = novel_results
        for r in novel_results:
            task = r.metadata.get("task", "?")
            print(f"    {task:20s}  {r.metric_id:40s}  {r.value:+.4f}")
        print(f"  {len(novel_results)} results in {time.time() - mt0:.1f}s")
    except Exception as e:
        print(f"  [kyle_lambda novel analysis] FAILED: {e}")
        result.metrics["kyle_lambda"] = []

    if run_cals:
        print(f"\n{'=' * 60}")
        print(f"  Calibrations ({len(CALIBRATIONS)})")
        print(f"{'=' * 60}")
        cal_tasks = tasks[:2]
        result.calibrations = run_calibrations(
            model, cal_tasks, CALIBRATIONS, n_prompts=n_prompts)

    result.elapsed_seconds = time.time() - t0
    return result


def kyle_lambda_analysis(result: ProtocolResult) -> list[str]:
    """Analyze results through the Kyle Lambda (market microstructure) lens.

    Kyle's Lambda (Kyle 1985) measures the "price impact" of each
    circuit component: how much does a unit increase in activation
    magnitude change the task-relevant logit difference?

    High |lambda| + high R-squared: precision instrument -- the
    component carries dense, reliable information per unit activation.
    Low |lambda|: noisy follower -- activates broadly but carries
    little per-unit information.
    Negative lambda: inhibitory -- activation reduces the target
    logit rather than increasing it.

    The signal-to-noise ratio (|lambda| / residual_std) is the circuit
    analog of Kyle's sigma_v / sigma_u: how much of the component's
    activation variance is "informed" (task-relevant) vs random.

    Combined with activation_patching and EAP:
    - HIGH LAMBDA + HIGH PATCHING: core circuit -- both per-unit
      and total causal effect are large
    - HIGH LAMBDA + LOW PATCHING: latent circuit -- high information
      density but low total effect (perhaps rarely active)
    - LOW LAMBDA + HIGH PATCHING: volume circuit -- large total effect
      but diffuse (needs large activation to matter)
    - LOW LAMBDA + LOW PATCHING: noise
    """
    lines = ["\n  Kyle Lambda Analysis:", "  --------------------"]

    novel = result.metrics.get("kyle_lambda", [])

    for task in result.tasks:
        lines.append(f"\n    {task}:")

        ap_r = _find(result.metrics.get("activation_patching", []), task)
        es_r = _find(result.metrics.get("effect_size", []), task)

        joint_r2 = _find_by_metric_id(novel, "WC_M11.joint_r_squared", task)
        max_lam = _find_by_metric_id(novel, "WC_M11.max_abs_lambda", task)
        frac_sig = _find_by_metric_id(novel, "WC_M11.fraction_significant_components", task)
        mean_snr = _find_by_metric_id(novel, "WC_M11.mean_signal_to_noise", task)

        if joint_r2:
            if joint_r2.value > 0.5:
                label = "activations strongly predict logits"
            elif joint_r2.value > 0.2:
                label = "moderate predictive power"
            else:
                label = "weak predictive power"
            lines.append(f"      Joint R-squared:          {joint_r2.value:.4f} — {label}")

        if max_lam:
            lines.append(f"      Max |lambda|:             {max_lam.value:.4f} "
                         f"(top: {max_lam.metadata.get('top_component', '?')})")

        if frac_sig:
            lines.append(f"      Significant components:   {frac_sig.value:.4f} "
                         f"({frac_sig.metadata.get('n_significant', '?')} total)")

        if mean_snr:
            if mean_snr.value > 1.0:
                label = "high information density"
            elif mean_snr.value > 0.3:
                label = "moderate information density"
            else:
                label = "low information density"
            lines.append(f"      Mean signal-to-noise:     {mean_snr.value:.4f} — {label}")

        if ap_r:
            label = "high causal importance" if ap_r.value > THRESHOLDS["activation_patching"] else "low causal importance"
            lines.append(f"      Activation patching:      {ap_r.value:.4f} — {label}")

        # Verdict
        high_lambda = max_lam is not None and max_lam.value > 0.5
        high_patching = ap_r is not None and ap_r.value > THRESHOLDS["activation_patching"]

        if high_lambda and high_patching:
            verdict = "CORE CIRCUIT — high per-unit impact + high total causal effect"
        elif high_lambda and not high_patching:
            verdict = "LATENT CIRCUIT — high information density but low total effect"
        elif not high_lambda and high_patching:
            verdict = "VOLUME CIRCUIT — large total effect but diffuse per-unit impact"
        elif max_lam is None or ap_r is None:
            verdict = "INSUFFICIENT DATA"
        else:
            verdict = "NOISE — low per-unit impact, low total effect"
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

    lines.extend(kyle_lambda_analysis(result))

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
