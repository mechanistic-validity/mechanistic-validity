"""Protocol WC_M12 --- Free Energy Decomposition (Active Inference)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Lens:         Information Theory
Validity Type: Measurement
Framework:    Wildcard --- Computational Neuroscience (Friston FEP)
Family:       Variational Free Energy / Active Inference
Validity:     Causal --- per-head accuracy/complexity decomposition

References:
    Friston (2010) "The free-energy principle: a unified brain theory?"
        --- the Free Energy Principle (FEP)
    Ren et al. (NeurIPS 2025) "Transformers as Intrinsic Optimizers"
        --- exact derivation of softmax attention from Helmholtz free energy
    Parr & Friston (2019) "Generalised free energy and active inference"
        --- active inference as expected free energy minimization

Question:
    Can we decompose each attention head's contribution into an accuracy
    term (how much it reduces cross-entropy loss) and a complexity term
    (how much its attention pattern diverges from uniform)?

    Free energy = Complexity - Accuracy.

    A low free-energy head is doing useful work efficiently: high
    accuracy with minimal distributional complexity. A high free-energy
    head is either not helping predictions (low accuracy) or using
    very peaked attention (high complexity).

    The Pareto frontier of accuracy vs complexity identifies the
    thermodynamically optimal circuit components --- the heads that
    achieve the most behavioral improvement per unit of distributional
    cost.

Metrics:
    activation_patching --- Component-level causal importance
    effect_size         --- Overall circuit importance
    sigma_ablation      --- Noise tolerance

Calibrations:
    STRUCTURAL_CALIBRATIONS

Usage:
    uv run python free_energy.py                       # all tasks, CPU
    uv run python free_energy.py --device cuda          # GPU
    uv run python free_energy.py --tasks ioi induction  # specific tasks

    # As a callable module:
    from protocols.wildcard.free_energy import run_protocol
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
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from mechval.metrics.common import CIRCUIT_TASKS, EvalResult, get_circuit_heads, load_model

from protocols import ProtocolResult, import_metric_runner
from protocols.calibration_runner import STRUCTURAL_CALIBRATIONS, run_calibrations, summarize_calibrations

PROTOCOL_ID = "WC_M12"
PROTOCOL_NAME = "Free Energy Decomposition (Active Inference)"
METRICS = ["activation_patching", "effect_size", "sigma_ablation"]
CALIBRATIONS = STRUCTURAL_CALIBRATIONS
OUTPUT_DIR = Path(__file__).parent / "results" / "wc_m12_free_energy"

THRESHOLDS = {
    "activation_patching": 0.5,
    "effect_size": 0.8,
    "sigma_ablation": 0.5,
}


# ---------------------------------------------------------------------------
# Novel analysis: Free Energy decomposition per attention head
# ---------------------------------------------------------------------------

def free_energy_decomposition(
    model,
    tokens: torch.Tensor,
    target_position: int = -1,
) -> dict:
    """Decompose each attention head into accuracy and complexity terms.

    Accuracy: how much CE loss increases when this head is zero-ablated
    (positive = head was reducing loss = accurate).

    Complexity: KL divergence of the head's attention pattern from
    uniform (how peaked/non-uniform the attention is).

    Free energy = Complexity - Accuracy.
    Efficiency = Accuracy / (Complexity + epsilon).

    Categories:
    - efficient_circuit: accuracy > 0.1 AND complexity < 2.0
    - noisy_circuit: accuracy > 0.1 AND complexity >= 2.0
    - irrelevant: accuracy <= 0.1
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    attn_hook_names = [f"blocks.{l}.attn.hook_pattern" for l in range(n_layers)]
    result_hook_names = [f"blocks.{l}.attn.hook_result" for l in range(n_layers)]
    all_hooks = list(set(attn_hook_names + result_hook_names))

    with torch.no_grad():
        baseline_logits, cache = model.run_with_cache(tokens, names_filter=all_hooks)

    # Baseline CE loss at target_position
    target_tokens = tokens[:, target_position]  # (N,)
    # For target_position=-1, the logit predicting that token is at position -2
    # But for simplicity, use logits at target_position predicting any token
    logits_at_pos = baseline_logits[:, target_position, :]  # (N, vocab)
    # Use the most likely token as "target" for a task-agnostic metric
    baseline_probs = torch.softmax(logits_at_pos, dim=-1)
    baseline_ce = -torch.log(baseline_probs.max(dim=-1).values + 1e-8)  # (N,)

    head_results = {}

    for layer in range(n_layers):
        attn_pattern = cache[f"blocks.{layer}.attn.hook_pattern"]  # (N, n_heads, seq, seq)

        for head in range(n_heads):
            # === Accuracy term: zero-ablation effect ===
            def make_zero_hook(h):
                def hook_fn(result, hook):
                    result_copy = result.clone()
                    result_copy[:, :, h, :] = 0.0
                    return result_copy
                return hook_fn

            with torch.no_grad():
                ablated_logits = model.run_with_hooks(
                    tokens,
                    fwd_hooks=[(f"blocks.{layer}.attn.hook_result",
                                make_zero_hook(head))],
                )
            ablated_logits_at_pos = ablated_logits[:, target_position, :]
            ablated_probs = torch.softmax(ablated_logits_at_pos, dim=-1)
            ablated_ce = -torch.log(ablated_probs.max(dim=-1).values + 1e-8)

            accuracy_contribution = float((ablated_ce - baseline_ce).mean().item())

            # === Complexity term: KL from uniform attention ===
            head_attn = attn_pattern[:, head, target_position, :]  # (N, seq)
            head_attn = head_attn.clamp_min(1e-8)
            seq_len = head_attn.shape[-1]
            uniform = torch.ones_like(head_attn) / seq_len

            # KL(head_attn || uniform)
            kl_from_uniform = float(F.kl_div(
                uniform.log(), head_attn, reduction='batchmean', log_target=False
            ).item())

            free_energy = kl_from_uniform - accuracy_contribution
            efficiency = accuracy_contribution / (kl_from_uniform + 1e-4)

            if accuracy_contribution > 0.1 and kl_from_uniform < 2.0:
                category = "efficient_circuit"
            elif accuracy_contribution > 0.1:
                category = "noisy_circuit"
            else:
                category = "irrelevant"

            head_results[(layer, head)] = {
                "accuracy_contribution": float(accuracy_contribution),
                "complexity_cost": float(kl_from_uniform),
                "free_energy": float(free_energy),
                "efficiency": float(efficiency),
                "category": category,
            }

    # Pareto frontier: high accuracy AND low complexity
    pareto_frontier = sorted(
        [(k, v) for k, v in head_results.items() if v["accuracy_contribution"] > 0],
        key=lambda x: -x[1]["efficiency"]
    )

    total_fe = sum(v["free_energy"] for v in head_results.values())

    return {
        "component_free_energies": head_results,
        "pareto_frontier": pareto_frontier[:10],
        "total_free_energy": total_fe,
    }


def run_novel_analysis(model, tasks: list[str], n_prompts: int = 40) -> list[EvalResult]:
    """Run Free Energy novel analysis, returning EvalResults per task."""
    results = []

    for task in tasks:
        try:
            vocab_size = model.cfg.d_vocab
            tokens = torch.randint(0, vocab_size, (n_prompts, 16),
                                   device=next(model.parameters()).device)

            fe = free_energy_decomposition(model, tokens, target_position=-1)

            # Total free energy
            results.append(EvalResult(
                metric_id="WC_M12.total_free_energy",
                value=fe["total_free_energy"],
                n_samples=n_prompts,
                metadata={"task": task},
            ))

            # Fraction of efficient circuit heads
            all_heads = fe["component_free_energies"]
            n_efficient = sum(1 for v in all_heads.values() if v["category"] == "efficient_circuit")
            frac_efficient = n_efficient / max(len(all_heads), 1)
            results.append(EvalResult(
                metric_id="WC_M12.fraction_efficient_heads",
                value=frac_efficient,
                n_samples=n_prompts,
                metadata={"task": task, "n_efficient": n_efficient, "n_total": len(all_heads)},
            ))

            # Mean efficiency of top-5 Pareto frontier heads
            if fe["pareto_frontier"]:
                top_eff = [v["efficiency"] for _, v in fe["pareto_frontier"][:5]]
                mean_top_eff = float(np.mean(top_eff))
            else:
                mean_top_eff = 0.0
            results.append(EvalResult(
                metric_id="WC_M12.mean_pareto_efficiency",
                value=mean_top_eff,
                n_samples=n_prompts,
                metadata={"task": task},
            ))

            # Mean accuracy contribution (across all heads)
            mean_accuracy = float(np.mean([v["accuracy_contribution"] for v in all_heads.values()]))
            results.append(EvalResult(
                metric_id="WC_M12.mean_accuracy_contribution",
                value=mean_accuracy,
                n_samples=n_prompts,
                metadata={"task": task},
            ))

        except Exception as e:
            print(f"  [WC_M12 novel analysis] {task} FAILED: {e}")

    return results


# ---------------------------------------------------------------------------
# Protocol runner
# ---------------------------------------------------------------------------

def run_protocol(model, tasks: list[str] | None = None, n_prompts: int = 40,
                 device: str = "cpu", run_cals: bool = True) -> ProtocolResult:
    """Run all WC_M12 metrics + novel Free Energy analysis. Returns a ProtocolResult."""
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

    # Novel Free Energy analysis
    print(f"\n{'─' * 60}")
    print(f"  Free Energy novel analysis — {len(tasks)} tasks, {n_prompts} prompts")
    print(f"{'─' * 60}")
    mt0 = time.time()
    try:
        novel_results = run_novel_analysis(model, tasks, n_prompts=n_prompts)
        result.metrics["free_energy"] = novel_results
        for r in novel_results:
            task = r.metadata.get("task", "?")
            print(f"    {task:20s}  {r.metric_id:40s}  {r.value:+.4f}")
        print(f"  {len(novel_results)} results in {time.time() - mt0:.1f}s")
    except Exception as e:
        print(f"  [free_energy novel analysis] FAILED: {e}")
        result.metrics["free_energy"] = []

    if run_cals:
        print(f"\n{'=' * 60}")
        print(f"  Calibrations ({len(CALIBRATIONS)})")
        print(f"{'=' * 60}")
        cal_tasks = tasks[:2]
        result.calibrations = run_calibrations(
            model, cal_tasks, CALIBRATIONS, n_prompts=n_prompts)

    result.elapsed_seconds = time.time() - t0
    return result


def free_energy_analysis(result: ProtocolResult) -> list[str]:
    """Analyze results through the Free Energy Principle lens.

    The Free Energy Principle (Friston 2010) decomposes neural
    computation into two terms:
    - Accuracy: reduction in prediction error (how much removing a
      head increases CE loss).
    - Complexity: KL divergence from a uniform prior (how peaked the
      attention pattern is).

    Free Energy = Complexity - Accuracy. A thermodynamically efficient
    circuit component achieves high accuracy at low complexity.

    The Pareto frontier of accuracy vs complexity identifies the
    optimal circuit: heads that achieve maximal behavioral improvement
    per unit of distributional cost. Heads above the frontier are
    wasteful (high complexity for their accuracy level). Heads below
    the frontier are the thermodynamically optimal circuit.

    Combined with activation_patching and sigma_ablation:
    - EFFICIENT CIRCUIT: low free energy + high causal importance
      = Pareto-optimal component, robust
    - NOISY CIRCUIT: high free energy + high causal importance
      = important but computationally expensive
    - ROBUST BACKGROUND: low free energy + low causal importance
      = efficient but not needed for this task
    - FRAGILE NOISE: high free energy + low sigma_ablation
      = complex, unimportant, and noise-sensitive
    """
    lines = ["\n  Free Energy Decomposition Analysis:", "  -----------------------------------"]

    novel = result.metrics.get("free_energy", [])

    for task in result.tasks:
        lines.append(f"\n    {task}:")

        ap_r = _find(result.metrics.get("activation_patching", []), task)
        sa_r = _find(result.metrics.get("sigma_ablation", []), task)

        total_fe = _find_by_metric_id(novel, "WC_M12.total_free_energy", task)
        frac_eff = _find_by_metric_id(novel, "WC_M12.fraction_efficient_heads", task)
        pareto_eff = _find_by_metric_id(novel, "WC_M12.mean_pareto_efficiency", task)
        mean_acc = _find_by_metric_id(novel, "WC_M12.mean_accuracy_contribution", task)

        if total_fe:
            if total_fe.value < 0:
                label = "net accuracy dominates (efficient overall)"
            elif total_fe.value < 5.0:
                label = "moderate free energy budget"
            else:
                label = "high free energy budget (complexity-dominated)"
            lines.append(f"      Total free energy:        {total_fe.value:.4f} — {label}")

        if frac_eff:
            lines.append(f"      Efficient heads:          {frac_eff.value:.4f} "
                         f"({frac_eff.metadata.get('n_efficient', '?')}/"
                         f"{frac_eff.metadata.get('n_total', '?')})")

        if pareto_eff:
            if pareto_eff.value > 1.0:
                label = "strong Pareto efficiency"
            elif pareto_eff.value > 0.3:
                label = "moderate Pareto efficiency"
            else:
                label = "weak Pareto efficiency"
            lines.append(f"      Pareto frontier mean:     {pareto_eff.value:.4f} — {label}")

        if mean_acc:
            lines.append(f"      Mean accuracy contrib:    {mean_acc.value:.4f}")

        if ap_r:
            label = "high causal importance" if ap_r.value > THRESHOLDS["activation_patching"] else "low causal importance"
            lines.append(f"      Activation patching:      {ap_r.value:.4f} — {label}")

        if sa_r:
            label = "noise-robust" if sa_r.value > THRESHOLDS["sigma_ablation"] else "noise-sensitive"
            lines.append(f"      Sigma ablation:           {sa_r.value:.4f} — {label}")

        # Verdict
        efficient = frac_eff is not None and frac_eff.value > 0.2
        causal = ap_r is not None and ap_r.value > THRESHOLDS["activation_patching"]

        if efficient and causal:
            verdict = "EFFICIENT CIRCUIT — Pareto-optimal heads with high causal importance"
        elif not efficient and causal:
            verdict = "NOISY CIRCUIT — causally important but computationally expensive"
        elif efficient and not causal:
            verdict = "ROBUST BACKGROUND — efficient but not needed for this task"
        elif frac_eff is None or ap_r is None:
            verdict = "INSUFFICIENT DATA"
        else:
            verdict = "FRAGILE NOISE — inefficient and causally unimportant"
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

    lines.extend(free_energy_analysis(result))

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
