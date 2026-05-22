"""Protocol WC_M9 --- Hawkes Process (Self-Exciting Point Process)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Lens:         Information Theory
Validity Type: Internal
Framework:    Wildcard --- Stochastic Processes (Hawkes / Point Processes)
Family:       Multivariate Hawkes Process
Validity:     Structural --- co-activation interaction matrix and branching ratio

References:
    Hawkes (1971) "Spectra of some self-exciting and mutually exciting
        point processes" --- the original Hawkes process
    Ogata (1981) "On Lewis' simulation method for point processes"
        --- EM fitting procedure
    Zhou et al. (2013) "Learning triggering kernels for multi-
        dimensional Hawkes processes" --- multivariate EM algorithm

Question:
    Can we model component activations as a self-exciting point process
    over layer depth? When a component fires at layer l, does it
    increase the probability of other components firing at layer l+1?

    The interaction matrix Phi[i,j] captures how component j's
    activation excites (positive) or inhibits (negative) component i.
    The branching ratio (spectral radius of Phi) determines whether the
    circuit is supercritical (self-sustaining cascade, R > 1) or
    subcritical (damped, R < 1).

Metrics:
    activation_patching --- Component-level causal importance
    eap                 --- Edge attribution for cross-validation
    effect_size         --- Overall circuit importance

Calibrations:
    STRUCTURAL_CALIBRATIONS

Usage:
    uv run python hawkes_process.py                       # all tasks, CPU
    uv run python hawkes_process.py --device cuda          # GPU
    uv run python hawkes_process.py --tasks ioi induction  # specific tasks

    # As a callable module:
    from protocols.wildcard.hawkes_process import run_protocol
    result = run_protocol(model, tasks=["ioi"], n_prompts=40)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from mechval.metrics.common import CIRCUIT_TASKS, EvalResult, get_circuit_heads, load_model

from protocols import ProtocolResult, import_metric_runner
from protocols.calibration_runner import STRUCTURAL_CALIBRATIONS, run_calibrations, summarize_calibrations

PROTOCOL_ID = "WC_M9"
PROTOCOL_NAME = "Hawkes Process (Self-Exciting Point Process)"
METRICS = ["activation_patching", "eap", "effect_size"]
CALIBRATIONS = STRUCTURAL_CALIBRATIONS
OUTPUT_DIR = Path(__file__).parent / "results" / "wc_m9_hawkes_process"

THRESHOLDS = {
    "activation_patching": 0.5,
    "eap": 0.3,
    "effect_size": 0.8,
}


# ---------------------------------------------------------------------------
# Novel analysis: Multivariate Hawkes process over layer depth
# ---------------------------------------------------------------------------

@dataclass
class HawkesEvent:
    """A single component activation event."""
    layer: int
    component_idx: int
    activation: float


def collect_hawkes_events(
    model,
    tokens: torch.Tensor,
    target_position: int = -1,
    activation_threshold: float = 0.1,
) -> list[list[HawkesEvent]]:
    """Collect activation events per prompt.

    For each prompt, an "event" is a component (attention head or MLP)
    that produces an activation norm above threshold at *target_position*.
    Returns a list (per prompt) of lists of HawkesEvents sorted by layer.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    N = tokens.shape[0]

    attn_hooks = {i: f"blocks.{i}.attn.hook_result" for i in range(n_layers)}
    mlp_hooks = {i: f"blocks.{i}.hook_mlp_out" for i in range(n_layers)}
    all_hooks = list(attn_hooks.values()) + list(mlp_hooks.values())

    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=all_hooks)

    all_event_sequences = []

    for prompt_idx in range(N):
        events = []

        for layer_idx in range(n_layers):
            # Attention heads
            attn_result = cache[attn_hooks[layer_idx]]  # (N, seq, n_heads, d_head)
            head_activations = attn_result[prompt_idx, target_position, :, :].norm(dim=-1)

            for head_idx in range(n_heads):
                act = head_activations[head_idx].item()
                if act > activation_threshold:
                    events.append(HawkesEvent(
                        layer=layer_idx,
                        component_idx=head_idx,
                        activation=act,
                    ))

            # MLP
            mlp_out = cache[mlp_hooks[layer_idx]]  # (N, seq, d_model)
            mlp_act = mlp_out[prompt_idx, target_position, :].abs().mean().item()
            if mlp_act > activation_threshold:
                events.append(HawkesEvent(
                    layer=layer_idx,
                    component_idx=n_heads + layer_idx,
                    activation=mlp_act,
                ))

        all_event_sequences.append(sorted(events, key=lambda e: e.layer))

    return all_event_sequences


def fit_hawkes_interaction_matrix(
    event_sequences: list[list[HawkesEvent]],
    n_components: int,
    n_layers: int,
    bandwidth: float = 1.5,
    n_em_steps: int = 30,
) -> dict:
    """Fit a multivariate Hawkes process interaction matrix via EM.

    E-step: assign each event's responsibility to background vs excitation.
    M-step: update mu (base rates) and Phi (interaction matrix).

    Returns mu, Phi, branching_ratio, and strongest circuit edges.
    """
    beta = 1.0 / bandwidth
    K = n_components
    N = len(event_sequences)

    mu = np.ones(K) * 0.1
    Phi = np.zeros((K, K))

    def kernel(delta_layer):
        return beta * np.exp(-beta * delta_layer)

    for em_step in range(n_em_steps):
        mu_sum = np.zeros(K)
        Phi_sum = np.zeros((K, K))
        norm = np.zeros(K)

        for seq in event_sequences:
            if not seq:
                continue

            evlist = [(e.layer, e.component_idx) for e in seq]

            for j, (lj, kj) in enumerate(evlist):
                past_contrib = np.zeros(K)
                for i_past, (li, ki) in enumerate(evlist[:j]):
                    dl = lj - li
                    if dl <= 0:
                        continue
                    past_contrib[ki] += Phi[kj, ki] * kernel(dl)

                lambda_bg = mu[kj]
                lambda_ex = max(past_contrib.sum(), 0.0)
                total = lambda_bg + lambda_ex + 1e-10

                p_bg = lambda_bg / total
                p_ex = (past_contrib.clip(0) / total) if lambda_ex > 0 else np.zeros(K)

                mu_sum[kj] += p_bg
                Phi_sum[kj, :] += p_ex
                norm[kj] += 1.0

        total_obs_time = n_layers * N
        mu = (mu_sum + 1e-6) / (total_obs_time + 1)

        for i in range(K):
            if norm[i] > 0:
                Phi[i, :] = Phi_sum[i, :] / (norm[i] + 1e-6)

    spectral_radius = float(np.max(np.abs(np.linalg.eigvals(Phi))))

    edges = []
    for i in range(K):
        for j in range(K):
            if abs(Phi[i, j]) > 0.01:
                edges.append({
                    "from_component": int(j),
                    "to_component": int(i),
                    "strength": float(Phi[i, j]),
                    "type": "excitation" if Phi[i, j] > 0 else "inhibition",
                })
    edges = sorted(edges, key=lambda x: -abs(x["strength"]))

    return {
        "mu": mu,
        "Phi": Phi,
        "branching_ratio": spectral_radius,
        "is_supercritical": spectral_radius > 1.0,
        "circuit_edges": edges[:20],
    }


def run_novel_analysis(model, tasks: list[str], n_prompts: int = 40) -> list[EvalResult]:
    """Run Hawkes process novel analysis, returning EvalResults per task."""
    results = []
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    n_components = n_heads + n_layers  # heads + per-layer MLP

    for task in tasks:
        try:
            vocab_size = model.cfg.d_vocab
            tokens = torch.randint(0, vocab_size, (n_prompts, 16),
                                   device=next(model.parameters()).device)

            event_sequences = collect_hawkes_events(
                model, tokens, target_position=-1, activation_threshold=0.1)

            hawkes = fit_hawkes_interaction_matrix(
                event_sequences, n_components=n_components,
                n_layers=n_layers, bandwidth=1.5, n_em_steps=20)

            # Branching ratio
            results.append(EvalResult(
                metric_id="WC_M9.branching_ratio",
                value=hawkes["branching_ratio"],
                n_samples=n_prompts,
                metadata={
                    "task": task,
                    "is_supercritical": hawkes["is_supercritical"],
                },
            ))

            # Mean base rate
            mean_mu = float(np.mean(hawkes["mu"]))
            results.append(EvalResult(
                metric_id="WC_M9.mean_base_rate",
                value=mean_mu,
                n_samples=n_prompts,
                metadata={"task": task},
            ))

            # Number of strong excitatory edges
            n_strong = sum(1 for e in hawkes["circuit_edges"]
                           if e["strength"] > 0.1 and e["type"] == "excitation")
            results.append(EvalResult(
                metric_id="WC_M9.n_strong_excitatory_edges",
                value=float(n_strong),
                n_samples=n_prompts,
                metadata={"task": task, "top_edges": hawkes["circuit_edges"][:5]},
            ))

        except Exception as e:
            print(f"  [WC_M9 novel analysis] {task} FAILED: {e}")

    return results


# ---------------------------------------------------------------------------
# Protocol runner
# ---------------------------------------------------------------------------

def run_protocol(model, tasks: list[str] | None = None, n_prompts: int = 40,
                 device: str = "cpu", run_cals: bool = True) -> ProtocolResult:
    """Run all WC_M9 metrics + novel Hawkes process analysis. Returns a ProtocolResult."""
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

    # Novel Hawkes process analysis
    print(f"\n{'─' * 60}")
    print(f"  Hawkes process novel analysis — {len(tasks)} tasks, {n_prompts} prompts")
    print(f"{'─' * 60}")
    mt0 = time.time()
    try:
        novel_results = run_novel_analysis(model, tasks, n_prompts=n_prompts)
        result.metrics["hawkes_process"] = novel_results
        for r in novel_results:
            task = r.metadata.get("task", "?")
            print(f"    {task:20s}  {r.metric_id:40s}  {r.value:+.4f}")
        print(f"  {len(novel_results)} results in {time.time() - mt0:.1f}s")
    except Exception as e:
        print(f"  [hawkes_process novel analysis] FAILED: {e}")
        result.metrics["hawkes_process"] = []

    if run_cals:
        print(f"\n{'=' * 60}")
        print(f"  Calibrations ({len(CALIBRATIONS)})")
        print(f"{'=' * 60}")
        cal_tasks = tasks[:2]
        result.calibrations = run_calibrations(
            model, cal_tasks, CALIBRATIONS, n_prompts=n_prompts)

    result.elapsed_seconds = time.time() - t0
    return result


def hawkes_process_analysis(result: ProtocolResult) -> list[str]:
    """Analyze results through the Hawkes process lens.

    The multivariate Hawkes process treats component activations as events
    in a self-exciting point process over layer depth. The interaction
    matrix Phi captures how each component's firing affects subsequent
    components' firing probabilities.

    Key metric: the branching ratio (spectral radius of Phi).
    - Supercritical (R > 1): the circuit is self-sustaining -- component
      activations create cascades of downstream activity.
    - Subcritical (R < 1): the circuit is damped -- activations die out
      without external driving input.

    Components with high self-excitation (Phi[i,i]) are recurrent
    computation nodes. Components with high cross-excitation to many
    targets are "hub" nodes in the circuit.

    Combined with activation_patching and EAP:
    - SUPERCRITICAL + HIGH CAUSAL: self-amplifying circuit core
    - SUBCRITICAL + HIGH CAUSAL: externally-driven circuit
    - SUPERCRITICAL + LOW CAUSAL: self-sustaining but task-irrelevant
    - SUBCRITICAL + LOW CAUSAL: background noise
    """
    lines = ["\n  Hawkes Process Analysis:", "  -----------------------"]

    novel = result.metrics.get("hawkes_process", [])

    for task in result.tasks:
        lines.append(f"\n    {task}:")

        ap_r = _find(result.metrics.get("activation_patching", []), task)
        eap_r = _find(result.metrics.get("eap", []), task)
        es_r = _find(result.metrics.get("effect_size", []), task)

        br = _find_by_metric_id(novel, "WC_M9.branching_ratio", task)
        base_rate = _find_by_metric_id(novel, "WC_M9.mean_base_rate", task)
        n_edges = _find_by_metric_id(novel, "WC_M9.n_strong_excitatory_edges", task)

        if br:
            if br.value > 1.0:
                label = "SUPERCRITICAL — self-sustaining cascade"
            elif br.value > 0.5:
                label = "near-critical — moderate self-excitation"
            else:
                label = "SUBCRITICAL — damped, externally driven"
            lines.append(f"      Branching ratio:          {br.value:.4f} — {label}")

        if base_rate:
            lines.append(f"      Mean base firing rate:    {base_rate.value:.4f}")

        if n_edges:
            lines.append(f"      Strong excitatory edges:  {n_edges.value:.0f}")

        if ap_r:
            label = "high causal importance" if ap_r.value > THRESHOLDS["activation_patching"] else "low causal importance"
            lines.append(f"      Activation patching:      {ap_r.value:.4f} — {label}")

        # Verdict
        supercrit = br is not None and br.value > 1.0
        causal = ap_r is not None and ap_r.value > THRESHOLDS["activation_patching"]

        if supercrit and causal:
            verdict = "SELF-AMPLIFYING CIRCUIT CORE — cascade dynamics with causal importance"
        elif not supercrit and causal:
            verdict = "EXTERNALLY-DRIVEN CIRCUIT — causal but requires driving input"
        elif supercrit and not causal:
            verdict = "SELF-SUSTAINING BUT TASK-IRRELEVANT — cascade without causal effect"
        elif br is None or ap_r is None:
            verdict = "INSUFFICIENT DATA"
        else:
            verdict = "BACKGROUND NOISE — no cascade, no causal effect"
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

    lines.extend(hawkes_process_analysis(result))

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
