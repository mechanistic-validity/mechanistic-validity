"""Protocol I14 --- Settling Depth
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Lens:         Control Theory
Validity Type: Internal
Framework:    Causal (perturbation propagation and recovery)
Family:       Causal (Settling / Transient Analysis)
Validity:     Internal --- I3 Transient dynamics; Construct --- C4 Recovery

References:
    Ogata (2010) "Modern Control Engineering" --- settling time in LTI systems
    Elhage et al. (2021) "A Mathematical Framework for Transformer Circuits"
    McGrath et al. (2023) "The Hydra Effect: Emergent Self-repair in
        Language Model Computations"

Question:
    When a circuit component is perturbed at layer L, how quickly does
    the residual stream recover? If the circuit is modular and self-
    contained, perturbations should either persist (the component is
    necessary) or settle quickly (the network compensates). Long settling
    depths indicate distributed processing; short settling depths suggest
    localized computation or self-repair.

Metrics:
    mean_settling_depth  --- average number of layers until the
                            perturbation decays below 20% of initial
    steady_state_error   --- residual perturbation at the final layer
                            (normalized L2 distance)

Calibrations:
    CAUSAL_CALIBRATIONS

Usage:
    uv run python a14_settling_depth.py                       # all tasks, CPU
    uv run python a14_settling_depth.py --device cuda          # GPU
    uv run python a14_settling_depth.py --tasks ioi induction  # specific tasks

    # As a callable module:
    from protocols.causal.a14_settling_depth import run_protocol
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

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    generate_prompts,
    get_circuit_heads,
    get_token_ids,
    load_model,
    log,
)

from protocols import ProtocolResult
from protocols.calibration_runner import CAUSAL_CALIBRATIONS, run_calibrations, summarize_calibrations

PROTOCOL_ID = "I14"
PROTOCOL_NAME = "Settling Depth"
METRICS = ["mean_settling_depth", "steady_state_error"]
CALIBRATIONS = CAUSAL_CALIBRATIONS
OUTPUT_DIR = Path(__file__).parent / "results" / "a14_settling_depth"

THRESHOLDS = {
    "mean_settling_depth": 3.0,
    "steady_state_error": 0.1,
}

SETTLING_THRESHOLD = 0.2  # perturbation fraction considered "settled"


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_settling_depth(model, tasks: list[str], n_prompts: int = 40,
                       device: str = "cpu") -> list[EvalResult]:
    """Measure perturbation settling depth for each circuit layer.

    For each task:
    1. Group circuit heads by layer.
    2. For each circuit layer L:
       a. Run clean forward pass, cache residual stream at all layers.
       b. Run perturbed forward pass (zero circuit heads at layer L),
          cache residual stream at all layers.
       c. At each subsequent layer L+k, measure normalized L2 distance
          between clean and perturbed residual streams.
       d. Find first layer where distance < SETTLING_THRESHOLD * initial distance.
    3. Report mean settling depth and steady-state error across circuit layers.
    """
    results = []
    n_layers = model.cfg.n_layers

    resid_hook_names = [f"blocks.{i}.hook_resid_post" for i in range(n_layers)]

    for task in tasks:
        try:
            heads = get_circuit_heads(task)
            if not heads:
                log(f"  [I14] {task}: no circuit heads, skipping")
                continue

            prompts = generate_prompts(task, model.tokenizer, n_prompts=n_prompts)
            if len(prompts) < 3:
                log(f"  [I14] {task}: too few prompts, skipping")
                continue

            # Group heads by layer
            heads_by_layer: dict[int, list[int]] = {}
            for L, H in heads:
                heads_by_layer.setdefault(L, []).append(H)

            circuit_layers = sorted(heads_by_layer.keys())
            if not circuit_layers:
                continue

            n_valid = min(len(prompts), n_prompts)
            all_settling_depths = []
            all_steady_errors = []

            for prompt_idx in range(n_valid):
                tokens = model.to_tokens(prompts[prompt_idx].text)

                # Clean forward pass
                _, clean_cache = model.run_with_cache(tokens, names_filter=resid_hook_names)
                clean_resids = {
                    i: clean_cache[f"blocks.{i}.hook_resid_post"][0, -1, :].clone()
                    for i in range(n_layers)
                }

                for circuit_layer in circuit_layers:
                    layer_heads = heads_by_layer[circuit_layer]

                    # Build perturbation hook: zero the circuit heads at this layer
                    def _make_hook(target_layer, target_heads):
                        def hook_fn(z, hook):
                            for H in target_heads:
                                z[0, :, H, :] = 0.0
                            return z
                        return (f"blocks.{target_layer}.attn.hook_z", hook_fn)

                    fwd_hooks = [_make_hook(circuit_layer, layer_heads)]

                    # Perturbed forward pass
                    _, perturbed_cache = model.run_with_cache(
                        tokens,
                        names_filter=resid_hook_names,
                        fwd_hooks=fwd_hooks,
                    )
                    perturbed_resids = {
                        i: perturbed_cache[f"blocks.{i}.hook_resid_post"][0, -1, :].clone()
                        for i in range(n_layers)
                    }

                    # Measure perturbation propagation at subsequent layers
                    subsequent_layers = [i for i in range(n_layers) if i > circuit_layer]
                    if not subsequent_layers:
                        continue

                    # Initial perturbation distance (at the first layer after circuit_layer)
                    first_after = subsequent_layers[0]
                    initial_dist = torch.norm(
                        perturbed_resids[first_after] - clean_resids[first_after]
                    ).item()
                    clean_norm = torch.norm(clean_resids[first_after]).item()

                    if initial_dist < 1e-10:
                        # No perturbation detected
                        all_settling_depths.append(0)
                        all_steady_errors.append(0.0)
                        continue

                    # Track distances at each subsequent layer
                    settling_depth = len(subsequent_layers)  # default: never settled
                    for k, layer_idx in enumerate(subsequent_layers):
                        dist = torch.norm(
                            perturbed_resids[layer_idx] - clean_resids[layer_idx]
                        ).item()
                        normalized_dist = dist / initial_dist if initial_dist > 1e-10 else 0.0

                        if normalized_dist < SETTLING_THRESHOLD:
                            settling_depth = k
                            break

                    all_settling_depths.append(settling_depth)

                    # Steady-state error: distance at the final layer
                    final_dist = torch.norm(
                        perturbed_resids[n_layers - 1] - clean_resids[n_layers - 1]
                    ).item()
                    final_clean_norm = torch.norm(clean_resids[n_layers - 1]).item()
                    steady_error = final_dist / final_clean_norm if final_clean_norm > 1e-10 else 0.0
                    all_steady_errors.append(steady_error)

            if not all_settling_depths:
                log(f"  [I14] {task}: no settling data collected, skipping")
                continue

            mean_depth = float(np.mean(all_settling_depths))
            mean_error = float(np.mean(all_steady_errors))

            results.append(EvalResult(
                metric_id="I14.mean_settling_depth",
                value=mean_depth,
                n_samples=len(all_settling_depths),
                metadata={
                    "task": task,
                    "n_circuit_layers": len(circuit_layers),
                    "circuit_layers": circuit_layers,
                    "settling_threshold": SETTLING_THRESHOLD,
                    "passed": mean_depth < THRESHOLDS["mean_settling_depth"],
                },
            ))
            results.append(EvalResult(
                metric_id="I14.steady_state_error",
                value=mean_error,
                n_samples=len(all_steady_errors),
                metadata={
                    "task": task,
                    "passed": mean_error < THRESHOLDS["steady_state_error"],
                },
            ))

            log(f"  [I14] {task}: settling_depth={mean_depth:.2f}, "
                f"steady_error={mean_error:.4f}")

        except Exception as e:
            log(f"  [I14] {task} FAILED: {e}")

    return results


# ---------------------------------------------------------------------------
# Protocol runner
# ---------------------------------------------------------------------------

def run_protocol(model, tasks: list[str] | None = None, n_prompts: int = 40,
                 device: str = "cpu", run_cals: bool = True,
                 protocol_results: dict | None = None) -> ProtocolResult:
    """Run I14 settling depth analysis. Returns a ProtocolResult."""
    tasks = tasks or CIRCUIT_TASKS
    t0 = time.time()
    result = ProtocolResult(
        protocol_id=PROTOCOL_ID,
        protocol_name=PROTOCOL_NAME,
        tasks=tasks,
    )

    print(f"\n{'─' * 60}")
    print(f"  Settling Depth — {len(tasks)} tasks, {n_prompts} prompts")
    print(f"{'─' * 60}")

    mt0 = time.time()
    try:
        all_results = run_settling_depth(model, tasks, n_prompts=n_prompts,
                                         device=device)
        result.metrics["mean_settling_depth"] = [
            r for r in all_results if r.metric_id == "I14.mean_settling_depth"
        ]
        result.metrics["steady_state_error"] = [
            r for r in all_results if r.metric_id == "I14.steady_state_error"
        ]
        for r in all_results:
            task = r.metadata.get("task", "?")
            passed = r.metadata.get("passed", None)
            tag = " PASS" if passed else (" FAIL" if passed is not None else "")
            print(f"    {task:20s}  {r.metric_id:30s}  {r.value:+.4f}{tag}")
        print(f"  {len(all_results)} results in {time.time() - mt0:.1f}s")
    except Exception as e:
        print(f"  [settling_depth] FAILED: {e}")
        result.metrics["mean_settling_depth"] = []
        result.metrics["steady_state_error"] = []

    if run_cals:
        print(f"\n{'=' * 60}")
        print(f"  Calibrations ({len(CALIBRATIONS)})")
        print(f"{'=' * 60}")
        cal_tasks = tasks[:2]
        result.calibrations = run_calibrations(
            model, cal_tasks, CALIBRATIONS, n_prompts=n_prompts)

    result.elapsed_seconds = time.time() - t0
    return result


# ---------------------------------------------------------------------------
# Analysis and display
# ---------------------------------------------------------------------------

def settling_analysis(result: ProtocolResult) -> list[str]:
    """Analyze results through the settling depth lens."""
    lines = ["\n  Settling Depth Analysis:", "  ────────────────────────"]

    for task in result.tasks:
        lines.append(f"\n    {task}:")

        sd = _find(result.metrics.get("mean_settling_depth", []), task)
        se = _find(result.metrics.get("steady_state_error", []), task)

        if sd:
            depth = sd.value
            if depth < 1.0:
                label = "immediate recovery (self-repair / compensation)"
            elif depth < 3.0:
                label = "fast settling (localized computation)"
            elif depth < 6.0:
                label = "moderate settling (distributed processing)"
            else:
                label = "slow settling (perturbation persists through many layers)"
            lines.append(f"      Mean settling depth: {depth:.2f} layers — {label}")

        if se:
            error = se.value
            if error < 0.05:
                label = "full recovery (network compensates completely)"
            elif error < 0.15:
                label = "near recovery (small residual effect)"
            else:
                label = "persistent perturbation (component is strongly necessary)"
            lines.append(f"      Steady-state error:  {error:.4f} — {label}")

        if sd and se:
            fast = sd.value < 3.0
            recovered = se.value < 0.1

            if fast and recovered:
                verdict = "SELF-REPAIR — fast settling with full recovery (Hydra effect)"
            elif fast and not recovered:
                verdict = "LOCALIZED DAMAGE — fast settling but permanent effect"
            elif not fast and recovered:
                verdict = "SLOW RECOVERY — distributed compensation across layers"
            else:
                verdict = "PERSISTENT DISRUPTION — component is strongly necessary"
            lines.append(f"      VERDICT: {verdict}")

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

    lines.extend(settling_analysis(result))

    if result.calibrations:
        lines.append("")
        lines.append(summarize_calibrations(result.calibrations))

    lines.append(f"\n  Elapsed: {result.elapsed_seconds:.1f}s")

    text = "\n".join(lines)
    print(text)
    return text


def save_protocol_results(result: ProtocolResult, output_dir: Path | None = None):
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
                          device=args.device,
                          run_cals=not args.no_calibrations)
    summarize(result)

    if not args.no_save:
        save_protocol_results(result, output_dir)

    n = sum(len(r) for r in result.metrics.values())
    nc = sum(len(r) for r in result.calibrations.values())
    print(f"\nTotal: {n} metric + {nc} calibration results in {result.elapsed_seconds:.1f}s")


if __name__ == "__main__":
    main()
