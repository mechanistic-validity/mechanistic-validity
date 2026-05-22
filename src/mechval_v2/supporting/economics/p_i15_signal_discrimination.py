"""Protocol I15 --- Signal Discrimination
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Lens:         Economics
Validity Type: Internal
Framework:    Internal Validity --- Causal Specificity
Family:       Signal vs Noise Discrimination
Validity:     Internal --- does the circuit respond specifically to
              task-relevant perturbations?

References:
    Woodward (2003) "Making Things Happen" --- interventionist theory;
        causes must be specific (targeted perturbation > random)
    Conant & Ashby (1970) "Every good regulator of a system must be a
        model of that system" --- the circuit must discriminate signals
    Vig et al. (2020) "Causal Mediation Analysis for Interpreting
        Neural NLP" --- contrasting task-relevant vs random interventions

Question:
    Does the circuit respond more strongly to task-relevant perturbations
    than to random perturbations of the same magnitude? A circuit that
    responds equally to both is not specifically computing the task ---
    it is just generally sensitive to input changes. A high discrimination
    ratio (task_effect / random_effect >> 1) indicates the circuit is
    tuned to task-relevant signals.

Metrics:
    activation_patching --- Component-level causal importance
    effect_size         --- Overall circuit importance
    cka                 --- Representational similarity

Calibrations:
    CAUSAL_CALIBRATIONS

Usage:
    uv run python signal_discrimination.py                       # all tasks, CPU
    uv run python signal_discrimination.py --device cuda          # GPU
    uv run python signal_discrimination.py --tasks ioi induction  # specific tasks

    # As a callable module:
    from protocols.signal_discrimination import run_protocol
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
    generate_prompts,
    get_circuit_heads,
    get_token_ids,
    heads_to_layer_dict,
    load_model,
    logit_diff_from_logits,
)

from protocols import ProtocolResult, import_metric_runner
from protocols.calibration_runner import CAUSAL_CALIBRATIONS, run_calibrations, summarize_calibrations

PROTOCOL_ID = "I15"
PROTOCOL_NAME = "Signal Discrimination"
METRICS = ["activation_patching", "effect_size", "cka"]
CALIBRATIONS = CAUSAL_CALIBRATIONS
OUTPUT_DIR = Path(__file__).parent / "results" / "i15_signal_discrimination"

THRESHOLDS = {
    "activation_patching": 0.5,
    "effect_size": 0.8,
    "cka": 0.5,
}


# ---------------------------------------------------------------------------
# Novel analysis: Signal discrimination ratio
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_signal_discrimination(
    model,
    task: str,
    n_prompts: int = 40,
) -> tuple[float, float, list[float], list[float]]:
    """Compute signal discrimination ratio for a task.

    For each prompt:
    1. Compute clean logit diff.
    2. Task-relevant perturbation: patch circuit heads from a corrupted prompt
       (different answer). Measure |logit_diff_change|.
    3. Random perturbation: patch circuit heads with random vectors of the
       same L2 norm. Measure |logit_diff_change|.

    Returns (discrimination_ratio, p_value, task_effects, random_effects).
    """
    tokenizer = model.tokenizer
    heads = get_circuit_heads(task)
    if not heads:
        return float("nan"), float("nan"), [], []

    prompts = generate_prompts(task, tokenizer, n_prompts=n_prompts)
    if len(prompts) < 4:
        return float("nan"), float("nan"), [], []

    correct_ids, incorrect_ids = get_token_ids(prompts, tokenizer)
    if len(correct_ids) < 4:
        return float("nan"), float("nan"), [], []

    heads_by_layer = heads_to_layer_dict(heads)
    hook_names = [f"blocks.{L}.attn.hook_z" for L in heads_by_layer]

    task_effects = []
    random_effects = []

    # Use pairs of prompts: each prompt is "clean", the next is "corrupted"
    n_pairs = min(len(prompts) // 2, len(correct_ids) // 2)

    for i in range(n_pairs):
        clean_idx = 2 * i
        corrupt_idx = 2 * i + 1

        if clean_idx >= len(correct_ids) or corrupt_idx >= len(correct_ids):
            break

        clean_tokens = model.to_tokens(prompts[clean_idx].text)
        corrupt_tokens = model.to_tokens(prompts[corrupt_idx].text)

        # Clean logit diff
        clean_logits = model(clean_tokens)
        clean_ld = logit_diff_from_logits(
            clean_logits, correct_ids[clean_idx], incorrect_ids[clean_idx])

        # Get corrupted activations
        _, corrupt_cache = model.run_with_cache(corrupt_tokens, names_filter=hook_names)

        # Task-relevant perturbation: patch from corrupted prompt
        task_hooks = []
        patch_norms = {}
        for layer, head_list in heads_by_layer.items():
            corrupt_z = corrupt_cache[f"blocks.{layer}.attn.hook_z"]

            def _task_hook(z, hook, _layer=layer, _heads=head_list, _corrupt_z=corrupt_z):
                for H in _heads:
                    z[0, -1, H, :] = _corrupt_z[0, -1, H, :].to(z.device)
                return z

            task_hooks.append((f"blocks.{layer}.attn.hook_z", _task_hook))

            # Record norms for random perturbation matching
            for H in head_list:
                diff = corrupt_z[0, -1, H, :] - clean_logits.new_zeros(corrupt_z.shape[-1])
                patch_norms[(layer, H)] = float(torch.norm(
                    corrupt_cache[f"blocks.{layer}.attn.hook_z"][0, -1, H, :]).item())

        patched_logits = model.run_with_hooks(clean_tokens, fwd_hooks=task_hooks)
        patched_ld = logit_diff_from_logits(
            patched_logits, correct_ids[clean_idx], incorrect_ids[clean_idx])
        task_effect = abs(clean_ld - patched_ld)
        task_effects.append(task_effect)

        # Random perturbation: replace circuit heads with random vectors
        # of the same L2 norm as the corrupted activations
        _, clean_cache = model.run_with_cache(clean_tokens, names_filter=hook_names)

        random_hooks = []
        for layer, head_list in heads_by_layer.items():
            clean_z = clean_cache[f"blocks.{layer}.attn.hook_z"]

            def _random_hook(z, hook, _layer=layer, _heads=head_list, _clean_z=clean_z):
                for H in _heads:
                    clean_act = _clean_z[0, -1, H, :]
                    norm = torch.norm(clean_act).item()
                    random_vec = torch.randn_like(z[0, -1, H, :])
                    random_vec = random_vec / (torch.norm(random_vec) + 1e-8) * norm
                    z[0, -1, H, :] = random_vec.to(z.device)
                return z

            random_hooks.append((f"blocks.{layer}.attn.hook_z", _random_hook))

        random_logits = model.run_with_hooks(clean_tokens, fwd_hooks=random_hooks)
        random_ld = logit_diff_from_logits(
            random_logits, correct_ids[clean_idx], incorrect_ids[clean_idx])
        random_effect = abs(clean_ld - random_ld)
        random_effects.append(random_effect)

    if not task_effects or not random_effects:
        return float("nan"), float("nan"), [], []

    mean_task = np.mean(task_effects)
    mean_random = np.mean(random_effects)
    discrimination_ratio = mean_task / max(mean_random, 1e-8)

    # Wilcoxon signed-rank test (or fall back to simple comparison)
    try:
        from scipy.stats import wilcoxon
        diffs = np.array(task_effects) - np.array(random_effects)
        if np.any(diffs != 0):
            _, p_value = wilcoxon(diffs, alternative="greater")
        else:
            p_value = 1.0
    except ImportError:
        # Fallback: simple paired t-test approximation
        diffs = np.array(task_effects) - np.array(random_effects)
        n = len(diffs)
        if n > 1 and np.std(diffs) > 1e-8:
            t_stat = np.mean(diffs) / (np.std(diffs) / np.sqrt(n))
            # Approximate one-sided p-value from t-distribution
            p_value = float(0.5 * np.exp(-0.5 * t_stat ** 2)) if t_stat > 0 else 1.0
        else:
            p_value = 1.0

    return float(discrimination_ratio), float(p_value), task_effects, random_effects


def run_novel_analysis(
    model,
    tasks: list[str],
    n_prompts: int = 40,
) -> list[EvalResult]:
    """Run signal discrimination analysis for each task."""
    results = []

    for task in tasks:
        try:
            ratio, p_value, task_effs, rand_effs = run_signal_discrimination(
                model, task, n_prompts=n_prompts)

            results.append(EvalResult(
                metric_id="I15.discrimination_ratio",
                value=ratio,
                n_samples=len(task_effs),
                metadata={
                    "task": task,
                    "mean_task_effect": float(np.mean(task_effs)) if task_effs else float("nan"),
                    "mean_random_effect": float(np.mean(rand_effs)) if rand_effs else float("nan"),
                    "n_pairs": len(task_effs),
                },
            ))

            results.append(EvalResult(
                metric_id="I15.p_value",
                value=p_value,
                n_samples=len(task_effs),
                metadata={
                    "task": task,
                    "significant_005": p_value < 0.05 if not np.isnan(p_value) else False,
                },
            ))

        except Exception as e:
            print(f"  [I15 novel analysis] {task} FAILED: {e}")

    return results


# ---------------------------------------------------------------------------
# Protocol runner
# ---------------------------------------------------------------------------

def run_protocol(model, tasks: list[str] | None = None, n_prompts: int = 40,
                 device: str = "cpu", run_cals: bool = True) -> ProtocolResult:
    """Run all I15 metrics + novel signal discrimination analysis. Returns a ProtocolResult."""
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

    # Novel signal discrimination analysis
    print(f"\n{'─' * 60}")
    print(f"  Signal discrimination novel analysis — {len(tasks)} tasks, {n_prompts} prompts")
    print(f"{'─' * 60}")
    mt0 = time.time()
    try:
        novel_results = run_novel_analysis(model, tasks, n_prompts=n_prompts)
        result.metrics["signal_discrimination"] = novel_results
        for r in novel_results:
            task = r.metadata.get("task", "?")
            print(f"    {task:20s}  {r.metric_id:40s}  {r.value:+.4f}")
        print(f"  {len(novel_results)} results in {time.time() - mt0:.1f}s")
    except Exception as e:
        print(f"  [signal_discrimination novel analysis] FAILED: {e}")
        result.metrics["signal_discrimination"] = []

    if run_cals:
        print(f"\n{'=' * 60}")
        print(f"  Calibrations ({len(CALIBRATIONS)})")
        print(f"{'=' * 60}")
        cal_tasks = tasks[:2]
        result.calibrations = run_calibrations(
            model, cal_tasks, CALIBRATIONS, n_prompts=n_prompts)

    result.elapsed_seconds = time.time() - t0
    return result


def signal_discrimination_analysis(result: ProtocolResult) -> list[str]:
    """Analyze results through the signal discrimination lens.

    A circuit that specifically computes a task should respond much more
    strongly to task-relevant perturbations (patching from a different-answer
    prompt) than to random perturbations of the same magnitude. The
    discrimination ratio quantifies this:

    - ratio >> 1: circuit is specifically tuned to task signals
    - ratio ~ 1: circuit responds equally to any perturbation (not specific)
    - ratio < 1: circuit is more sensitive to noise than signal (broken)

    Combined with activation_patching and effect_size:
    - HIGH ratio + HIGH AP: specific and causally important
    - HIGH ratio + LOW AP: specific but not critical (backup circuit?)
    - LOW ratio + HIGH AP: causally important but not specific (general processor)
    - LOW ratio + LOW AP: neither specific nor important
    """
    lines = ["\n  Signal Discrimination Analysis:",
             "  --------------------------------"]

    novel = result.metrics.get("signal_discrimination", [])

    for task in result.tasks:
        lines.append(f"\n    {task}:")

        ap_r = _find(result.metrics.get("activation_patching", []), task)
        es_r = _find(result.metrics.get("effect_size", []), task)
        ratio_r = _find_by_metric_id(novel, "I15.discrimination_ratio", task)
        pval_r = _find_by_metric_id(novel, "I15.p_value", task)

        if ratio_r:
            if np.isnan(ratio_r.value):
                lines.append(f"      Discrimination ratio: N/A")
            elif ratio_r.value > 3.0:
                label = "strongly discriminating"
            elif ratio_r.value > 1.5:
                label = "moderately discriminating"
            elif ratio_r.value > 1.0:
                label = "weakly discriminating"
            else:
                label = "not discriminating (responds more to noise)"
            if not np.isnan(ratio_r.value):
                lines.append(f"      Discrimination ratio:     {ratio_r.value:.4f} — {label}")
                meta = ratio_r.metadata
                lines.append(f"        mean task effect:   {meta.get('mean_task_effect', '?'):.4f}")
                lines.append(f"        mean random effect: {meta.get('mean_random_effect', '?'):.4f}")

        if pval_r and not np.isnan(pval_r.value):
            sig = "***" if pval_r.value < 0.001 else ("**" if pval_r.value < 0.01 else ("*" if pval_r.value < 0.05 else "n.s."))
            lines.append(f"      p-value:                  {pval_r.value:.4f} {sig}")

        if ap_r:
            label = "high" if ap_r.value > THRESHOLDS["activation_patching"] else "low"
            lines.append(f"      Activation patching:      {ap_r.value:.4f} — {label}")

        # Verdict
        if ratio_r is None or np.isnan(ratio_r.value):
            verdict = "INSUFFICIENT DATA"
        elif ratio_r.value > 2.0 and ap_r is not None and ap_r.value > THRESHOLDS["activation_patching"]:
            verdict = "SPECIFIC CIRCUIT — discriminates signal from noise and is causally important"
        elif ratio_r.value > 2.0:
            verdict = "SIGNAL-TUNED — discriminates well but causal role unclear"
        elif ap_r is not None and ap_r.value > THRESHOLDS["activation_patching"]:
            verdict = "GENERAL PROCESSOR — causally important but not signal-specific"
        else:
            verdict = "NONSPECIFIC — neither discriminating nor causally important"
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

    lines.extend(signal_discrimination_analysis(result))

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
