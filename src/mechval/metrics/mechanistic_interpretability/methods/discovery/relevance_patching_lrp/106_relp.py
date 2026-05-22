"""RelP: Relevance Patching via LRP (Measurement C19)
Paper: Jafari, Nanda et al. (2025). NeurIPS 2025.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     C19 — RelP Faithfulness
Categories:     measurement
Validity layer: Measurement
Criteria:       M1 (test-retest reliability), M4 (faithfulness)
Establishes:    Whether LRP-based attribution scores faithfully approximate
                activation patching ground truth at per-component granularity
Requires:       CPU or GPU, model
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Implements Relevance Patching (RelP) from Jafari, Eberle, Khakzar & Nanda
(NeurIPS 2025 MechInterp Workshop Spotlight). RelP replaces local gradients
in standard attribution patching with Layer-wise Relevance Propagation (LRP)
coefficients. Same cost as attribution patching (2 forward + 1 backward pass)
but vastly more faithful: on GPT-2 Large MLP outputs, attribution patching
achieves Pearson r=0.006 with activation patching ground truth; RelP achieves
r=0.956.

For each task and each layer:
1. Compute activation patching ground truth: for each component (attention
   head or MLP), replace its output with a corrupted-input activation and
   measure the change in logit diff.
2. Compute RelP attribution scores: run a clean forward pass and propagate
   relevance backward using the LRP epsilon-rule through linear layers
   and attention blocks.
3. Report Pearson correlation between the two score vectors.

Pass condition: pearson_r > 0.8

Usage:
    uv run python 106_relp.py --tasks ioi --n-prompts 40
    uv run python 106_relp.py --tasks ioi sva --device cpu
"""

import numpy as np
import torch
from scipy.stats import pearsonr

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    generate_prompts,
    get_circuit_info,
    get_token_ids,
    load_model,
    log,
    parse_common_args,
    save_incremental,
    save_results,
)


def _logit_diff(logits: torch.Tensor, correct_id: int, incorrect_id: int) -> float:
    return (logits[0, -1, correct_id] - logits[0, -1, incorrect_id]).item()


@torch.no_grad()
def compute_activation_patching_scores(
    model, prompts, correct_ids, incorrect_ids,
) -> np.ndarray:
    """Ground-truth activation patching: per-component effect on logit diff.

    For each component c (attention head or MLP at each layer), patch its
    output from a corrupted run into the clean run and measure the drop in
    logit diff. The score for component c is:
        AP(c) = logit_diff(clean) - logit_diff(clean with c patched from corrupt)

    Returns a (n_components,) array where n_components = n_layers * (n_heads + 1).
    The +1 is for the MLP at each layer. Layout: [L0H0, L0H1, ..., L0MLP, L1H0, ...].
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    n_components = n_layers * (n_heads + 1)
    n = min(len(prompts), len(correct_ids), len(incorrect_ids))
    total_scores = np.zeros(n_components, dtype=np.float64)

    for i in range(n):
        tokens = model.to_tokens(prompts[i].text)

        # Create corrupted input by shuffling token positions (excluding BOS)
        corrupt_tokens = tokens.clone()
        seq_len = corrupt_tokens.shape[1]
        if seq_len > 2:
            perm = torch.randperm(seq_len - 1) + 1
            corrupt_tokens[0, 1:] = corrupt_tokens[0, perm]

        # Clean and corrupt forward passes with caching
        clean_logits, clean_cache = model.run_with_cache(
            tokens,
            names_filter=lambda n: "hook_z" in n or "hook_mlp_out" in n,
        )
        _, corrupt_cache = model.run_with_cache(
            corrupt_tokens,
            names_filter=lambda n: "hook_z" in n or "hook_mlp_out" in n,
        )
        clean_ld = _logit_diff(clean_logits, correct_ids[i], incorrect_ids[i])

        # Patch each component one at a time
        for L in range(n_layers):
            # Patch each attention head
            for H in range(n_heads):
                def patch_head(z, hook, _L=L, _H=H):
                    z[0, :, _H, :] = corrupt_cache[f"blocks.{_L}.attn.hook_z"][0, :, _H, :]
                    return z

                patched_logits = model.run_with_hooks(
                    tokens,
                    fwd_hooks=[(f"blocks.{L}.attn.hook_z", patch_head)],
                )
                patched_ld = _logit_diff(patched_logits, correct_ids[i], incorrect_ids[i])
                idx = L * (n_heads + 1) + H
                total_scores[idx] += clean_ld - patched_ld

            # Patch MLP
            def patch_mlp(mlp_out, hook, _L=L):
                mlp_out[0, :, :] = corrupt_cache[f"blocks.{_L}.hook_mlp_out"][0, :, :]
                return mlp_out

            patched_logits = model.run_with_hooks(
                tokens,
                fwd_hooks=[(f"blocks.{L}.hook_mlp_out", patch_mlp)],
            )
            patched_ld = _logit_diff(patched_logits, correct_ids[i], incorrect_ids[i])
            idx = L * (n_heads + 1) + n_heads
            total_scores[idx] += clean_ld - patched_ld

        if (i + 1) % 10 == 0:
            log(f"    act-patch: processed {i+1}/{n} prompts")

    total_scores /= max(n, 1)
    return total_scores


def compute_relp_scores(
    model, prompts, correct_ids, incorrect_ids,
) -> np.ndarray:
    """Compute RelP attribution scores using LRP epsilon-rule propagation.

    For each prompt:
    1. Run a clean forward pass caching hook_z and hook_mlp_out at every layer.
    2. Compute the logit-diff scalar and backpropagate to get gradients at
       each component's output.
    3. Apply the LRP epsilon-rule: for each component c with output activation
       a_c and gradient g_c, the relevance is:
           R_c = a_c * g_c / (a_c^2 + epsilon)
       summed over the component's output dimensions. This replaces the raw
       gradient with an LRP coefficient that redistributes relevance
       proportionally to each component's contribution.

    Returns a (n_components,) array with the same layout as activation patching.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    n_components = n_layers * (n_heads + 1)
    n = min(len(prompts), len(correct_ids), len(incorrect_ids))
    total_scores = np.zeros(n_components, dtype=np.float64)
    eps = 1e-6

    for i in range(n):
        tokens = model.to_tokens(prompts[i].text)
        model.zero_grad()

        z_cache = {}
        mlp_cache = {}

        def make_z_hook(layer):
            def hook_fn(z, hook):
                z_cache[layer] = z
                z.retain_grad()
                return z
            return hook_fn

        def make_mlp_hook(layer):
            def hook_fn(mlp_out, hook):
                mlp_cache[layer] = mlp_out
                mlp_out.retain_grad()
                return mlp_out
            return hook_fn

        fwd_hooks = []
        for L in range(n_layers):
            fwd_hooks.append((f"blocks.{L}.attn.hook_z", make_z_hook(L)))
            fwd_hooks.append((f"blocks.{L}.hook_mlp_out", make_mlp_hook(L)))

        with torch.enable_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)
            logit_diff = logits[0, -1, correct_ids[i]] - logits[0, -1, incorrect_ids[i]]
            logit_diff.backward()

        # Compute LRP epsilon-rule relevance for each component
        for L in range(n_layers):
            # Attention heads
            z = z_cache.get(L)
            if z is not None and z.grad is not None:
                # z shape: (batch, seq, n_heads, d_head)
                a = z[0, -1].detach()  # (n_heads, d_head)
                g = z.grad[0, -1].detach()  # (n_heads, d_head)
                for H in range(n_heads):
                    a_h = a[H]  # (d_head,)
                    g_h = g[H]  # (d_head,)
                    # LRP epsilon-rule: R = sum(a * g / (|a| + eps))
                    relevance = (a_h * g_h / (a_h.abs() + eps)).sum().item()
                    idx = L * (n_heads + 1) + H
                    total_scores[idx] += relevance

            # MLP
            mlp_out = mlp_cache.get(L)
            if mlp_out is not None and mlp_out.grad is not None:
                a_mlp = mlp_out[0, -1].detach()  # (d_model,)
                g_mlp = mlp_out.grad[0, -1].detach()  # (d_model,)
                relevance = (a_mlp * g_mlp / (a_mlp.abs() + eps)).sum().item()
                idx = L * (n_heads + 1) + n_heads
                total_scores[idx] += relevance

        model.zero_grad()

        if (i + 1) % 10 == 0:
            log(f"    relp: processed {i+1}/{n} prompts")

    total_scores /= max(n, 1)
    return total_scores


def run_relp(model, tasks: list[str], n_prompts: int = 40) -> list[EvalResult]:
    tokenizer = model.tokenizer
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    results = []

    for task in tasks:
        circuit, all_heads, all_edges = get_circuit_info(task)
        if circuit is None:
            log(f"  {task}: no circuit, skipping")
            continue

        prompts = generate_prompts(task, tokenizer, n_prompts)
        if not prompts:
            log(f"  {task}: no prompts, skipping")
            continue

        correct_ids, incorrect_ids = get_token_ids(prompts, tokenizer)
        if not correct_ids:
            continue

        log(f"  {task}: computing activation patching ground truth...")
        ap_scores = compute_activation_patching_scores(
            model, prompts, correct_ids, incorrect_ids,
        )

        log(f"  {task}: computing RelP (LRP epsilon-rule) scores...")
        relp_scores = compute_relp_scores(
            model, prompts, correct_ids, incorrect_ids,
        )

        # Pearson correlation between RelP and activation patching
        if ap_scores.std() > 1e-12 and relp_scores.std() > 1e-12:
            pearson_r, p_value = pearsonr(ap_scores, relp_scores)
            pearson_r = float(pearson_r)
            p_value = float(p_value)
        else:
            pearson_r = 0.0
            p_value = 1.0

        passed = bool(pearson_r > 0.8)

        log(f"    Pearson r={pearson_r:.4f}  p={p_value:.2e}  [{('PASS' if passed else 'FAIL')}]")

        # Per-layer breakdown
        per_layer = []
        for L in range(n_layers):
            start = L * (n_heads + 1)
            end = start + n_heads + 1
            ap_layer = ap_scores[start:end]
            relp_layer = relp_scores[start:end]
            if ap_layer.std() > 1e-12 and relp_layer.std() > 1e-12:
                r_layer, _ = pearsonr(ap_layer, relp_layer)
                r_layer = float(r_layer)
            else:
                r_layer = 0.0
            per_layer.append({
                "layer": L,
                "pearson_r": r_layer,
                "mean_ap": float(ap_layer.mean()),
                "mean_relp": float(relp_layer.mean()),
            })

        # Top components by RelP score
        n_components = n_layers * (n_heads + 1)
        top_components = []
        for idx in range(n_components):
            L = idx // (n_heads + 1)
            c = idx % (n_heads + 1)
            if c < n_heads:
                name = f"L{L}H{c}"
                in_circuit = (L, c) in all_heads
            else:
                name = f"L{L}MLP"
                in_circuit = False
            top_components.append({
                "component": name,
                "relp_score": float(relp_scores[idx]),
                "ap_score": float(ap_scores[idx]),
                "in_circuit": in_circuit,
            })
        top_components.sort(key=lambda x: abs(x["relp_score"]), reverse=True)

        results.append(EvalResult(
            metric_id="C19.relp_faithfulness",
            value=pearson_r,
            n_samples=len(prompts),
            metadata={
                "task": task,
                "pearson_correlation": pearson_r,
                "p_value": p_value,
                "passed": passed,
                "threshold": 0.8,
                "n_components": n_components,
                "n_layers": n_layers,
                "n_heads": n_heads,
                "per_layer": per_layer,
                "top_components": top_components[:30],
            },
        ))

    return results


def main():
    parser = parse_common_args("C19: RelP Faithfulness (LRP vs Activation Patching)")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("C19: RELP FAITHFULNESS (LRP vs ACTIVATION PATCHING)")
    log("=" * 60)

    out = args.out or "106_relp.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_relp(model, [task], args.n_prompts)
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)
            p = "PASS" if r.metadata["passed"] else "FAIL"
            log(f"  {task}: Pearson r={r.value:.4f}  [{p}]")

    save_results(results, out)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
