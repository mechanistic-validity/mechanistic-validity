"""GradSAE Causal Influence Score (Evaluation EX17)
Paper: Shu, Wu, Zhao, Du, Liu (2025). EMNLP 2025. arXiv:2505.08080
=============================================
Instrument:     EX17 --- GradSAE Causal Influence Score
Categories:     evaluation
Validity layer: Internal
Criteria:       I1 Component Necessity, I2 Compositional Sufficiency
Establishes:    Whether SAE latent activations are faithful proxies for
                causal influence on model output, or whether high-activation
                features can be causally inert (dissociated)
Requires:       CPU or GPU, model, SAE (optional)
=============================================

Implements the activation x gradient causal influence scoring from
Shu, Wu, Zhao, Du, Liu (EMNLP 2025, arXiv:2505.08080).

For each prompt, computes per-SAE-latent causal influence as the product
of the latent activation and the gradient of the target output probability
with respect to that activation. This yields a ranking of features by
causal contribution rather than by activation magnitude alone.

The key diagnostic is the causal dissociation rate: the fraction of
top-by-activation features that do NOT appear in the top-by-causal-influence
ranking. High dissociation means activation magnitude is a poor proxy for
causal importance --- the model has features that fire strongly but do not
matter for the output.

Core logic:
1. Run model forward through an SAE hook point, capturing latent activations.
2. Compute gradients of target-token log-probability w.r.t. latent activations.
3. Causal influence score = activation * gradient (per feature, per prompt).
4. Rank features by activation magnitude and by causal influence.
5. Compute dissociation rate = |top_k_by_act - top_k_by_causal| / top_k.

Pass condition: causal_dissociation_rate < 0.3

Usage:
    uv run python 113_gradsae_causal.py --n-prompts 50
    uv run python 113_gradsae_causal.py --model gpt2 --device cpu --top-k 20
"""

import numpy as np
import torch

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    InstrumentInfo,
    generate_prompts,
    get_token_ids,
    load_model,
    log,
    parse_common_args,
    save_results,
)

INSTRUMENT_INFO = InstrumentInfo(
    name="GradSAE Causal Influence Score",
    paper_ref="Shu et al. EMNLP 2025",
    paper_cite=(
        "Shu, Wu, Zhao, Du, Liu 2025, "
        "GradSAE: Gradient-based Feature Attribution for Sparse Autoencoders "
        "(EMNLP 2025, arXiv:2505.08080)"
    ),
    description=(
        "For each active SAE latent, computes activation x gradient "
        "(gradient of output probability w.r.t. latent activation) to "
        "produce per-feature causal influence scores. The dissociation "
        "rate between activation-ranked and causal-ranked features is "
        "the primary diagnostic."
    ),
    category="evaluation",
    tier="established",
    origin="external",
)

DISSOCIATION_THRESHOLD = 0.3


def _get_hook_name(model, hook_name: str | None) -> str:
    """Resolve the hook point for capturing latent activations.

    Defaults to the MLP output of the middle layer, which is a
    standard choice for SAE analysis on GPT-2 scale models.
    """
    if hook_name is not None:
        return hook_name
    mid_layer = model.cfg.n_layers // 2
    return f"blocks.{mid_layer}.hook_mlp_out"


def _compute_gradsae_scores(
    model,
    prompt_text: str,
    target_token_id: int,
    hook_name: str,
    artifact=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute activation magnitudes and causal influence scores at hook_name.

    Returns:
        activations: (d_hook,) activation magnitudes at the last sequence position.
        gradients: (d_hook,) gradients of target log-prob w.r.t. activations.
        causal_scores: (d_hook,) element-wise activation * gradient.
    """
    tokens = model.to_tokens(prompt_text)

    # Storage for the captured activation
    captured = {}

    def fwd_hook(value, hook):
        # value shape: (batch, seq, d_hook)
        # Keep the last-token activation and retain grad
        act = value[:, -1, :]  # (1, d_hook)
        act.retain_grad()
        captured["act"] = act
        return value

    # Forward with hook, gradients enabled
    model.zero_grad()
    logits = model.run_with_hooks(
        tokens,
        fwd_hooks=[(hook_name, fwd_hook)],
    )

    # Target log-probability at the last position
    last_logits = logits[0, -1]  # (vocab,)
    log_probs = torch.log_softmax(last_logits, dim=-1)
    target_log_prob = log_probs[target_token_id]

    # Backward to get gradient w.r.t. the captured activation
    target_log_prob.backward()

    act = captured["act"].detach().squeeze(0)  # (d_hook,)
    grad = captured["act"].grad.detach().squeeze(0)  # (d_hook,)

    # If an SAE artifact is provided, project through encoder
    if artifact is not None:
        # artifact assumed to have W_enc (d_hook, d_sae) and b_enc (d_sae)
        W_enc = artifact.W_enc.detach()  # (d_hook, d_sae)
        b_enc = artifact.b_enc.detach() if hasattr(artifact, "b_enc") else torch.zeros(W_enc.shape[1], device=act.device)
        latent_act = act @ W_enc + b_enc  # (d_sae,)
        latent_grad = grad @ W_enc  # (d_sae,) chain rule through linear encoder
        causal_scores = latent_act * latent_grad
        return latent_act, latent_grad, causal_scores

    # Without SAE: treat the raw hook activations as "latents"
    causal_scores = act * grad
    return act, grad, causal_scores


def run_gradsae_causal(
    model,
    tasks: list[str] | None = None,
    n_prompts: int = 50,
    artifact=None,
    hook_name: str | None = None,
    top_k: int = 20,
) -> list[EvalResult]:
    """Compute GradSAE causal influence dissociation across tasks.

    For each task, generates prompts and computes the fraction of
    top-by-activation features that are NOT in the top-by-causal-influence
    set (the causal dissociation rate).

    Args:
        model: HookedTransformer instance.
        tasks: list of task names (default: CIRCUIT_TASKS).
        n_prompts: number of prompts per task.
        artifact: optional SAE object with W_enc / b_enc attributes.
        hook_name: TransformerLens hook point name for activation capture.
        top_k: number of top features to compare between rankings.

    Returns:
        List of EvalResult, one per task plus an aggregate.
    """
    if tasks is None:
        tasks = CIRCUIT_TASKS
    resolved_hook = _get_hook_name(model, hook_name)
    log(f"  GradSAE causal influence at hook: {resolved_hook}")
    log(f"  top_k={top_k}, n_prompts={n_prompts}")

    results = []
    all_dissociation_rates = []

    for task in tasks:
        prompts = generate_prompts(task, model.tokenizer, n_prompts=n_prompts)
        if not prompts:
            log(f"    {task}: no prompts, skipping")
            continue

        correct_ids, incorrect_ids = get_token_ids(prompts, model.tokenizer)
        if not correct_ids:
            log(f"    {task}: no valid token ids, skipping")
            continue

        task_dissociations = []
        task_mean_causal = []
        per_prompt_details = []

        for i, prompt in enumerate(prompts):
            if i >= len(correct_ids):
                break

            target_id = correct_ids[i]
            try:
                act, grad, causal = _compute_gradsae_scores(
                    model, prompt.text, target_id, resolved_hook, artifact
                )
            except Exception as e:
                log(f"    {task} prompt {i}: error {e}")
                continue

            n_features = act.shape[0]
            k = min(top_k, n_features)

            # Rank by activation magnitude (absolute value)
            top_by_act = set(torch.topk(act.abs(), k).indices.tolist())
            # Rank by causal influence (absolute value)
            top_by_causal = set(torch.topk(causal.abs(), k).indices.tolist())

            # Dissociation: fraction of top-by-activation NOT in top-by-causal
            dissociation = len(top_by_act - top_by_causal) / k
            task_dissociations.append(dissociation)
            task_mean_causal.append(causal.abs().mean().item())

            per_prompt_details.append({
                "prompt_index": i,
                "dissociation": dissociation,
                "mean_causal_influence": causal.abs().mean().item(),
                "max_causal_influence": causal.abs().max().item(),
                "top_act_indices": sorted(top_by_act),
                "top_causal_indices": sorted(top_by_causal),
                "overlap_count": len(top_by_act & top_by_causal),
            })

        if not task_dissociations:
            log(f"    {task}: no valid results")
            continue

        mean_dissociation = float(np.mean(task_dissociations))
        std_dissociation = float(np.std(task_dissociations))
        passed = mean_dissociation < DISSOCIATION_THRESHOLD
        all_dissociation_rates.append(mean_dissociation)

        log(f"    {task}: dissociation={mean_dissociation:.4f} "
            f"+/- {std_dissociation:.4f} "
            f"({'PASS' if passed else 'FAIL'})")

        results.append(EvalResult(
            metric_id="EX17.gradsae_causal_influence",
            value=mean_dissociation,
            n_samples=len(task_dissociations),
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": task,
                "causal_dissociation_rate": mean_dissociation,
                "causal_dissociation_std": std_dissociation,
                "mean_causal_influence": float(np.mean(task_mean_causal)),
                "n_prompts_evaluated": len(task_dissociations),
                "top_k": top_k,
                "hook_name": resolved_hook,
                "passed": passed,
                "threshold": DISSOCIATION_THRESHOLD,
                "per_prompt": per_prompt_details,
            },
        ))

    # Aggregate result across all tasks
    if all_dissociation_rates:
        agg_mean = float(np.mean(all_dissociation_rates))
        agg_std = float(np.std(all_dissociation_rates))
        agg_passed = agg_mean < DISSOCIATION_THRESHOLD
        log(f"  Aggregate: dissociation={agg_mean:.4f} +/- {agg_std:.4f} "
            f"({'PASS' if agg_passed else 'FAIL'})")

        results.append(EvalResult(
            metric_id="EX17.gradsae_causal_influence",
            value=agg_mean,
            n_samples=sum(r.n_samples for r in results),
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": "aggregate",
                "causal_dissociation_rate": agg_mean,
                "causal_dissociation_std": agg_std,
                "n_tasks_evaluated": len(all_dissociation_rates),
                "per_task_rates": {
                    r.metadata["task"]: r.metadata["causal_dissociation_rate"]
                    for r in results if r.metadata.get("task") != "aggregate"
                },
                "top_k": top_k,
                "hook_name": resolved_hook,
                "passed": agg_passed,
                "threshold": DISSOCIATION_THRESHOLD,
            },
        ))

    return results


def main():
    parser = parse_common_args("EX17: GradSAE Causal Influence Score")
    parser.add_argument("--top-k", type=int, default=20,
                        help="Number of top features to compare (default: 20)")
    parser.add_argument("--hook-name", default=None,
                        help="Hook point for activation capture (default: mid-layer MLP out)")
    args = parser.parse_args()

    model = load_model(args.model, args.device)

    log("=" * 60)
    log("EX17: GRADSAE CAUSAL INFLUENCE SCORE")
    log("=" * 60)

    tasks = args.tasks or CIRCUIT_TASKS
    results = run_gradsae_causal(
        model,
        tasks=tasks,
        n_prompts=args.n_prompts,
        hook_name=args.hook_name,
        top_k=args.top_k,
    )

    out = args.out or "113_gradsae_causal.json"
    save_results(results, out)
    log(f"\nDone. {len(results)} results.")


if __name__ == "__main__":
    main()
