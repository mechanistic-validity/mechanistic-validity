"""DMSAE Core Stability Score (Measurement M9)
Paper: Martin-Linares, Ling (2025). arXiv:2512.24975
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     M09 — DMSAE Core Stability
Categories:     measurement
Validity layer: Measurement
Criteria:       M1 Reliability
Establishes:    Which SAE features are stable under iterative
                distillation — a reliability diagnostic for the
                decomposition
Requires:       Model + hook point (artifact optional)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Based on Martin-Linares & Ling, arXiv:2512.24975.

Operationalizes M1 Reliability via iterative distillation:
    1. Train a small SAE on activations at a hook point.
    2. Identify high gradient-times-activation features, mark as "core".
    3. Reinitialize non-core features, retrain.
    4. After n cycles, record which features converged into the stable
       core. Core membership rate = reliability score.

Key finding from the paper: only 197/65,000 features in a 65k SAE
are stable across distillation cycles.

Pass condition: report-only (no pass/fail). Uses value > 0 as trivial
pass since any nonzero core fraction is informative.

Usage:
    uv run python 115_core_stability.py --model gpt2 --device cpu
    uv run python 115_core_stability.py --hook blocks.6.hook_resid_pre
"""

import numpy as np
import torch
import torch.nn as nn

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    InstrumentInfo,
    generate_prompts,
    get_token_ids,
    load_model,
    log,
    parse_common_args,
    save_incremental,
    save_results,
)

INSTRUMENT_INFO = InstrumentInfo(
    name="DMSAE Core Stability (Martin-Linares & Ling 2025)",
    paper_ref="Martin-Linares & Ling, arXiv:2512.24975",
    paper_cite="Martin-Linares & Ling 2025, DMSAE Core Stability",
    description=(
        "Iterative distillation identifies which SAE features are "
        "reliably recovered across training cycles — a reliability "
        "diagnostic. Only a small fraction of features form a stable core."
    ),
    category="measurement",
    tier="measurement_theory",
    origin="established",
)


class _MiniSAE(nn.Module):
    """Minimal sparse autoencoder for core stability evaluation.

    Single-layer ReLU encoder with L1 sparsity, tied decoder.
    """

    def __init__(self, d_model: int, n_features: int, l1_coeff: float = 1e-3):
        super().__init__()
        self.d_model = d_model
        self.n_features = n_features
        self.l1_coeff = l1_coeff

        self.encoder = nn.Linear(d_model, n_features)
        self.decoder = nn.Linear(n_features, d_model, bias=False)
        self.b_dec = nn.Parameter(torch.zeros(d_model))

        # Xavier init
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.encoder.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.encoder(x - self.b_dec))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encode(x)
        x_hat = self.decoder(h) + self.b_dec
        recon_loss = (x - x_hat).pow(2).mean()
        l1_loss = h.abs().mean()
        return x_hat, recon_loss + self.l1_coeff * l1_loss

    def reinit_features(self, keep_mask: torch.Tensor) -> None:
        """Reinitialize encoder/decoder rows for features NOT in keep_mask."""
        drop = ~keep_mask
        n_drop = drop.sum().item()
        if n_drop == 0:
            return
        with torch.no_grad():
            nn.init.xavier_uniform_(
                self.encoder.weight[drop].unsqueeze(0)
            ).squeeze_(0)
            self.encoder.weight[drop] = self.encoder.weight[drop].clone()
            nn.init.zeros_(self.encoder.bias[drop])
            # Decoder columns correspond to feature indices
            nn.init.xavier_uniform_(
                self.decoder.weight[:, drop].unsqueeze(0)
            ).squeeze_(0)
            self.decoder.weight[:, drop] = self.decoder.weight[:, drop].clone()


def _collect_activations(
    model,
    prompts: list,
    hook_name: str,
    max_tokens: int = 4096,
) -> torch.Tensor:
    """Collect activations at hook_name, concatenated over prompts.

    Returns a (total_tokens, d_model) tensor on CPU.
    """
    all_acts = []
    total = 0
    for p in prompts:
        if total >= max_tokens:
            break
        tokens = model.to_tokens(p.text)
        _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
        acts = cache[hook_name].detach().cpu()  # (1, seq, d_model)
        acts = acts.reshape(-1, acts.shape[-1])
        all_acts.append(acts)
        total += acts.shape[0]
    if not all_acts:
        return torch.empty(0)
    result = torch.cat(all_acts, dim=0)[:max_tokens]
    return result


def _train_sae(
    sae: _MiniSAE,
    data: torch.Tensor,
    n_steps: int = 200,
    batch_size: int = 64,
    lr: float = 3e-4,
) -> _MiniSAE:
    """Train SAE on data for n_steps mini-batch steps."""
    device = next(sae.parameters()).device
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
    n = data.shape[0]
    for step in range(n_steps):
        idx = torch.randint(0, n, (min(batch_size, n),))
        batch = data[idx].to(device)
        _, loss = sae(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return sae


def _identify_core_features(
    sae: _MiniSAE,
    data: torch.Tensor,
    top_frac: float = 0.1,
) -> torch.Tensor:
    """Identify core features via gradient-times-activation importance.

    Returns a boolean mask of shape (n_features,) marking the top
    fraction of features by importance.
    """
    device = next(sae.parameters()).device
    # Use a subset for importance scoring
    subset = data[:min(512, data.shape[0])].to(device).requires_grad_(False)

    sae.zero_grad()
    h = sae.encode(subset)  # (batch, n_features)
    x_hat = sae.decoder(h) + sae.b_dec
    recon_loss = (subset - x_hat).pow(2).mean()
    recon_loss.backward()

    # Gradient w.r.t. encoder weight: importance = |grad * activation|
    # activation magnitude per feature = mean |h| over batch
    act_magnitude = h.detach().abs().mean(dim=0)  # (n_features,)
    grad_magnitude = sae.encoder.weight.grad.abs().mean(dim=1)  # (n_features,)
    importance = act_magnitude * grad_magnitude

    k = max(1, int(top_frac * sae.n_features))
    _, top_idx = importance.topk(k)
    mask = torch.zeros(sae.n_features, dtype=torch.bool, device=device)
    mask[top_idx] = True
    return mask.cpu()


def _run_distillation_cycles(
    data: torch.Tensor,
    d_model: int,
    n_features: int,
    n_cycles: int,
    device: str,
    train_steps: int = 200,
    top_frac: float = 0.1,
) -> tuple[float, list[int], list[float]]:
    """Run iterative distillation and return core stability results.

    Returns:
        core_fraction: fraction of features in stable core after all cycles.
        core_sizes: list of core sizes at each cycle.
        cycle_overlaps: Jaccard overlap of core sets between consecutive cycles.
    """
    sae = _MiniSAE(d_model, n_features).to(device)
    _train_sae(sae, data, n_steps=train_steps)

    prev_core: torch.Tensor | None = None
    core_sizes: list[int] = []
    cycle_overlaps: list[float] = []

    for cycle in range(n_cycles):
        core_mask = _identify_core_features(sae, data, top_frac=top_frac)
        core_size = int(core_mask.sum().item())
        core_sizes.append(core_size)
        log(f"      cycle {cycle+1}/{n_cycles}: core size = {core_size}/{n_features}")

        if prev_core is not None:
            # Jaccard overlap between consecutive core sets
            intersection = (prev_core & core_mask).sum().float()
            union = (prev_core | core_mask).sum().float()
            overlap = (intersection / union).item() if union > 0 else 0.0
            cycle_overlaps.append(overlap)
            log(f"        overlap with previous core: {overlap:.4f}")

        prev_core = core_mask.clone()

        # Reinitialize non-core features and retrain
        sae.reinit_features(core_mask.to(device))
        _train_sae(sae, data, n_steps=train_steps)

    # Final core identification after last retrain
    final_core = _identify_core_features(sae, data, top_frac=top_frac)
    final_core_size = int(final_core.sum().item())

    # Stable core: features that were in core in BOTH last cycle and final
    if prev_core is not None:
        stable = (prev_core & final_core).sum().item()
    else:
        stable = final_core_size

    core_fraction = stable / n_features if n_features > 0 else 0.0

    return core_fraction, core_sizes, cycle_overlaps


@torch.no_grad()
def run_core_stability(
    model,
    tasks: list[str] | None = None,
    n_prompts: int = 50,
    artifact=None,
    hook_name: str | None = None,
    n_cycles: int = 3,
    stability_threshold: float = 0.8,
) -> list[EvalResult]:
    """Run DMSAE core stability analysis.

    Trains a small SAE on model activations and iteratively distills
    to identify the stable feature core.

    Args:
        model: HookedTransformer model.
        tasks: List of task names to collect prompts from.
        n_prompts: Number of prompts per task for activation collection.
        artifact: Optional artifact adapter (unused, for interface compat).
        hook_name: Hook point for activation collection.
        n_cycles: Number of distillation cycles.
        stability_threshold: Jaccard overlap threshold for "stable" label.

    Returns:
        List of EvalResult with core_fraction scores.
    """
    if tasks is None:
        tasks = CIRCUIT_TASKS

    tokenizer = model.tokenizer
    device = str(model.cfg.device)
    d_model = model.cfg.d_model

    # Scale SAE width relative to d_model (8x expansion, capped for speed)
    n_features = min(d_model * 8, 4096)

    effective_hook = hook_name or "blocks.0.hook_resid_pre"

    results = []

    for task in tasks:
        log(f"  {task}: collecting activations at {effective_hook}")
        prompts = generate_prompts(task, tokenizer, n_prompts)
        if not prompts:
            log(f"  {task}: no prompts, skipping")
            continue

        # We need activations, so temporarily enable grad for SAE training
        with torch.set_grad_enabled(True):
            data = _collect_activations(model, prompts, effective_hook)

        if data.shape[0] == 0:
            log(f"  {task}: no activations collected, skipping")
            continue

        log(f"    collected {data.shape[0]} tokens, d_model={d_model}, "
            f"n_features={n_features}")

        with torch.set_grad_enabled(True):
            core_fraction, core_sizes, cycle_overlaps = _run_distillation_cycles(
                data=data,
                d_model=d_model,
                n_features=n_features,
                n_cycles=n_cycles,
                device=device,
            )

        # Stability: did the last overlap exceed threshold?
        mean_overlap = float(np.mean(cycle_overlaps)) if cycle_overlaps else 0.0
        is_stable = mean_overlap >= stability_threshold

        passed = core_fraction > 0  # Trivial pass: report-only metric

        log(f"    core_fraction={core_fraction:.4f}  "
            f"mean_overlap={mean_overlap:.4f}  "
            f"stable={is_stable}  [{'PASS' if passed else 'FAIL'}]")

        results.append(EvalResult(
            metric_id="M9.core_stability",
            value=core_fraction,
            n_samples=int(data.shape[0]),
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": task,
                "core_fraction": core_fraction,
                "core_sizes": core_sizes,
                "cycle_overlaps": cycle_overlaps,
                "mean_overlap": mean_overlap,
                "is_stable": is_stable,
                "stability_threshold": stability_threshold,
                "passed": passed,
                "n_cycles": n_cycles,
                "n_features": n_features,
                "d_model": d_model,
                "hook_name": effective_hook,
                "n_prompts_used": len(prompts),
            },
        ))

    return results


def main():
    parser = parse_common_args("M9: DMSAE Core Stability Score")
    parser.add_argument("--hook", default=None,
                        help="Hook point for activation collection")
    parser.add_argument("--n-cycles", type=int, default=3,
                        help="Number of distillation cycles (default: 3)")
    parser.add_argument("--stability-threshold", type=float, default=0.8,
                        help="Jaccard overlap threshold for stability (default: 0.8)")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("M9: DMSAE CORE STABILITY SCORE")
    log("=" * 60)

    out = args.out or "115_core_stability.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_core_stability(
            model, [task],
            n_prompts=args.n_prompts,
            hook_name=args.hook,
            n_cycles=args.n_cycles,
            stability_threshold=args.stability_threshold,
        )
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)
            log(f"  {task}: core_fraction={r.value:.4f}")

    save_results(results, out, args=args)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
