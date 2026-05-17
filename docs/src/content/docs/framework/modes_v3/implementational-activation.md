---
title: "Implementational Mode: Activation-Statistical"
description: "What are the distributional properties of activations at these components? — characterizing behavior without committing to function."
---


# Implementational Mode: Activation-Statistical

| | |
|---|---|
| Origin | Descriptive neuroscience — [Hubel & Wiesel (1962)](https://doi.org/10.1113/jphysiol.1962.sp006837) tuning curves, firing-rate histograms, receptive-field mapping |
| Question | What are the distributional properties of activations at the identified components? |
| Licensing evidence | Summary statistics on a representative corpus + stability across subsamples |
| Interpretive-validity risk | Interpreting statistics as function — a bimodal distribution is not evidence of a binary computation |
| Position in partial order | Between $I_{\text{top}}$ and $I_{\text{fun}}$ — richer than "which components" but weaker than "what they compute" |

## What this mode claims

A verdict tagged `[implementational-statistical]` characterizes *what the activations look like* — their distribution, sparsity, clustering, geometric properties, or temporal dynamics — without committing to what computation produces them or what they represent. It is the MI analogue of recording a neuron's firing-rate histogram without proposing a computational role.

This mode sits between topographic (which components matter) and functional (what they do). A statistical characterization can *suggest* function — bimodality might suggest a gating role, heavy tails might suggest rare-event detection — but the suggestion requires separate causal evidence to become a functional claim.

## Formal characterization

Let $a_i(x) \in \mathbb{R}^d$ be the activation of component $c_i$ on input $x$. An activation-statistical claim characterizes the distribution:

$$P(a_i) = \mathbb{E}_{x \sim \mathcal{D}}[\delta(a - a_i(x))]$$

through summary statistics: mean $\mu$, variance $\sigma^2$, sparsity (fraction of inputs where $\|a_i(x)\| > \tau$), distribution shape (unimodal/bimodal/heavy-tailed), or geometric properties (effective rank of the covariance $\text{Cov}(a_i)$).

The claim is explicitly *about the distribution on a specific corpus* $\mathcal{D}$, not about the component's "nature." Statistics on the Pile-10k are statistics on the Pile-10k — extrapolation to "the component's behavior in general" requires domain-coverage arguments.

## What licenses an `[implementational-statistical]` tag

1. **Summary statistics** computed on a stated, representative corpus — mean, variance, sparsity, distribution shape (histogram or kernel density), effective rank, tail behavior.

2. **Corpus and coverage stated** — the statistics are about the component's behavior *on this data*. The corpus composition matters: Pile-10k vs. OpenWebText vs. code vs. multilingual will produce different statistics for the same component.

3. **Stability** — the statistics are consistent across random subsamples of the corpus. Bootstrap confidence intervals should be reported for key quantities. An unstable statistic (wide CI) is a characterization of noise, not of the component.

4. **Context examples (optional but strengthening)** — top-activating and bottom-activating contexts that illustrate what drives the distribution. These are illustrative, not evidential — they help readers build intuition but do not constitute functional claims.

## What does NOT license an `[implementational-statistical]` tag

- **Statistics on a non-representative corpus.** If you compute statistics only on IOI prompts, you have statistics for IOI prompts, not for the component's general behavior. Task-specific statistics are fine but must be labeled as such.
- **Interpreting distributional properties as computation.** "SAE feature 1247 has bimodal activations, so it implements a binary gate." Bimodality is a statistical observation. The gating claim is functional and requires causal evidence (does forcing the feature to one mode change behavior accordingly?).
- **Activation magnitude as importance.** High mean activation does not imply causal importance. A component can be highly active but causally irrelevant (its output might be canceled downstream).
- **Single-example statistics.** Reporting that "on this one prompt, head L5H3 has attention entropy 0.2" is an observation, not a statistical characterization. The claim requires distributional evidence.

<details class="worked-example">
<summary>Worked example: SAE feature activation profile</summary>

**Claim.** SAE feature 1247 in GPT-2 Small (layer 8, 32k SAE) has the following activation-statistical profile on Pile-10k: mean activation $\mu = 0.31 \pm 0.02$, fires above threshold ($|a| > 1.0$) on 12.3% of tokens (95% CI: 11.8%-12.8%), bimodal distribution with peaks at $-0.5$ and $+1.2$, effective rank of output covariance = 3.2. Top-activating contexts are predominantly tokens following opening parentheses. `[implementational-statistical]`

**Why this is statistical, not functional:** The profile describes *what the activations look like* — distribution shape, sparsity, context correlation. It does not claim the feature "detects parentheses" (that would be functional) or "encodes nesting depth" (representational). The correlation with parentheses is descriptive, not causal.

**Upgrade path:** To make a functional claim, you would need to show that *intervening* on the feature (clamping it to one mode vs. the other) changes the model's behavior in a manner consistent with a proposed function. The statistical profile generates hypotheses; causal experiments test them.
</details>

<details class="worked-example">
<summary>Anti-pattern: statistics as explanation</summary>

**Bad claim:** "Head L9H9 has high mean activation on IOI prompts (mean DLA = 1.8), confirming its role as the name-mover head."

Why this fails on two levels: (1) High activation magnitude is not evidence of a specific function — it could be high for many reasons. (2) The "confirmation" conflates a statistical observation with an algorithmic characterization that requires separate evidence (OV analysis, attention pattern characterization). The statistical mode says only what the activations look like, not what they mean.
</details>

## Upgrade and downgrade

| Direction | What's required |
|---|---|
| $I_{\text{top}} \to I_{\text{stat}}$ (from topographic) | Go beyond "these components matter" to characterize *how they behave statistically*. |
| $I_{\text{stat}} \to I_{\text{fun}}$ (to functional) | Demonstrate that the statistical properties correspond to a specific input-output function via causal intervention. Does forcing the activation to a specific value produce predictable output changes? |
| $I_{\text{stat}} \to R$ (to representational) | Show that the activation distribution *tracks a specific variable* — not just that it has a specific shape, but that the shape corresponds to a causal variable in the data. |

## Instruments that provide statistical-level evidence

- **E01 (PCA dimensionality)** — effective rank and variance explained per component
- **E02 (Participation ratio)** — spectral concentration of output covariance
- **E05 (Intrinsic dimension)** — manifold dimensionality of activation clouds
- **E06 (Persistent homology)** — topological features of activation geometry
- **D05 (Per-token NLL)** — positional statistics of where the circuit is most active

## Key references

- Olah, C., et al. (2020). ["Zoom In: An Introduction to Circuits."](https://distill.pub/2020/circuits/zoom-in/) *Distill.* — The activation-pattern characterization tradition; feature visualization as statistical description.
- Bricken, T., et al. (2023). ["Towards Monosemanticity: Decomposing Language Models With Dictionary Learning."](https://transformer-circuits.pub/2023/monosemantic-features/index.html) *Anthropic Transformer Circuits Thread.* — Activation-statistical characterization of SAE features (top contexts, frequency, distribution shape).
- Cunningham, H., et al. (2023). ["Sparse Autoencoders Find Highly Interpretable Features in Language Models."](https://arxiv.org/abs/2309.08600) *ICLR 2024.* — Feature dashboards as statistical characterization.
- Gurnee, W., et al. (2023). ["Finding Neurons in a Haystack: Case Studies with Sparse Probing."](https://arxiv.org/abs/2305.01610) *TMLR.* — Single-neuron activation profiles and their limits.
- Facco, E., et al. (2017). ["Estimating the intrinsic dimension of datasets by a minimal neighborhood information."](https://doi.org/10.1038/s41598-017-11873-y) *Scientific Reports.* — Two-NN estimator for intrinsic dimension of activation manifolds.
