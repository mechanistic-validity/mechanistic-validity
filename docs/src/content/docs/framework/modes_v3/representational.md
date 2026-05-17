---
title: "Representational Mode"
description: "What information does the system encode, in what format, and where? — content and geometry without committing to procedure."
---


# Representational Mode

| | |
|---|---|
| Origin | [Pylyshyn (1984)](https://doi.org/10.7551/mitpress/2004.001.0001) — content vs. functional architecture; extended by [Marr (1982)](https://doi.org/10.7551/mitpress/9780262514620.001.0001)'s "representation and algorithm" |
| Question | What information does the system encode, in what format, and where? |
| Licensing evidence | Probing with selectivity baselines (decodable) or IIA/DAS with stated alignment map (causal) |
| Interpretive-validity risk | Conflating "decodable" with "encoded" — the first is a measurement capability, the second is a causal claim |
| Position in partial order | $C > A > R > I_{\text{fun}} > I_{\text{con}} > I_{\text{top}}$ — above implementational, below algorithmic |

## What this mode claims

A verdict tagged `[representational]` specifies three things: (1) *content* — what variables or features are represented, (2) *format* — how that content is encoded (linear direction, subspace, distributed, nonlinear manifold), and (3) *location* — where in the model (layer, position, component) the encoding is accessible.

A representational claim does not commit to how the representation is *used* — that would be algorithmic. It commits only to what is *there*. "The model linearly encodes syntactic number at layer 8, position $p$, in direction $\hat{v}$" is representational. "The model uses the number encoding to route agreement information to the verb position" is algorithmic.

## Formal characterization

Let $h_l^p \in \mathbb{R}^{d_{\text{model}}}$ be the residual-stream activation at layer $l$, position $p$. A linear representational claim asserts the existence of a direction $\hat{v}$ such that:

$$\langle h_l^p, \hat{v} \rangle \text{ tracks a causal variable } Z \text{ with } d' = \frac{\mu_+ - \mu_-}{\sigma_{\text{pooled}}} > \tau$$

for some threshold $\tau$, where $\mu_+$ and $\mu_-$ are the mean projections for the two values of $Z$.

**Decodable vs. causal representational claims:**

- *Decodable*: a probe can extract $Z$ from $h_l^p$ with accuracy above a selectivity baseline ([Hewitt & Liang 2019](https://arxiv.org/abs/1909.03368)). This establishes that the information is *accessible*, not that it is *used*.
- *Causal*: intervening on $h_l^p$ along $\hat{v}$ changes the model's behavior in a manner consistent with changing $Z$. This is the IIA/DAS standard ([Geiger et al. 2021](https://arxiv.org/abs/2106.02997)) — the representation is not merely decodable but load-bearing.

The causal version is strictly stronger. A decodable representation might be an epiphenomenal byproduct; a causal representation is part of the computation.

## What licenses a `[representational]` tag

1. **Geometric form stated explicitly** — linear direction, $k$-dimensional subspace, distributed activation pattern, or nonlinear manifold. The claim must commit to the encoding format.

2. **Decodability demonstrated with baselines:**
   - For probing: selectivity ([Hewitt & Liang 2019](https://arxiv.org/abs/1909.03368)) — accuracy on the target task minus accuracy on a control task of equal difficulty
   - For IIA: random-vector baseline and untrained-model baseline

3. **If the claim is causal (stronger):** Interchange intervention (IIA/DAS) demonstrating that the representation is load-bearing. Swapping the direction $\hat{v}$ between two inputs that differ only in $Z$ should change the model's output accordingly.

4. **Alignment map capacity stated and varied** ([Wu et al. 2024](https://arxiv.org/abs/2305.08809)): If IIA is high only with unconstrained nonlinear maps, the finding is about map flexibility, not linear geometry. The claim should specify what alignment architecture was used and show how IIA degrades as capacity is reduced.

## What does NOT license a `[representational]` tag

- **Probe accuracy without a control task.** A powerful probe can decode almost anything from high-dimensional activations. Without selectivity, high probe accuracy is not evidence of encoding.
- **IIA without stating the alignment map architecture.** A 3-layer MLP alignment map can brute-force almost any interchange intervention. The claim must specify the map's capacity.
- **"The model represents X" based on behavioral evidence alone.** If the model gets the right answer on sentences requiring knowledge of X, that is behavioral (computational/algorithmic), not representational. A representational claim must point to a specific geometric structure in activation space.
- **Reporting PCA directions without testing causal relevance.** PCA finds variance, not computation. High-variance directions may be irrelevant to the task.

<details class="worked-example">
<summary>Worked example: Number encoding in GPT-2 (decodable → causal)</summary>

**Decodable claim (weaker):** "A linear probe trained on GPT-2 Small residual stream at layer 8, subject position, achieves 94% accuracy at classifying singular vs. plural number. Selectivity = 94% - 51% (control task: random binary label) = 43 percentage points." This establishes decodability with a proper baseline.

**Causal claim (stronger):** "IIA with a 1-dimensional linear alignment map at layer 8 achieves interchange accuracy 0.72. Swapping the number direction between 'The cat sits' and 'The cats sit' causes the model to change verb agreement predictions accordingly. Baseline (random direction swap): 0.03." This establishes that the linear encoding is causally load-bearing.

**Upgrade to algorithmic:** "The number representation at layer 8 is *read* by attention heads at layers 9-10 via QK composition, which route agreement information to the verb position." This crosses from representational (what is encoded) to algorithmic (how it is used).
</details>

<details class="worked-example">
<summary>Anti-pattern: representation without geometry</summary>

**Bad claim:** "The model represents the indirect object."

Why this fails: No geometric form is specified (linear? subspace? distributed?). No location (which layer? which position?). No format commitment. This is a computational-level statement ("the model tracks this variable") dressed up as representational. A proper representational claim would be: "The IO name is linearly decodable from the residual stream at layer 7, IO position, in a 3-dimensional subspace with $d' = 2.4$."
</details>

## Upgrade and downgrade

| Direction | What's required |
|---|---|
| $I_{\text{fun}} \to R$ (upgrade from functional) | Demonstrate that the component's function is *representational* — encoding a specific variable — rather than just performing a mathematical operation. A head that suppresses a direction is functional; showing that direction encodes a specific causal variable makes it representational. |
| $R \to A$ (upgrade to algorithmic) | Show ordering, composition, and procedure: the representations flow through specific steps that jointly produce the behavior. How is the representation *read* and *used*? |
| $R \to I_{\text{fun}}$ (downgrade) | If the causal test fails (IIA is at baseline), the claim downgrades to "this component produces outputs with this geometric structure" — a functional/statistical characterization, not a representational one. |

## Instruments that provide representational-level evidence

- **E01 (PCA dimensionality)** — geometry of activation subspaces per layer
- **E03 (RSA)** — whether neural similarity structure matches task-variable structure
- **E05 (Intrinsic dimension)** — manifold complexity of representations
- **E10 (Subspace alignment)** — whether circuit heads share representational geometry
- **A03 (Interchange intervention)** — causal relevance of specific directions

## Key references

- Pylyshyn, Z. (1984). [*Computation and Cognition.*](https://doi.org/10.7551/mitpress/2004.001.0001) MIT Press. — Representation vs. functional architecture distinction.
- Hewitt, J. & Liang, P. (2019). ["Designing and Interpreting Probes with Control Tasks."](https://arxiv.org/abs/1909.03368) *EMNLP 2019.* — Selectivity baseline for probing.
- Geiger, A., et al. (2021). ["Causal Abstractions of Neural Networks."](https://arxiv.org/abs/2106.02997) *NeurIPS 2021.* — IIA/DAS as causal representational evidence.
- Wu, Z., et al. (2024). ["Interpretability at Scale: Identifying Causal Mechanisms in Alpaca."](https://arxiv.org/abs/2305.08809) *NeurIPS 2024.* — DAS/boundless DAS; alignment map capacity and its confounds.
- Li, K., et al. (2023). ["Emergent World Representations: Exploring a Sequence Model Trained on a Synthetic Task."](https://arxiv.org/abs/2210.13382) *ICLR 2023.* — Linear probing + intervention on Othello-GPT as a representational + causal claim.
