---
title: "Implementational Mode: Topographic"
description: "Which components are involved? — a map of where the computation happens without explaining what happens there."
---


# Implementational Mode: Topographic

| | |
|---|---|
| Origin | [Craver (2007)](https://doi.org/10.1093/acprof:oso/9780199299317.001.0001) — decomposition; [Bechtel & Richardson (2010)](https://doi.org/10.7551/mitpress/8328.001.0001) — localization |
| Question | Which components are causally involved in this behavior? |
| Licensing evidence | Ablation (necessity) + locus specificity + counterfactual baseline |
| Interpretive-validity risk | Treating a map as an explanation — knowing *where* is not knowing *what* or *how* |
| Position in partial order | $I_{\text{top}}$ — lowest implementational commitment (a set, not a graph) |

## What this mode claims

A verdict tagged `[implementational-topographic]` identifies a set of components (heads, neurons, layers, SAE features) that participate in producing a behavior. It makes no commitment to what they do, how they are connected, or what procedure they execute. It is a *map* — a spatial answer to the question "where does this happen?"

This is the most commonly established claim in circuit discovery and the appropriate default for any ablation-based result. There is nothing lesser about it. Most of the field's most important findings — the IOI heads, induction heads, Greater-Than components — were first established topographically before being characterized at higher modes.

## Formal characterization

Let $C = \{c_1, \ldots, c_k\}$ be a proposed circuit (a set of attention heads and/or MLPs). A topographic claim asserts:

$$\forall c_i \in C: \quad \mathbb{E}_{x \sim \mathcal{D}} \left[ m(M, x) - m(M_{\setminus \{c_i\}}, x) \right] > \delta$$

where $M_{\setminus \{c_i\}}$ is the model with component $c_i$ ablated and $\delta$ is a meaningful effect threshold. Additionally, for locus specificity:

$$\forall c_j \notin C \text{ (adjacent)}: \quad \mathbb{E}_{x \sim \mathcal{D}} \left[ m(M, x) - m(M_{\setminus \{c_j\}}, x) \right] < \delta$$

The claim is about *membership in a causal set*, not about the structure of that set.

## What licenses an `[implementational-topographic]` tag

1. **Necessity** — ablating each component degrades the behavior reliably, under at least two ablation methods (zero + mean, or zero + resample). Single-method ablation is insufficient because zero ablation introduces distributional shift.

2. **Locus specificity** — the effect is localized to the named components, not attributable to collateral disruption. Adjacent heads (same layer, not in circuit) should show negligible effect when ablated individually.

3. **Counterfactual baseline** — the ablation is compared against a baseline (resample, mean, or zero), and the magnitude of degradation is reported relative to the baseline, not just as a raw number.

4. **Discovery procedure named** — different procedures (activation patching, ACDC, EAP-IG, manual) can return different component sets for the same behavior ([Conmy et al. 2023](https://arxiv.org/abs/2304.14997)). The procedure is part of the finding.

## What does NOT license a `[implementational-topographic]` tag

- **Single-method single-example ablation.** One head, one prompt, one method. Not enough for a claim about general causal involvement.
- **Attribution scores without causal validation.** Gradient-based attribution, DLA, or saliency maps are *candidate topographies* until validated by causal intervention.
- **High activation ≠ causal involvement.** A head can be highly active on a task without being necessary for it. Activity is a heuristic for circuit membership, not evidence.
- **Complement ablation alone.** Showing the circuit *suffices* (complement ablated, behavior preserved) is stronger than topographic — it's a sufficiency result that supports the upgrade to connectomic.

<details class="worked-example">
<summary>Worked example: IOI topographic claim</summary>

**Claim.** Heads L5H1, L5H5, L6H9, L7H3, L7H10, L8H6, L8H10, L8H11, L9H6, and L9H9 are causally involved in IOI behavior in GPT-2 Small. `[implementational-topographic]`

**Evidence:**
- Necessity: each head shows $|\Delta \text{logit diff}| > 0.3$ under both zero and mean ablation
- Locus specificity: adjacent heads (L5H0, L5H2, L6H0, etc.) show $|\Delta| < 0.05$ under the same interventions
- Discovery procedure: activation patching ([Wang et al. 2022](https://arxiv.org/abs/2211.00593)), confirmed by ACDC ([Conmy et al. 2023](https://arxiv.org/abs/2304.14997))

**What this is not:** This does not tell us that L9H9 "moves names" or that L7H3 "inhibits the subject." Those are functional/algorithmic characterizations requiring additional evidence. The topographic claim says only: these 10 heads are where the action is.
</details>

## Upgrade paths

| Direction | What's required |
|---|---|
| $I_{\text{top}} \to I_{\text{con}}$ (→ connectomic) | Path-level causal evidence of directed connections between the identified components. Path patching, not just activation patching. |
| $I_{\text{top}} \to I_{\text{stat}}$ (→ activation-statistical) | Characterize the distributional properties of activations at the identified components. |
| $I_{\text{top}} \to I_{\text{fun}}$ (→ functional) | Specify the input-output function of individual components — what each one *does*, not just that it matters. |

## Metrics that provide topographic-level evidence

- **A01 (Activation patching)** — single-node necessity via mean/resample ablation
- **A04 (Resample ablation / complement ablation)** — sufficiency of the proposed set
- **A09 (Hyperparameter sensitivity)** — does the discovered set change with method parameters?
- **F01 (Seed variance)** — does the set replicate across random seeds?
- **B08 (Edge Jaccard)** — agreement between discovery methods on the component set

## Key references

- Wang, K., et al. (2022). ["Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 Small."](https://arxiv.org/abs/2211.00593) *ICLR 2023.* — IOI circuit discovery via activation patching.
- Conmy, A., et al. (2023). ["Towards Automated Circuit Discovery for Mechanistic Interpretability."](https://arxiv.org/abs/2304.14997) *NeurIPS 2023.* — ACDC; automated topographic discovery.
- Craver, C. F. (2007). [*Explaining the Brain.*](https://doi.org/10.1093/acprof:oso/9780199299317.001.0001) Oxford University Press. — Decomposition as the first step of mechanistic explanation.
- Bechtel, W. & Richardson, R. C. (2010). [*Discovering Complexity.*](https://doi.org/10.7551/mitpress/8328.001.0001) MIT Press. — Localization strategy and its failure modes.
