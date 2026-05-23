---
title: "Implementational Mode: Functional"
description: "What input-output transformation does this specific component perform? — the strongest implementational sub-mode."
---


# Implementational Mode: Functional

| | |
|---|---|
| Origin | [Craver (2007)](https://doi.org/10.1093/acprof:oso/9780199299317.001.0001) — activities of entities; [Cummins (1975)](https://doi.org/10.2307/2024796) — functional analysis |
| Question | What input-output transformation does this specific component perform? |
| Licensing evidence | Operation specification + held-out predictions + replacement test (sufficiency + minimality) |
| Interpretive-validity risk | Ascribing a function based on when a component fires rather than testing what it *does* to its input |
| Position in partial order | $I_{\text{fun}} > I_{\text{con}} > I_{\text{top}}$ — the strongest implementational sub-mode; just below representational |

## What this mode claims

A verdict tagged `[implementational-functional]` specifies what a single component does to its input to produce its output — the *function* it implements at the component level (not the system level). This is the strongest implementational sub-mode because it commits to *what the part does*, not just where it is (topographic), what it's connected to (connectomic), or what its activations look like (statistical).

The claim is about a single component's transformation. It is *not* about what the component represents (that's representational) or how it contributes to a multi-step procedure (that's algorithmic). A functional claim says: "given this input, this component produces this output, and here is the mathematical characterization of that mapping."

## Formal characterization

Let $c_i$ be a component (head or MLP) with input $h_{\text{in}} \in \mathbb{R}^{d_{\text{model}}}$ (the residual stream it reads from) and output $\Delta h \in \mathbb{R}^{d_{\text{model}}}$ (its contribution to the residual stream). A functional claim asserts:

$$\Delta h = g_i(h_{\text{in}}) \approx \hat{g}_i(h_{\text{in}})$$

where $\hat{g}_i$ is the claimed function, specified with enough precision to generate quantitative predictions on held-out inputs.

For attention heads, the function is typically decomposed as:

$$\hat{g}_{\text{head}}(h_{\text{in}}) = \sum_q \text{attn}(p, q) \cdot W_{OV} \cdot h_q$$

where the functional claim specifies *what* the attention pattern selects (the attention function) and *what* the OV circuit does to the selected content (the value function).

## What licenses an `[implementational-functional]` tag

1. **Operation specification** — the function stated with enough precision to generate quantitative predictions. "It copies tokens" is too vague; "It projects the attended token's embedding through a rank-1 $W_{OV}$ matrix aligned with the token's unembedding direction, adding $\Delta \text{logit}[t] \approx \text{attn}(p, q) \cdot \sigma_1 \cdot \langle u_1, W_U[t] \rangle$" is a functional claim.

2. **Held-out prediction** — the function's predicted output compared to the component's actual output on inputs not used to derive the function. If the characterization was derived from IOI prompts, test it on SVA prompts or random text. The function should generalize beyond the discovery distribution.

3. **Sufficiency (replacement test)** — replacing the component with its claimed function should preserve the behavior. If you substitute a synthetic implementation of $\hat{g}_i$ for the actual component and performance is maintained, the function is sufficient.

4. **Minimality (optional but strengthening)** — the claimed function is not achievable by a simpler description. If "$g$ copies tokens" and "$g$ multiplies by a rank-1 matrix" both fit, the simpler description (rank-1 multiplication) is preferred unless copying adds predictive power.

## What does NOT license an `[implementational-functional]` tag

- **Naming convention as function.** "L9H9 is the name-mover head" assigns a function by name. The name is a hypothesis; the functional claim requires showing the head actually performs a token-copying operation, testing this on held-out inputs, and demonstrating the replacement test.
- **When-it-fires != what-it-does.** "This head fires on code tokens, so its function is code detection." Firing patterns are statistical. Function is about the input-output transformation: what does it *do* to the code tokens it attends to?
- **Attention pattern alone.** "Head L5H1 attends to the previous token" describes the attention function but not the value function. A complete functional claim must specify both *what* the head attends to and *what it does with the attended content*.
- **Ablation effect without operation specification.** "Ablating this head reduces IOI performance by 40%" is topographic (it matters), not functional (what it does). The effect magnitude tells you importance, not function.

<details class="worked-example">
<summary>Worked example: L10H7 as suppression head (functional)</summary>

**Claim.** Head L10H7 in GPT-2 Small implements a linear suppression function: for token $t$ at position $p$, if $t$ has high probability in the output distribution before this head acts, the head subtracts a vector proportional to $t$'s unembedding direction from the residual stream:

$$\Delta h \approx -\text{attn}(p, q) \cdot \sigma_1 \cdot \langle W_{OV} h_q, W_U[t] \rangle \cdot \hat{u}_t$$

`[implementational-functional]`

**Evidence:**
- Operation specification: rank-1 SVD of $W_{OV}$ shows dominant singular value with left-singular vector anti-aligned to high-probability token unembeddings
- Held-out prediction: the formula predicts $\Delta \text{logit}[t]$ with $R^2 = 0.87$ on random text (not IOI)
- Replacement test: substituting the rank-1 approximation for the full head preserves IOI logit difference to within 95%
- Minimality: rank-2 approximation does not meaningfully improve $R^2$

**What this is not:** This does not claim the head "suppresses repeated names" (that's an algorithmic claim about its role in a procedure). It claims only what transformation the head applies to its input — a specific linear operation. The algorithmic role (suppressing the subject in IOI) requires additionally showing how this function contributes to the multi-step procedure.
</details>

<details class="worked-example">
<summary>Anti-pattern: role-name as function</summary>

**Bad claim:** "L9H9 is a name-mover head — its function is to move names."

Why this fails: "Moving names" is not a mathematical specification of an input-output transformation. A proper functional claim would be: "L9H9 applies its OV matrix to the embedding of the token it attends to, projecting it through a rank-$k$ matrix whose top singular vectors align with name-token unembedding directions, effectively copying the attended name's logit contribution to the output position." This can be tested quantitatively on held-out inputs.
</details>

## Upgrade and downgrade

| Direction | What's required |
|---|---|
| $I_{\text{con}} \to I_{\text{fun}}$ (upgrade from connectomic) | For each node in the graph, specify the input-output function with quantitative predictions and the replacement test. |
| $I_{\text{fun}} \to R$ (upgrade to representational) | Demonstrate that the component's function is specifically *representational* — that its input or output encodes a specific causal variable in a specific geometric form. A head that suppresses a direction is functional; showing that direction encodes "repeated name identity" makes it representational. |
| $I_{\text{fun}} \to A$ (upgrade to algorithmic) | Combine multiple components' functions into an ordered procedure with information flow. An algorithm = multiple functional components + their composition + sufficiency of the joint procedure. |

## Metrics that provide functional-level evidence

- **B02 (OV/QK composition analysis)** — weight-space characterization of the head's operation
- **B05 (NMF/ICA on weight matrices)** — decomposition of the operation into interpretable factors
- **A05 (Weight-extended analysis)** — connecting weight structure to functional claims
- **B06 (Norm trajectory)** — how component output magnitude evolves (operational fingerprint)
- **D01 (Logit attribution)** — direct contribution to output (one aspect of function)

## Key references

- Elhage, N., et al. (2021). ["A Mathematical Framework for Transformer Circuits."](https://transformer-circuits.pub/2021/framework/index.html) *Transformer Circuits Thread.* — Rank-1 OV analysis as functional characterization.
- Craver, C. F. (2007). [*Explaining the Brain.*](https://doi.org/10.1093/acprof:oso/9780199299317.001.0001) Oxford University Press. — "Activities" of mechanistic entities as the functional level.
- Cummins, R. (1975). ["Functional Analysis."](https://doi.org/10.2307/2024796) *Journal of Philosophy* 72(20), 741-765. — Functions as contributions to system capacities.
- Olsson, C., et al. (2022). ["In-context Learning and Induction Heads."](https://arxiv.org/abs/2209.11895) *Transformer Circuits Thread.* — Functional characterization (copying + QK composition) leading to algorithmic claim.
