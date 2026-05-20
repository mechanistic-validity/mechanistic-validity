---
title: "Algorithmic Mode"
description: "What procedure does the system execute to produce the output? — specifying operations, ordering, and information flow."
---


# Algorithmic Mode

| | |
|---|---|
| Origin | [Marr (1982)](https://doi.org/10.7551/mitpress/9780262514620.001.0001), Level 2 |
| Question | What procedure does the system execute to produce the output? |
| Licensing evidence | Path patching demonstrating information flow + operation specification + sufficiency of the claimed procedure |
| Interpretive-validity risk | Naming components sequentially and calling it an "algorithm" without specifying what each step *does* |
| Position in partial order | $C > A > R > I_{\text{fun}} > I_{\text{con}} > I_{\text{top}}$ — second highest |

## What this mode claims

A verdict tagged `[algorithmic]` specifies a sequence of operations, their ordering, and how intermediate representations flow between them. The algorithm is stated with enough precision that it could be re-implemented — it is a description of a *procedure*, not just a naming of components.

An algorithmic claim says the model follows a specific step-by-step procedure. It must specify: (1) what operation each step performs on its input, (2) what output that operation produces, (3) how outputs flow between steps, and (4) that executing the claimed procedure on the claimed inputs produces the claimed outputs.

## Formal characterization

An algorithmic claim asserts the existence of a sequence of operations $\{o_1, \ldots, o_k\}$ such that:

$$y = o_k \circ o_{k-1} \circ \cdots \circ o_1(x)$$

where each $o_i$ is specified at the component level (which head/MLP performs it), with its input domain (what it reads from the residual stream) and output range (what it writes). The composition must be *sufficient*: executing the claimed procedure should reproduce the behavior without the rest of the model ([Craver 2007](https://doi.org/10.1093/acprof:oso/9780199299317.001.0001)'s sufficiency criterion).

The ordering is testable: if the algorithm claims step $A$ feeds step $B$, then interventions on $A$ at the correct position should affect $B$'s output, and interventions at other positions should not.

## What licenses an `[algorithmic]` tag

1. **Operation specification** — each step states *what transformation* is applied, not just *which component* is active. "L5H1 attends to the previous token" is not enough; "L5H1 copies position information backward via its OV circuit, writing the attended token's embedding into the residual stream at the current position" specifies the operation.

2. **Path-level causal evidence** — path patching (not just activation patching) demonstrating the claimed information flow. Activation patching establishes necessity; path patching establishes *directed dependency between specific steps*.

3. **Timing consistency** — if step $A$ at layer $l_1$ feeds step $B$ at layer $l_2 > l_1$, then corrupting $A$'s output at layer $l_1$ should degrade $B$'s behavior, and corrupting at $l_2$ directly should have a different (larger) effect.

4. **Sufficiency** — the claimed procedure, executed in isolation (minimal circuit with complement ablated), must reproduce the behavior. This is the difference between "these components are important" (implementational) and "they jointly execute this procedure" (algorithmic).

## What does NOT license an `[algorithmic]` tag

- **Naming components and calling their sequential activation an "algorithm."** An algorithm must specify what *operation* each step performs on its input to produce its output, not just which components are active in which order.
- **Activation patching alone.** Activation patching establishes causal necessity (implementational). It does not establish directed information flow between steps (algorithmic).
- **Post-hoc narrative.** "First the model attends to X, then it processes Y, then it outputs Z" narrated from attention patterns is not an algorithm unless each step's operation is specified and the information flow is causally demonstrated.

<details class="worked-example">
<summary>Worked example: Induction heads as algorithmic</summary>

[Olsson et al. (2022)](https://arxiv.org/abs/2209.11895) describe induction heads with an algorithmic claim:

**Step 1 (previous-token heads, L0-L1):** Copy position information backward — the OV circuit writes the embedding of the previous token into the current position's residual stream.

**Step 2 (induction heads, L5-L6):** Use QK composition with step 1's output to attend to the token *following* the previous occurrence of the current token. The Q vectors at the current position compose with K vectors that now contain previous-token information (from step 1), creating an attention pattern that targets the post-match position.

**Step 3 (copying via OV):** The OV circuit of the induction head copies the attended token's embedding to the output, boosting its logit.

**Why this is algorithmic, not just implementational:** Each step specifies an *operation* (copy, compose, attend). The information flow is directional (step 1 output feeds step 2 input via QK composition). The procedure is sufficient — the two-layer induction circuit reproduces in-context learning behavior in isolation.

**Evidence:** QK composition scores demonstrate the directed dependency. Patching step 1's output disrupts step 2's attention pattern. The minimal two-layer circuit reproduces the behavior.
</details>

<details class="worked-example">
<summary>Anti-pattern: IOI "algorithm" that's actually topographic</summary>

**Pseudo-algorithmic claim:** "The IOI algorithm works as follows: duplicate-token heads fire, then S-inhibition heads fire, then name-mover heads fire."

**Why this fails:** It lists components in layer order and calls it an algorithm. But it doesn't specify: What operation do duplicate-token heads perform on their input? What representation do they write? How does S-inhibition read that output? The temporal sequence of layer computation is *architecture*, not *algorithm*. Every transformer processes layers sequentially — that fact alone is not an algorithmic claim.

**Correct version:** "Duplicate-token heads compute the identity of repeated name tokens by attending from the second occurrence to the first and writing a same-token signal into the residual stream. S-inhibition heads read this signal and suppress the corresponding name's contribution to the output logits by writing a negative direction aligned with that name's unembedding vector. Name-mover heads then copy the *remaining* (unsuppressed) name to the output."

Now each step has a specified operation, input, and output.
</details>

## Upgrade and downgrade

| Direction | What's required |
|---|---|
| $I_{\text{fun}} \to A$ (upgrade from functional) | You have component-level functions — now demonstrate *ordering, composition, and sufficiency*: the functions flow into each other through a specific directed procedure that jointly produces the behavior. |
| $A \to C$ (upgrade to computational) | Provide the normative account: why does this algorithm solve the *right* problem? Show error analysis consistent with problem boundaries. |
| $A \to I_{\text{fun}}$ (downgrade) | If path-level causal evidence fails — information flow is not directional or the procedure is not sufficient in isolation — the claim is at best functional (individual operations) without the algorithmic composition. |

## Metrics that provide algorithmic-level evidence

- **A02 (Path patching)** — directed causal evidence of information flow
- **A04 (Resample ablation)** — sufficiency of the proposed procedure
- **B02 (OV/QK composition)** — weight-space evidence for step composition
- **B08 (Edge Jaccard)** — stability of the claimed information-flow graph

## Key references

- Marr, D. (1982). [*Vision.*](https://doi.org/10.7551/mitpress/9780262514620.001.0001) MIT Press. — Algorithmic level as procedure specification.
- Pylyshyn, Z. (1984). [*Computation and Cognition.*](https://doi.org/10.7551/mitpress/2004.001.0001) MIT Press. — Functional architecture vs. algorithm distinction; what counts as a procedure specification.
- Craver, C. F. (2007). [*Explaining the Brain.*](https://doi.org/10.1093/acprof:oso/9780199299317.001.0001) Oxford University Press. — Sufficiency criterion for mechanistic explanation.
- Olsson, C., et al. (2022). ["In-context Learning and Induction Heads."](https://arxiv.org/abs/2209.11895) *Transformer Circuits Thread.* — Canonical algorithmic-mode claim with QK composition evidence.
- Wang, K., et al. (2022). ["Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 Small."](https://arxiv.org/abs/2211.00593) *ICLR 2023.* — IOI circuit with partial algorithmic characterization.
- Conmy, A., et al. (2023). ["Towards Automated Circuit Discovery for Mechanistic Interpretability."](https://arxiv.org/abs/2304.14997) *NeurIPS 2023.* — ACDC; automated circuit-as-algorithm discovery via edge patching.
- Goldowsky-Dill, N., et al. (2023). ["Localizing Model Behavior with Path Patching."](https://arxiv.org/abs/2304.05969) — Path patching as algorithmic-level metric.
