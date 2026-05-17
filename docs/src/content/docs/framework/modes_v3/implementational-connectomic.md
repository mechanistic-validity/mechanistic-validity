---
title: "Implementational Mode: Connectomic"
description: "How are the components wired to each other? — a directed graph of information flow without specifying what each node computes."
---


# Implementational Mode: Connectomic

| | |
|---|---|
| Origin | [Craver (2007)](https://doi.org/10.1093/acprof:oso/9780199299317.001.0001) — organization; neuroscience connectomics ([Sporns et al. 2005](https://doi.org/10.1371/journal.pcbi.0010042)) |
| Question | How are the identified components wired to each other? |
| Licensing evidence | Path patching (edge-level) + specificity of claimed pathways vs. alternatives |
| Interpretive-validity risk | Confusing temporal ordering (layer sequence) with causal wiring (information flow) |
| Position in partial order | $I_{\text{con}} > I_{\text{top}}$ — asserts structure (a graph), not just membership (a set) |

## What this mode claims

A verdict tagged `[implementational-connectomic]` identifies directed connections between components — which feeds into which, through what pathway. This is stronger than topographic because it asserts *structure* (a directed graph) rather than just *membership* (a set). The connectomic claim says: "head $A$ sends information to head $B$ through the residual stream, and $B$'s computation depends on receiving $A$'s output."

The analogy to neuroscience is deliberate: a connectome is a wiring diagram. It tells you the structure of the network without specifying what each neuron computes. In MI, a circuit graph (nodes = heads, edges = information-flow dependencies) is a connectomic claim. It is more than a list of important heads, and less than an algorithm.

## Formal characterization

Let $G = (V, E)$ be a directed graph where $V = C$ (the circuit components) and $E \subseteq V \times V$ (directed edges). A connectomic claim asserts:

$$\forall (c_i, c_j) \in E: \quad \text{PathEffect}(c_i \to c_j) > \delta \quad \text{AND} \quad \text{PathEffect}(c_i \to c_j) \gg \text{PathEffect}(c_i \to c_k) \text{ for } (c_i, c_k) \notin E$$

where $\text{PathEffect}(c_i \to c_j)$ is the causal effect of patching $c_i$'s output *specifically at the path to $c_j$* (holding other paths fixed). This is what path patching measures.

The key distinction from topographic: a topographic claim is invariant to permutation of the component set. A connectomic claim is not — it asserts directed relationships between specific pairs.

## What licenses an `[implementational-connectomic]` tag

1. **Path-level causal evidence** — path patching ([Goldowsky-Dill et al. 2023](https://arxiv.org/abs/2304.05969)) or edge attribution demonstrating that the claimed pathway is load-bearing. Activation patching alone establishes node importance (topographic), not edge importance (connectomic).

2. **Specificity of claimed paths** — patching along the claimed path has substantially more effect than patching along alternative paths of the same length. If every path from $A$ to downstream has similar effect, the claim is not connectomic — it's just that $A$ matters (topographic).

3. **Directionality** — the effect must be asymmetric. If corrupting $A$ affects $B$ and corrupting $B$ equally affects $A$ (after controlling for layer order), the "wiring" claim is not established.

4. **Convergent structural evidence (optional but strengthening)** — weight-space composition scores (QK/OV composition, virtual weights) that agree with the causal graph. When path patching and weight-space analysis agree on the same edges, the connectomic claim is substantially stronger.

## What does NOT license a `[implementational-connectomic]` tag

- **Layer ordering alone.** In a transformer, every earlier layer's output is accessible to every later layer via the residual stream. The fact that $A$ is in layer 5 and $B$ is in layer 9 does not mean $A$ connects to $B$ — every layer-5 head "connects" to every layer-9 head in this trivial sense. A connectomic claim must show *specific, load-bearing* connections above this background.
- **Attention pattern inspection.** That head $B$ attends to positions where head $A$ has written is suggestive but not causal. The residual stream contains contributions from many components at each position.
- **Correlation between head activations.** Two heads being co-active is statistical, not structural. They might both respond to the same input feature without being wired to each other.
- **ACDC edges without effect validation.** ACDC discovers edges via iterative patching, but the discovered graph should be validated with held-out path-patching to confirm edge-level effects.

<details class="worked-example">
<summary>Worked example: IOI QK composition edges</summary>

**Claim.** In the IOI circuit, the S-inhibition heads (L7H3, L7H10, L8H6, L8H10, L8H11) receive directed input from the duplicate-token heads (L5H1, L5H5) via the residual stream, and this connection is load-bearing for the suppression of the repeated name. `[implementational-connectomic]`

**Evidence:**
- Path patching from L5H1/L5H5 output *specifically at the path to* L7-8 S-inhibition heads shows $\Delta \text{logit diff} = 0.4$-$0.7$
- Alternative paths (L5H1 → name-mover heads directly) show $\Delta < 0.05$ — the information flows through S-inhibition first
- QK composition scores: $\langle W_{OV}^{5.1}, W_{QK}^{7.3} \rangle$ is high relative to random head pairs (top 5% of all pairwise compositions)
- Directionality: corrupting L5H1 degrades L7H3's attention pattern; corrupting L7H3 does not affect L5H1's attention pattern

**What this is not:** This does not specify *what* the duplicate-token heads compute or *what operation* S-inhibition performs. It says only that information flows directionally from one group to the other, and that this flow is necessary for the behavior.
</details>

## Upgrade and downgrade

| Direction | What's required |
|---|---|
| $I_{\text{top}} \to I_{\text{con}}$ (upgrade from topographic) | Path-level causal evidence that specific edges carry information, not just that nodes matter. |
| $I_{\text{con}} \to I_{\text{fun}}$ (upgrade to functional) | Specify what each node in the graph *does* to its input to produce its output. The graph tells you the wiring; the functional claim tells you the components. |
| $I_{\text{con}} \to A$ (upgrade to algorithmic) | Combine the graph (connectomic) with the node functions (functional) and demonstrate sufficiency of the procedure. An algorithm = wiring + operations + sufficiency. |

## Instruments that provide connectomic-level evidence

- **A02 (Path patching)** — direct causal evidence of edge-level effects
- **B02 (OV/QK composition)** — weight-space evidence for compositional wiring
- **B08 (Edge Jaccard)** — agreement between methods on the edge set
- **A13 (PC algorithm)** — observational causal discovery of the graph structure
- **C01 (Transfer entropy)** — directional information flow between components

## Key references

- Goldowsky-Dill, N., et al. (2023). ["Localizing Model Behavior with Path Patching."](https://arxiv.org/abs/2304.05969) — Path patching as edge-level causal instrument.
- Conmy, A., et al. (2023). ["Towards Automated Circuit Discovery for Mechanistic Interpretability."](https://arxiv.org/abs/2304.14997) *NeurIPS 2023.* — ACDC; edge-based circuit discovery.
- Elhage, N., et al. (2021). ["A Mathematical Framework for Transformer Circuits."](https://transformer-circuits.pub/2021/framework/index.html) *Transformer Circuits Thread.* — QK/OV composition as structural wiring evidence.
- Sporns, O., Tononi, G., & Kotter, R. (2005). ["The Human Connectome: A Structural Description of the Human Brain."](https://doi.org/10.1371/journal.pcbi.0010042) *PLoS Computational Biology.* — Connectomics framework.
