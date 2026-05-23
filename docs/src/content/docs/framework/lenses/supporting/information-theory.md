---
title: "Information Theory"
description: "The network science lens: does information flow through the circuit in directed, efficient, self-sustaining patterns?"
---

# The Information Theory Lens

This lens asks one question: **does information flow through the circuit in directed, efficient, self-sustaining patterns?**

When we claim that a transformer circuit "processes" information, we are making a claim about communication channels: information enters the circuit at specific components, is transformed and transmitted through intermediate components, and arrives at the output in a form that determines the model's behavior. Information theory (Shannon 1948; Cover & Thomas 2006) provides the mathematical framework for quantifying how much information flows, how efficiently it is compressed, and whether the flow is directed or merely correlated. Network science extends this by treating the circuit as a graph and asking whether information propagation through it has the epidemiological structure of a self-sustaining process or the fragile structure of a signal that requires constant external reinforcement.

Together, these tools address a gap that the other lenses leave open. The neuroscience lens asks whether a component is causally necessary. The pharmacology lens asks whether the effect scales. The dynamical systems lens asks whether the dynamics are structured. But none of them directly ask whether the circuit's components communicate with each other in a directed, efficient way — whether the claimed "flow" of information is actually a flow, with measurable direction, quantifiable throughput, and identifiable bottlenecks. A set of components that are each individually causally relevant but share no directed information transfer is not a circuit in any information-theoretic sense. It is a coincidence of co-located effects.

There is a disanalogy worth naming. Shannon's channel model assumes a sender, a channel, and a receiver with well-defined codebooks. A transformer circuit has none of these — there is no explicit encoder, no negotiated code, and no clean separation between channel and noise. The "information" in a residual stream is not a message in Shannon's sense; it is the full state of a distributed computation. Information-theoretic quantities (mutual information, transfer entropy) can still be computed on activations, but they measure statistical dependence, not intentional communication. The gap between statistical dependence and computational role is real, and this lens does not close it by itself. What it does is provide evidence that is orthogonal to ablation-based evidence: two components can be statistically independent (no information flow) yet both causally necessary (ablating either degrades performance). The combination of causal evidence from other lenses and information-theoretic evidence from this one is stronger than either alone.

:::note
For the full metrics and protocols reference, see [Information Theory -- Metrics & Protocols](/framework/lenses/supporting/information-theory-metrics).
:::

## Key Distinctions

### Correlation vs directed information

Mutual information $I(X; Y)$ is symmetric — it tells you that $X$ and $Y$ share information, but not which causes which. Two attention heads can have high mutual information because they both read from the same position in the residual stream, not because one sends information to the other. Transfer entropy (Schreiber 2000) and Granger causality (Granger 1969) add directionality by conditioning on the past: $X$ Granger-causes $Y$ if knowing $X$'s past reduces uncertainty about $Y$'s future, beyond what $Y$'s own past provides.

In MI: two heads in a proposed circuit may show high mutual information between their activations — but this could reflect shared input rather than directed communication. Granger causality applied to layer-by-layer activations (S08) tests the stronger claim: does the earlier component's activation predict the later component's activation beyond what the later component's own history predicts? If so, there is directed information transfer along the claimed circuit edge. If not, the "connection" may be an artifact of shared context.

### Sufficient statistics vs raw representations

The information bottleneck principle (Tishby, Pereira & Bialek 1999) formalizes a tradeoff: the optimal representation of input $X$ for predicting output $Y$ is the one that compresses $X$ maximally while preserving all information about $Y$. The sufficient statistic $T(X)$ satisfies $I(T(X); Y) = I(X; Y)$ — it loses no task-relevant information — while minimizing $I(T(X); X)$ — it discards everything irrelevant.

In MI: a circuit's activations are an intermediate representation between input and output. If they are a good sufficient statistic — preserving most of $I(\text{input}; \text{output})$ while being much lower-dimensional than the input — the circuit is performing genuine compression, extracting the task-relevant signal from the noise. If the circuit's activations preserve little of $I(\text{input}; \text{output})$, the circuit is losing task-relevant information and cannot be the primary mechanism for the behavior. If the circuit's activations preserve $I(\text{input}; \text{output})$ but are as high-dimensional as the input itself, the circuit is not compressing — it is passing everything through, and "circuit" is a misleading label for "the whole model."

### Endemic vs epidemic propagation

The SIR model (Kermack & McKendrick 1927) distinguishes two regimes of disease spread based on the basic reproduction number $R_0$. When $R_0 > 1$, each infected individual infects more than one other on average — the disease is self-sustaining and spreads exponentially. When $R_0 < 1$, the disease dies out without external reintroduction.

In MI: if we model information as "spreading" through circuit components — each component receiving signal from its predecessors and transmitting to its successors — the circuit's $R_0$ measures whether the signal is self-sustaining through the layers. A circuit with $R_0 > 1$ amplifies its signal as it propagates: the information grows stronger or at least maintains itself through the circuit's depth. A circuit with $R_0 < 1$ attenuates: the signal weakens at each stage and would vanish without constant reinforcement from the residual stream outside the circuit. The first pattern is consistent with a genuine computational pathway. The second suggests the "circuit" is not self-contained — it depends on non-circuit components to keep the signal alive, which undermines claims of circuit sufficiency.

### Efficiency vs accuracy

The free energy principle, adapted from statistical mechanics and variational inference, frames computation as a tradeoff between accuracy (how well a component's output predicts the target) and complexity (how much of the input the component uses to achieve that prediction). Free energy $F = -\text{accuracy} + \text{complexity}$, and the optimal component minimizes $F$ — achieving good predictions without unnecessary complexity.

In MI: some attention heads may achieve high attribution scores by using a large fraction of the residual stream's capacity — they are accurate but wasteful. Others achieve comparable attribution with much lower complexity. The Pareto frontier of accuracy vs complexity identifies the heads that are genuinely efficient: no other head achieves equal accuracy at lower complexity. Heads below the frontier are either redundant (another head does the same job more efficiently) or vestigial (they add complexity without proportionate accuracy). A circuit composed primarily of Pareto-efficient components is well-engineered. A circuit that includes many sub-Pareto components is carrying dead weight.

## Analytical Constructs

### The information flow graph

The signature artifact of information-theoretic evaluation is the information flow graph: a directed graph where nodes are circuit components and edges are weighted by directed information transfer measures (Granger F-statistics, transfer entropy values, or conditional mutual information).

The graph reveals structure that no single ablation can:

- **Sources** — components with high out-degree and low in-degree in the information flow graph. These are where task-relevant information enters the circuit. In a well-defined circuit, sources should correspond to components that attend to or extract the task-relevant tokens.
- **Sinks** — components with high in-degree and low out-degree. These are where the circuit's computation converges to produce the output. Sinks should correspond to components with high direct logit attribution.
- **Hubs** — components with both high in-degree and high out-degree. These are the bottlenecks through which information must pass. Removing a hub should collapse $R_0$ below 1.0 (I12).
- **Topology match** — the empirically measured information flow graph should agree with the circuit's claimed topology. If the circuit claim says "head A sends to head B," the edge $A \to B$ should have significant directed information transfer. If the information flow graph reveals strong edges not present in the claimed circuit, the circuit description is incomplete.

To construct the graph: collect layer-by-layer activations for circuit components across a prompt set. For each pair of components $(A, B)$ where $A$ is in an earlier layer than $B$, compute the Granger F-statistic or transfer entropy from $A$'s activations to $B$'s activations, conditioning on all other components in intermediate layers. Apply Bonferroni correction for multiple comparisons. Retain edges with $p < 0.05$. Report the graph, its degree distribution, and its overlap with the claimed circuit topology.

### Effective number of components (Hill diversity)

Raw Shannon entropy $H$ of a circuit's activation distribution or contribution weights is difficult to interpret — is $H = 2.3$ bits concentrated or diffuse? The Hill number ${}^1D = 2^H$ (Hill 1973; Jost 2006) converts entropy to an **effective number of equally-contributing components**: an intuitive count. A circuit with $H = 3.2$ bits behaves as if it had $2^{3.2} \approx 9.2$ equally-active components.

This transformation is standard in ecology (where it converts species-abundance entropy to "effective number of species") and provides a natural language for localization claims in MI:

- **Low effective count** (${}^1D \leq 5$) — the computation is concentrated in a few components. The circuit is localized.
- **High effective count** (${}^1D \geq 20$) — the computation is spread across many components. The circuit is distributed.
- **Effective count close to total count** — every component contributes roughly equally. There is no circuit in any meaningful sense; the "circuit" is the whole model.

Report ${}^1D$ alongside entropy whenever characterizing a circuit's concentration or diffuseness. It turns a number that requires expertise to interpret into one that does not.

## Sources

| Source | Year | Field | Principle |
|---|---|---|---|
| [Shannon, "A mathematical theory of communication"](https://doi.org/10.1002/j.1538-7305.1948.tb01338.x) | 1948 | Information Theory | **Channel capacity and mutual information** — the fundamental limits on reliable communication through a noisy channel |
| [Granger, "Investigating causal relations by econometric models and cross-spectral methods"](https://doi.org/10.2307/1912791) | 1969 | Econometrics | **Granger causality** — $X$ Granger-causes $Y$ if $X$'s past reduces prediction error for $Y$ beyond $Y$'s own past |
| [Kermack & McKendrick, "A contribution to the mathematical theory of epidemics"](https://doi.org/10.1098/rspa.1927.0118) | 1927 | Epidemiology | **SIR model and $R_0$** — self-sustaining propagation ($R_0 > 1$) vs die-out ($R_0 < 1$); threshold determines whether a signal maintains itself through a network |
| [Hawkes, "Spectra of some self-exciting and mutually exciting point processes"](https://doi.org/10.1093/biomet/58.1.83) | 1971 | Statistics | **Hawkes process** — self-exciting point process where past events increase the rate of future events; models temporal cascading in neural circuits |
| [Tishby, Pereira & Bialek, "The information bottleneck method"](https://arxiv.org/abs/physics/0004057) | 1999 | Information Theory | **Information bottleneck** — optimal representations compress input while preserving output-relevant information; the sufficient statistic is the endpoint |
| [Schreiber, "Measuring information transfer"](https://doi.org/10.1103/PhysRevLett.85.461) | 2000 | Information Theory | **Transfer entropy** — non-parametric, model-free measure of directed information transfer; generalizes Granger causality beyond linear models |
| [Cover & Thomas, *Elements of Information Theory*](https://doi.org/10.1002/047174882X) | 2006 | Information Theory | **Data processing inequality** — post-processing cannot increase information; downstream circuit components cannot contain more task-relevant information than upstream ones |
| [Hill, "Diversity and evenness: a unifying notation and its consequences"](https://doi.org/10.2307/1934352) | 1973 | Ecology | **Hill numbers** — ${}^qD$ converts entropy of order $q$ to an effective number of equally-abundant types; ${}^1D = 2^H$ for Shannon entropy |
| [Jost, "Entropy and diversity"](https://doi.org/10.1111/j.2006.0030-1299.14714.x) | 2006 | Ecology / Information Theory | **True diversity** — entropy is not diversity; exponentiation is required for a measure that doubles when two equally-diverse communities are pooled |

## Validity type: [Construct validity](/framework/validity-types/construct)

> **Data processing inequality (Cover & Thomas 2006):** For any Markov chain $X \to Y \to Z$, $I(X; Z) \leq I(X; Y)$. Processing cannot create information. In MI: if a circuit is a pipeline $A \to B \to C$, the information about the task at $C$ cannot exceed the information at $B$. Any apparent increase indicates that $C$ receives information from outside the circuit — the circuit description is incomplete.

A set of causally relevant components with no measurable directed information transfer between them is not a circuit — it is a list of individually important parts. To call it a circuit is to claim that the components communicate, that the communication is directional, and that the signal is strong enough to sustain the claimed computation through the circuit's depth. Information theory provides the tools to test each of these claims independently of ablation evidence.

The primary contribution of this lens is to [construct validity](/framework/validity-types/construct) (C9) and [internal validity](/framework/validity-types/internal) (I11, I12), with a secondary contribution to [measurement validity](/framework/validity-types/measurement) (M8). Together, these criteria ask whether the circuit has the information-theoretic properties that would make the claimed computation coherent: compression (C9), directed flow (I11), self-sustaining propagation (I12), and efficient use of model capacity (M8).

## Criteria

| Code | Criterion | What it asks | Validity type |
|---|---|---|---|
| C9 | Information bottleneck | Does the circuit compress input to a sufficient statistic for the output? | Construct |
| I11 | Directed information flow | Does Granger causality confirm directed transfer along claimed edges? | Internal |
| I12 | Transmission criticality | Does the circuit's information propagation self-sustain, and do hub removals collapse it? | Internal |
| M8 | Free energy efficiency | Are the circuit's components Pareto-efficient in accuracy vs complexity? | Measurement |

### Information bottleneck (C9)

The circuit's activations should compress the input to a low-dimensional sufficient statistic for the output — preserving task-relevant information while discarding irrelevant variation.

**What it establishes.** The circuit is performing genuine information processing: extracting the signal from the noise, reducing dimensionality, and retaining what matters for the task. This is the information-theoretic analog of structural plausibility (C2) — it asks whether the circuit's representations have the right information content for the claimed computation, not just the right causal profile.

**What it does not establish.** That the compression is causally necessary (that is I1), that the compressed representation is interpretable (that is V1), or that the same compression occurs in other models (that is E6). A circuit can compress efficiently and still be one of multiple circuits that achieve the same output.

**Threshold.** Mutual information ratio $I(\text{circuit activations}; \text{output}) / I(\text{input}; \text{output}) \geq 0.7$, with compression ratio $\dim(\text{circuit}) / \dim(\text{input}) \leq 0.3$. The circuit preserves at least 70% of the task-relevant information in at most 30% of the input's dimensionality.

**Minimum reporting.** $I(\text{circuit}; \text{output})$, $I(\text{input}; \text{output})$, their ratio, the dimensionality of the circuit representation, the dimensionality of the input, and the compression ratio. Estimation method for mutual information (binning, KSG estimator, or variational bound) must be stated, as MI estimates are sensitive to the estimator.

### Directed information flow (I11)

Granger causality or transfer entropy should confirm directed information transfer along the circuit's claimed edges.

**What it establishes.** The circuit's components actually communicate — earlier components send information that later components use, beyond what the later components could predict from their own context. This is evidence for the "flow" in "information flow," which ablation studies assume but do not directly test. A circuit with confirmed directed edges has a verified communication topology, not just a list of individually important components.

**What it does not establish.** That the directed information is task-relevant (Granger causality detects any predictive relationship, not just task-relevant ones), or that the flow is necessary for the behavior (that requires ablation of specific edges, which is I1). A circuit can have strong directed information transfer along edges that carry non-task-relevant signal.

**Threshold.** At least 50% of claimed circuit edges show significant directed information transfer after Bonferroni correction ($p < 0.05$). Edge Jaccard similarity between the claimed circuit graph and the empirically measured information flow graph $\geq 0.3$. The Jaccard threshold is deliberately low — circuits are typically specified at a coarser granularity than the information flow graph, so partial overlap is expected.

**Minimum reporting.** Number of claimed edges tested, number significant after correction, Jaccard similarity with claimed topology, the correction method, and the information transfer measure used (Granger F-statistic or transfer entropy). Any strong empirical edges not present in the claimed circuit should be noted — they indicate missing circuit components.

### Transmission criticality (I12)

The circuit's information propagation, modeled as a network SIR process (WC_M13), should be self-sustaining ($R_0 > 1$), and removing hub components should collapse propagation ($R_0 < 1$).

**What it establishes.** The circuit is a self-sustaining computational pathway: signal entering the circuit maintains or amplifies itself through the circuit's depth without requiring constant reinforcement from non-circuit components. Hub components — those whose removal drops $R_0$ below 1 — are the circuit's structural bottlenecks, the components whose presence is necessary not just for their own contribution but for the entire circuit's ability to propagate signal.

**What it does not establish.** That the self-sustaining signal is the task-relevant computation (it could be a general-purpose signal that happens to traverse the circuit), or that hub components are more important than non-hub components for the task output (a non-hub component might have higher direct logit attribution). Transmission criticality characterizes the circuit's network structure, not the content of what it computes.

**Threshold.** Circuit $R_0 > 1.0$ (self-sustaining propagation). Removing each identified hub component individually should drop $R_0$ below $1.0$. If no single hub removal drops $R_0$ below $1.0$, the circuit has distributed robustness and no single structural bottleneck — which is informative but does not satisfy the criterion as stated.

**Minimum reporting.** Circuit $R_0$, the SIR model parameters (transmission rate, recovery rate), the identity of hub components (those with highest $R_0$ sensitivity), and the $R_0$ after removing each hub. The Hawkes process variant (WC_M9) may be used to model temporal cascading within layers — if used, report the branching ratio and the excitation kernel parameters.

### Free energy efficiency (M8)

The circuit's components should achieve a good accuracy-complexity tradeoff, as measured by their position on the Pareto frontier (WC_M12).

**What it establishes.** The circuit is composed of components that are not wasteful — each one achieves its contribution to the task without using more of the residual stream's capacity than necessary. This is a measurement-validity criterion because it evaluates whether the circuit's components are well-chosen, not whether the circuit's causal claims hold. A circuit composed entirely of Pareto-efficient components is a minimal description of the computation. A circuit containing many sub-Pareto components may include redundant or vestigial elements that inflate the circuit size without improving its explanatory power.

**What it does not establish.** That the Pareto-efficient components are causally necessary (a component can be efficient and still be backed up by another equally efficient component), or that the inefficient components are unimportant (they may carry non-task-relevant computations that are important for other behaviors).

**Threshold.** At least 50% of circuit heads lie on or near the Pareto frontier (no other head in the model achieves equal accuracy at lower complexity). Mean Pareto efficiency across circuit components $\geq 0.6$, where Pareto efficiency is defined as the ratio of a component's accuracy to the accuracy of the nearest Pareto-optimal component at the same complexity level. A component on the frontier has efficiency $1.0$; a component far below it has efficiency approaching $0$.

**Minimum reporting.** The accuracy-complexity scatter plot for all heads in the model, with circuit heads highlighted. The Pareto frontier. The fraction of circuit heads on the frontier. Mean Pareto efficiency. The definitions of "accuracy" (e.g., direct logit attribution, task probability contribution) and "complexity" (e.g., effective rank of the head's $W_{OV}$, fraction of residual stream variance used) must be stated — these are not standardized quantities, and the result depends on the definitions.

## Evidence Patterns

| Evidence pattern | What it establishes | Recommended language |
|---|---|---|
| High compression + directed edges + $R_0 > 1$ | Coherent, directed, self-sustaining information pipeline | "The circuit compresses input to a sufficient statistic and transmits it through directed, self-sustaining information flow" |
| Directed edges without compression | Information flows but the circuit does not reduce dimensionality | "Directed information transfer confirmed along [fraction] of claimed edges; the circuit passes information through without compressing it" |
| High compression without directed edges | The circuit compresses but components do not demonstrably communicate | "Circuit activations compress task-relevant information; directed flow between components is not confirmed" |
| $R_0 < 1$ | Signal attenuates through the circuit; not self-sustaining | "The circuit's information propagation is sub-critical ($R_0 = X$); the signal requires external reinforcement" |
| Most components sub-Pareto | Circuit includes inefficient components | "Only [fraction] of circuit components are Pareto-efficient; the circuit description may include redundant elements" |

## Verdicts

- **Proposed to Causally suggestive:** C9 (information bottleneck) alone does not gate the transition. The information theory lens contributes to construct and internal validity but does not substitute for causal evidence from the neuroscience lens.
- **Causally suggestive to Mechanistically supported:** I11 (directed information flow) strengthens the case that the circuit is not just a set of individually causally relevant components but a connected communication pathway. This contributes to internal validity beyond what ablation alone provides.
- **Mechanistically supported to Triangulated:** I12 (transmission criticality) and C9 (information bottleneck) provide convergent structural evidence from an information-theoretic evidence family, independent of ablation-based evidence. M8 (free energy efficiency) contributes to measurement validity by evaluating whether the circuit's composition is parsimonious.
- **Triangulated to Validated:** All four criteria satisfied, with directed information flow confirmed on held-out prompts and cross-checkpoint stability of $R_0$ and compression ratio.

## Protocol

1. **Compression.** Estimate $I(\text{circuit activations}; \text{output})$ and $I(\text{input}; \text{output})$ on the prompt set. Report the ratio and the compression ratio. State the MI estimator used and its hyperparameters.
2. **Directed flow.** For each claimed circuit edge, compute the Granger F-statistic or transfer entropy from the earlier component's activations to the later component's activations. Apply Bonferroni correction. Report the fraction of significant edges and Jaccard similarity with the claimed topology.
3. **Transmission criticality.** Fit an SIR model to the circuit's information propagation graph. Report $R_0$. Identify hub components. Remove each hub individually and report the resulting $R_0$. Optionally fit a Hawkes process (WC_M9) for temporal cascading analysis.
4. **Free energy efficiency.** Compute accuracy and complexity for all heads in the model. Plot the Pareto frontier. Report the fraction of circuit heads on the frontier and mean Pareto efficiency.
5. **Integration.** Construct the information flow graph. Compare its topology to the claimed circuit structure. Note any strong empirical edges absent from the claimed circuit.
6. A skipped step must be named in the verdict.

## Case Studies

- [IOI](/framework/examples/examples/examples-ioi) — the IOI circuit can be analyzed with this lens via S08 (Granger causality) and WC_M13 (SIR transmission) protocols; the claimed name-mover to S-inhibition edges are testable as directed information flow
- [Induction Heads](/framework/examples/examples/examples-induction-heads) — the two-head composition (previous-token head to induction head) is a natural target for directed information transfer analysis; $R_0$ should exceed 1.0 for the two-layer pipeline
