---
title: "Information Theory -- Metrics & Protocols"
description: "Detailed reference for all information-theoretic metrics (IT1-IT3) and protocols used to evaluate mechanistic interpretability claims."
---

# Information Theory -- Metrics & Protocols

This page is the detailed reference companion to the [Information Theory lens overview](/framework/lenses/supporting/information-theory). Where the overview explains *why* information theory matters for evaluating circuit claims, this page documents the specific metrics and protocols that implement the evaluation: what each one measures, how it works, what thresholds it uses, and how to run it.

The information theory lens contributes to [construct validity](/framework/validity-types/construct), [internal validity](/framework/validity-types/internal), and [measurement validity](/framework/validity-types/measurement). Its metrics and protocols divide into three groups: **core metrics** (IT1--IT3) that directly quantify information-theoretic properties of circuit components, **structural protocols** (C01--C03, C09) that evaluate the circuit's information flow topology and decomposition, and **extended protocols** (IT_CD, IT_AC, WC_M9, WC_M12, WC_M13, SM_PP) that apply tools from adjacent fields — epidemiology, active inference, point processes, predictive coding — through an information-theoretic lens.

---

## Core Metrics

These three metrics form the foundation of the information-theoretic evaluation. Each targets a different aspect of Shannon's communication framework applied to circuit components.

### IT1 — Channel Capacity

**What it measures.** How much task-relevant information each circuit head transmits. Mutual information between discretized head activations and the correct output token is computed, then normalized by the maximum possible MI to yield a utilization ratio.

**How it works.** For each circuit head and each prompt in the evaluation set, the activation norm at the final sequence position is recorded. These activations are discretized into 10 equal-width bins. The mutual information $I(X; Y)$ between the binned activations $X$ and the categorical output labels $Y$ is computed from the joint and marginal distributions:

$$I(X; Y) = \sum_{x, y} p(x, y) \log_2 \frac{p(x, y)}{p(x) \cdot p(y)}$$

The utilization ratio is $I(X; Y) / \log_2(N_{\text{bins}})$, where $\log_2(10) \approx 3.32$ bits is the theoretical maximum for 10 bins.

**Threshold.** Mean utilization across circuit heads $> 0.3$. A head that uses less than 30% of its channel capacity for the task is transmitting mostly noise or task-irrelevant signal.

**What it establishes.** That the circuit's components carry measurable task-relevant information. A circuit composed of heads with near-zero utilization is not functioning as a communication channel for the task, regardless of what ablation studies show.

**What it does not establish.** That the information is causally used by the model (an external decoder can extract information the model ignores), or that the channel is efficient (high utilization does not imply high compression).

**Reference.** Shannon (1948), "A Mathematical Theory of Communication," *Bell System Technical Journal* 27:379--423.

**Usage:**
```bash
uv run python IT1_channel_capacity.py --tasks ioi sva --n-prompts 40
```

**Source:** `mechval_v2/supporting/information_theory/IT1_channel_capacity.py`

---

### IT2 — Rate-Distortion

**What it measures.** The minimum bits needed to describe circuit activations while preserving output quality. Progressive quantization of circuit head activations reveals the compression-fidelity tradeoff: how aggressively can the circuit's representations be compressed before task performance degrades?

**How it works.** Baseline logit difference is measured on clean (unmodified) forward passes. Then, for each bit level in $\{8, 4, 2, 1\}$, circuit head activations at `hook_z` are uniformly quantized to $2^{\text{bits}}$ levels during the forward pass, and the resulting logit difference is compared to baseline. The retention ratio is $\text{LD}_{\text{quantized}} / \text{LD}_{\text{baseline}}$.

The quantization procedure for each head:

1. Record $v_{\min}$ and $v_{\max}$ across the head's activation tensor
2. Normalize to $[0, 1]$
3. Round to the nearest of $2^{\text{bits}} - 1$ evenly spaced levels
4. Rescale back to $[v_{\min}, v_{\max}]$

This produces a rate-distortion curve: a monotonically decreasing function mapping compression level to task fidelity.

**Threshold.** Retention $> 0.8$ at 4-bit quantization. A circuit that loses more than 20% of its logit difference when compressed to 16 levels is encoding task-relevant information in fine-grained activation patterns — either the circuit is more complex than claimed, or the quantization is destroying structure that the circuit genuinely needs.

**What it establishes.** How much precision the circuit's representations require. A circuit that survives aggressive quantization implements a robust, coarse-grained computation. One that fails under mild quantization relies on precise numerical relationships between activations.

**What it does not establish.** That the compression limit is tight (the quantization scheme may not be optimal), or that the circuit is efficiently compressed (surviving quantization does not mean the circuit is already compressed — it means it *could* be).

**Reference.** Shannon (1959), "Coding Theorems for a Discrete Source with a Fidelity Criterion," *IRE National Convention Record* 7:142--163.

**Usage:**
```bash
uv run python IT2_rate_distortion.py --tasks ioi sva --n-prompts 40
```

**Source:** `mechval_v2/supporting/information_theory/IT2_rate_distortion.py`

---

### IT3 — Kolmogorov Complexity Proxy

**What it measures.** The compressibility of circuit weight matrices as a proxy for algorithmic complexity. Weight matrices for each circuit head ($W_Q$, $W_K$, $W_V$, $W_O$) are flattened to float16 bytes and gzip-compressed. The compression ratio (raw size / compressed size) indicates whether the weights have exploitable structure.

**How it works.** For each circuit head, the four weight matrices are extracted, converted to float16, serialized to bytes, and gzip-compressed. The compression ratio is:

$$\text{ratio} = \frac{|\text{raw bytes}|}{|\text{gzip}(\text{raw bytes})|}$$

A ratio of 1.0 means no compression (random-looking weights). Higher ratios indicate structured, patterned weights that a compression algorithm can exploit.

**Threshold.** Mean compression ratio across circuit heads $> 1.5$. Circuit heads with compression ratios near 1.0 have weights that look random — they lack the low-complexity structure expected of a clean algorithmic implementation.

**What it establishes.** That the circuit's weight matrices are structured, not random. Kolmogorov complexity is uncomputable in general, but gzip compression provides a practical upper bound: if the weights compress well, their Kolmogorov complexity is bounded above by the compressed size. Structured weights are consistent with the circuit implementing a simple, describable algorithm.

**What it does not establish.** That the structure is task-relevant (the weights could be structured for reasons unrelated to the task), or that low complexity implies interpretability (a simple algorithm can still be opaque). The gzip proxy is also sensitive to the choice of compression algorithm and float precision.

**Reference.** Kolmogorov (1965), "Three Approaches to the Definition of the Concept of Information," *Problems of Information Transmission* 1:1--7.

**Usage:**
```bash
uv run python IT3_kolmogorov_complexity.py --tasks ioi sva --n-prompts 40
```

**Source:** `mechval_v2/supporting/information_theory/IT3_kolmogorov_complexity.py`

---

## Structural Protocols

These protocols evaluate the information-theoretic structure of the circuit as a whole: how information flows between components, how it decomposes across them, and whether the circuit's causal structure can be recovered from observational data.

### C01 — Information Flow

**Protocol ID:** `C01`
**Validity type:** Construct (C1 Convergent), Internal (I1 Necessity)

**Question.** How does information flow through the circuit? Do circuit components share mutual information? Is there directed information transfer (transfer entropy) between layers? Does Granger causality confirm the hypothesized information flow direction?

**Metrics.**

| Metric | What it measures | Threshold |
|---|---|---|
| `mutual_information` | MI between circuit component activations | $> 0.1$ |
| `conditional_mi` | Conditional MI controlling for confounders | $> 0.05$ |
| `transfer_entropy` | Directed information transfer between components | $> 0.05$ |
| `granger_causality` | Granger-causal relationships between components | $> 0.05$ |

**How the metrics relate.** Mutual information is symmetric — it detects shared information but not direction. Transfer entropy (Schreiber 2000) adds directionality by conditioning on the target's own past. Granger causality (Granger 1969) provides the same insight in a parametric (regression-based) framework. Conditional MI controls for confounders: two heads may share MI simply because they both read from the same residual stream position, not because one sends to the other. Conditioning on the shared context isolates genuine inter-component communication.

**Calibrations.** Measurement invariance (stability across prompt samples), convergent validity (agreement with causal metrics from other lenses), discriminant validity (distinguishing circuits from non-circuits).

**References.** Cover & Thomas (2006), *Elements of Information Theory*; Schreiber (2000), "Measuring Information Transfer"; Granger (1969), "Investigating Causal Relations by Econometric Models."

**Source:** `protocols/information_theory/c01_information_flow.py`

---

### C02 — Information Decomposition

**Protocol ID:** `C02`
**Validity type:** Construct (C1 Convergent), Internal (I1 Necessity, I2 Sufficiency)

**Question.** How is information shared across circuit components? Is it redundant (many components carry the same signal), synergistic (information emerges only from combinations), or unique (each component contributes something distinct)?

**Metrics.**

| Metric | What it measures |
|---|---|
| `pid` | Partial Information Decomposition: redundancy, synergy, and unique information across circuit heads |
| `o_information` | O-information: net synergy vs redundancy across components (Rosas et al. 2019) |
| `info_bottleneck` | Information bottleneck: compression vs prediction tradeoff (Tishby et al. 2000) |

**Interpreting the decomposition.**

- **High synergy** indicates composition — the circuit's computational value exceeds the sum of its parts. This is the information-theoretic signature of a genuine multi-component mechanism.
- **High redundancy** suggests over-specification or robustness. Multiple components carry the same signal, which may reflect backup circuits or distributed encoding.
- **High unique information** localizes function to individual components. Each head contributes something no other head provides.

**References.** Williams & Beer (2010), "Nonnegative Decomposition of Multivariate Information"; Rosas et al. (2019), "Quantifying High-Order Interdependencies via Multivariate Extensions of the Mutual Information"; Tishby et al. (2000), "The Information Bottleneck Method."

**Source:** `protocols/information_theory/c02_decomposition.py`

---

### C03 — Causal Structure Discovery

**Protocol ID:** `C03`
**Validity type:** Construct (C1 Convergent, C2 Discriminant), Internal (I1 Necessity)

**Question.** Can the circuit's causal structure be recovered from observational data alone? Does the learned DAG match the hypothesized circuit graph?

**Metrics.**

| Metric | What it measures |
|---|---|
| `notears` | NOTEARS DAG recovery: does continuous optimization recover the circuit graph from activations? |
| `ocse` | Observational-Counterfactual Structural Equivalence: is the circuit structurally equivalent under observational and counterfactual distributions? |

**Why this matters.** If a circuit's causal graph can be recovered from observational data (without interventions), the circuit structure is not an artifact of the intervention procedure. NOTEARS (Zheng et al. 2018) uses continuous optimization with an acyclicity constraint to learn a DAG from activation data. If the learned DAG matches the hypothesized circuit topology, the claim gains support from a completely independent methodology. OCSE (Rubenstein et al. 2017) tests the stronger condition: does the circuit's structure remain the same when evaluated under counterfactual (corrupted) inputs?

**References.** Zheng et al. (2018), "DAGs with NO TEARS"; Rubenstein et al. (2017), "Causal Consistency of Structural Equation Models."

**Source:** `protocols/information_theory/c03_causal_structure.py`

---

### C09 — Information Bottleneck

**Protocol ID:** `C09`
**Validity type:** Construct (C3 Information), Internal (I2 Sufficiency)

**Question.** How much of the model's input-output mutual information is captured by the circuit's internal activations? If the circuit is a faithful bottleneck, then $I(\text{circuit activations}; \text{output})$ should approach $I(\text{input}; \text{output})$.

**Metrics.**

| Metric | What it measures | Threshold |
|---|---|---|
| `sufficiency_ratio` | $R^2$ of linear prediction from circuit head activations to output logit differences | $\geq 0.7$ |
| `compression_ratio` | $\dim(\text{circuit activations}) / \dim(\text{residual stream})$ | $\leq 0.3$ |

**How it works.** The sufficiency ratio is approximated by the $R^2$ of a linear model predicting output logit differences from concatenated circuit head activations. This is a practical lower bound on $I(\text{circuit}; \text{output}) / I(\text{input}; \text{output})$ — if a linear model can predict the output from circuit activations, the circuit captures at least that much of the task-relevant information. The compression ratio measures how much smaller the circuit's representation is compared to the full residual stream.

**What the combination means.** A circuit that achieves high sufficiency ($\geq 0.7$) with low compression ($\leq 0.3$) is performing genuine information compression: extracting the task-relevant signal into a representation that is much smaller than the input. A circuit with high sufficiency but high compression (close to 1.0) is passing everything through — "circuit" is a misleading label for "the whole model." A circuit with low sufficiency at any compression level is losing task-relevant information and cannot be the primary mechanism.

**References.** Tishby, Pereira & Bialek (2000), "The Information Bottleneck Method"; Shwartz-Ziv & Tishby (2017), "Opening the Black Box of Deep Neural Networks via Information."

**Source:** `protocols/information_theory/c09_information_bottleneck.py`

---

## Directed Flow Protocols

These protocols test whether the "flow" in "information flow" is genuine: directed, temporally structured, and consistent with the claimed circuit edges.

### A07 — Granger Causality and Transfer Entropy

**Protocol ID:** `A07`
**Validity type:** Internal (I1 Necessity)

**Question.** Do circuit components carry directed information flow consistent with the hypothesized causal structure? Does knowledge of upstream circuit activations improve prediction of downstream behavior beyond what the target's own past provides?

**Metrics.**

| Metric | What it measures |
|---|---|
| `transfer_entropy` | Directed information flow between components (Schreiber 2000) |
| `granger_causality` | Temporal precedence in activation sequences (Granger 1969) |
| `cross_task_transfer` | Cross-task IIA transfer: does a circuit learned for one task generalize causally to another? |
| `ocse` | Observational Causal Structure Estimation |

**Calibrations.** Bootstrap stability, seed variance, ablation invariance (results hold across zero/mean/resample ablation), method invariance, convergent validity with other causal metrics.

**Why both Granger and transfer entropy.** Granger causality is parametric (assumes a linear VAR model) and therefore more powerful when linearity holds. Transfer entropy is nonparametric and captures nonlinear dependencies that Granger misses. Agreement between the two strengthens the claim. Disagreement (Granger significant, transfer entropy not) suggests the relationship is linear; the reverse suggests nonlinear dependence.

**References.** Granger (1969), "Investigating Causal Relations by Econometric Models"; Schreiber (2000), "Measuring Information Transfer."

**Source:** `protocols/information_theory/a07_granger_te.py`

---

### A08 — Partial Information Decomposition

**Protocol ID:** `A08`
**Validity type:** Internal (I1 Necessity, I2 Sufficiency)

**Question.** How is the information about the task output decomposed across circuit components? What fraction is redundant, unique, or synergistic?

**Metric:** `pid` — the Williams & Beer (2010) partial information decomposition applied to circuit heads.

**Interpreting the PID.**

| Component | Interpretation for circuit evaluation |
|---|---|
| **Redundancy** | Multiple heads carry the same task information. High redundancy suggests the circuit is robust to single-head failure but may be over-specified. |
| **Unique information** | A head carries task information that no other head provides. High unique information means the head is irreplaceable — its ablation should cause a large performance drop. |
| **Synergy** | Task information that is only available when considering multiple heads jointly. High synergy is the information-theoretic signature of composition: the heads compute something together that neither computes alone. |

**What the balance reveals.** A circuit dominated by unique information is modular — each head has a distinct role. A circuit dominated by synergy is compositional — the computation emerges from interaction. A circuit dominated by redundancy is distributed — the same signal is encoded many times. Most real circuits show a mixture, and the balance characterizes the circuit's computational architecture.

**Reference.** Williams & Beer (2010), "Nonnegative Decomposition of Multivariate Information."

**Source:** `protocols/information_theory/a08_pid.py`

---

## Extended Protocols

These protocols apply tools from adjacent fields to circuit evaluation, unified by their information-theoretic foundations.

### IT_CD — Capacity and Distortion

**Protocol ID:** `IT_CD`
**Validity type:** Measurement (External)

**Question.** What is the information-theoretic capacity of the circuit's communication channels, and what is the compression-fidelity tradeoff?

This is the protocol-level orchestrator for the IT1 (channel capacity) and IT2 (rate-distortion) core metrics. It runs both metrics, applies structural calibrations, and produces a unified capacity-distortion profile for the circuit.

**Metrics:** `channel_capacity` (threshold $> 0.5$), `rate_distortion` (threshold $> 0.5$).

**References.** Shannon (1948); Cover & Thomas (2006), *Elements of Information Theory*; Berger (1971), *Rate Distortion Theory*.

**Source:** `protocols/information_theory/capacity_distortion.py`

---

### IT_AC — Algorithmic Complexity

**Protocol ID:** `IT_AC`
**Validity type:** Construct (External)

**Question.** What is the algorithmic complexity of the circuit's computation? Low Kolmogorov complexity suggests the circuit implements a simple, compressible algorithm. High complexity suggests the computation is inherently complex or the circuit is over-parameterized.

This is the protocol-level orchestrator for the IT3 (Kolmogorov complexity proxy) core metric.

**Metric:** `kolmogorov_complexity` (threshold $> 0.5$).

**References.** Kolmogorov (1965); Solomonoff (1964), "A Formal Theory of Inductive Inference"; Li & Vitanyi (2008), *An Introduction to Kolmogorov Complexity and Its Applications*.

**Source:** `protocols/information_theory/algorithmic_complexity.py`

---

### WC_M9 — Hawkes Process (Self-Exciting Point Process)

**Protocol ID:** `WC_M9`
**Validity type:** Internal (Structural)

**Question.** Can component activations be modeled as a self-exciting point process over layer depth? When a component fires at layer $l$, does it increase the probability of other components firing at layer $l+1$?

**Core concept.** A Hawkes process (Hawkes 1971) is a point process where past events increase the rate of future events. Applied to circuits, the interaction matrix $\Phi[i,j]$ captures how component $j$'s activation excites (positive) or inhibits (negative) component $i$ at the next layer. The spectral radius of $\Phi$ is the branching ratio:

- **Branching ratio $> 1$** (supercritical): the circuit generates self-sustaining cascades. Information entering the circuit amplifies as it propagates.
- **Branching ratio $< 1$** (subcritical): the circuit's signal is damped. Without external reinforcement, the signal dies out.
- **Branching ratio $\approx 1$** (critical): the circuit is at the edge — signal maintains itself but does not amplify.

**Metrics:** `activation_patching` (threshold $> 0.5$), `eap` (threshold $> 0.3$), `effect_size` (threshold $> 0.8$).

**What the Hawkes analysis adds beyond ablation.** Ablation tells you which components matter. The Hawkes process tells you *how they interact over depth*: which components excite which others, whether the excitation is sufficient for self-sustaining propagation, and which components are the cascade initiators vs the cascade followers.

**References.** Hawkes (1971), "Spectra of Some Self-Exciting and Mutually Exciting Point Processes"; Ogata (1981), "On Lewis' Simulation Method for Point Processes"; Zhou et al. (2013), "Learning Triggering Kernels for Multi-Dimensional Hawkes Processes."

**Source:** `protocols/information_theory/hawkes_process.py`

---

### WC_M12 — Free Energy Decomposition (Active Inference)

**Protocol ID:** `WC_M12`
**Validity type:** Measurement (Causal)

**Question.** Can each attention head's contribution be decomposed into an accuracy term (how much it reduces cross-entropy loss) and a complexity term (how much its attention pattern diverges from uniform)?

**Core concept.** Drawing on Friston's Free Energy Principle (2010) and Ren et al.'s derivation of softmax attention from Helmholtz free energy (NeurIPS 2025), this protocol decomposes each head's contribution as:

$$F = \text{Complexity} - \text{Accuracy}$$

- **Accuracy:** how much the head reduces cross-entropy loss when included (measured via activation patching).
- **Complexity:** KL divergence of the head's attention pattern from the uniform distribution, measuring how much distributional "work" the head does.

A **low free-energy head** is doing useful work efficiently: high accuracy with minimal distributional complexity. A **high free-energy head** is either not helping predictions or using very peaked attention patterns without proportionate benefit.

**The Pareto frontier.** Plotting accuracy vs complexity for all heads in the model identifies the thermodynamically optimal circuit components — the heads achieving the most behavioral improvement per unit of distributional cost. Circuit heads should cluster on or near this frontier.

**Metrics:** `activation_patching` (threshold $> 0.5$), `effect_size` (threshold $> 0.8$), `sigma_ablation` (threshold $> 0.5$).

**References.** Friston (2010), "The Free-Energy Principle: A Unified Brain Theory?"; Ren et al. (NeurIPS 2025), "Transformers as Intrinsic Optimizers"; Parr & Friston (2019), "Generalised Free Energy and Active Inference."

**Source:** `protocols/information_theory/free_energy.py`

---

### WC_M13 — SIR Transmission (Epidemiological Circuit Model)

**Protocol ID:** `WC_M13`
**Validity type:** Internal (Structural)

**Question.** Can information spreading through the circuit be modeled as a contagion process? Each component is a node that can be susceptible, infected (active for the task), or recovered. The transmission rate $\beta_{AB}$ measures how much component $A$'s activation increases the probability that component $B$ also activates.

**Core concept.** The basic reproduction number $R_0$ per component measures how much it "spreads" task-relevant activation to downstream components:

- **$R_0 > 1$**: the component is a "superspreader" — its activation triggers a self-sustaining cascade through the circuit.
- **$R_0 < 1$**: the component is a transient activator — its signal dies out without external reinforcement from the residual stream.

**Component classification.**

| Classification | $R_0$ | Excess activation | Role |
|---|---|---|---|
| Hub / superspreader | High ($> 1$) | Variable | Propagates task information through the circuit |
| Terminal / output head | Low ($< 1$) | High | Receives but does not propagate — likely the circuit's output interface |
| Relay | $\approx 1$ | Moderate | Passes information through without amplification |

**Connection to criterion I12.** This protocol directly tests the [transmission criticality criterion (I12)](/framework/lenses/supporting/information-theory) from the information theory lens: the circuit's $R_0$ should exceed 1.0 (self-sustaining), and removing hub components should collapse $R_0$ below 1.0.

**Metrics:** `activation_patching` (threshold $> 0.5$), `eap` (threshold $> 0.3$), `effect_size` (threshold $> 0.8$).

**References.** Kermack & McKendrick (1927), "A Contribution to the Mathematical Theory of Epidemics"; Pastor-Satorras & Vespignani (2001), "Epidemic Spreading in Scale-Free Networks."

**Source:** `protocols/information_theory/sir_transmission.py`

---

### SM_PP — Predictive Processing

**Protocol ID:** `SM_PP`
**Validity type:** Construct (External)

**Question.** Does the circuit minimize prediction error? Is there a free energy gradient across layers? Does entropy cascade through the circuit hierarchy?

**Core concept.** Predictive processing (Clark 2013; Rao & Ballard 1999) frames computation as hierarchical prediction error minimization. Applied to circuits, three properties should hold:

1. **Free energy gradient.** Free energy (surprise) should decrease across layers: each layer reduces uncertainty about the output. A negative gradient indicates the circuit progressively reduces surprise.
2. **Prediction error.** The circuit should generate accurate top-down predictions. Low prediction error between what each layer "expects" and what it receives from the layer below.
3. **Entropy cascade.** Entropy of the circuit's representations should decrease across the hierarchy: early layers carry high-entropy, ambiguous representations; later layers carry low-entropy, refined representations.

**Metrics.**

| Metric | What it measures | Threshold |
|---|---|---|
| `free_energy_gradient` | Layer-wise free energy gradient (negative = reducing surprise) | $< 0.0$ |
| `prediction_error` | Prediction error magnitude between circuit predictions and targets | $< 0.5$ |
| `entropy_cascade` | Entropy change across circuit hierarchy (decreasing = refinement) | $< 0.0$ |

**What the combination means.** A circuit with a negative free energy gradient, low prediction error, and decreasing entropy implements a predictive processing pipeline: it progressively refines an uncertain input into a confident prediction. A circuit that fails all three is not performing hierarchical prediction — its layers are not organized as a refinement cascade. Mixed results (e.g., negative gradient but high prediction error) suggest the circuit refines representations but the refinement does not converge to the correct prediction.

**References.** Friston (2010), "The Free-Energy Principle: A Unified Brain Theory?"; Clark (2013), "Whatever Next? Predictive Brains, Situated Agents, and the Future of Cognitive Science"; Rao & Ballard (1999), "Predictive Coding in the Visual Cortex."

**Source:** `protocols/information_theory/predictive_processing.py`

---

## Umbrella Protocol: IT — Information Theory (Full)

**Protocol ID:** `IT`

The full information theory protocol combines all three core metrics — channel capacity, rate-distortion, and Kolmogorov complexity — into a single evaluation run. This provides a unified information-theoretic characterization: how much task information the circuit transmits (IT1), how compressible its representations are (IT2), and how structured its weights are (IT3).

**Usage:**
```bash
uv run python information_theory.py --tasks ioi sva --n-prompts 40
```

**Source:** `protocols/information_theory/information_theory.py`

---

## Summary Table

| ID | Name | Type | Validity | Key question |
|---|---|---|---|---|
| IT1 | Channel Capacity | Core metric | Construct | How much task information does each head transmit? |
| IT2 | Rate-Distortion | Core metric | Construct | How compressible are the circuit's representations? |
| IT3 | Kolmogorov Complexity | Core metric | Construct | How structured are the circuit's weights? |
| C01 | Information Flow | Protocol | Construct, Internal | Is there directed information transfer along claimed edges? |
| C02 | Decomposition | Protocol | Construct, Internal | Is information redundant, synergistic, or unique? |
| C03 | Causal Structure | Protocol | Construct, Internal | Can the circuit DAG be recovered from data? |
| C09 | Information Bottleneck | Protocol | Construct, Internal | Does the circuit compress input to a sufficient statistic? |
| A07 | Granger / Transfer Entropy | Protocol | Internal | Does upstream activity predict downstream beyond self-history? |
| A08 | PID | Protocol | Internal | How does task information decompose across heads? |
| IT_CD | Capacity-Distortion | Protocol | Measurement | Combined capacity and compression profile |
| IT_AC | Algorithmic Complexity | Protocol | Construct | Combined algorithmic complexity profile |
| WC_M9 | Hawkes Process | Protocol | Internal | Are cross-layer activations self-exciting? |
| WC_M12 | Free Energy | Protocol | Measurement | Are circuit heads Pareto-efficient in accuracy vs complexity? |
| WC_M13 | SIR Transmission | Protocol | Internal | Is information propagation self-sustaining ($R_0 > 1$)? |
| SM_PP | Predictive Processing | Protocol | Construct | Does the circuit minimize prediction error hierarchically? |

---

## Evidence Patterns

These patterns emerge from combining multiple information-theoretic metrics. No single metric is decisive; the diagnostic power comes from convergent or divergent signals across the suite.

| Pattern | Metrics involved | Interpretation |
|---|---|---|
| High channel capacity + high compression ratio + high weight structure | IT1, IT2, IT3 | The circuit transmits task information efficiently through structured, compressible representations |
| High MI flow + confirmed directionality + $R_0 > 1$ | C01, A07, WC_M13 | Coherent, directed, self-sustaining information pipeline |
| High synergy in PID + negative free energy gradient | A08, SM_PP | Compositional computation with hierarchical refinement |
| Low channel capacity + high activation patching | IT1, C01 | Components are causally important but do not transmit task information — the mechanism may be inhibitory rather than transmissive |
| High sufficiency ratio + high compression ratio ($\approx 1$) | C09 | The "circuit" is the whole model — no genuine compression |
| $R_0 < 1$ + hub removal has no effect | WC_M13 | Signal is not self-sustaining; the circuit depends on external reinforcement |
| Granger significant but transfer entropy not | A07 | The inter-component relationship is linear |
| Transfer entropy significant but Granger not | A07 | The inter-component relationship is nonlinear |

---

## How This Lens Connects to Validity Criteria

The information theory lens maps onto four criteria defined in the [lens overview](/framework/lenses/supporting/information-theory):

| Criterion | Code | Primary metrics |
|---|---|---|
| Information bottleneck | C9 | C09 (sufficiency + compression), C02 (info bottleneck sub-metric) |
| Directed information flow | I11 | C01, A07 (Granger + transfer entropy) |
| Transmission criticality | I12 | WC_M13 (SIR $R_0$), WC_M9 (Hawkes branching ratio) |
| Free energy efficiency | M8 | WC_M12 (Pareto frontier) |

A circuit that satisfies all four criteria has the information-theoretic properties expected of a genuine computational pathway: it compresses input to a sufficient statistic (C9), transmits it through directed channels (I11), sustains the signal through its depth (I12), and does so without wasting capacity (M8).
