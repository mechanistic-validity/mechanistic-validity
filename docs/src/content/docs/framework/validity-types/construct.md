---
title: "Construct Validity"
description: "Formal specification: quantitative thresholds, pass conditions, and calibration data for construct validity criteria C1–C5."
---

# Construct Validity — Formal Specification

| | |
|---|---|
| Question | Is the thing being claimed a coherent theoretical entity? |
| Lens | [Philosophy of Science](/framework/lenses/core/philosophy-of-science) |
| Criteria | C1–C5 |
| Dependency | Construct validity is prior to all other validity types — ambiguity here propagates downstream |
| Status in MI | Most neglected type; most circuit papers name the construct without specifying it |

Construct validity asks whether the entity being claimed exists as a well-defined theoretical object. The [Philosophy of Science lens](/framework/lenses/core/philosophy-of-science) explains the intellectual background and shows the criteria applied to real cases. This page gives the formal definitions, quantitative thresholds, and calibration data.

## C1 — Falsifiability

> [Full criterion page →](/framework/criteria/construct/falsifiability)

A claim is falsifiable when a disconfirming observation is specified before evidence collection. The specification must name three things:

$$\text{Falsifiability condition} = (\text{metric } m, \; \text{threshold } \tau, \; \text{dataset } D)$$

**Pass condition:** All three components stated in advance. If retrospective, this is disclosed.

**Formal requirement:** There exists a measurement $m(C, D)$ of circuit $C$ on dataset $D$ such that:

$$m(C, D) < \tau \implies \text{claim is disconfirmed}$$

**Examples of valid conditions:**
- $\text{IIA}(C, D_{\text{held-out}}) < 0.10$
- $\text{Faithfulness}(C, D_{\text{paraphrase}}) < 0.50$ under resample ablation
- $\text{Logit diff recovery} < 0.30$ on template-varied prompts

**Examples of invalid conditions:**
- "If the circuit doesn't work" (no metric, no threshold, no dataset)
- "If faithfulness is low" (no threshold)
- "If the ablation fails on the same prompts used for discovery" (discovery set, not held-out)

**Calibration:** No published circuit paper we are aware of states a quantitative falsifiability condition in advance. This criterion is aspirational but enforceable going forward.

## C2 — Structural Plausibility

> [Full criterion page →](/framework/criteria/construct/structural-plausibility)

A component's weight-space signature must match its claimed computational role.

**Pass condition:** For every named component role, the weight-space measurement is consistent with the claim.

**Formal requirements by role type:**

*Copying head (name-mover, induction head):* The $W_{OV}$ matrix should approximate a copying operation. We measure the copying score:

$$\text{CopyScore}(h) = \frac{1}{|V|} \sum_{t \in V} \frac{(W_U \, W_{OV}^{(h)} \, W_E)_{t,t}}{\max_j (W_U \, W_{OV}^{(h)} \, W_E)_{t,j}}$$

where $V$ is a relevant token vocabulary, $W_E$ is the embedding, and $W_U$ is the unembedding. A copying head should have $\text{CopyScore} > 0.5$.

*Ordinal head (successor, Greater-Than):* The $W_{OV}$ should encode monotonic ordering:

$$\text{effect}(y_1, y_2) = e_{y_2}^\top \, W_U \, W_{OV}^{(h)} \, W_E \, e_{y_1}$$

Structural plausibility requires $\text{Corr}(\text{effect}(y_1, y_2), \; \text{sign}(y_2 - y_1)) > 0.7$ across relevant token pairs.

*Inhibition head (S-inhibition):* Attention pattern should peak at the position of the repeated subject. Measured as:

$$\text{AttnFrac}(h, \text{pos}_S) = \frac{A^{(h)}_{\text{final}, \text{pos}_S}}{\sum_j A^{(h)}_{\text{final}, j}}$$

Structural plausibility requires $\text{AttnFrac} > 0.3$ on clean IOI prompts (above uniform attention $\approx 0.07$ for a 15-token sequence).

**Failure threshold:** Any mismatch between role label and weight-space signature must be flagged. A "name-mover" with $\text{CopyScore} < 0.3$ fails C2.

## C3 — Task Specificity

> [Full criterion page →](/framework/criteria/construct/task-specificity)

The circuit should not score highly on unrelated tasks under the same evaluation.

**Pass condition:** The selectivity ratio is positive on at least one related off-task.

**Selectivity ratio:** For circuit $C$ discovered on task $T_{\text{disc}}$ and evaluated on related task $T_{\text{off}}$:

$$S(C) = \frac{F(C, T_{\text{disc}}) - F(C, T_{\text{off}})}{F(C, T_{\text{disc}})}$$

| $S$ value | Interpretation |
|---|---|
| $S > 0.5$ | Strong task specificity — circuit is substantially more faithful on its discovery task |
| $0 < S \leq 0.5$ | Moderate specificity — circuit has some off-task faithfulness but favors discovery task |
| $S \approx 0$ | No specificity — circuit is equally faithful on both tasks (bottleneck or general-purpose) |
| $S < 0$ | Inverted specificity — circuit is *more* faithful on the off-task (red flag) |

**Off-task selection:** The off-task must be *related*, not trivially distinct.

| Discovery task | Informative off-task | Trivial off-task (too easy) |
|---|---|---|
| IOI | Subject-verb agreement | Modular arithmetic |
| Greater-Than | Successor | Translation |
| Gendered pronouns | IOI | Factual recall |

**Calibration:** No published circuit paper reports a selectivity ratio. This is the gap C3 is designed to close.

## C4 — Minimality

> [Full criterion page →](/framework/criteria/construct/minimality)

Every component must be individually necessary given the others.

**Pass condition:** For every $c_i \in C$, ablating $c_i$ while leaving all other members intact produces a performance decrease exceeding threshold $\delta$.

**Formal definition:** Circuit $C = \{c_1, \ldots, c_n\}$ is minimal if and only if:

$$\forall \, c_i \in C: \quad F(C) - F(C \setminus \{c_i\}) > \delta$$

where $F$ is the faithfulness score and $\delta$ is the minimum meaningful effect. A reasonable default is $\delta = 0.02$ (2% faithfulness drop).

**Joint vs individual necessity:** Two components $c_i, c_j$ are jointly redundant if:

$$F(C \setminus \{c_i\}) \approx F(C) \quad \text{and} \quad F(C \setminus \{c_j\}) \approx F(C) \quad \text{but} \quad F(C \setminus \{c_i, c_j\}) \ll F(C)$$

This pattern indicates backup mechanisms. [Wang et al. (2022)](https://arxiv.org/abs/2211.00593) found this with IOI backup name-movers. Both components and their relationship should be reported.

**Calibration:**

| Circuit | Components | After pruning | Redundant members found |
|---|---|---|---|
| IOI ([Wang et al. 2022](https://arxiv.org/abs/2211.00593)) | 26 heads | ~20 core + 6 backup | Yes — backup name-movers |
| Greater-Than ([Hanna et al. 2023](https://arxiv.org/abs/2305.00586)) | ~12 heads | Not reported | Not tested |

## C5 — Convergent Validity

> [Full criterion page →](/framework/criteria/construct/convergent-validity)

Multiple independent metrics should identify the same components.

**Pass condition:** $J(C_A, C_B) \geq 0.5$ between metrics from different evidence families.

**Jaccard similarity:**

$$J(C_A, C_B) = \frac{|C_A \cap C_B|}{|C_A \cup C_B|}$$

| $J$ value | Interpretation |
|---|---|
| $J > 0.6$ | Strong convergent validity — methods agree on most components |
| $0.3 \leq J \leq 0.6$ | Moderate — partial agreement, investigate discrepancies |
| $J < 0.3$ | Weak — circuit is method-dependent |
| $J \approx 0$ | Failed — methods identify different components entirely |

**Independence requirement:** The two metrics must come from different [evidence families](/framework/evidence-families/) with non-overlapping major assumptions.

| Valid pair | Why independent |
|---|---|
| Activation patching + weight classifier | Causal (interventionist) vs structural (static weights) |
| DAS-IIA + SVD spectral analysis | Representational (learned subspace) vs structural (spectral) |
| EAP + linear probe | Causal (gradient-based) vs representational (supervised) |

| Invalid pair | Why dependent |
|---|---|
| Zero ablation + mean ablation | Both causal, both interventionist, share confound structure |
| Activation patching + path patching | Same framework, one is a refinement of the other |

**MTMM inequality ([Campbell & Fiske 1959](https://doi.org/10.1037/h0046016)):** For trait $i$ measured by methods $a$ and $b$, convergent validity requires:

$$r_{ia, ib} > r_{ia, jb} \quad \text{for all } j \neq i$$

Two methods should agree more about the same circuit than about different circuits measured by the same method. When this inequality fails, the method is driving the result more than the mechanism.

**Calibration:**

| Circuit pair | Methods | $J$ | Interpretation |
|---|---|---|---|
| IOI: patching vs weight classifier | Causal vs structural | ~0.67 (project estimate) | Strong convergent validity |
| SVA: weight circuit vs EAP circuit | Structural vs causal | ~0.0 (observed in this project) | Failed — underdetermined |
| Induction heads: behavioral vs structural | Behavioral vs structural | High (qualitative) | Cross-model agreement supports convergent validity |

## Partial-pass interpretation

| Pattern | Criteria met | Interpretation | Recommended language |
|---|---|---|---|
| Pre-registered, structurally coherent, but single-method | C1, C2 | Well-defined construct, method-dependent identification | "Coherent construct, convergence not yet tested" |
| Convergent, but not task-specific | C1, C5 | Real entity, but may be general-purpose | "Convergent but non-discriminant" |
| Minimal and specific, but no convergence | C3, C4 | Task-specific finding from one method | "Task-specific by one metric, convergence needed" |
| All met except falsifiability | C2–C5 | Strong post-hoc case, but not pre-registered | "Retrospectively well-supported, not prospectively falsifiable" |
| None met | — | Label without construct backing | "Named but not validated as a construct" |

## Protocol

For a proposed circuit $C$ and behavior $B$:

1. **C1.** State $(m, \tau, D)$ before collecting evidence.
2. **C2.** For every named role, compute the relevant weight-space metric (CopyScore, attention fraction, or effect correlation). Flag mismatches.
3. **C3.** Evaluate $F(C, T_{\text{off}})$ on at least one related task. Compute $S(C)$.
4. **C4.** Per-component leave-one-out ablation. Report $F(C) - F(C \setminus \{c_i\})$ for each $c_i$.
5. **C5.** Identify one method from a different evidence family. Compute $J(C_{\text{method 1}}, C_{\text{method 2}})$.
