---
title: "Verdicts"
description: "A five-tier evidential grading system for mechanistic claims — what each tier requires and how claims move between them."
---

# Verdicts

A verdict is a **composite assessment of evidential status** — it answers the question "how well-established is this mechanistic claim, given all available evidence across all five validity types?" The verdict is not a quality judgment on the paper. It is a characterization of where the claim stands on the path from initial proposal to full validation, with specific gaps named.

## Why tiers, not scores

A continuous score (e.g., "this claim has validity 0.73") implies a precision the evidence does not support and obscures qualitative transitions. The difference between a claim with one causal experiment and a claim with five convergent lines of evidence is not well-captured by assigning them 0.4 and 0.8 on a scale — the second has crossed a qualitative threshold (convergence) that changes what the claim means.

The tier system makes these thresholds explicit. Each tier has a *minimum evidence package* — a set of criteria that must be satisfied (not merely partially addressed) for the claim to occupy that tier. Movement between tiers is governed by specific upgrade conditions that name what additional evidence is required.

## Intellectual origins

| Source | Year | Contribution |
|---|---|---|
| [Hill](https://doi.org/10.1177/003591576505800503) | 1965 | Nine criteria for causal inference in epidemiology (strength, consistency, specificity, temporality, biological gradient, plausibility, coherence, experiment, analogy). Hill did not intend these as a checklist — he intended them as considerations that, in aggregate, make a causal claim more or less credible. The tier system operationalizes this aggregate judgment. |
| [Grading of Recommendations Assessment (GRADE)](https://doi.org/10.1136/bmj.328.7454.1490) | 2004 | Evidence grading in clinical medicine: High / Moderate / Low / Very Low. Each grade has specific upgrade and downgrade conditions (inconsistency, indirectness, imprecision, publication bias). The framework demonstrated that tiered grading with named transitions is more actionable than continuous scores. |
| [Shadish, Cook & Campbell](https://psycnet.apa.org/record/2001-18082-000) | 2002 | Threats to validity as *reasons for doubt* — each threat, if present and unaddressed, downgrades the evidential status of the claim. The tier system formalizes which threats block which transitions. |
| [Lakatos, *The Methodology of Scientific Research Programmes*](https://doi.org/10.1017/CBO9780511621123) | 1978 | Progressive vs. degenerative research programmes — a programme is progressive if it generates novel predictions that are subsequently confirmed, degenerative if it only accommodates known facts. The Proposed → Triangulated progression tracks this: a claim that only explains the data it was discovered from is at a lower tier than one that generates and confirms novel predictions. |

## The five tiers

### Tier 1: Proposed

**What it means:** A mechanistic claim has been stated with enough precision to be evaluated, but the evidence does not yet establish causal relevance. The claim is on the table — it is falsifiable and interesting — but it has not been causally tested.

**Minimum evidence package:**
- A defined construct (the entity is named and its boundaries are stated)
- A falsifiable prediction (at least one testable consequence of the claim)
- A measurement (at least one number has been produced, with metric stated)

**What is NOT required:** Causal evidence. A claim can be Proposed based purely on correlational, structural, or statistical evidence — e.g., "this probe achieves 90% accuracy on syntactic number at layer 8" or "these SAE features have cosine similarity > 0.7 with known direction $\hat{v}$."

**Characteristic occupants:** Probing results without intervention. SAE feature descriptions without causal testing. Weight-space structural analyses without behavioral confirmation. Linear decodability claims (Othello board state, syntactic features) prior to interchange intervention.

**What holds claims here:** The absence of causal evidence (I1–I2) is the primary barrier. A claim cannot exit Proposed without at least necessity evidence.

<details class="worked-example">
<summary>Worked example: SAE features at Tier 1</summary>

A sparse autoencoder trained on GPT-2 Small residual stream activations produces a feature $f_{42}$ whose decoder direction has high cosine similarity with the "is_noun" probing direction, whose top-activating contexts are predominantly nouns, and whose activation magnitude correlates with the model's confidence on syntactic tasks.

This is a Proposed claim. The evidence is correlational and structural: the feature *looks like* it represents noun-hood. But no intervention has been performed. We do not know whether the feature is *causally relevant* to noun-related computation (I1), whether it is *sufficient* (I2), or whether it is *specific* to noun-hood rather than a correlated property like word frequency (I3).

The claim is well-posed (falsifiable, with a defined construct and quantitative measurements). It simply hasn't been causally tested. Moving to Tier 2 requires ablating or patching the feature and demonstrating a noun-specific behavioral change.
</details>

---

### Tier 2: Causally suggestive

**What it means:** There is causal evidence that the claimed mechanism is involved in the behavior — removing or disrupting it changes the output. The evidence establishes *necessity* but not yet sufficiency, specificity, or convergence.

**Minimum evidence package (in addition to Tier 1):**
- Necessity (I1): Ablating or disrupting the claimed mechanism changes the target behavior by more than a size-matched random control
- Effect magnitude (E4): The absolute effect size is large enough that the computational story is coherent (not a large fraction of a tiny signal)
- Level declaration (V1): The claim is stated at a level consistent with the evidence type

**Upgrade condition from Proposed:** At least one well-controlled causal experiment demonstrating that the claimed mechanism is necessary for the behavior. "Well-controlled" means: (a) the behavioral change is measured against a baseline, (b) a random-component control establishes that the effect is specific to the claimed components, and (c) the ablation method is named as part of the claim.

**What holds claims here:** Sufficiency (I2) is the most common missing piece. Most circuits demonstrate necessity without sufficiency — ablating them hurts, but the circuit alone does not reproduce the behavior. Additionally, specificity (I3) is often untested: the ablation hurts the target behavior, but does it also hurt everything else? If so, the component is a general bottleneck, not a specific mechanism.

**Characteristic occupants:** Most published circuit findings in MI. The IOI circuit (Wang et al. 2022) under its original evaluation. Individual head ablation studies. Knowledge neuron editing (Meng et al. 2022). Most activation-patching-based claims.

---

### Tier 3: Mechanistically supported

**What it means:** The mechanism is established as both necessary and sufficient for the behavior under at least one well-characterized ablation method, with specificity evidence demonstrating that the effect is mechanism-specific rather than reflecting general model degradation. The claim has moved from "this is involved" to "this is specifically and sufficiently responsible."

**Minimum evidence package (in addition to Tier 2):**
- Sufficiency (I2): The claimed mechanism, operating alone or with minimal support, reproduces the target behavior to within a stated tolerance
- Specificity (I3): The effect is selective — the ablation changes the target behavior substantially more than unrelated behaviors (selectivity index $SI > 10$, or the related-task comparison demonstrates meaningful separation)
- Consistency (I4): The result replicates across at least two of: prompt templates, ablation methods, or random seeds
- Measurement reliability (M1): Bootstrap CI on the principal metric demonstrates $\rho_{XX'} \geq 0.7$
- Calibration (M5): The reported metric is located against at least one published reference point on the same task and model

**Upgrade condition from Causally suggestive:** Sufficiency + specificity. The claim must demonstrate both that the mechanism is enough (sufficiency) and that it is the right thing (specificity). Either alone is insufficient: a mechanism can be sufficient but non-specific (a large chunk of the model reproduces any behavior), or specific but not sufficient (the component does exactly one thing, but removing it doesn't fully explain the behavior because other components also contribute).

**Formal characterization of sufficiency:**

Let $B$ be the target behavior measured by metric $M$. Let $C$ be the claimed circuit. Let $\bar{C}$ denote the complement (all components not in $C$). Sufficiency requires:

$$R = \frac{M(C)}{M(\text{full})} \geq \tau$$

where $M(C)$ is the metric with only $C$ active (complement ablated) and $\tau$ is a stated threshold. Wang et al. use $\tau = 0.8$ for the IOI circuit; we recommend stating $\tau$ rather than using an implicit standard.

Note that sufficiency is *method-dependent*: the complement ablation method (zero, mean, resample) is part of the claim. Miller et al. (2024) demonstrated that IOI's $R \approx 0.87$ under mean ablation drops below 0.50 under resample ablation, changing the verdict.

**Characteristic occupants:** Induction heads (Olsson et al. 2022) — necessity, sufficiency, and specificity all demonstrated. Greater-Than circuit (Hanna et al. 2023) — strong structural plausibility and specificity evidence. Copy suppression (McDougall et al. 2023) — unusually clean specificity.

---

### Tier 4: Triangulated

**What it means:** Multiple independent lines of evidence converge on the same mechanistic account. The mechanism has been confirmed by methods with non-overlapping assumptions — weight-space analysis, activation-based patching, behavioral intervention, and/or cross-model comparison. No single method's failure would collapse the claim.

**Minimum evidence package (in addition to Tier 3):**
- Multi-method convergence (C5): At least two methods with genuinely different assumptions (not just two variants of patching) confirm the same mechanism. The Jaccard similarity between their identified components is reported.
- External robustness (E5): The mechanism appears across at least two prompt distributions not used during discovery, or across two model sizes within the same family.
- Cross-procedure agreement (V2): Different discovery procedures return overlapping circuits, and the overlap is characterized quantitatively.
- Nomological network density: The construct makes at least three independently testable predictions, of which at least two have been confirmed by different methods.

**Upgrade condition from Mechanistically supported:** Convergence. A single methodology, no matter how well-executed, produces a finding that is *conditional on that methodology's assumptions*. Triangulation means the finding survives the failure of any single method's assumptions.

**Formal characterization of convergence:**

Let $C_1, C_2, \ldots, C_k$ be circuits identified by $k$ different procedures $P_1, \ldots, P_k$ (where the $P_i$ differ in their core assumptions — e.g., weight-based vs. activation-based vs. behavioral). Convergence is:

$$J_{\text{all}} = \frac{|C_1 \cap C_2 \cap \ldots \cap C_k|}{|C_1 \cup C_2 \cup \ldots \cup C_k|} \geq \tau_J$$

The *robust core* $C_1 \cap \ldots \cap C_k$ contains the components that every method agrees on. Claims about the robust core are more strongly supported than claims about the full union.

**What convergence is NOT:** Running the same method twice (e.g., activation patching with different hyperparameters) is replication, not triangulation. The methods must have *genuinely non-overlapping failure modes* — if one method fails due to a distributional assumption, the other must not share that assumption.

**Characteristic occupants:** Induction heads are the strongest candidate — confirmed by attention pattern analysis, QK composition analysis, training dynamics (phase transition), cross-model search, and behavioral ablation. Each of these methods has a different failure mode, and they all converge.

---

### Tier 5: Validated (within scope)

**What it means:** The mechanism is fully characterized within a stated scope. All five validity types pass at their respective criteria. The mechanistic account is complete: every component's function is known, the information flow is demonstrated, the account makes quantitative predictions that have been tested, and the scope of the claim is explicitly bounded. This is what "fully understood" looks like.

**Minimum evidence package (in addition to Tier 4):**
- All five validity types pass at their primary criteria
- Component-level function (mode $I_{\text{fun}}$): The input-output function of each component in the circuit is characterized
- Quantitative prediction: The mechanistic account generates at least one novel quantitative prediction that was confirmed after the prediction was stated
- Scope declaration: The domain over which the mechanism operates is explicitly bounded, and the boundary is tested (cases just outside scope should show the mechanism failing)
- Coverage $\kappa > 0.9$ on a representative distribution

**Upgrade condition from Triangulated:** Completeness. The account must be *closed* — there are no uncharacterized components, no gaps in the information flow, no untested predictions. This does not mean omniscient — it means the scope is stated and within that scope, the account is complete.

**Why "within scope":** Validated is not "true of all language models" or even "true of this model on all inputs." It is "true of this model, on this class of inputs, within this explanatory scope." The scope restriction is not a weakness — it is honesty about what has actually been established.

**Characteristic occupants:** Grokking / modular addition (Nanda et al. 2023) — a toy transformer trained on modular addition, where every weight matrix is explained by the Fourier algorithm, quantitative predictions are confirmed, and the scope (one-layer toy model, single arithmetic task) is explicit. Superposition theory (Elhage et al. 2022) in toy models — validated as a mathematical framework within the scope of toy models with known feature statistics.

**Why so few claims reach this tier:** Validated requires *completeness*, not just *correctness*. A circuit can be correctly identified (every component it names is genuinely involved) without being completely characterized (every component's function is known and the information flow is fully traced). Completeness is expensive — it requires explaining not just what the circuit does but how each part contributes. For real-model circuits with dozens of components, this remains difficult. That difficulty is real, not an artifact of high standards.

---

## Lateral verdicts

Two verdicts sit outside the progression. They are not lower or higher — they are *different kinds of conclusions*.

### Underdetermined

**What it means:** The evidence is consistent with multiple mechanistic accounts and does not distinguish between them. The claim is not wrong — it is *underdetermined*. Multiple explanations survive the available evidence.

**When to assign:** When two or more mechanistic accounts have comparable evidential support and no available experiment distinguishes them. This is not a failure of the research — it is a characterization of the current state. The informative response is to name the competing accounts and identify what experiment would distinguish them.

**Formal characterization:** Let $H_1, H_2, \ldots, H_n$ be competing mechanistic hypotheses for behavior $B$. Underdetermination holds when:

$$\forall i, j: \quad P(\mathcal{E} | H_i) \approx P(\mathcal{E} | H_j)$$

That is, the available evidence $\mathcal{E}$ is approximately equally likely under all competing hypotheses. The posterior ratio $P(H_i | \mathcal{E}) / P(H_j | \mathcal{E})$ is determined primarily by priors, not evidence.

**Example:** The Docstring Circuit (Heimersheim & Janiak 2023) — is the mechanism "variable binding" (tracking which variable name corresponds to which argument position) or "positional copying" (copying from a fixed offset regardless of variable identity)? Both accounts are consistent with the observed activation patching results. The experiment that would distinguish them (testing on prompts where the two accounts predict different outputs) has not been performed.

### Disconfirmed

**What it means:** The evidence actively contradicts the mechanistic claim. Not "insufficient evidence" but "evidence against."

**When to assign:** When a specific prediction of the claimed mechanism has been tested and failed, OR when the mechanism has been shown to be an artifact of the measurement procedure.

**Types of disconfirmation:**
- **Prediction failure**: The mechanism predicts behavior $X$ and the model produces behavior $\neg X$ in the relevant conditions.
- **Artifact demonstration**: The claimed mechanism is shown to be an artifact of the measurement (e.g., the patching result disappears when the distributional assumption of mean ablation is corrected).
- **Construct dissolution**: The entity named by the claim is shown not to be a coherent construct (e.g., "the bias circuit" is indistinguishable from "the gender knowledge circuit" — the construct cannot be separated from legitimate processing).

**Disconfirmation is not failure.** A disconfirmed claim is informative — it narrows the space of possible mechanisms. A field that never disconfirms is not doing science. The lateral position of Disconfirmed (rather than placing it below Proposed) reflects this: disconfirmation is a *different kind of conclusion*, not a worse one.

## Upgrade mechanics

### Upgrade conditions are conjunctive

To move from Tier $n$ to Tier $n+1$, *all* conditions for Tier $n+1$ must be satisfied simultaneously. Partial satisfaction of Tier $n+1$ conditions does not produce a position "between" tiers — the claim remains at Tier $n$ with the partial evidence noted.

### Downgrades are possible

New evidence can move a claim to a lower tier. Miller et al. (2024) effectively downgraded the IOI circuit's internal validity by demonstrating method-conditional results — what was presented as a stable finding is actually conditional on the ablation choice. Downgrades are not punitive — they are the system updating on new evidence.

### The weakest-link principle

The tier is determined by the *weakest gating validity type*, not the average. A claim with excellent internal, external, and interpretive validity but unreliable measurement is bounded at Proposed (measurement failure blocks all upgrades). This prevents impressive evidence in one dimension from masking fundamental problems in another.

Formally: let $V_C, V_I, V_E, V_M, V_V$ be the per-type assessments. The maximum achievable tier is:

$$\text{Tier}_{\max} = \min(\text{tier allowed by } V_C, \text{tier allowed by } V_I, \ldots, \text{tier allowed by } V_V)$$

where each $V_x$ imposes a ceiling based on its assessment level.

## Verdict statement format

A verdict is not just a tier label — it is a structured statement:

```
Verdict: [Tier] — [Mode tag]

Strongest evidence: [Validity type]: [Specific criterion]
Weakest evidence: [Validity type]: [Specific criterion]
Primary gap: [What would be needed for the next tier]
Scope: [Explicit bounds of the claim]
```

The mode tag (from the [Description Modes](/framework/description-modes/)) identifies the level at which the claim is stated. The strongest/weakest pairing identifies where the claim is most and least secure. The primary gap names a specific next experiment. The scope bounds the claim.

This format ensures that a verdict is actionable — it tells the reader not just where the claim stands but what would change its standing.
