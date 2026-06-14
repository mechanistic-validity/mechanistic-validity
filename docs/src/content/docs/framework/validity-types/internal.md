---
title: "Internal Validity"
description: "Formal specification: quantitative thresholds, pass conditions, and calibration data for internal validity criteria I1–I5."
---

# Internal Validity — Formal Specification

| | |
|---|---|
| Question | Does the evidence establish that the component implements the computation, not merely participates in it? |
| Lens | [Neuroscience](/framework/lenses/core/neuroscience) |
| Criteria | I1–I6 |
| Dependency | Internal validity is the workhorse — most MI evidence is internal-validity evidence. But it says nothing about whether the finding generalizes ([external](/framework/validity-types/external)), whether the metric is reliable ([measurement](/framework/validity-types/measurement)), whether the construct is coherent ([construct](/framework/validity-types/construct)), or whether the narrative is correct ([interpretive](/framework/validity-types/interpretive)). |
| Status in MI | Best-addressed by existing methods; still routinely method-conditional |

Internal validity asks whether the causal inference from intervention to behavior is licensed within the experimental setup. The [Neuroscience lens](/framework/lenses/core/neuroscience) explains the intellectual background. This page gives the formal definitions, quantitative thresholds, and calibration data.

## I1 — Necessity

> [Full criterion page →](/framework/criteria/internal/necessity)

Removing the component should degrade the behavior. For circuit $C$, model $f$, input $x$, and counterfactual value $\bar{C}$:

$$\text{Necessity}(C) = \frac{f(x) - f(x \mid \text{do}(C := \bar{C}))}{f(x)}$$

**Pass condition:** $\text{Necessity}(C) > 0.10$ with an equal-size random-component baseline producing $\text{Necessity}(C_{\text{random}}) < 0.05$.

| $\text{Necessity}$ value | Interpretation |
|---|---|
| $> 0.80$ | Strong necessity — component is critical for the behavior |
| $0.30 - 0.80$ | Moderate — component contributes but is not the sole driver |
| $0.10 - 0.30$ | Weak — component participates but may be one of many |
| $< 0.10$ | Not necessary — indistinguishable from random components |

**Ablation method is part of the claim.** Necessity scores are a joint property of the component and the ablation type. [Miller, Chughtai & Saunders (2024)](https://arxiv.org/abs/2404.01945) show that the same circuit's faithfulness varies from 87% under mean ablation to below 50% under other methods. The full claim must state the ablation method.

**Common confounds:**
- **Bottleneck confound.** A component that many computations route through is necessary for all of them, but implements none in particular.
- **Off-manifold confound.** Zero and mean ablation push activations to values the model never encounters during training.

**Calibration:**

| Circuit | Method | $\text{Necessity}$ | Notes |
|---|---|---|---|
| IOI name-movers ([Wang et al. 2022](https://arxiv.org/abs/2211.00593)) | Mean ablation | $\approx 0.87$ | Drops logit diff from 3.56 to 0.46 |
| IOI name-movers | Resample ablation | $< 0.50$ | Method-dependent; same circuit, weaker score |
| Induction heads ([Olsson et al. 2022](https://arxiv.org/abs/2209.11895)) | Mean ablation | High (qualitative) | Stronger on repeated sequences, weaker on non-repeated |

## I2 — Sufficiency

> [Full criterion page →](/framework/criteria/internal/sufficiency)

Isolating or restoring the component should reproduce the behavior. The recovery fraction is:

$$R = \frac{f_{\text{circuit}}(x)}{f_{\text{full}}(x)}$$

**Pass condition:** $R \geq 0.70$ on held-out prompts, with the complement ablation method stated.

| $R$ value | Interpretation |
|---|---|
| $R > 0.90$ | Strong sufficiency — circuit reproduces nearly all of the behavior in isolation |
| $0.70 \leq R \leq 0.90$ | Moderate — circuit captures most of the behavior |
| $0.50 \leq R < 0.70$ | Weak — circuit contributes substantially but something is missing |
| $R < 0.50$ | Not sufficient — circuit alone does not drive the behavior |

**The asymmetry with necessity.** Necessity requires ablating the circuit. Sufficiency requires ablating everything *outside* the circuit. Resample ablation of the complement is a stricter test than mean ablation, since mean ablation leaves systematic residual signal.

**Two forms of sufficiency:**
- *Isolation sufficiency:* Run only the circuit; ablate the complement. This is what $R$ measures.
- *Restoration sufficiency:* In a corrupted prompt where the behavior fails, restoring only the circuit restores the behavior. This is the activation-patching form and typically yields higher $R$ because the rest of the model remains intact.

**Calibration:**

| Circuit | Method | $R$ | Notes |
|---|---|---|---|
| IOI ([Wang et al. 2022](https://arxiv.org/abs/2211.00593)) | Mean ablation of complement | $\approx 0.87$ | 87% of logit diff recovered |
| Greater-Than ([Hanna et al. 2023](https://arxiv.org/abs/2305.00586)) | Mean ablation of complement | $\approx 0.895$ | 89.5% of probability diff recovered |

## I3 — Specificity

> [Full criterion page →](/framework/criteria/internal/specificity)

The component should be more necessary for the target behavior than for unrelated behaviors.

**Pass condition:** $\text{Specificity}(C) > 1.0$ against at least one related off-task behavior.

$$\text{Specificity}(C, B, B') = \frac{\text{Necessity}(C, B)}{\text{Necessity}(C, B')}$$

| Specificity value | Interpretation |
|---|---|
| $> 3.0$ | Strong specificity — component is much more necessary for $B$ than $B'$ |
| $1.5 - 3.0$ | Moderate specificity |
| $1.0 - 1.5$ | Weak specificity |
| $< 1.0$ | Inverted — component is *more* necessary for the control behavior (red flag) |

**Off-task selection matters.** The control behavior $B'$ must be *related*, not trivially distinct.

| Target task | Informative off-task | Trivial off-task |
|---|---|---|
| IOI | Subject-verb agreement | Modular arithmetic |
| Greater-Than | Successor | Translation |
| Gendered pronouns | IOI | Factual recall |

**The double dissociation test.** The strongest specificity evidence is a double dissociation: ablating circuit $A$ impairs behavior $X$ but not $Y$, and ablating circuit $B$ impairs $Y$ but not $X$.

**Calibration:** No published circuit paper reports a formal specificity ratio against a related task. Induction heads have implicit specificity (stronger on repeated sequences than non-repeated), but this is not quantified as a ratio.

## I4 — Consistency

> [Full criterion page →](/framework/criteria/internal/consistency)

The effect should replicate across contexts sufficient to rule out an artifact of the discovery distribution.

**Pass condition:** Replication across at least two of three axes, with bootstrap confidence intervals on the principal metrics.

| Axis | What it tests | Example |
|---|---|---|
| Cross-prompt | Template or paraphrase robustness | IOI with varied syntactic structures |
| Cross-seed | Independence from random initialization | Same circuit found in independently trained copies |
| Cross-checkpoint | Stability across training | Circuit present at step 50k, 100k, and 200k |

**Calibration:**

| Circuit | Cross-prompt | Cross-seed | Cross-checkpoint | Assessment |
|---|---|---|---|---|
| IOI ([Wang et al. 2022](https://arxiv.org/abs/2211.00593)) | Partial (name substitutions, ABBA/BABA) | Not tested | Not tested | One axis, partially |
| Induction heads ([Olsson et al. 2022](https://arxiv.org/abs/2209.11895)) | Yes (any repeated sequence) | Yes (multiple model families) | Yes (training dynamics) | All three axes — unusually strong |
| Greater-Than ([Hanna et al. 2023](https://arxiv.org/abs/2305.00586)) | Partial (year ranges) | Not tested | Not tested | One axis, partially |

## I5 — Confound Control

> [Full criterion page →](/framework/criteria/internal/confound-control)

The observed effect should not be explained by collateral disruption to non-circuit components.

**Pass condition:** At least two ablation methods compared, with consistent results.

| Confound | Mechanism | Mitigation |
|---|---|---|
| **Off-manifold ablation** | Zero and mean ablation push activations to out-of-distribution values | Use resample ablation against a counterfactual distribution |
| **Backup suppression** | Ablating one component can suppress or activate backup mechanisms | Test individual and joint ablation; report backup activation |
| **Layer-norm redistribution** | Ablating a component changes layer-norm statistics for all subsequent components | Compare effects with and without freezing layer-norm parameters |

**Method comparison protocol:** Report the same metric under at least two ablation methods. If the results diverge substantially, the finding is method-conditional — flag it as such.

**Calibration:**

| Circuit | Methods compared | Consistent? | Notes |
|---|---|---|---|
| IOI ([Wang et al. 2022](https://arxiv.org/abs/2211.00593)) | Mean ablation only | N/A — single method | [Miller et al. (2024)](https://arxiv.org/abs/2404.01945) later showed method-dependence |
| IOI ([Miller et al. 2024](https://arxiv.org/abs/2404.01945)) | Mean vs. resample vs. others | No — substantial divergence | Faithfulness ranges 87% to below 50% depending on method |

## Partial-pass interpretation

| Evidence pattern | Criteria met | Interpretation | Recommended language |
|---|---|---|---|
| Necessary but not sufficient | I1 | Distributed or incomplete circuit | "Causally implicated, not localized" |
| Sufficient but not necessary | I2 | Redundancy or forced route | "A capable route, not shown necessary" |
| Necessary + sufficient, not specific | I1, I2 | General-capability component | "Real mechanism, not task-specific" |
| Necessary + sufficient + specific, not consistent | I1, I2, I3 | Benchmark artifact possible | "Locally established, not yet robust" |
| Strong I1 + I2, single ablation method | I1, I2 (conditional) | Method-conditional claim | "Sufficient under [method]; not tested under alternatives" |
| All six met | I1–I6 | Full internal validity | Upgrade to external validity testing |

## Protocol

For circuit $C$ and behavior $B$:

1. **I1.** Ablate $C$; record $\text{Necessity}(C)$ under at least two methods. Compare to equal-size random baseline.
2. **I2.** Ablate complement; record $R$. Use held-out prompts not used for discovery.
3. **I3.** Compute $\text{Necessity}(C, B')$ for one related off-task $B'$. Report specificity ratio.
4. **I4.** Replicate across at least two of: cross-prompt, cross-seed, cross-checkpoint.
5. **I5.** Compare results across ablation methods. If inconsistent, report the range and flag as method-conditional.
6. **I6.** Test at least one rival component set of comparable size. Report faithfulness gap. If rivals achieve comparable faithfulness, scope claim to "a sufficient mechanism."
