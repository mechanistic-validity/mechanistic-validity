---
title: "External Validity"
description: "Does the effect generalize beyond the specific conditions of discovery? — evaluating reach, scaling, and cross-architecture transfer."
---

# External Validity — Formal Specification

| | |
|---|---|
| Question | Does the claim generalize beyond the conditions in which it was tested? |
| Lens | [Pharmacology](/framework/lenses_v6/pharmacology) |
| Criteria | E1–E6 |
| Dependency | External validity converts an internally valid result into a property of the model rather than a property of the experiment |
| Status in MI | Improving with recent benchmarks; still routinely overstated |

External validity asks whether the claim made on the conditions tested generalizes to other prompt distributions, other model sizes, other model families, and other intervention strengths. A result that passes all internal-validity tests on one prompt distribution at one intervention strength is a *local result*. External validity is what makes it a *finding*.

The [Pharmacology lens](/framework/lenses_v6/pharmacology) operationalizes external validity because pharmacology has developed the most rigorous standards for exactly this set of questions: dose-response curves, therapeutic windows, selectivity ratios, and cross-population generalization. The MI analogs map cleanly onto this framework.

## Why external validity is distinct from internal consistency

External validity overlaps with the I4 (Consistency) criterion of internal validity, but they are not the same. Consistency is the internal-validity criterion that asks whether the causal pattern replicates across contexts *within the same experimental setup*. External validity is broader: it includes the quantitative shape of the effect across intervention strengths (the dose-response curve), the selectivity ratio between on-task and off-task effects, the absolute magnitude of the effect, and the transfer of the effect across model architectures. Consistency answers whether the result replicates; external validity answers whether the mechanism generalizes.

## E1 — Intervention Reach

> [Full criterion page →](/framework/criteria/external/intervention-reach)

The intervention must demonstrably modify the proposed target component, separately from whether the behavioral outcome changed.

**Pass condition:** Direct measurement of the target activation (or weight-based proxy) confirms the intervention reached the intended component with specificity above baseline collateral disruption.

**Formal requirement:**

$$\text{Reach}(C) = \frac{\Delta \text{activation}(C)}{\Delta \text{activation}(C_{\text{adjacent}})} > 5.0$$

where $C_{\text{adjacent}}$ is a same-layer adjacent component not in the circuit.

**Calibration:** Activation patching has high reach (intervention is localized by construction). Zero and mean ablation have lower reach (distributional shift propagates through layer norm). Weight-based interventions have perfect reach in principle but require verification that the modified weights are the ones driving the behavior.

## E2 — Graded Response

> [Full criterion page →](/framework/criteria/external/graded-response)

The effect must scale monotonically with intervention strength, with a measurable threshold and plateau.

**Pass condition:** At least 5 intervention strengths tested, spanning subthreshold to saturating. The dose-response curve is monotonic with $R^2 > 0.8$.

**Formal requirement:** Let $\lambda$ be intervention strength (0 = no intervention, 1 = full ablation). The graded response is:

$$\text{Effect}(\lambda) = f_\lambda(x) - f_0(x)$$

A graded response requires $\frac{d}{d\lambda}\text{Effect}(\lambda) \geq 0$ for all $\lambda$, with a defined threshold $\lambda^*$ below which effect is negligible and a plateau $\lambda^{**}$ above which additional strength produces no additional effect.

**MI-specific threats:**
- *Single-dose reporting.* Most MI papers report one ablation strength (usually $\lambda = 1$, full ablation). A mechanism that only appears at extreme intervention strengths — where the model is generally degraded — is less specific than one that appears at low strengths with a wide therapeutic window.
- *Non-monotonic responses.* Some circuits are suppressed by over-intervention (backup mechanisms activate). This would appear as a non-monotonic curve and must be reported rather than hidden.

## E3 — Selectivity

> [Full criterion page →](/framework/criteria/external/selectivity)

The ratio of on-task to off-task effect at matched intervention strength must be substantial.

**Pass condition:** $\text{Selectivity}(C) > 3.0$ at the threshold intervention strength $\lambda^*$.

$$\text{Selectivity}(C, \lambda) = \frac{\text{Effect}(C, T_{\text{disc}}, \lambda)}{\text{Effect}(C, T_{\text{off}}, \lambda)}$$

| Selectivity value | Interpretation |
|---|---|
| $> 5.0$ | High selectivity — circuit is primarily task-specific |
| $3.0 - 5.0$ | Moderate selectivity — acceptable for most MI claims |
| $1.5 - 3.0$ | Low selectivity — circuit has substantial off-task effects |
| $< 1.5$ | Insufficient — effect is not task-selective |

**Selectivity-by-default failure:** On-target measurements are reported without off-target measurements, and the absence of a reported off-target effect is treated as evidence of selectivity. This is the most common external-validity failure in current MI.

## E4 — Effect Magnitude

> [Full criterion page →](/framework/criteria/external/effect-magnitude)

The absolute effect must be large enough to support the computational story being told.

**Pass condition:** Recovery fraction $R \geq 0.70$ on held-out prompts (from I2); logit difference or behavioral metric at least 1 SD above the population mean effect of random same-size ablations.

**Magnitude underspecification failure:** A statistically reliable but small effect is reported as evidence of the mechanism without the absolute recovery fraction. Statistical significance does not establish computational relevance.

## E5 — Robustness

> [Full criterion page →](/framework/criteria/external/robustness)

The result must survive paraphrase, alternative templates, and within-family scale transfer.

**Pass condition:** The effect magnitude degrades by no more than 30% across at least two prompt variants, and is identifiable (possibly at reduced magnitude) in at least one other model in the same family (e.g., GPT-2 Small → GPT-2 Medium).

**MI-specific threats:**
- *Template overfitting.* A circuit discovered on "When Mary and John went to the store, John gave a drink to ___" may not transfer to paraphrased versions. The circuit may be a template-specific shortcut rather than a general IOI mechanism.
- *Within-family overreach.* A mechanism identified in GPT-2 Small is treated as a transformer-wide property without explicit cross-family testing.

**Calibration:**

| Circuit | Template variants | Cross-scale | Assessment |
|---|---|---|---|
| IOI ([Wang et al. 2022](https://arxiv.org/abs/2211.00593)) | ABBA/BABA, name substitutions | Not tested | Partial — template robustness only |
| Induction heads ([Olsson et al. 2022](https://arxiv.org/abs/2209.11895)) | Any repeated sequence | Across GPT families, Pythia | Strong — most externally robust MI result |

## E6 — Cross-Architecture Generalization

> [Full criterion page →](/framework/criteria/external/cross-architecture)

The mechanism should be identifiable in at least one other model family, against a stated matching criterion.

**Pass condition:** A quantitative matching criterion is stated before the match is attempted (e.g., Jaccard similarity of component sets $J \geq 0.4$, or cosine similarity of weight-space signatures $\geq 0.6$). The match is reported as a score, not a binary yes/no.

**MI-specific gap:** There is no agreed criterion by which a circuit match across architectures counts. The project's recommendation is that the matching criterion be stated before the match is attempted. Cross-model matches reported without a pre-stated criterion are analogous to underpowered clinical trials reporting significance without pre-specified endpoints.

**Calibration:**

| Circuit | Matched in | Matching criterion | Assessment |
|---|---|---|---|
| Induction heads | Multiple GPT families, Pythia, LLaMA | Behavioral (prefix matching score) | Strong — criterion is pre-specified and cross-family |
| IOI | Not yet matched cross-family | None stated | No cross-architecture evidence |
| Weight-space circuit signatures | Gemma, Qwen (initial exploration) | Cosine similarity of weight directions | Preliminary — criterion partially specified |

## Partial-pass interpretation

| Evidence pattern | Criteria met | Interpretation | Recommended language |
|---|---|---|---|
| Robust, no cross-architecture test | E1–E5 | Robust within one family | "Robust in GPT-2; cross-family transfer not tested" |
| Strong graded response, low selectivity | E2, E4 | Effect is real but not task-specific | "Dose-responsive but non-selective" |
| High selectivity, single template | E3 | Template-specific selectivity | "Selective on discovery template; paraphrase robustness needed" |
| Cross-architecture match, no graded response | E6 | Cross-family presence established, mechanism unclear | "Present across architectures; dose-response not characterized" |

## Gaps in current practice

- *No standard for the dose-response curve.* The project recommends at least five intervention strengths spanning threshold and plateau, with off-task degradation reported on the same axes.
- *No standard for cross-architecture matching.* The project recommends stating the matching criterion before the match is attempted.
- *Underreporting of within-family scale transfer.* Cross-scale transfer is a relatively cheap test that is routinely omitted. The project recommends treating within-family scale transfer as the minimum external-validity test.

## Protocol

For circuit $C$ and behavior $B$:

1. **E1.** Verify that the intervention reaches the target (measure $\Delta$ activation at the target vs. adjacent components).
2. **E2.** Test at least five intervention strengths. Plot and fit a dose-response curve. Identify threshold and plateau.
3. **E3.** Compute $\text{Selectivity}(C, \lambda^*)$ against one related off-task behavior.
4. **E4.** Report the absolute recovery fraction $R$ from I2. Report the effect in SD units relative to random same-size ablations.
5. **E5.** Test at least two prompt variants. Test at least one other model in the same family.
6. **E6.** State the matching criterion in advance. Attempt the match in at least one other architecture. Report the match as a quantitative score.
