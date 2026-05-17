---
title: "Measurement Validity"
description: "Are the instruments that produced the evidence reliable, calibrated, and selective? — formal specification of M1–M6."
---

# Measurement Validity — Formal Specification

| | |
|---|---|
| Question | Are the instruments that produced the evidence reliable, calibrated, and selective? |
| Lens | [Measurement Theory](/framework/lenses_v6/measurement-theory) |
| Criteria | M1–M6 |
| Source theory | Classical test theory ([Lord & Novick 1968](https://psycnet.apa.org/record/1969-02031-000)); multi-trait multi-method ([Campbell & Fiske 1959](https://doi.org/10.1037/h0046016)); signal detection theory ([Green & Swets 1966](https://doi.org/10.1901/jeab.1966.9-649)) |
| Dependency | Measurement validity evaluates the *instrument*, not the *claim*; it is prior to interpreting what the instrument measures |
| Status in MI | Most easily corrected failures; baselines often omitted |

Measurement validity asks whether the instruments used to produce evidence for a circuit claim are themselves valid measurements. It is the type whose failures are most correctable — typically because the remedy is a baseline that was always feasible to compute but was not reported.

## Why measurement validity is independent of the claim's validity

The other validity types evaluate the claim being made. Measurement validity evaluates the instruments that produced the evidence. The distinction matters because failures in the two have different remedies. A failure of construct validity is remedied by clarifying the construct. A failure of internal validity is remedied by adding interventions. A failure of measurement validity is remedied by validating the instrument — typically by adding controls or baselines — without changing the claim itself.

The pharmacology analogy is the distinction between assay validation and drug efficacy. A pharmacologist validates the assay (instrument) before drawing conclusions about the drug (claim). An assay whose precision and selectivity have not been characterized cannot support a drug-efficacy conclusion regardless of how strong the measured effect appears.

## M1 — Reliability

> [Full criterion page →](/framework/criteria/measurement/reliability)

Scores from the instrument should be stable across prompt subsamples, random seeds, and checkpoints.

**Formal requirement ([Lord & Novick 1968](https://psycnet.apa.org/record/1969-02031-000)):** An observed score $X = T + E$ where $T$ is the true score and $E$ is measurement error. Reliability is:

$$\rho_{XX'} = \frac{\sigma^2_T}{\sigma^2_T + \sigma^2_E}$$

An instrument with $\rho_{XX'} < 0.7$ contributes more noise than signal; any validity claim built on it is attenuated.

**Pass condition:** Test-retest reliability $\rho_{XX'} \geq 0.7$ across two independent prompt subsamples of equal size, or bootstrap 95% CI on the principal metric spans $\leq 0.10$.

**MI-specific threat:** *Single-split reliability.* Scores reported from a single prompt split without bootstrap or test-retest evaluation hide instability. A faithfulness score of 0.87 computed on 50 prompts may have a bootstrap 95% CI of $\pm 0.15$, making it a property of the specific sample rather than the circuit.

## M2 — Invariance

> [Full criterion page →](/framework/criteria/measurement/invariance)

The instrument should produce comparable scores across model sizes within a family, or the score difference should be attributable to a measurable difference in the model rather than to a property of the instrument.

**Pass condition:** Instrument scores on matched circuits from the same model family differ by no more than the empirical standard deviation across seeds. If scores differ substantially across sizes, the instrument must be recalibrated per model size.

**MI-specific threat:** *Scale contamination.* An instrument calibrated on GPT-2 Small may systematically over- or under-estimate faithfulness in GPT-2 Medium because absolute logit differences scale with model size. Reports should always state the model on which the instrument is calibrated.

## M3 — Baseline Separation

> [Full criterion page →](/framework/criteria/measurement/baseline-separation)

The instrument's output on a real construct must be substantially above its output on a random or untrained control.

**This is the measurement criterion MI most consistently fails.** Three baselines are required:

**Random-vector baseline:** Replace the circuit's component activations with random unit vectors of the same dimensionality. The IIA or faithfulness score under this substitution is the floor.

$$\text{BaselineGap} = \text{Score}(C_\text{real}) - \text{Score}(C_\text{random})$$

**Pass condition:** $\text{BaselineGap} > 0.10$. Without this gap, the score is dominated by the alignment map's capacity rather than the circuit's structure.

**Untrained-model baseline:** Compute the same metric on an identically architected but randomly initialized (untrained) model. [Sutter et al. (2025)](https://arxiv.org/abs/2412.09659) proved that unconstrained nonlinear IIA achieves near-perfect scores on random-initialization models. Without this baseline, a high IIA score says nothing about the trained model's representations — it measures the map's degrees of freedom.

**Published reference baseline:** Report the instrument's output on published reference circuits (IOI, Greater-Than, induction heads) to establish where the claimed result sits on a common scale.

**The canonical example:**

> An IIA of 0.48 at layer 8 MLP is right in the published SAE/transcoder baseline range of 0.40–0.60 for GPT-2 Small SVA ([Lazo et al. 2025](https://arxiv.org/abs/2502.xxxxx)). Without the random-vector baseline and the published reference range, 0.48 is an uninterpretable number. With both baselines, it is a competitive and publishable result.

**Three required baselines for IIA:**

| Baseline | What it controls | Required? |
|---|---|---|
| Random-vector baseline | Alignment map capacity | Yes |
| Untrained-model baseline | Architecture-induced correlation | Yes |
| Published reference range | Calibration to literature | Strongly recommended |

## M4 — Sensitivity

> [Full criterion page →](/framework/criteria/measurement/sensitivity)

The instrument should detect real circuits at acceptable hit rates without excessive false positives.

**Pass condition:** $\text{AUROC} \geq 0.80$ and $\text{AUPRC} \geq 0.50$ when the instrument is applied to a held-out set containing known circuits and known non-circuits.

**AUROC vs AUPRC for small circuits:** For circuits with few components in models with many components, AUROC can be high while precision is low. The precision-recall distinction is rarely reported but is essential for sparse discovery tasks.

**Formal requirement using signal detection theory ([Green & Swets 1966](https://doi.org/10.1901/jeab.1966.9-649)):**

$$d' = z(\text{hit rate}) - z(\text{false alarm rate})$$

A discriminating instrument has $d' > 1.5$ (roughly equivalent to AUROC $\approx 0.85$).

## M5 — Calibration

> [Full criterion page →](/framework/criteria/measurement/calibration)

The instrument's score should be interpretable against published reference values for the same metric on matched tasks.

**Pass condition:** At least one published reference value for the same metric on a comparable task is cited, and the claimed result is positioned relative to it.

**Reference calibration table for GPT-2 Small:**

| Task | Metric | Full model | Best circuit | Source |
|---|---|---|---|
| IOI | Logit diff | 3.56 | 3.10 (87% recovery) | [Wang et al. 2022](https://arxiv.org/abs/2211.00593) |
| Greater-Than | Prob diff | 81.7% | 72.7% (89.5% recovery) | [Hanna et al. 2023](https://arxiv.org/abs/2305.00586) |
| SVA | Logit diff | 0.70 | 0.65 (93% recovery) | [Lazo et al. 2025](https://arxiv.org/abs/2502.xxxxx) |
| SVA (BLiMP behavioral) | Accuracy | 95–97% | — | [Warstadt et al. 2020](https://doi.org/10.1162/tacl_a_00321) |
| Gendered pronoun (BLiMP) | Accuracy | ~99% | $\geq$ full model | [Mathwin 2023](https://arxiv.org/abs/2308.xxxxx) |

**MIB Causal Variable IIA reference:**

| Method | IIA range | Notes |
|---|---|---|
| DAS (best) | 86–95% | Dominates on MIB |
| SAE features | < DAS | MIB's headline finding: SAE features worse than raw neurons |
| Random baseline | ~10–15% | Instrument floor |

## M6 — Construct Coverage

> [Full criterion page →](/framework/criteria/measurement/construct-coverage)

The instrument should measure what it claims to measure rather than a property of the instrument's own degrees of freedom.

**Pass condition:** The instrument produces different scores on constructs that are theoretically distinguishable. If two circuits that should be distinct produce indistinguishable instrument scores, the instrument lacks construct coverage.

**Discovery-evaluation overlap failure:** Using the same prompts for circuit discovery and evaluation inflates apparent reliability because the circuit was optimized on those prompts. Discovery and evaluation sets must be disjoint.

**Alignment map capacity failure:** If IIA is high only with unconstrained nonlinear maps (MLP, large linear subspace), the finding is about map flexibility, not the circuit's geometry. The alignment architecture must be specified and the score must be reported as a function of alignment capacity — showing how IIA degrades as alignment is constrained toward a single linear direction.

## Partial-pass interpretation

| Pattern | Criteria met | Interpretation | Recommended language |
|---|---|---|---|
| Reliable but no baseline | M1 | Score is stable but uninterpretable | "Reliable; baseline gap not established" |
| Baseline gap established, not calibrated | M1, M3 | Score interpretable as above floor; not positioned in literature | "Above random; reference calibration needed" |
| Calibrated, no sensitivity test | M1, M3, M5 | Positioned in literature; false-positive rate unknown | "Calibrated; sensitivity not characterized" |
| All six met | M1–M6 | Full measurement validity | Proceed to interpret internal validity results |

## Protocol

For instrument $I$ and circuit $C$:

1. **M1.** Bootstrap the principal metric over at least 5 resampled prompt splits. Report 95% CI.
2. **M2.** If comparing across model sizes, verify scale comparability or recalibrate.
3. **M3.** Compute random-vector baseline and untrained-model baseline. Report $\text{BaselineGap}$.
4. **M4.** If using the instrument for discovery, compute AUROC and AUPRC on held-out known/unknown circuit sets.
5. **M5.** Cite at least one published reference value. Position the result on the calibration scale.
6. **M6.** Ensure discovery and evaluation sets are disjoint. Report IIA as a function of alignment capacity.
