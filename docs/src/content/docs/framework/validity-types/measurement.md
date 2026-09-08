---
title: "Measurement Validity"
---

# Measurement Validity

Are the instruments trustworthy? Once the construct is defined, the measurement tools used to detect it must be reliable (stable across repetitions), separable from random and untrained baselines, calibrated (the numbers map onto a known scale), and invariant (consistent across conditions). A metric that gives different answers when run with different random seeds is not evidence. Measurement validity evaluates the metric, not the claim — a distinction that matters because the two have different remedies. A construct-validity failure calls for a clearer construct; a measurement-validity failure calls for a better-characterized instrument.

## Position in the Dependency Chain

```
Construct → **Measurement** → Internal → External → Interpretive
```

Measurement validity occupies the second position. It depends on construct validity because a metric can only be assessed against a well-defined target — we cannot ask whether IIA reliably measures a construct that has not been specified. Every downstream validity type depends on measurement validity in turn, though the dependence takes two forms.

The strict form operates through Spearman's (1904) attenuation formula: reliability caps the correlation between any two measures at the geometric mean of their reliabilities. An unreliable metric cannot support a causal claim, regardless of how well the internal-validity design is executed. A faithfulness score with test-retest reliability of 0.5 attenuates any correlation it enters by a factor of √0.5 ≈ 0.71, placing a hard ceiling on what internal-validity evidence built from that score can establish.

The weak form operates through stability (M3) and invariance (M6). These criteria do not impose a mathematical ceiling on effect size estimates. They record that a reported quantity moves with analysis choices — seed, prompt sample, ablation method — which impugns the number that was published without bounding the effect it estimates.

## Theoretical Lineage

The measurement tradition in psychometrics predates construct validity by half a century. Spearman (1904) introduced the attenuation formula while developing the theory of test reliability, establishing that the observed correlation between two measures is the product of their true correlation and the geometric mean of their reliabilities. The formula provides the mathematical foundation for M1: a metric whose reliability has not been characterized cannot be interpreted, because its observed values confound signal with measurement error.

Borsboom, Mellenbergh, and van Heerden (2004) proposed that "a measure is valid if and only if the attribute exists and variation in it causally produces variation in the measurement outcome." This causal account of measurement validity reframes the question: we do not ask whether IIA "correlates with" circuit quality, but whether variation in circuit quality causally produces variation in IIA. The distinction matters because a metric can correlate with circuit quality for reasons that have nothing to do with the circuit — alignment map capacity, prompt distribution, or model scale.

Classical test theory (Lord & Novick, 1968) provides the formal framework: an observed score X = T + E, where T is the true score and E is measurement error. Reliability is the ratio of true-score variance to total variance. This decomposition applies directly to MI metrics. A faithfulness score computed on a single prompt split confounds the circuit's true faithfulness with the specific prompt sample. Bootstrap resampling or split-half reliability estimation separates the two.

The pharmacological analogy is assay validation. Before drawing conclusions about a drug's efficacy, pharmacologists validate the assay — characterizing its precision, selectivity, and dynamic range. An assay whose properties have not been established cannot support an efficacy conclusion regardless of how large the measured effect appears. MI metrics occupy the same position: they are assays whose validation is typically omitted.

## Criteria

| ID | Name | Question |
|---|---|---|
| M1 | Reliability | Do repeated measurements give the same answer? |
| M2 | Baseline separation | Is the score distinguishable from random or untrained baselines? |
| M3 | Stability | Is the classification robust to perturbation of analysis choices? |
| M4 | Calibration | Are the numbers meaningful — do they map onto a known scale? |
| M5 | Sensitivity | Can the instrument detect known-true effects? |
| M6 | Invariance | Does the metric behave consistently across conditions (model size, prompt distribution, ablation method)? |
| M7 | Selection correction | When k findings are selected from N candidates, is N reported and multiplicity controlled? |

M1–M3 address whether the metric produces stable outputs. M4–M5 address whether those outputs are interpretable. M6 addresses whether they generalize across conditions. M7 addresses whether they survive correction for the search that produced them.

The framework also defines 15 calibration meta-metrics (F01–F15) that map onto M1–M7. These are metrics of metrics: they take another metric's output as input and assess whether it is stable, reproducible, or distinguishable from baselines.

## Failure Examples

**Dead salmon fMRI (M2).** Bennett et al. scanned a dead Atlantic salmon with fMRI and found 16 voxels showing statistically significant activation. The study was a demonstration of false-positive rates in neuroimaging — up to 70% in some analysis pipelines — when baseline separation and multiple-comparison correction are omitted. The salmon had no neural activity to detect. The "significant" voxels were measurement artifacts indistinguishable from the baseline because no baseline separation criterion was applied.

**Vacuous nonlinear IIA (M2).** Sutter et al. demonstrated that unconstrained nonlinear interchange intervention accuracy (IIA) achieves near-perfect scores — approaching 100% — on randomly initialized, untrained models. The high IIA reflects the alignment map's degrees of freedom, not the model's learned representations. Without an untrained-model baseline, a high IIA score is uninterpretable: it may measure map flexibility rather than representational structure.

**SAEBench metrics (M2).** In the SAEBench evaluation suite, one metric scored higher on a randomly initialized model than on the trained model. A metric that assigns better scores to random weights than to learned weights fails baseline separation — it is measuring something other than what it claims.

**Emergent abilities (M4).** Schaeffer et al. showed that the apparent phase transitions in large language model "emergent abilities" were artifacts of the metric. When accuracy (a discontinuous metric) was replaced with a continuous metric measuring the same underlying quantity, the sharp transitions disappeared. The phase transition was a property of the measurement instrument, not the model. This is a calibration failure: the metric created the phenomenon it appeared to detect.

## Cross-Disciplinary Foundations

| Discipline | Transfer to measurement validity |
|---|---|
| Psychometrics | Test-retest reliability, split-half reliability, measurement invariance, the attenuation formula. Provides the formal apparatus for M1, M2, and M6. |
| Pharmacology | Assay validation protocols — characterize precision, selectivity, and dynamic range before interpreting results. The analogy to M1–M5 is direct: an unvalidated assay cannot support an efficacy claim. |
| Signal detection theory | Sensitivity (d′), receiver operating characteristic curves, the distinction between hit rate and false-alarm rate. Provides the formal apparatus for M5. |
| Philosophy of science | Severe testing (Mayo, 1996) — a test that a hypothesis was built to pass is not evidence for it. Applies to M2: a metric evaluated on its own discovery set has not been severely tested. |
