---
title: "Measurement-Theoretic Evidence"
description: "Meta-evidence evaluating whether metrics from other families produce trustworthy, stable, and meaningful measurements"
---

# Measurement-Theoretic Evidence

Measurement-theoretic evidence captures whether the measurements produced by other evidence families are trustworthy — evaluating metric reliability, stability, and validity rather than model properties directly.

## What this family measures

Measurement-theoretic evidence is meta-evidential: it evaluates the quality of evidence from all other families rather than producing first-order claims about circuits. These metrics ask: "If I run this analysis again with different random seeds, do I get the same answer?" "If I use two different methods that should measure the same thing, do they agree?" "Can this metric distinguish a real circuit from a baseline?"

This family draws on measurement theory — the science of whether measurements are reliable (consistent), valid (measuring what they claim to measure), and sensitive (able to detect real effects). In mechanistic interpretability, where ground truth is rarely available, these meta-evidential checks are critical for knowing how much confidence to place in any particular finding.

The metrics in this family do not tell you whether a circuit claim is correct. They tell you whether the methods used to evaluate that claim are functioning properly — whether the signal-to-noise ratio is sufficient, whether apparent findings replicate, and whether the metrics discriminate between genuinely different phenomena.

## Metrics

- **F01 Bootstrap Stability** — Resampling analysis testing whether findings are stable across data subsets
- **F02 Seed Variance** — Variation in results across random seeds in stochastic methods
- **F03 Convergent Validity** — Agreement between different metrics that should measure the same construct
- **F04 Discriminant Validity** — Disagreement between metrics that should measure different constructs
- **F05 Internal Consistency** — Whether sub-components of a composite measure agree with each other

## Characteristic strength

Measurement-theoretic evidence is the only family that evaluates the trustworthiness of all other evidence. Without it, researchers cannot distinguish between a strong finding measured reliably and a noise artifact measured once. This family provides the error bars, the replication checks, and the validity assessments that determine how much weight any other piece of evidence should receive.

This meta-evidential role is particularly important in mechanistic interpretability, where many methods have unknown reliability profiles. If an ablation study finds that head 9.1 is necessary for IOI, but bootstrap analysis shows the necessity score varies from 0.3 to 0.9 across data subsets, the original finding is substantially weakened. Without measurement-theoretic evidence, this instability would be invisible.

## Characteristic blind spot

Measurement-theoretic evidence cannot establish that a circuit claim is true. It can only establish that the metrics used to evaluate the claim are (or are not) reliable. A perfectly reliable metric can still be measuring the wrong thing — high test-retest reliability on a flawed metric is not progress.

This family also cannot create evidence where none exists. If no causal, structural, or representational evidence has been gathered, measurement-theoretic analysis has nothing to evaluate. It is inherently parasitic on first-order evidence from other families.

## Criteria served

- **M1 Reliability** — Bootstrap stability and seed variance directly assess measurement consistency
- **M2 Invariance** — Whether measurements are stable across irrelevant variations (prompt format, token order, dataset split)
- **M3 Baseline separation** — Whether metrics produce different scores for real circuits vs. random baselines
- **M4 Sensitivity** — Whether metrics can detect known differences (e.g., between a correct circuit and a slightly perturbed one)
- **M5 Calibration** — Whether confidence levels from metrics correspond to actual accuracy
- **M6 Construct coverage** — Whether the set of metrics used covers all relevant aspects of the claim being evaluated

## Convergent validity role

Measurement-theoretic evidence plays a unique role in convergent validity: it determines how much weight each other piece of evidence should receive in the overall assessment. Rather than combining additively with other families, it acts as a *multiplier* — high measurement quality amplifies confidence in other evidence, while low measurement quality attenuates it.

Convergent validity (F03) is itself a metric in this family, creating a recursive structure: the framework uses convergent validity between families as a criterion, and measurement-theoretic evidence formally assesses whether that convergence is real or artifactual. When causal and structural evidence appear to agree, measurement-theoretic analysis asks whether the agreement is robust (survives bootstrap, replicates across seeds) or fragile (depends on a specific data split or hyperparameter choice).

This family does not compete with others for evidential weight — it calibrates the weight of everything else.
