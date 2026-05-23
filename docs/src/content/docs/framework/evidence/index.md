---
title: "Evidence"
description: "Step 3: producing calibrated measurements through metrics, calibrations, protocols, and synthesis protocols."
---

# Evidence

This is where measurement happens. Steps 1 and 2 scoped the claim and identified which evidence families are relevant. Step 3 produces the actual measurements that criteria will be scored against.

There are four tools for generating evidence, ranging from atomic measurements to structured multi-protocol analyses. All four produce the same kind of output — scored measurements with metadata — and all feed into the same criteria-scoring step. The difference is scope and depth.

## Metrics

A metric is an atomic measurement. It takes a model, a circuit, and a dataset, and returns a number with metadata. Some metrics are simple — logit difference recovery is a single scalar. Others wrap complex procedures — ACDC discovers a circuit via iterative pruning, EAP computes edge attributions via gradient approximation — but they are still registered and scored the same way. The framework currently defines metrics across [six evidence families](/framework/evidence-families/):

| Family | What it measures | Example metrics |
|---|---|---|
| [A. Causal](/framework/metrics/causal/a01-scm-pearl) | Effects of interventions on behavior | Activation patching, path patching, causal scrubbing, DAS-IIA |
| [B. Structural](/framework/metrics/structural/b01-svd-spectral) | Weight-space properties without forward passes | SVD spectra, effective rank, OV/QK composition, weight alignment |
| [C. Information-theoretic](/framework/metrics/information/c01-mutual-information) | Information flow in bits | Mutual information, transfer entropy, PID, Granger causality |
| [D. Behavioral](/framework/metrics/behavioral/d01-faithfulness) | Input-output correspondence under controlled conditions | Faithfulness, logit diff recovery, KL divergence, cross-task transfer |
| [E. Representational](/framework/metrics/representational/e01-das-iia) | Information encoded in activations | DAS-IIA, linear probes, CKA, subspace alignment, intrinsic dimension |
| [F. Measurement-theoretic](/framework/metrics/measurement/f01-bootstrap-stability) | Meta-evidence about other metrics' reliability | Bootstrap stability, seed variance, convergent/discriminant validity |

Some metrics wrap well-known methods from the field — ACDC, EAP, CircuitLens, contrastive activation addition. In the codebase, these live under `metrics/mechanistic_interpretability/methods/` but are registered and scored identically to any other metric. The "methods" directory is an organizational convenience, not a separate concept.

## Calibrations

A calibration is a quality gate on a metric's output. It answers: *can we trust this measurement?*

Bootstrap stability checks whether the metric gives consistent results across prompt resamples. Seed variance checks reproducibility across random seeds. Convergent validity checks whether metrics from different evidence families agree. Discriminant validity checks whether the metric distinguishes the target circuit from controls. Measurement invariance checks whether results hold across model sizes.

Calibrations are themselves metrics — they return scored measurements and are registered in the same metric registry. The distinction is functional: calibrations evaluate the trustworthiness of other measurements rather than measuring the model directly. A metric result without calibration is a measurement without error bars.

## Protocols

A protocol is a curated bundle of metrics and calibrations, organized around a specific validity question, with domain-specific interpretation layered on top.

Protocol A01 (Pearl SCM), for example, runs four metrics — logit_diff, role_ablation, activation_patching, causal_scrubbing — plus a set of causal calibrations, and then interprets the results through Pearl's three-rung causal hierarchy. It answers the question: *is this circuit a valid structural causal model?* Running those four metrics and calibrations individually would produce the same raw numbers. The protocol adds two things: curation (these are the right metrics for this question) and interpretation (here is what the pattern of results means through this theoretical lens).

Protocols produce *new evidence* — they run metrics that may not have been run otherwise. They are not required for evaluating a claim (metrics and calibrations alone suffice), but they provide structured depth on specific validity dimensions. If criteria scoring reveals that internal validity criterion I3 (specificity) is weak, running protocol A01 targets exactly that gap.

There are currently 81 protocols across 10 families, covering all registered metrics. Each protocol is tagged with which validity criteria it strengthens, so a user can match weak criteria to targeted protocols. See the [protocols page](/framework/evidence/protocols) for the full inventory.

## Synthesis protocols

A synthesis protocol aggregates the outputs of multiple protocols into higher-order analyses. Where a protocol bundles metrics, a synthesis protocol bundles protocol results.

Functional parcellation (S01) takes results from 4+ protocols that each rank circuit components, then clusters components by convergent multi-signal agreement — the neural circuit analog of Glasser et al.'s multimodal brain parcellation. Dawid-Skene consensus (S02) jointly estimates the true circuit membership and each protocol's reliability using an EM algorithm. The meta-learner (S07) trains a logistic regression on labeled circuits to predict circuit membership from protocol features.

Synthesis protocols do not create a new type of evidence. They produce additional scored measurements that feed back into the same criteria-scoring pipeline. A Dawid-Skene consensus score might strengthen criterion C5 (convergent validity). A parcellation result might strengthen I3 (specificity) by revealing which components cluster together. The evidence is richer, but it enters the same scoring pathway.

There are currently 9 synthesis protocols. They are optional — useful when multiple protocols have been run and the analyst wants to extract structure from the combined results. See the [synthesis protocols page](/framework/evidence/synthesis-protocols) for details.
