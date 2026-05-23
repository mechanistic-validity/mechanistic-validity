---
title: "Methods Index"
description: "Cross-reference: which metrics use which MI techniques. Not a category -- a lookup table."
---

# Methods Index

This page maps MI techniques to the metrics that use them. Methods are not an evidence category -- they are implementation techniques that produce evidence in various families (causal, structural, representational, etc.). Each metric's canonical page is in its evidence-type category; this page is a cross-reference for readers who want to find "all metrics that use ACDC" or "all metrics that use DAS."

## Circuit Discovery Methods

| Method | Metric | Evidence family | Canonical page |
|---|---|---|---|
| ACDC (Conmy et al. 2023) | C10 ACDC | Causal | [MI Causal](/framework/lenses/core/mi-causal-metrics#c10----acdc) |
| EAP (Syed et al. 2023) | C14 Position-Aware EAP | Causal | [MI Causal](/framework/lenses/core/mi-causal-metrics#c14----position-aware-eap) |
| Information Bottleneck | C13 IB Circuit Discovery | Causal | [MI Causal](/framework/lenses/core/mi-causal-metrics#c13----information-bottleneck) |
| CircuitLens | C20 CircuitLens | Causal | [MI Causal](/framework/lenses/core/mi-causal-metrics#c20----circuitlens) |

## Attribution Methods

| Method | Metric | Evidence family | Canonical page |
|---|---|---|---|
| Sparse Feature Circuits | C08 SFC | Causal | [MI Causal](/framework/lenses/core/mi-causal-metrics#c08----sparse-feature-circuits) |
| Relevance Patching / LRP | C11, C19 RelP | Causal | [MI Causal](/framework/lenses/core/mi-causal-metrics#c11----relevance-patching) |
| Contextual Decomposition | C12 CD | Causal | [MI Causal](/framework/lenses/core/mi-causal-metrics#c12----contextual-decomposition) |
| VPD | C18 VPD | Causal | [MI Causal](/framework/lenses/core/mi-causal-metrics#c18----vpd) |
| Activation Patching | C2 | Causal | [Core A01 Pearl SCM](/framework/metrics/causal/a01-scm-pearl) |
| Path Patching | C33 | Causal | [MI Causal](/framework/lenses/core/mi-causal-metrics#c33----path-patching) |

## Steering & Editing Methods

| Method | Metric | Evidence family | Canonical page |
|---|---|---|---|
| CAA (Panickssery et al. 2024) | C09 CAA | Steering | [MI Steering](/framework/lenses/core/mi-steering-metrics) |
| LEACE (Belrose et al. 2023) | C15 Concept Erasure | Steering | [MI Steering](/framework/lenses/core/mi-steering-metrics) |
| RepE (Zou et al. 2023) | C16 RepE | Steering | [MI Steering](/framework/lenses/core/mi-steering-metrics) |
| Steering-Bench | B21 | Steering | [MI Steering](/framework/lenses/core/mi-steering-metrics) |

## Interchange Intervention Methods

| Method | Metric | Evidence family | Canonical page |
|---|---|---|---|
| DAS / IIA (Geiger et al. 2024) | C1 DAS-IIA | Causal | [MI Causal](/framework/lenses/core/mi-causal-metrics#c1----das-iia) |
| Causal Scrubbing (Chan et al. 2022) | C4 | Causal | [MI Causal](/framework/lenses/core/mi-causal-metrics#c4----causal-scrubbing) |

## Causal Discovery Methods

| Method | Metric | Evidence family | Canonical page |
|---|---|---|---|
| NOTEARS (Zheng et al. 2018) | C9 | Causal | [MI Causal](/framework/lenses/core/mi-causal-metrics#c9----notears) |
| oCSE (Sun et al. 2023) | C7 | Causal | [MI Causal](/framework/lenses/core/mi-causal-metrics#c7----ocse) |
| PC Algorithm | C42 | Causal | [MI Causal](/framework/lenses/core/mi-causal-metrics#c42----pc-algorithm) |
| Granger Causality | C56 | Information-theoretic | [MI Information](/framework/lenses/core/mi-information-metrics) |

## Decomposition Methods

| Method | Metric | Evidence family | Canonical page |
|---|---|---|---|
| SAE | AQ01-AQ09 | Artifact quality | [MI Artifact Quality](/framework/lenses/core/mi-artifact-quality-metrics) |
| Transcoder | AQ10-AQ12 | Artifact quality | [MI Artifact Quality](/framework/lenses/core/mi-artifact-quality-metrics) |
| Crosscoder | AQ13-AQ15 | Artifact quality | [MI Artifact Quality](/framework/lenses/core/mi-artifact-quality-metrics) |
| CLT / Circuit Tracing | FH01-FH05 | Faithfulness | [MI Faithfulness](/framework/lenses/core/mi-faithfulness-metrics) |
