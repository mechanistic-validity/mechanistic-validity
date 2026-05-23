---
title: "Extended MI Metrics Overview"
description: "Index of all extended mechanistic interpretability metrics across 10 evidence-type categories."
---

# Extended MI Metrics Overview

These are MI-specific metrics beyond the Core Metrics (the A-F evidence-family frameworks). They measure specific properties of circuits, features, and decompositions using tools from mechanistic interpretability. The metrics are organized by the type of evidence they produce, not by the tool or technique used -- a single technique like activation patching appears in the causal category regardless of which framework invokes it.

## Summary table

| Category | Count | What it covers | Page |
|---|---|---|---|
| Causal (incl. discovery) | 30 | Ablation, patching, scrubbing, causal discovery, ACDC, EAP | [MI Causal](/framework/lenses/core/mi-causal-metrics) |
| Structural | 22 | Weight decomposition, graph analysis, motifs, composition | [MI Structural](/framework/lenses/core/mi-structural-metrics) |
| Behavioral | 15 | Faithfulness variants, generalization, calibration | [MI Behavioral](/framework/lenses/core/mi-behavioral-metrics) |
| Information-Theoretic | 11 | MI, PID, transfer entropy, Granger, NOTEARS | [MI Information](/framework/lenses/core/mi-information-metrics) |
| Representational | 9 | Probes, RSA, CKA, attention entropy | [MI Representational](/framework/lenses/core/mi-representational-metrics) |
| Artifact Quality | 15 | SAE eval, transcoder, crosscoder validation | [MI Artifact Quality](/framework/lenses/core/mi-artifact-quality-metrics) |
| Faithfulness | 10 | CLT graph fidelity, circuit faithfulness | [MI Faithfulness](/framework/lenses/core/mi-faithfulness-metrics) |
| Steering | 5 | CAA, LEACE, RepE, cross-model transfer | [MI Steering](/framework/lenses/core/mi-steering-metrics) |
| Safety | 7 | Safety subspaces, adversarial ablation, claim reliability | [MI Safety](/framework/lenses/core/mi-safety-metrics) |
| Benchmarks | 6 | AxBench, SAEBench, CE-Bench, MIB | [MI Benchmarks](/framework/lenses/core/mi-benchmarks-metrics) |

In addition, the [MI Evaluation Metrics](/framework/lenses/core/mi-evaluation-metrics) page documents 43 evaluation metrics that cut across these categories, testing circuit faithfulness, feature quality, safety constructs, and decomposition completeness. The [MI Methods Index](/framework/evidence/methods-index) provides technique-based lookup into the same metrics.

## How these relate to Core Metrics

The Core Metrics (A01 through F08) document *frameworks* -- bundled metrics plus calibrations plus theoretical interpretation. Each framework draws on a scientific tradition (Pearl's SCM, Rubin's potential outcomes, Granger causality, etc.) and packages a curated set of metrics into a protocol with domain-specific pass/fail criteria.

The extended MI metrics listed here are individual metric implementations. Many of them are used *within* those frameworks. For example:

- **MET-activation-patching** is both a standalone extended metric (C2 on the [MI Causal](/framework/lenses/core/mi-causal-metrics) page) and a component of protocol A01 (Pearl SCM), A03 (Rubin CATE), A04 (Woodward), A10 (Regularity/INUS), and A11 (Actual Cause).
- **MET-das-iia** is both a standalone metric (C1 on [MI Causal](/framework/lenses/core/mi-causal-metrics)) and the central instrument of protocol A02 (Counterfactual DAS/IIA).
- **MET-mutual-information** appears on [MI Information](/framework/lenses/core/mi-information-metrics) as C01 and feeds into protocol A08 (PID).

The relationship is one-to-many: a single extended metric can appear in multiple protocols. The protocols add curation (which metrics to run together) and interpretation (what the pattern of results means through a specific theoretical lens). Running the extended metrics individually produces the same raw numbers; the protocols add structure and context.

## Cross-references

- **[Methods Index](/framework/evidence/methods-index)** -- technique-based lookup. If you know the method (ACDC, EAP, DAS, CAA) and want to find which metrics use it, start there.
- **[Calibrations](/framework/evidence/calibrations)** -- quality gates. Before trusting any extended metric's output, check which calibrations apply (bootstrap stability, convergent validity, measurement invariance, etc.).
- **[Protocols](/framework/evidence/protocols)** -- curated bundles. If you want structured depth on a specific validity question rather than individual metric scores, find the relevant protocol.
- **[Naming Convention](/framework/evidence/naming-convention)** -- how entity IDs (CRIT, MET, CAL, PROT, SYN) prevent namespace collisions across the framework.
