---
title: "Naming Convention"
description: "How entity IDs work in the framework: typed prefixes prevent collisions between criteria, metrics, calibrations, and protocols."
---

# Naming Convention

## The problem

The framework has five kinds of named entities: criteria, metrics, calibrations, protocols, and synthesis protocols. Legacy IDs use short codes -- C5, A01, M4 -- that are ambiguous across entity types. "C5" could refer to criterion C5 (convergent validity), calibration C5 (mediation analysis), or a metric with the C5 prefix. "A01" could be protocol A01 (Pearl SCM) or a metric in the causal family. When cross-referencing between pages or writing code that registers entities, these collisions produce confusion.

## The convention

Each entity type has a typed prefix that makes the ID unambiguous:

### CRIT -- Criteria

`CRIT-C1` through `CRIT-V5` (27 total). The sub-ID preserves the validity-type letter:

- **C** = Construct validity (C1 Falsifiability, C2 Structural plausibility, C3 Task specificity, C4 Minimality, C5 Convergent validity)
- **M** = Measurement validity (M1 Reliability, M2 Invariance, M3 Baseline separation, M4 Sensitivity, M5 Calibration, M6 Construct coverage)
- **I** = Internal validity (I1 Necessity, I2 Sufficiency, I3 Specificity, I4 Consistency, I5 Confound control)
- **E** = External validity (E1 Intervention reach, E2 Graded response, E3 Selectivity, E4 Effect magnitude, E5 Robustness, E6 Cross-architecture)
- **V** = Interpretive validity (V1 Level declaration, V2 Level-evidence match, V3 Narrative coherence, V4 Alternative exclusion, V5 Scope honesty)

### MET -- Metrics

`MET-activation-patching`, `MET-das-iia`, `MET-mutual-information`, etc. Full kebab-case slug, never abbreviated. The slug is the canonical name used in code and frontmatter. Examples:

| Typed ID | What it refers to |
|---|---|
| `MET-activation-patching` | Activation patching metric (C2 on the MI Causal page) |
| `MET-das-iia` | DAS interchange intervention accuracy (C1 on MI Causal) |
| `MET-rsa` | Representational similarity analysis (E03 on MI Representational) |
| `MET-caa-steering` | Contrastive activation addition (C09 on MI Steering) |
| `MET-axbench` | AxBench concept detection and steering (B20 on MI Benchmarks) |

### CAL -- Calibrations

`CAL-01` through `CAL-16` (16 total). Sequential numbers corresponding to the calibration checklist. Examples:

| Typed ID | Name | Legacy ID |
|---|---|---|
| `CAL-01` | Bootstrap Stability | C11 |
| `CAL-02` | Convergent Validity | C12 |
| `CAL-03` | Measurement Invariance | C13 |
| `CAL-04` | Derived Metrics (Sensitivity) | C14 |
| `CAL-05` | Reliability Suite | C16 |

### PROT -- Protocols

`PROT-{letter}{number}` format, where the letter indicates the evidence family. Examples:

| Typed ID | Name | Legacy ID |
|---|---|---|
| `PROT-A01` | Pearl SCM | A01 |
| `PROT-A02` | Counterfactual DAS/IIA | A02 |
| `PROT-B01` | Spectral/SVD | B01 |
| `PROT-D01` | Faithfulness | D01 |

### SYN -- Synthesis protocols

`SYN-01` through `SYN-09` (9 total). Sequential numbers. Examples:

| Typed ID | Name |
|---|---|
| `SYN-01` | Functional Parcellation |
| `SYN-02` | Dawid-Skene Consensus |
| `SYN-03` | Robust Rank Aggregation |

## In prose

Informal references are fine in running text. You can write "I3 Specificity" or "criterion C5" or "protocol A01" without the typed prefix -- context makes the type clear. The typed prefix is the canonical ID used in:

- Code (metric registry keys, config files, result schemas)
- Frontmatter (`id:` fields in page metadata)
- Formal cross-references (links between pages, summary tables)
- Machine-readable outputs (JSON results, scoring pipelines)

When writing prose, use whichever form reads most naturally. When writing code or structured data, use the typed prefix.

## Legacy IDs

Existing pages use untyped IDs (C1, M4, A01, EX12). These will be migrated to typed IDs over time. When encountering an untyped ID, context determines the type:

- If it appears on a **validity-type page** (construct, measurement, internal, external, interpretive), it is a **criterion**.
- If it appears on a **metric page** (mi-causal-metrics, mi-structural-metrics, etc.), it is a **metric**.
- If it appears on the **calibrations page**, it is a **calibration**.
- If it appears on the **protocols page**, it is a **protocol**.
- IDs starting with **EX** are evaluation metrics (a subset of MET).
- IDs starting with **S** followed by a number on the synthesis protocols page are **synthesis protocols**.

During the migration period, both forms are valid. The typed form is preferred for new content.

## Quick reference

| Prefix | Entity type | ID format | Count | Example |
|---|---|---|---|---|
| `CRIT` | Criterion | `CRIT-{letter}{number}` | 27 | `CRIT-I3` (Specificity) |
| `MET` | Metric | `MET-{kebab-slug}` | 130+ | `MET-activation-patching` |
| `CAL` | Calibration | `CAL-{number}` | 16 | `CAL-01` (Bootstrap Stability) |
| `PROT` | Protocol | `PROT-{letter}{number}` | 81 | `PROT-A01` (Pearl SCM) |
| `SYN` | Synthesis protocol | `SYN-{number}` | 9 | `SYN-02` (Dawid-Skene) |
