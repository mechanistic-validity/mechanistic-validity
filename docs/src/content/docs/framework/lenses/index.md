---
title: "Analytical Lenses"
description: "The intellectual traditions that ground the framework's criteria — five core lenses and six supporting lenses."
---

# Analytical Lenses

The criteria in the evaluation pipeline do not come from nowhere. Each one is grounded in a specific intellectual tradition that developed it for a reason — usually because a field got burned by ignoring it. Falsifiability is a criterion because philosophy of science spent a century learning what happens when claims cannot be disconfirmed. Necessity-plus-sufficiency is a criterion because neuroscience spent decades discovering that necessity alone does not establish implementation. Dose-response is a criterion because pharmacology learned that single-dose measurements hide compensatory dynamics.

We call these traditions *lenses*. A lens is not a method or a metric — it is an analytical vocabulary, a way of thinking about evidence that shapes which questions you ask and how you interpret the answers. The framework draws on eleven lenses from five fields.

## Core lenses

Five lenses map one-to-one to the five validity types. Each core lens provides the conceptual foundation for its validity type's criteria — the "why" behind the formal specifications on the [validity type pages](/framework/validity-types/).

| Core lens | Validity type | What it contributes |
|---|---|---|
| [Philosophy of Science](/framework/lenses/core/philosophy-of-science) | Construct | Falsifiability, convergent validity, the distinction between observables and theoretical entities |
| [Measurement Theory](/framework/lenses/core/measurement-theory) | Measurement | Reliability, invariance, the MTMM matrix, the distinction between the construct and the instrument |
| [Neuroscience](/framework/lenses/core/neuroscience) | Internal | Single vs double dissociation, lesion vs stimulation, constitutive relevance, multimodal parcellation |
| [Pharmacology](/framework/lenses/core/pharmacology) | External | Dose-response curves, functional selectivity, receptor reserve, the distinction between affinity and efficacy |
| [Mechanistic Interpretability](/framework/lenses/core/mechanistic-interpretability) | Interpretive | Description vs explanation, faithfulness vs understanding, Marr's levels, the description-mode hierarchy |

You do not need to read the lens pages to use the framework. The criteria stand on their own with formal pass conditions and thresholds. But when a criterion seems arbitrary — why require a double dissociation and not just a single one? why demand cross-family convergence and not just within-family agreement? — the lens page explains the intellectual history that motivates it.

## Supporting lenses

Six additional lenses provide analytical tools that strengthen specific validity types without defining new ones. They are not mapped to validity types one-to-one; instead, each contributes methods or concepts that apply across multiple types.

| Supporting lens | What it contributes | Validity types it supports |
|---|---|---|
| [Control Theory](/framework/lenses/supporting/control-theory) | Stability margins, settling depth, observability conditions, PID-inspired steering | Internal, External |
| [Dynamical Systems](/framework/lenses/supporting/dynamical-systems) | Renormalization group flow, Koopman/DMD analysis, topological data analysis, critical phenomena | Construct, Measurement |
| [Economics](/framework/lenses/supporting/economics) | Arbitrage search (can an alternative circuit substitute?), game-theoretic interaction models, price-impact linearity | Construct, Internal |
| [Genetics](/framework/lenses/supporting/genetics) | Knockout hierarchies, rescue experiments, Mendelian randomization, dose-response from molecular biology | Internal, External |
| [Geometry](/framework/lenses/supporting/geometry) | Fisher-Rao distances, angular steering, sheaf consistency, symmetry and equivariance | Measurement, Construct |
| [Information Theory](/framework/lenses/supporting/information-theory) | Mutual information, transfer entropy, PID, information bottleneck, Granger causality | Internal, Construct |

Supporting lenses are the source of many of the framework's protocols. The genetics lens contributes 16 molecular-biology protocols (knockout hierarchies, rescue experiments, Mendelian randomization, and others adapted from experimental biology). The economics lens contributes arbitrage search — a protocol that tests construct validity by asking whether an alternative set of components can substitute for the claimed circuit. If yes, the circuit is not uniquely necessary; the construct may not carve computation at its joints.

## Complete lens-to-criteria map

The table below shows every lens, the validity type it grounds (for core lenses) or supports (for supporting lenses), and the specific criteria it motivates. This is the full picture — every criterion in the framework traces back to at least one lens.

| Lens | Type | Validity type | Criteria | Key question |
|---|---|---|---|---|
| [Philosophy of Science](/framework/lenses/core/philosophy-of-science) | Core | [Construct](/framework/validity-types/construct) | C1 Falsifiability, C2 Structural plausibility, C3 Task specificity, C4 Minimality, C5 Convergent validity | Is the entity you named a real thing? |
| [Measurement Theory](/framework/lenses/core/measurement-theory) | Core | [Measurement](/framework/validity-types/measurement) | M1 Reliability, M2 Invariance, M3 Baseline separation, M4 Sensitivity, M5 Calibration, M6 Construct coverage | Can you trust the number? |
| [Neuroscience](/framework/lenses/core/neuroscience) | Core | [Internal](/framework/validity-types/internal) | I1 Necessity, I2 Sufficiency, I3 Specificity, I4 Consistency, I5 Confound control | Does the component implement the computation? |
| [Pharmacology](/framework/lenses/core/pharmacology) | Core | [External](/framework/validity-types/external) | E1 Intervention reach, E2 Graded response, E3 Selectivity, E4 Effect magnitude, E5 Robustness, E6 Cross-architecture | Does it generalize? |
| [Mechanistic Interpretability](/framework/lenses/core/mechanistic-interpretability) | Core | [Interpretive](/framework/validity-types/interpretive) | V1 Level declaration, V2 Level-evidence match, V3 Narrative coherence, V4 Alternative exclusion, V5 Scope honesty | Is the story right? |
| [Control Theory](/framework/lenses/supporting/control-theory) | Supporting | Internal, External | Settling depth, stability margins, observability | Is the circuit steerable and stable? |
| [Dynamical Systems](/framework/lenses/supporting/dynamical-systems) | Supporting | Construct, Measurement | Koopman/DMD modes, renormalization flow, TDA | Does the structure persist across scales? |
| [Economics](/framework/lenses/supporting/economics) | Supporting | Construct, Internal | Arbitrage search, game-theoretic interactions | Can something else substitute for it? |
| [Genetics](/framework/lenses/supporting/genetics) | Supporting | Internal, External | Knockout hierarchies, rescue, Mendelian randomization | Does it behave like a genetic pathway? |
| [Geometry](/framework/lenses/supporting/geometry) | Supporting | Measurement, Construct | Fisher-Rao distances, sheaf consistency, equivariance | Does the geometry match the computation? |
| [Information Theory](/framework/lenses/supporting/information-theory) | Supporting | Internal, Construct | MI, transfer entropy, PID, Granger causality | Does information flow through it? |

## How lenses relate to the pipeline

Lenses sit outside the five-step evaluation pipeline. They are reference material — the intellectual context that explains why the pipeline's criteria exist. A researcher who wants to evaluate a claim follows the pipeline (scope → evidence → criteria → verdict). A researcher who wants to understand *why* a criterion matters, or who wants to design a new protocol, reads the relevant lens.

The relationship is: **the pipeline is how you evaluate; lenses are how you think.**
