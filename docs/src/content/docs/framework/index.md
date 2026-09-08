---
title: "Mechanistic Validity"
description: "A framework for evaluating whether mechanistic interpretability claims are scientifically warranted."
---

# Overview

Most circuit claims in mechanistic interpretability rest on a single type of evidence: we ablated something and behavior changed. This is a causal observation — and it is real — but it is not enough to conclude that the component *implements* the computation, that the measurement is *trustworthy*, that the finding *generalizes*, or that the explanation is stated at the *right level of abstraction*. Each of these is a distinct way a claim can fail, and each requires its own evidence.

The mechanistic validity framework makes these failure modes explicit. It provides a six-layer pipeline for evaluating a circuit claim, from scoping the claim through issuing a verdict, and it names the five independent dimensions along which a claim can succeed or fail.

The framework applies to claims of the form *component C implements computation T in model M*. It does not rank circuits or privilege any particular discovery method. It produces a structured verdict — a pattern of which dimensions have evidence and which do not — rather than a scalar score.

## The evaluation pipeline

<p align="center">
  <img src="/mechanistic-validity/figures/v2/pipeline-horizontal.png" alt="Mechanistic Validity Pipeline — six layers from description mode through verdict" width="800"/>
</p>

The pipeline has six layers. Layers 1–2 scope the claim. Layer 3 produces evidence. Layers 4–5 score it. Layer 6 issues a verdict.

| Layer | Name | Question |
|---|---|---|
| 1 | [Description modes](/mechanistic-validity/framework/description-modes/) | At what level is the claim stated? |
| 2 | [Evidence families](/mechanistic-validity/framework/evidence-families/) | Which sources of signal support it? |
| 3 | [Metrics](/mechanistic-validity/framework/metrics/) | What was concretely measured? |
| 4 | [Criteria](/mechanistic-validity/framework/criteria/) | Does the evidence meet the stated conditions? |
| 5 | [Validity types](/mechanistic-validity/framework/validity-types/) | Which dimensions of validity does it address? |
| 6 | [Verdicts](/mechanistic-validity/framework/verdicts/) | What has the claim established? |

### Running example: activation patching on the IOI circuit

To make the pipeline concrete, consider a typical claim: "Head L9H9 implements name-moving in the IOI circuit (GPT-2 Small)." Walking through the six layers:

<p align="center">
  <img src="/mechanistic-validity/figures/v2/pipeline-vert-ioi.png" alt="Six-layer pipeline applied to IOI activation patching on L9H9" width="600"/>
</p>

1. **Description mode.** The claim names a component and attributes a function ("name-moving"), so it is `[implementational–functional]`. This is stronger than just saying which heads are involved (topographic) — it commits to *what* the head does, which requires evidence beyond ablation.

2. **Evidence families.** Activation patching produces interventional evidence on *activations*. The original analysis also includes observational evidence on *weights* (QK/OV composition scores). No training-history or behavioral-profiling evidence is present.

3. **Evidence.** Wang et al. (2022) run [activation patching](/mechanistic-validity/glossary/#activation-patching), [path patching](/mechanistic-validity/glossary/#path-patching), and mean [ablation](/mechanistic-validity/glossary/#ablation). They do not run calibrations — no bootstrap stability, no random-vector baseline, no seed variance.

4. **Criteria.** Necessity (I1) passes — ablating head L9H9 degrades logit difference. Sufficiency (I2) is partially addressed via circuit isolation. Specificity (I4) is weak — does ablating the IOI circuit also degrade unrelated tasks? This was not tested. Measurement reliability (M1) is unaddressed — the result is from a single random seed. Construct falsifiability (C1) is questionable — the circuit was defined by the same metrics used to evaluate it.

5. **Verdict.** *Causally suggestive* `[implementational–functional]` — necessity is established, but sufficiency is method-conditional ([Miller et al. 2024](https://arxiv.org/abs/2407.08734) showed it drops below 0.50 under resample ablation), specificity is untested, and the construct is circular. The primary gap is specificity: does ablating the IOI circuit leave other tasks intact?

This is not a failure of the original paper — it is an honest characterization of what activation patching alone establishes. Most published circuit findings land at this tier.

## The five validity types

Validity types are the five groupings of the 36 criteria — five independent ways a claim can fail, each rooted in a different scientific tradition. The ordering reflects logical precedence: construct validity is prior to all others, and a failure early in the chain limits what later evidence can establish.

| Validity type | Question | Criteria | Traditions |
|---|---|---|---|
| [Construct](/mechanistic-validity/framework/validity-types/construct/) | Is the target concept well-defined? | C1–C6 (6) | Philosophy of science, psychometrics |
| [Measurement](/mechanistic-validity/framework/validity-types/measurement/) | Are the instruments trustworthy? | M1–M7 (7) | Psychometrics, assay validation |
| [Internal](/mechanistic-validity/framework/validity-types/internal/) | Does the evidence support the causal claim? | I1–I12 (12) | Causal inference, neuroscience, genetics |
| [External](/mechanistic-validity/framework/validity-types/external/) | Does the mechanism generalize? | E1–E6 (6) | Pharmacology, transportability |
| [Interpretive](/mechanistic-validity/framework/validity-types/interpretive/) | Is the interpretation correct? | V1–V5 (5) | Marr's levels, unified validity |

These types have a dependency order: construct validity is prior to all others (an ambiguous construct cannot be measured), measurement validity gates internal validity (an unreliable metric cannot support a causal inference), internal validity gates external validity (a finding not established locally cannot be said to generalize), and interpretive validity is downstream of all four (a narrative cannot be evaluated until the mechanism has been established).

The dependency order does not mean work must proceed sequentially — it means a verdict at any level should name the types at which evidence is missing.

## Theoretical foundations

The framework draws its criteria from eight scientific traditions, each contributing a specific kind of reasoning to one or more validity types:

| Discipline | What it contributes | Criteria it grounds |
|---|---|---|
| [Philosophy of science](/mechanistic-validity/framework/lenses/core/philosophy-of-science/) | Falsifiability, severe testing, nomological networks, underdetermination | C1, C3, C4, C5, I5 |
| [Psychometrics](/mechanistic-validity/framework/lenses/core/measurement-theory/) | Classical test theory, generalizability theory, signal detection, MTMM matrix | M1–M7, C3, C4 |
| [Causal inference](/mechanistic-validity/framework/lenses/supporting/causal-inference/) | do-calculus, interventionism, transportability, potential outcomes | I1, I2, E4, E5 |
| [Neuroscience](/mechanistic-validity/framework/lenses/core/neuroscience/) | Lesion studies, double dissociation, mutual manipulability, multimodal parcellation | I1, I6, I12, C3 |
| [Pharmacology](/mechanistic-validity/framework/lenses/core/pharmacology/) | Dose-response, specificity, selectivity, therapeutic index | E1, E5, I4 |
| [Genetics](/mechanistic-validity/framework/lenses/supporting/genetics/) | Epistasis, rescue experiments, sensitivity analysis (E-value) | I8, I9, I10 |
| [Medical microbiology](/mechanistic-validity/framework/lenses/supporting/medical-microbiology/) | Koch's postulates and their three revisions; graded presence | I11, I12, V2 |
| [Mechanistic interpretability](/mechanistic-validity/framework/lenses/core/mechanistic-interpretability/) | The field's own metrics and evaluation practices | All criteria applied to MI evidence |

The foundation pages explain *why* each criterion is what it is. The pipeline is the operational workflow. A claim can be evaluated without reading the foundations — but they show why the criteria take the form they do.
