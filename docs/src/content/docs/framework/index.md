---
title: "Mechanistic Validity"
description: "A framework for evaluating whether mechanistic interpretability claims are scientifically warranted."
---

# Overview

Most circuit claims in mechanistic interpretability rest on a single type of evidence: we ablated something and behavior changed. This is a causal observation — and it is real — but it is not enough to conclude that the component *implements* the computation, that the measurement is *trustworthy*, that the finding *generalizes*, or that the explanation is stated at the *right level of abstraction*. Each of these is a distinct way a claim can fail, and each requires its own evidence.

The mechanistic validity framework makes these failure modes explicit. It provides five steps for evaluating a circuit claim, from scoping the claim through issuing a verdict, and it names the five independent dimensions along which a claim can succeed or fail.

The framework applies to claims of the form *component C implements computation T in model M*. It does not rank circuits or privilege any particular discovery method. It produces a structured verdict — a pattern of which dimensions have evidence and which do not — rather than a scalar score.

## The evaluation pipeline

<p align="center">
  <img src="/mechanistic-validity/figures/v2/pipeline-horizontal.png" alt="Mechanistic Validity Pipeline — five steps from description mode through verdict" width="800"/>
</p>

The pipeline has five steps, but they are not five equal things you do in sequence. Steps 1–2 are **scoping** — you do them once to constrain what the claim is and what evidence is relevant. Step 3 is the **work** — iteratively producing evidence. Steps 4–5 are **scoring** — deterministic given the evidence.

| Step | Name | Question |
|---|---|---|
| 1 | [Description mode](/framework/description-modes/) | At what level is the claim stated — computational, algorithmic, or implementational? |
| 2 | [Evidence families](/framework/evidence-families/) | Which types of evidence are relevant — causal, structural, representational, behavioral, information-theoretic, measurement-theoretic? |
| 3 | [Evidence](/framework/evidence/) | Run metrics, apply calibrations, optionally run protocols and synthesis protocols. |
| 4 | [Criteria](/framework/validity-types/) | Score 27 criteria against the evidence. The criteria are grouped into five validity types. |
| 5 | [Verdict](/framework/verdicts/) | Aggregate criteria scores into a tier, from *Proposed* through *Validated*. |

The real structure is a two-phase loop:

```
Scoping (once)
  1. Description Mode    ← what level is the claim at?
  2. Evidence Families   ← which metric families are relevant?

Phase 1 — Evidence (iterate)
  3. Run metrics, calibrations, protocols, synthesis protocols
     → check which criteria are weak → gather more if needed

Phase 2 — Scoring (deterministic)
  4. Score 27 criteria against the evidence
  5. Aggregate into verdict
```

You loop Phase 1 until you have enough evidence. Protocols and synthesis protocols are structured ways to do Phase 1 well — they don't come after the verdict, they feed *back into* the evidence body that criteria are scored against. Phase 2 is mechanical: the same evidence always produces the same verdict.

### Running example: activation patching on the IOI circuit

To make the pipeline concrete, consider a typical claim: "Head L9H9 implements name-moving in the IOI circuit (GPT-2 Small)." Walking through the five steps:

<p align="center">
  <img src="/mechanistic-validity/figures/v2/pipeline-vert-ioi.png" alt="Five-step pipeline applied to IOI activation patching on L9H9" width="600"/>
</p>

1. **Description mode.** The claim names a component and attributes a function ("name-moving"), so it is `[implementational–functional]`. This is stronger than just saying which heads are involved (topographic) — it commits to *what* the head does, which requires evidence beyond ablation.

2. **Evidence families.** Activation patching produces *causal* evidence. The original analysis also includes some *structural* evidence (QK/OV composition scores). No representational or information-theoretic evidence is present.

3. **Evidence.** Wang et al. (2022) run activation patching, path patching, and mean ablation. They do not run calibrations — no bootstrap stability, no random-vector baseline, no seed variance.

4. **Criteria.** Necessity (I1) passes — ablating head L9H9 degrades logit difference. Sufficiency (I2) is partially addressed via circuit isolation. Specificity (I3) is weak — does ablating the IOI circuit also degrade unrelated tasks? This was not tested. Measurement reliability (M1) is unaddressed — the result is from a single random seed. Construct falsifiability (C1) is questionable — the circuit was defined by the same metrics used to evaluate it.

5. **Verdict.** *Causally suggestive* `[implementational–functional]` — necessity is established, but sufficiency is method-conditional ([Miller et al. 2024](https://arxiv.org/abs/2407.08734) showed it drops below 0.50 under resample ablation), specificity is untested, and the construct is circular. The primary gap is specificity: does ablating the IOI circuit leave other tasks intact?

This is not a failure of the original paper — it is an honest characterization of what activation patching alone establishes. Most published circuit findings land at this tier.

## The five validity types

Validity types are not a pipeline step. They are the five groupings of the 27 criteria — five independent ways a claim can fail, each rooted in a different scientific tradition.

| Validity type | Question | Core lens |
|---|---|---|
| [Construct](/framework/validity-types/construct) | Is the claimed entity a coherent theoretical concept? | [Philosophy of Science](/framework/lenses/core/philosophy-of-science) |
| [Measurement](/framework/validity-types/measurement) | Is the metric that produced the evidence trustworthy? | [Measurement Theory](/framework/lenses/core/measurement-theory) |
| [Internal](/framework/validity-types/internal) | Does the evidence establish implementation, not just participation? | [Neuroscience](/framework/lenses/core/neuroscience) |
| [External](/framework/validity-types/external) | Does the claim generalize beyond the tested conditions? | [Pharmacology](/framework/lenses/core/pharmacology) |
| [Interpretive](/framework/validity-types/interpretive) | Is the narrative licensed by the evidence? | [Mechanistic Interpretability](/framework/lenses/core/mechanistic-interpretability) |

These types have a dependency order: construct validity is prior to all others (an ambiguous construct cannot be measured), measurement validity gates internal validity (an unreliable metric cannot support a causal inference), internal validity gates external validity (a finding not established locally cannot be said to generalize), and interpretive validity is downstream of all four (a narrative cannot be evaluated until the mechanism has been established).

The dependency order does not mean work must proceed sequentially — it means a verdict at any level should name the types at which evidence is missing.

## Analytical lenses

Each validity type draws its criteria from a specific intellectual tradition — its *core lens*. The [Philosophy of Science](/framework/lenses/core/philosophy-of-science) lens explains why falsifiability and convergent validity matter for construct validity. The [Neuroscience](/framework/lenses/core/neuroscience) lens explains why necessity and sufficiency together are weaker than Craver's mutual manipulability for internal validity. These lenses are the "why" behind the criteria.

Six [supporting lenses](/framework/lenses/) — from control theory, dynamical systems, economics, genetics, geometry, and information theory — provide additional analytical tools that strengthen specific validity types without defining new ones. The [Genetics](/framework/lenses/supporting/genetics) lens contributes knockout/rescue experimental designs that strengthen internal validity. The [Economics](/framework/lenses/supporting/economics) lens contributes arbitrage-search methodology that probes construct validity by testing whether alternative component sets can substitute for the claimed circuit.

The lenses are the intellectual foundations. The pipeline is the operational workflow. You can evaluate a claim without reading the lenses — but the lenses explain *why* the criteria are what they are.
