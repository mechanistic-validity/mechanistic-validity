---
title: "Overview"
description: "Mechanistic Validity: a framework for evaluating circuit claims in mechanistic interpretability."
---

# Mechanistic Validity

Mechanistic Validity is a framework for evaluating claims about circuits in neural networks. It provides a seven-layer evaluation pipeline, five validity types (each rooted in a distinct scientific tradition), 36 criteria, and a structured verdict system. Its purpose is to make explicit which part of a mechanistic claim a given measurement supports and which parts remain unaddressed.

The framework does not introduce new measurement methods. It organizes existing methods — ablation, activation patching, DAS-IIA, causal scrubbing, weight analysis, baseline calibration — under a common evaluative vocabulary drawn from the standard typology of validity in philosophy of science and adapted to the conditions of mechanistic interpretability.

## What the framework is for

The framework applies to claims of the form *component C implements computation T in model M*. Such claims are made routinely in circuit-discovery papers and are typically supported by a small number of metrics — most often activation patching plus one form of ablation. The framework specifies what additional evidence is required for the claim to be considered validated under each of five named dimensions, and what verdict is licensed when only a subset of that evidence is present.

The framework's central commitment is that a single high score does not validate a circuit claim. Validation is a pattern of evidence across multiple dimensions, and a claim is only as strong as the dimension on which it has the weakest support.

## The evaluation pipeline

The pipeline has seven layers. Layers 1–2 scope the claim. Layer 3 produces evidence. Layers 4–6 score it. Layer 7 issues a verdict.

| Layer | Name | Question |
|---|---|---|
| 1 | [Description modes](/mechanistic-validity/framework/description-modes/) | At what level is the claim stated? |
| 2 | [Evidence families](/mechanistic-validity/framework/evidence-families/) | Which sources of signal support it? |
| 3 | [Metrics](/mechanistic-validity/framework/metrics/) | What was concretely measured? |
| 4 | [Criteria](/mechanistic-validity/framework/criteria/) | Does the evidence meet the stated conditions? |
| 5 | [Validity types](/mechanistic-validity/framework/validity-types/) | Which dimensions of validity does it address? |
| 6 | Synthesis | How is evidence aggregated across methods? |
| 7 | [Verdicts](/mechanistic-validity/framework/verdicts/) | What has the claim established? |

The five **validity types** form a dependency chain: Construct → Measurement → Internal → External → Interpretive. A failure early in the chain limits what later evidence can establish: an ambiguous construct cannot be reliably measured, an unreliable measurement cannot support a causal inference, and so on.

The 36 **criteria** — 6 construct, 7 measurement, 12 internal, 6 external, 5 interpretive — are the specific, falsifiable conditions within each validity type. Each draws from a distinct scientific tradition: philosophy of science and psychometrics for construct validity, causal inference and neuroscience for internal validity, pharmacology for external validity, and Marr's levels for interpretive validity.

## Sixteen audited case studies

We audit sixteen published mechanistic claims across fifteen papers. No claim reaches Validated. One claim — induction heads for token copying — reaches Triangulated. Seven reach Mechanistically Supported. The capping criterion for every Mechanistically Supported claim is I6 (double dissociation): the field rarely attempts crossed designs. I8 (confounding sensitivity) is untested in all sixteen.

See the [case studies](/mechanistic-validity/framework/examples/) for the full verdicts and per-criterion scoring.

## What the framework is not

The framework does not rank circuits. It produces a structured verdict — a pattern of which dimensions have evidence and which do not — rather than a scalar score. Two circuits with the same scalar faithfulness can have very different verdict structures, and the framework's value is in making that difference visible.

The framework also does not assume any particular discovery method is correct. Activation patching, EAP, DAS-IIA, weight classifiers, and causal scrubbing all appear as metrics that produce evidence for one or more criteria. None is privileged. The framework's role is to specify what each metric actually establishes.
