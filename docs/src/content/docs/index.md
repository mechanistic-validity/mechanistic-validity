---
title: "Overview"
description: "Mechanistic Validity: a framework for evaluating circuit claims in mechanistic interpretability."
---

# Mechanistic Validity

Mechanistic Validity is a framework for evaluating claims about circuits in neural networks. It defines five validity types, the criteria within each type, and the instruments that produce evidence for each criterion. Its purpose is to make explicit which part of a mechanistic claim a given measurement supports and which parts remain unaddressed.

The framework does not introduce new measurement methods. It organizes existing methods — ablation, activation patching, IIA, causal scrubbing, weight analysis, baseline calibration — under a common evaluative vocabulary drawn from the standard typology of validity in philosophy of science and adapted to the conditions of mechanistic interpretability.

## What the framework is for

The framework applies to claims of the form *component C implements computation T in model M*. Such claims are made routinely in circuit-discovery papers and are typically supported by a small number of instruments — most often activation patching plus one form of ablation. The framework specifies what additional evidence is required for the claim to be considered validated under each of five named dimensions, and what verdict is licensed when only a subset of that evidence is present.

The framework's central commitment is that a single high score does not validate a circuit claim. Validation is a pattern of evidence across multiple dimensions, and a claim is only as strong as the dimension on which it has the weakest support.

## The two-layer structure

The framework has two layers. The upper layer is the five **validity types** — construct, internal, external, measurement, interpretive. These are the abstract questions a claim must answer. The lower layer is the five **lenses** — Philosophy of Science, Neuroscience, Pharmacology, Measurement Theory, Mechanistic Interpretability. These are the operational toolkits, one per validity type, that translate the abstract question into criteria, instruments, and reporting rules.

| Validity type | Lens | What the lens provides |
|---|---|---|
| Construct | Philosophy of Science | Falsifiability, structural plausibility, task specificity, minimality, convergent validity |
| Internal | Neuroscience | Necessity, sufficiency, specificity, consistency |
| External | Pharmacology | Intervention reach, graded response, selectivity, effect magnitude, robustness, cross-architecture generalization |
| Measurement | Measurement Theory | Reliability, invariance, baseline separation, sensitivity, calibration, construct coverage |
| Interpretive | Mechanistic Interpretability | Level declaration, level-evidence match, narrative coherence, alternative exclusion, scope honesty |

The validity-type pages explain what each type asks of a claim and where current MI practice falls short. The lens pages give the operational criteria, the instruments that produce evidence for each criterion, the failure modes that appear in practice, and a minimum reporting protocol.

## What the framework is not

The framework does not rank circuits. It produces a structured verdict — a pattern of which dimensions have evidence and which do not — rather than a scalar score. Two circuits with the same scalar faithfulness can have very different verdict structures under the framework, and the framework's value is in making that difference visible.

The framework also does not assume any particular discovery method is correct. Activation patching, EAP, IIA, weight classifiers, and causal scrubbing all appear in the lens pages as instruments that produce evidence for one or more criteria. None is privileged. The framework's role is to specify what each instrument actually establishes.

## How to get started

A reader new to the framework should begin with [Reading this site](start/reading-this-site.md), which explains how the pages link together and how to use the framework to audit a circuit claim. Readers who already know which validity type they are concerned with can skip to that type's page; readers who already know which method they are using can skip to the corresponding lens.
