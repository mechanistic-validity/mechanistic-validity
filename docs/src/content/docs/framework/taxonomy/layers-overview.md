---
title: "Layers Overview"
description: "One-paragraph summary of each layer in the Mechanistic Validity hierarchy, plus the mode tag."
---

# Layers Overview

The framework has five layers plus a verdict annotation. Each layer asks a different question and occupies a different epistemic role. The layers are summarized here in bottom-up order — the order in which a researcher builds a claim.

## Layer A — Instruments

Layer A is the concrete runnable test: the thing you actually execute on the model. Instruments include activation patching, resample ablation, DAS-IIA, weight classifiers, bootstrap stability checks, and spectral SVD of weight matrices. An instrument produces a number or a set of numbers. By itself, a number from Layer A is not a finding — it is data that must be interpreted upward through the hierarchy. The same instrument (for example, causal scrubbing) can contribute evidence for criteria in multiple validity types depending on what question it is being used to answer.

## Layer B — Evidence Families

Layer B classifies an instrument's output by the *kind of signal* it produces, independent of the specific tool used. There are six families: causal (interventions and counterfactuals), structural (weight-space analysis with no forward pass), representational (latent geometry), behavioral (held-out task and generalization tests), information-theoretic (coding properties), and measurement (reliability and calibration audits). The evidence-family classification is what allows the framework to check convergent validity: two instruments from different families that agree on the same components constitute stronger evidence than two instruments from the same family, because they have structurally different failure modes.

## Layer C — Criteria

Layer C specifies the exact condition that must be met for a claim to count as satisfying a given dimension of validity. There are approximately 27 criteria across the five validity types. Each criterion is a falsifiable, operationally defined condition with a minimum-reporting rule. For example, *baseline separation* (in measurement validity) requires that the IIA score exceed both the random-vector baseline and the untrained-model baseline by a practically meaningful margin — it is not satisfied by pointing at a high absolute score.

## Layer D — Validity Types

Layer D is the level at which the framework's evaluative logic lives. The five validity types — construct, internal, external, measurement, interpretive — are the five abstract questions a circuit claim must answer. A validity type is satisfied when all its criteria are met; partial satisfaction is reported as partial validity. The types are not equivalent: a claim can be internally valid (the causal inference is licensed within the experiment) without being externally valid (the result may not generalize), and a claim can pass all four traditional validity types while failing interpretive validity (the verdict may be stated at the wrong description level).

## Layer E — Verdict

Layer E is the claim itself, stated with explicit scope. A verdict names: (1) the component or set of components, (2) the computation or behavior attributed to them, (3) the model and task, (4) the verdict-strength tier (Proposed, Causally suggestive, Mechanistically supported, Triangulated, Validated, Underdetermined, or Disconfirmed), and (5) the description-mode tag. The verdict is not a score. It is a structured statement about which validity types have been addressed, which criteria within those types have been met, and what remains open.

## Mode Tag — Description Level

The mode tag is an annotation on Layer E, not a separate layer. It names the type of claim the verdict is making: `[computational]`, `[algorithmic]`, `[representational]`, `[implementational]`, `[architectural]`, `[structural]`, or `[transportable]`. The tag is applied last, after the verdict has been assembled from Layers A–D, because description level is a property of the *claim* rather than of the evidence that supports it. Interpretive validity exists specifically to check whether the declared mode tag is licensed by the evidence.
