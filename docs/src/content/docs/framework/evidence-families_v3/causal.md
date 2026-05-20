---
title: "Causal Evidence"
description: "Evidence from interventions that test what happens when components are forced to specific values"
---

# Causal Evidence

Causal evidence captures what happens when you actively intervene on a model's internal state — forcing a component to take a specific value and observing downstream consequences.

## What this family measures

Causal evidence is produced by metrics that perform interventions: ablations, activations patches, interchange interventions, and mediation analyses. The core logic is counterfactual — what would the model's output have been if this component had taken a different value?

This makes causal evidence the most direct test of whether a component is necessary or sufficient for a behavior. When you ablate a component and behavior degrades, you have evidence of necessity. When you patch a component's activation from one context into another and behavior transfers, you have evidence of sufficiency.

The interventional nature of this evidence distinguishes it from purely observational approaches. Rather than asking "does this component correlate with the behavior?" causal metrics ask "does changing this component change the behavior?" This is a stronger claim, but comes with its own limitations around what interventions can and cannot reveal about underlying computation.

## Metrics

- **A01 Pearl SCM** — Structural causal model formalization of circuit graphs; do-calculus interventions
- **A02 Counterfactual DAS** — Distributed Alignment Search with counterfactual pairs
- **A03 Rubin CATE** — Conditional Average Treatment Effect estimation for component-level causal impact
- **A06 Mediation** — Indirect effect decomposition through specific pathways
- **A11 Actual Cause** — Halpern-Pearl actual causation applied to single-input traces

## Characteristic strength

Causal evidence directly tests necessity and sufficiency — the two foundational properties any circuit claim must establish. No amount of correlational evidence (representational, behavioral, or information-theoretic) can substitute for showing that intervening on a component changes the output in the predicted way.

This family also provides the clearest operationalization of "this component does X" claims. When a researcher says "head 9.1 performs name mover behavior," causal evidence is what grounds that claim: patching the name signal through head 9.1 transfers the behavior, ablating head 9.1 destroys it.

## Characteristic blind spot

Interventions cannot distinguish between a component that *computes* something and a component that merely *transmits* it. If you ablate a wire carrying a signal, the signal disappears downstream — but the wire didn't compute anything. Similarly, intervention destroys the distinction between "this component extracts feature X from the input" and "this component passes along feature X that was already computed upstream."

This is particularly problematic for residual stream components, where every downstream layer has access to every upstream computation. Ablating an early component disrupts all later components that read from it, making it appear more causally important than it may be in terms of the specific computation it performs.

## Criteria served

- **I1 Necessity** — Ablation and knockout interventions directly test whether a component is required for behavior
- **I2 Sufficiency** — Activation patching tests whether a component (or set of components) is enough to produce behavior
- **I3 Specificity** — Targeted interventions can test whether a component's causal role is specific to the claimed task or general
- **I5 Confound control** — Causal methods can rule out confounding pathways through careful intervention design

## Convergent validity role

Causal evidence is strongest when combined with structural or representational evidence from different families. When causal ablation shows a component is necessary AND weight-space analysis shows the component has the right structure to perform the computation AND probing shows the relevant variable is encoded there, the three independent lines of evidence triangulate on the same conclusion.

Causal + causal (e.g., ablation + patching) is valuable but provides less triangulation than causal + structural, because both metrics share the same fundamental assumption: that single-site interventions cleanly isolate component contributions. When causal evidence agrees with evidence from a family that makes entirely different assumptions, the resulting confidence is multiplicative rather than additive.
