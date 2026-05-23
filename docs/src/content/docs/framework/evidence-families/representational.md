---
title: "Representational Evidence"
description: "Evidence from latent geometry revealing what information is encoded in activation space and how"
---

# Representational Evidence

Representational evidence captures what variables are encoded in a model's internal activation spaces — the geometry of latent representations and whether specific information is linearly or nonlinearly accessible at specific locations.

## What this family measures

Representational evidence comes from probing, alignment, and similarity analyses that ask: "Is variable X decodable from the activations at location Y?" This includes linear probes that test whether a classification boundary exists in activation space, distributed alignment searches that identify subspaces encoding specific variables, and representational similarity analyses that compare the geometry of internal representations to external structure.

The focus is on *encoding* — whether information is present and in what geometric form. A linear probe achieving high accuracy on "is the indirect object a male name?" at the residual stream after layer 8 establishes that this information is linearly encoded at that location. Representational similarity analysis showing that the internal geometry at a specific layer mirrors the semantic similarity structure of the input tokens establishes that semantic information is preserved in that representation.

This family sits between structural evidence (which asks what the weights can compute) and causal evidence (which asks what matters for the output). Representational evidence asks what information is *available* at each point in the computation — which constrains but does not determine what is actually used.

## Metrics

- **E01 DAS-IIA** — Distributed Alignment Search with Interchange Intervention Accuracy
- **E02 Linear Probe** — Linear classifiers trained on intermediate activations to decode variables
- **E03 RSA** — Representational Similarity Analysis comparing internal and external similarity structures
- **E04 CKA** — Centered Kernel Alignment for comparing representation spaces across layers or models
- **E05 Subspace Alignment** — Principal angle and overlap metrics between activation subspaces

## Characteristic strength

Representational evidence uniquely tests whether specific variables are encoded at specific locations in a model's computation. This is critical for mechanistic claims because it establishes the *information-processing substrate* — you cannot claim a component "computes X from Y" unless Y is represented in the component's input and X is represented in its output.

The geometric specificity of this evidence is also valuable: not just "is the information there?" but "in what form?" A variable encoded in a 1-dimensional subspace is processed differently than one distributed across 50 dimensions. This geometric characterization constrains the space of possible computational mechanisms.

## Characteristic blind spot

Encoding does not imply use. A representation can be perfectly decodable by a probe yet causally inert — never read by any downstream computation. This is the "decodability fallacy": just because a linear probe can extract information from activations does not mean the model's own downstream weights do extract it.

This is not a hypothetical concern. Residual stream representations accumulate information from all prior layers, so many variables remain decodable long after they were last causally relevant. Representational evidence alone cannot distinguish between "this variable is encoded here because it was just computed and will be used downstream" and "this variable is encoded here as a residual trace that nothing reads."

## Criteria served

- **C3 Task specificity** — Probing can test whether a component's representations are specific to the task or encode general information
- **C5 Convergent validity** — Representational similarity across models or training runs provides evidence that circuits converge on similar solutions

## Convergent validity role

Representational evidence combines most powerfully with causal evidence: if a probe shows a variable is encoded at a location (representational) AND patching that variable's subspace transfers behavior (causal), the encoding is confirmed to be causally active rather than merely decodable. This combination directly addresses the decodability fallacy.

Representational + structural is also informative: if weight analysis shows a downstream component reads from the subspace where a variable is encoded, this provides a mechanistic link even without direct intervention. Representational + representational (e.g., probing + RSA) strengthens the characterization of what is encoded but does not address whether the encoding is used.
