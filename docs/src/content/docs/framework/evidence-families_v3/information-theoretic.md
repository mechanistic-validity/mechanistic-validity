---
title: "Information-Theoretic Evidence"
description: "Evidence from coding properties quantifying how much information flows through components and in what form"
---

# Information-Theoretic Evidence

Information-theoretic evidence captures how much information flows through model components and in what form — quantifying communication between parts of a circuit without assuming a specific computational mechanism.

## What this family measures

Information-theoretic evidence comes from estimating mutual information, conditional dependencies, and information decompositions between components of a model's computation. These instruments ask: "How many bits about variable X are carried by component Y?" and "How does information about the task flow from input to output through the network?"

The distinctive feature of this family is its agnosticism about *mechanism*. Mutual information between a hidden layer and a task label quantifies how much the layer "knows" about the task without specifying how that knowledge is encoded or what computation produced it. Transfer entropy quantifies directed information flow without assuming linearity. Partial information decomposition separates redundant, unique, and synergistic contributions without requiring a specific functional form.

This abstraction is both the family's power and its limitation. By measuring information content rather than specific computations, these instruments can detect relationships that more targeted methods miss — including nonlinear encodings, distributed representations, and synergistic interactions between components.

## Instruments

- **C01 Mutual Information** — Bits shared between a component's activations and a task-relevant variable
- **C02 Conditional MI** — Information about a target variable given knowledge of other components
- **C03 Transfer Entropy** — Directed information flow from one component to another over time
- **C04 PID** — Partial Information Decomposition separating redundant, unique, and synergistic contributions
- **C05 Info Bottleneck** — Optimal compression characterization of what information components retain

## Characteristic strength

Information-theoretic evidence quantifies information flow without assuming a specific computational mechanism. This makes it uniquely suited to detecting unexpected relationships, measuring the efficiency of information transmission, and characterizing how information is distributed across components (redundantly, uniquely, or synergistically).

PID in particular offers something no other family provides: formal separation of redundant and synergistic contributions. When two components carry redundant information about a variable, either alone suffices. When they carry synergistic information, neither alone contains the signal — it emerges only from their combination. This distinction is critical for understanding circuit structure but invisible to causal or representational methods applied to individual components.

## Characteristic blind spot

High mutual information between a component and a task variable does not establish a causal role. Correlation in information space is still correlation. A component can carry high mutual information about a task variable because it receives that information from upstream but never passes it downstream — a dead-end branch that happens to be correlated with the output.

Additionally, information-theoretic quantities are notoriously difficult to estimate accurately in high-dimensional continuous spaces. Neural network activations live in spaces where naive MI estimators have high bias, and even modern neural estimators (MINE, InfoNCE) provide bounds rather than exact values. The resulting estimates may be noisy or systematically biased, complicating interpretation.

## Criteria served

- **I3 Specificity** — Conditional MI can test whether a component's information about the task is specific (high MI with target, low MI with distractors) or general
- **I4 Consistency** — Information flow patterns should be stable across different inputs from the same task; inconsistency signals unreliable circuit identification

## Convergent validity role

Information-theoretic evidence combines most naturally with causal evidence: if MI shows that information about variable X flows through component Y (information-theoretic) AND ablating Y destroys the model's use of X (causal), the correlation is confirmed as causal. This addresses the "correlation in information space" blind spot directly.

Information-theoretic + representational is also productive: MI quantifies *how much* information is present while probing characterizes *what form* it takes. Together they provide both the quantity and geometry of information at each computational stage.

Information-theoretic + information-theoretic (e.g., MI + PID) refines the characterization of information flow but inherits the shared limitation of not establishing causal relevance. Cross-family combinations are strongly preferred for triangulation.
