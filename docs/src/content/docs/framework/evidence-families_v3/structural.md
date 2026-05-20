---
title: "Structural Evidence"
description: "Evidence from weight-space analysis revealing architectural capacity independent of any input"
---

# Structural Evidence

Structural evidence captures what a model's parameters reveal about computational capacity before any input is processed — properties of the weights themselves.

## What this family measures

Structural evidence comes from analyzing the model's weight matrices directly: their spectra, rank structure, alignment between projections, and decomposition into interpretable components. This is the only evidence family that requires no forward pass — it characterizes what the architecture *can* compute rather than what it *does* compute on specific inputs.

The key insight is that weight matrices encode computational structure. The singular value decomposition of an OV matrix reveals what information an attention head can move. The alignment between a query matrix and a key matrix reveals what features an attention head can attend to. The effective rank of a projection reveals how many independent directions it uses.

Because this evidence is input-independent, it provides a distinct epistemic contribution: it reveals architectural *capacity*. A weight structure that cannot perform a computation rules out that component from the circuit, regardless of what activation-based methods might suggest (which could be confounded by information flowing through other paths).

## Metrics

- **B01 SVD/Spectral** — Singular value decomposition and spectral analysis of weight matrices
- **B02 Effective Rank** — Numerical rank and dimensionality of projection subspaces
- **B03 OV/QK Decomposition** — Factored analysis of attention head behavior through composed weight matrices
- **B04 Weight Alignment** — Cosine similarity and subspace overlap between different projections

## Characteristic strength

Structural evidence is unique in being entirely input-independent. This means it cannot be confounded by distributional artifacts in the evaluation set, cannot overfit to a specific prompt distribution, and provides ground truth about what computations are *architecturally possible*.

When structural analysis shows that a head's OV matrix has a rank-1 component aligned with the "copy previous token" direction, this is a statement about the head's inherent capacity — not about what it happens to do on a particular set of prompts. This permanence makes structural evidence particularly valuable for claims about circuit identity (what a component *is*) rather than circuit behavior (what a component *does* in context).

## Characteristic blind spot

A weight structure that CAN compute something does not mean it DOES compute it on actual inputs. Capacity is necessary but not sufficient for function. A head might have the right OV structure to perform name-moving but never encounter inputs that activate that pathway in practice.

This means structural evidence alone cannot establish that a circuit is *used* — only that it *could* be used. The gap between capacity and function must be bridged by other evidence families (causal, behavioral, or representational) that examine what happens when the model actually processes inputs.

## Criteria served

- **C2 Structural plausibility** — Directly tests whether the proposed circuit has the right weight-space structure to perform the claimed computation
- **C4 Minimality** — Rank and spectral analysis can reveal whether a circuit uses more components than necessary, or whether a simpler structural explanation exists

## Convergent validity role

Structural evidence combines powerfully with causal evidence: if ablation shows a component is necessary (causal) AND weight analysis shows it has the right structure for the computation (structural), the two lines of evidence address each other's blind spots. The causal evidence rules out "has structure but doesn't use it" while the structural evidence rules out "is causally important but only as a relay."

Structural + structural (e.g., SVD + rank analysis) strengthens confidence in the structural characterization but does not address the capacity-vs-use gap. The most informative combinations pair structural evidence with families that require actual input processing.
