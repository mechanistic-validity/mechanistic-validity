---
title: "Cross-Architecture Generalization"
validity_type: "External"
criterion_id: "E6"
---

# Criterion E6 — Cross-Architecture Generalization

| | |
|---|---|
| Validity type | External |
| Pass condition | The mechanism appears in at least one other model family (not just a different checkpoint or size of the same model) |
| Evidence family | Structural, Behavioral |
| Minimum reporting | Model family tested; matching criterion used; structural alignment score or behavioral replication result; null expectation |
| Common failure mode | Claiming generalization after testing only different sizes of the same model family |

## What this criterion requires

Cross-architecture generalization is the strongest form of external validity. The mechanism must be found in a model family with different architecture, training data, or training procedure.

Satisfied when:

1. **A second model *family* is tested.** GPT-2 variants (Small, Medium, Large, XL) are the same family. GPT-2 Small + Pythia-160M is a cross-family test. GPT-2 Small + Gemma-2B is a stronger cross-family test (architectural differences: grouped-query attention, RMSNorm, SwiGLU).

2. **An explicit matching criterion is stated.** How are circuit components identified as "the same" across models? Acceptable criteria: cosine similarity of W_OV decoder directions ≥ 0.7, structural classifier F1 on cross-model component identification, IIA transfer on the same task.

3. **The result meets the matching criterion** with a stated null expectation (e.g., expected cosine between random heads from different models).

## Why cross-architecture matters

A circuit finding that generalizes across model families is substantially more significant than one found in a single model. Cross-architecture replication rules out the possibility that the mechanism is an artifact of a specific training procedure or architecture choice. When the same mechanism appears in models with different attention patterns, normalization schemes, and activation functions, the finding is about the task, not the model.

## Minimum reporting rule

- Second model family tested and its architectural differences from the discovery model.
- Matching criterion stated explicitly.
- Structural alignment score or behavioral replication result with null comparison.
- Distinguish "same family, different size" (robustness, E5) from "different family" (cross-architecture, E6).
