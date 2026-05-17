---
title: "Verdict Anatomy"
description: "How to read and write a Mechanistic Validity verdict: the four required fields and the mode tag."
---

# Verdict Anatomy

A verdict is a structured statement with five required fields. A verdict that substitutes a scalar for a structured statement has discarded the information the framework is designed to preserve.

## The five fields

```
[component]  [behavior/computation]  [in model, on task]  [verdict-strength]  [mode tag]
```

**Example:**

> L8.MLP causally mediates subject-verb number agreement in GPT-2 Small on the Linzen SVA dataset — *Mechanistically supported* `[implementational]`

| Field | Value |
|---|---|
| Component | L8.MLP |
| Behavior/computation | causally mediates subject-verb number agreement |
| Scope (model, task) | GPT-2 Small, Linzen SVA dataset |
| Verdict-strength | Mechanistically supported |
| Mode tag | `[implementational]` |

## The verdict-strength tiers

| Tier | Meaning | Minimum evidence |
|---|---|---|
| Proposed | Hypothesis-generating; no intervention evidence yet | Structural or representational evidence only |
| Causally suggestive | Partial internal validity; necessity shown, sufficiency not established | Ablation degrades behavior; no restoration test |
| Mechanistically supported | Necessity and sufficiency established; specificity and consistency partially addressed | Ablation + patching; ≥2 ablation variants |
| Triangulated | Internal validity fully addressed; ≥1 external and ≥1 construct criterion met | All five internal criteria + cross-task or cross-scale |
| Validated | All five validity types addressed with explicit baselines and convergent evidence | Full audit passed |
| Underdetermined | Evidence consistent with multiple mechanisms; construct validity cannot resolve | Active evidence present; interpretation ambiguous |
| Disconfirmed | Fails decisively on a key validity criterion | Explicit falsification criterion met |

## Mode tag decision

After the verdict-strength tier is assigned, add the mode tag:

| If the evidence establishes... | Licensed tag |
|---|---|
| The locus of an effect (which component, which layer) | `[implementational]` |
| The operation performed (what the component computes) | `[algorithmic]` |
| The encoding geometry (what the activations represent) | `[representational]` |
| The weight-space structure (without forward pass) | `[structural]` |
| The distribution of labor across components | `[architectural]` |
| Cross-model survival of the mechanism | `[transportable]` |
| The computational problem and why it is solved | `[computational]` |

## Partial verdicts

A partial verdict names the validity types that have been addressed and explicitly marks the others as open:

> L8.MLP causally mediates SVA in GPT-2 Small — *Mechanistically supported* `[implementational]`; external validity and interpretive validity unaddressed.

The "unaddressed" clause is not optional. Its purpose is to prevent silent upgrading by selective citation.

## What a verdict is not

- A score. "IIA = 0.48" is not a verdict.
- A confidence. "We are confident that L8.MLP is a key node" is not a verdict.
- A universal claim. A verdict is always scoped to a model and a task.
