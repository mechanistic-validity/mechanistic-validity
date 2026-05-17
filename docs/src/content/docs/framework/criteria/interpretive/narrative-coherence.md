---
title: "Narrative Coherence"
validity_type: "Interpretive"
criterion_id: "V3"
---

# Criterion V3 — Narrative Coherence

| | |
|---|---|
| Validity type | Interpretive |
| Pass condition | The prose description of the finding is consistent with and entailed by the mode-tagged claim |
| Evidence family | N/A (criterion is about writing, not experiments) |
| Minimum reporting | No specific reporting format — checked by auditing the prose against the mode-tagged verdict |
| Common failure mode | Prose says "implements" or "is responsible for" when the verdict tag is `[representational]`; prose says "suggests" when the verdict tag is `[causal-mechanistic]` |

## What this criterion requires

Narrative coherence checks that the words used to describe a finding are consistent with the mode tag declared in the verdict (V1) and the evidence that licenses it (V2).

Every mode tag has permitted and prohibited prose patterns:

| Tag | Permitted | Prohibited |
|---|---|---|
| `[functional]` | "predicts," "correlates with," "is associated with" | "implements," "causes," "is responsible for" |
| `[representational]` | "encodes," "represents," "contains information about," "is linearly accessible at" | "implements," "causes," "is responsible for" |
| `[causal-mechanistic]` | "causally implements," "is necessary and sufficient for," "mediates" | "fully explains," "is the only mechanism" (unless minimality is established) |
| `[structural-mechanistic]` | "is structurally consistent with," "has parameters that implement" | Causal language without causal evidence |
| `[transportable]` | "generalizes to," "transfers to," "appears in" (with scope restriction) | Universal claims ("always," "in all models") without supporting evidence |

## Why this is a separate criterion from level–evidence match (V2)

Level–evidence match (V2) is about the logical relationship between evidence and the mode tag. Narrative coherence (V3) is about the relationship between the mode tag and the prose. Both can fail independently:

- V2 fails, V3 passes: the evidence does not license the declared tag, but the prose correctly describes what the evidence shows (the tag is wrong, the prose is accurate about what was found).
- V2 passes, V3 fails: the evidence licenses the declared tag, but the prose overclaims (using stronger language than the tag permits).

The most common pattern in published MI is V2 partially passes and V3 fails — the evidence is borderline for the declared tag, and the prose uses language that overstates what is established.

## Minimum reporting rule

Audit every sentence in the results and discussion sections that makes a claim about a circuit component. For each sentence, identify the strongest mode tag that the evidence would license. If the sentence's language implies a stronger mode than the evidence licenses, revise the prose.
