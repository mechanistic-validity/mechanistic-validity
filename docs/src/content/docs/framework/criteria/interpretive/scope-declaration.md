---
title: "Scope Declaration"
validity_type: "Interpretive"
criterion_id: "V5"
---

# Criterion V5 — Scope Declaration

| | |
|---|---|
| Validity type | Interpretive |
| Pass condition | The claim states which models, tasks, prompt distributions, and description modes it does not cover |
| Evidence family | N/A (criterion is about claim boundaries, not experiments) |
| Minimum reporting | Explicit non-coverage: the models, tasks, and distributions the claim does not extend to |
| Common failure mode | A result on one model and a handful of prompts stated as a claim about "language models" or "in-context learning" in general |

## What this criterion requires

A mechanistic claim needs a boundary as much as it needs a body. Scope declaration requires stating which models, which tasks, which prompt distributions, and which description mode a claim covers — and, symmetrically, what it does not. Without a stated boundary, a finding in one model on a handful of prompts is read as a finding about the class of systems the model belongs to.

Satisfied when:

1. **Coverage is named.** The model(s), task(s), and prompt distribution(s) tested are stated.
2. **Non-coverage is named.** Models, tasks, and conditions not tested are stated explicitly, not left implicit.
3. **The claim does not exceed the coverage.** Generalizing beyond what was tested requires either additional evidence or an explicit statement that the generalization is speculative.

A scope declaration is not a limitation. A limitation states what the evidence cannot show given how the study was run — a small sample, an untested confound. Scope states what the claim does not assert in the first place, independent of whether stronger evidence could later extend it.

## MI example

Knowledge neurons: the paper reports that its localization method "can be easily generalized" to other models, with no second model tested. The claim moves from one architecture (BERT-base) to a property asserted of neural networks in general — the held-out architectures and training regimes are never named. A scope-compliant version states the one model and the one task tested and stops there, leaving generalization a separate, untested claim rather than a stated conclusion.

## Relation to other interpretive criteria

Scope declaration (V5) closes the interpretive chain: V1 states the mode, V2 checks the evidence supports it, V3 rules out a simpler mode, V4 checks the name does not smuggle in an unmeasured property, and V5 states where all of that stops applying. A claim that passes V1–V4 but skips V5 is locally sound and globally overclaimed.
