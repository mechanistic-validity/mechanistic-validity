---
title: "Prompt Generalization"
validity_type: "External"
criterion_id: "E2"
---

# Criterion E2 — Prompt Generalization

| | |
|---|---|
| Validity type | External |
| Pass condition | The mechanism operates on diverse prompts, not only the specific examples used for discovery |
| Evidence family | Behavioral |
| Minimum reporting | Number and diversity of test prompts, effect size distribution across prompts |
| Common failure mode | Discovering and testing on the same small prompt set |

## What this criterion requires

Prompt generalization asks whether the mechanism works beyond the specific inputs that were used to find it. A circuit discovered on 20 IOI sentences and tested on the same 20 sentences has not demonstrated generalization.

Satisfied when:

1. **Test prompts differ from discovery prompts.** The evaluation set is distinct from the set used to identify the mechanism.
2. **Prompts are diverse.** They vary in length, structure, and lexical content, not just token substitution.
3. **The effect holds across the distribution.** The mechanism's effect size is consistent, not driven by a few outlier prompts.

## MI example

IOI circuit studies typically use template-generated sentences ("When Mary and John went to the store, John gave a drink to ___"). The circuit is discovered and tested on variants of this template. Prompt generalization requires testing on naturally occurring indirect-object constructions with different syntactic structures — relative clauses, passives, embedded contexts — not just slot-filled versions of one template.

## Connection to the chain

Required for Validated tier. Without prompt generalization, the mechanism may be an artifact of the specific input distribution used for discovery, not a general property of the model.
