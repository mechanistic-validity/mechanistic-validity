---
title: "Offset Coupling"
validity_type: "Internal"
criterion_id: "I12"
---

# Criterion I12 — Offset Coupling

| | |
|---|---|
| Validity type | Internal |
| Pass condition | The mechanism disappears when the capability is removed |
| Evidence family | Training (interventional) |
| Minimum reporting | Method of capability removal, mechanism measurement before and after, behavioral confirmation |
| Common failure mode | Never testing what happens to the mechanism when the capability is trained away |

## What this criterion requires

Offset coupling is the converse of onset coupling (I11). It asks: if the capability is removed — by fine-tuning, by [ablation](/mechanistic-validity/glossary/#ablation), by catastrophic forgetting — does the mechanism go away too? A mechanism that persists after the capability is gone is not tightly coupled to it.

Satisfied when:

1. **The capability is removed.** Fine-tuning on a different task, training past the phase transition, or targeted unlearning eliminates the behavior.
2. **The mechanism is measured after removal.** The structural or activation signature of the mechanism is checked in the modified model.
3. **The mechanism is absent or degraded.** If the mechanism persists unchanged after the capability is gone, offset coupling fails.

## MI example

The induction heads (token copying) case study is capped partly by I12. Induction heads have been shown to appear when in-context learning appears (onset coupling), but the converse — fine-tuning away the copying behavior and checking whether induction heads disappear — has not been tested. Without this, the mechanism could be an architectural feature that persists regardless of the capability.

## Connection to the chain

Part of the developmental coupling block (I11–I12). Required for Validated tier. Offset coupling is the top-down leg of Craver's (2007) mutual manipulability criterion: if the capacity goes, its mechanism should go with it.
