---
title: "Onset Coupling"
validity_type: "Internal"
criterion_id: "I11"
---

# Criterion I11 — Onset Coupling

| | |
|---|---|
| Validity type | Internal |
| Pass condition | The mechanism appears when the capability appears during training |
| Evidence family | Training (observational + interventional) |
| Minimum reporting | Training checkpoints examined, capability metric, mechanism metric, co-emergence evidence |
| Common failure mode | Studying only the trained model without checking when the mechanism formed |

## What this criterion requires

Onset coupling asks whether the mechanism and the capability it supposedly implements emerge together during training. If the capability appears at checkpoint 500 but the mechanism is already present at checkpoint 100, the mechanism may be an architectural prior rather than a learned computation.

Satisfied when:

1. **Multiple training checkpoints are examined.** The mechanism is measured across training, not only in the final model.
2. **The capability onset is identified.** The checkpoint at which the behavior first appears is located.
3. **The mechanism onset coincides.** The mechanism appears at or near the same checkpoint. Temporal coupling supports a causal link; temporal dissociation weakens it.

## MI example

Olsson et al. studied induction heads across training and showed that in-context learning loss improves at the same checkpoint where induction heads form — a phase change. This is onset coupling evidence. It supports the claim that induction heads implement token copying, though it does not by itself establish causation (the mechanism and capability could be co-effects of a third change).

## Connection to the chain

Part of the developmental coupling block (I11–I12). Required for Validated tier. Onset coupling is the bottom-up leg of Craver's (2007) mutual manipulability criterion: if the mechanism is a genuine part of the system, it should appear when the system acquires the capacity.
