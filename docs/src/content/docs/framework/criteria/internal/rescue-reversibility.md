---
title: "Rescue Reversibility"
validity_type: "Internal"
criterion_id: "I7"
---

# Criterion I7 — Rescue Reversibility

| | |
|---|---|
| Validity type | Internal |
| Pass condition | Restoration recovers at least 80% of clean performance (fraction of clean-corrupt gap) |
| Evidence family | Causal |
| Minimum reporting | Clean performance, corrupt performance, rescue performance, recovery fraction, corruption method, restoration method, number of prompts |
| Common failure mode | Corrupting and restoring the same component (trivial); must corrupt broadly and restore selectively |
| Lens | Genetics |

## What this criterion requires

Rescue reversibility tests whether corrupting the model broadly and then selectively restoring only the proposed circuit recovers the target behavior. This is a stronger test than sufficiency (I2): sufficiency isolates the circuit in an otherwise-ablated model, while rescue corrupts the model first and then asks whether restoring the circuit undoes the damage.

Satisfied when:

1. **Broad corruption degrades behavior.** A corruption applied to the full model (or a superset of the circuit) produces substantial degradation on the target metric.
2. **Selective restoration recovers behavior.** Restoring only the proposed circuit's activations (from the clean run) into the corrupted model recovers at least 80% of the clean-corrupt gap.
3. **Corruption and restoration target different scopes.** The corruption must be broader than the restoration. Corrupting and restoring the exact same component is trivially reversible and does not count.

Rescue reversibility does not establish that the circuit is the only mechanism capable of the behavior, nor does it establish necessity. A circuit can be rescue-reversible without being necessary if redundant pathways exist.

## Distinction from I2 — Sufficiency

Sufficiency ablates everything outside the circuit and checks whether the circuit alone produces the behavior. Rescue corrupts everything (or a broad region) and checks whether restoring the circuit undoes the corruption. Rescue is strictly harder: it requires the circuit to override active interference from corrupted components, not merely operate in isolation.

## Minimum reporting rule

- Clean performance on the target metric.
- Corrupt performance and the corruption method used.
- Rescue performance and the restoration method used.
- Recovery fraction: (rescue - corrupt) / (clean - corrupt).
- Scope of corruption vs. scope of restoration (must differ).
- Number of prompts tested.
