---
title: "ADR-015: Proxy Circuit Validation Protocol"
date: 2026-05-19
status: accepted
---

# ADR-015: Proxy Circuit Validation Protocol

## Context

10 of 54 tasks have `circuit_status="proxy_circuit"` — they inherit a circuit from a related "parent" task without independent verification. Three families exist:

- **RTI family** (5 tasks): rti, rti_pattern, token_flood, buffalo, mib_rti — share 15-head circuit
- **IOI family** (4 tasks): ioi, centering_theory, resumptive, self_allo — share 15-head circuit
- **Induction family** (4 tasks): induction, sequence_internal, alternating_pair, novel_song — share 7-head circuit

The proxy assumption is: "the same heads implement both tasks because the tasks are behavioral variants of the same phenomenon." This is a testable hypothesis, not a known fact.

## Decision

Formalize proxy circuit validation as a V2 (Transportability) protocol:

1. **Structural test** (no GPU): Compare prompt structure, linguistic phenomenon, output type
2. **Behavioral test** (GPU): Run activation patching per variant, compare causal head sets
3. **Verdict**: Jaccard > 0.7 = transfers (promote), 0.3-0.7 = partial (needs own circuit), < 0.3 = doesn't transfer

## Rationale

- Proxy circuits are the framework's weakest link — untested assumptions about circuit transfer
- V2 Transportability is the natural home for this validation
- Different prompt types may recruit different heads (polysemanticity)
- The IOI family is highest priority: syntactic structures differ significantly between ioi and centering_theory/resumptive

## Consequences

- IOI family: HIGH priority for GPU testing (syntactic variation)
- RTI family: MEDIUM priority (structural similarity is high, but token_flood may differ)
- Induction family: LOW priority (well-characterized general mechanism)
- Need V2 transportability metrics in the metric registry
- Parent claim specs should be runnable on proxy task prompts
