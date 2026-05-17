---
title: "Mechanistic Validity Taxonomy"
description: "The spine of the Mechanistic Validity framework: how the five layers and seven description modes fit together."
---


# Mechanistic Validity Taxonomy

| | |
|---|---|
| Role | Organizing spine of the entire framework |
| Layers | A (Instruments) → B (Evidence families) → C (Criteria) → D (Validity types) → E (Verdicts) |
| Verdict annotation | Mode tag: the seven description levels |

Every circuit claim in mechanistic interpretability is a chain from a concrete measurement to a conclusion. The taxonomy names every link in that chain. Reading bottom-up is how you *build* a claim. Reading top-down is how you *evaluate* one.

## The five-layer hierarchy

```
Mode tag ── [computational] [algorithmic] [representational]
             [implementational] [architectural] [structural] [transportable]
                │
                ▼
Layer E  ── Verdict ────────── The claim, stated with explicit scope and mode tag
                │
Layer D  ── Validity types ─── The five abstract questions a claim must answer
             │  ├── Construct
             │  ├── Internal
             │  ├── External
             │  ├── Measurement
             │  └── Interpretive
             │
Layer C  ── Criteria ─────────  ~27 specific, falsifiable conditions, grouped by type
             │
Layer B  ── Evidence families ─ The six kinds of signal an instrument can produce
             │  Causal | Structural | Representational | Behavioral | Info-theoretic | Measurement
             │
Layer A  ── Instruments ──────  The concrete runnable tests
```

A claim that skips Layer D is not a finding. It is a measurement with a story attached.

## How to read the taxonomy

| Direction | Use case |
|---|---|
| Bottom-up (A → E) | Building a new claim: what does my instrument establish? |
| Top-down (E → A) | Auditing an existing claim: what evidence would this verdict require? |
| Sideways (across Layer B) | Checking convergent validity: do independent evidence families agree? |
| Mode tag last | After the verdict is assembled, check whether the declared description level is licensed |

## How the layers gate each other

The layers form a dependency order:
1. Layer A must be run before Layer B can be assigned.
2. Layer C cannot be assessed before Layer B.
3. Layer D cannot be satisfied before Layer C.
4. Layer E cannot be written before Layer D.
5. The mode tag is applied to Layer E last.

The dependency order is the reason the audit procedure runs construct validity first and interpretive validity last.

## The mode tag

Every verdict carries a bracketed description-mode tag that names the level of description the claim operates at.

| Tag | Claim type | Page |
|---|---|---|
| `[computational]` | What is being computed and why | [Computational](/framework/modes_v3/computational) |
| `[algorithmic]` | What operation is performed | [Algorithmic](/framework/modes_v3/algorithmic) |
| `[representational]` | What is encoded, where, how | [Representational](/framework/modes_v3/representational) |
| `[implementational]` | Which weights/components carry it | [Implementational](/framework/modes_v3/implementational-topographic) |
| `[architectural]` | How computational labor is distributed | — |
| `[structural]` | What the weights say before any input | — |
| `[transportable]` | Which features survive cross-model shift | — |

## Where to go from here

- [Dependency order](/framework/taxonomy/dependency-order) — why construct validity gates internal, and how the five types constrain each other
- [Verdict anatomy](/framework/taxonomy/verdict-anatomy) — how to read and write a verdict
- [Design principles](/framework/taxonomy/design-principles) — the commitments behind the framework
