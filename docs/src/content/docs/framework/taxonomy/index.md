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
| Last updated | 16 May 2026 |

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
| `[computational]` | What is being computed and why | [A_computational.md](../01_modes/A_computational.md) |
| `[algorithmic]` | What operation is performed | [B_algorithmic.md](../01_modes/B_algorithmic.md) |
| `[representational]` | What is encoded, where, how | [C_representational.md](../01_modes/C_representational.md) |
| `[implementational]` | Which weights/components carry it | [D_implementational.md](../01_modes/D_implementational.md) |
| `[architectural]` | How computational labor is distributed | [E_architectural.md](../01_modes/E_architectural.md) |
| `[structural]` | What the weights say before any input | [F_structural.md](../01_modes/F_structural.md) |
| `[transportable]` | Which features survive cross-model shift | [G_transportable.md](../01_modes/G_transportable.md) |

For how the seven modes extend Marr's three levels, see [H_marr-comparison.md](../01_modes/H_marr-comparison.md).

## Where to go from here

- Dependency order in detail: [B_dependency-order.md](B_dependency-order.md)
- How to read and write a verdict: [C_verdict-anatomy.md](C_verdict-anatomy.md)
- Design commitments behind the framework: [D_design-principles.md](D_design-principles.md)
