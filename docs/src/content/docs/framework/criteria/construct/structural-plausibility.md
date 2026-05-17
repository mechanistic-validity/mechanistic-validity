---
title: "Structural Plausibility"
validity_type: "Construct"
criterion_id: "C2"
---

# Criterion C2 — Structural Plausibility

| | |
|---|---|
| Validity type | Construct |
| Pass condition | Circuit components are at the layers and positions theory predicts, with weight-space signatures consistent with the claimed computational role |
| Evidence family | Structural (weight-space) |
| Minimum reporting | Layer/position of each component; relevant weight-space metric with value; comparison to null (random same-layer heads) |
| Common failure mode | Reporting only ablation results; never checking whether nominated heads have the right structural properties |

## What this criterion requires

Structural plausibility asks: if this component were doing what the circuit claim says, what would its weights look like — and do they?

Satisfied when:

1. **Layer and position are theoretically expected.** Name-mover heads should be late-layer (layers 9–11 in GPT-2 Small). S-inhibition heads should attend to the subject token. An SVA MLP should be downstream of primary syntactic integration heads.
2. **Weight-space signatures are consistent.** A copying head should have low-rank W_OV (near rank-1, singular value gap ≥ 0.8). An induction head should have predictable W_QK structure. An SVA head should have W_in directions geometrically close to the number-information subspace.
3. **The signature is specific to nominated components** — not just any same-layer head.

## This project's results

- An SAE direction aligned at cos = 0.82 with L8H6 W_OV — consistent with an output-writing role in IOI. **This is a structural plausibility result, not a causal result.**
- Another direction aligned at cos = 0.72 with L3H0 — consistent with an early syntactic role.

These findings satisfy structural plausibility for the IOI circuit claim (Finding 25). They do not establish causation; they make the construct coherent.

## Primary instruments

- B01 SpectralSVD: spectral gap of W_OV for copying/inhibition heads
- B02 OVQK analysis: eigenvalue analysis for attention pattern detection
- B03 Weight alignment: cosine similarity of W_dec directions to known-circuit templates
- B07 Effective rank: participation ratio of W matrices per head
- B08 Template distance: Frobenius distance to known-circuit W_OV templates

## Minimum reporting rule

For each component: layer/position + why theoretically expected + one weight-space metric + comparison to null. If a component is at an unexpected layer or has an unexpected signature, this must be noted — not silently omitted.

## Common failures

**Ablation-only circuit discovery:** Component added because ablating it degrades performance, with no check of whether its weights are consistent with the claimed role.

**Layer anomalies ignored:** A circuit method nominates a very early-layer head for a late-stage role without explanation.
