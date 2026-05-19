---
title: "ADR-016: Epistemic Expanded Claim Spec"
date: 2026-05-19
status: implemented
---

# ADR-016: Epistemic Expanded Claim Spec (4th rival)

## Context

The epistemic framing task now has 4 rival claim specs from different discovery methods:

| Spec | Method | Heads | Steps | Key feature |
|------|--------|-------|-------|-------------|
| epistemic_framing | Manual (core) | 4 | 3 | Minimal, high confidence |
| epistemic_tight | Activation patching | 13 | 5 | Includes suppressors |
| epistemic_eap | Edge attribution patching | 15 | 4 | Edge-centric, different heads |
| epistemic_expanded | Broad activation patching | 32 | 6 | Maximally inclusive |

## Decision

Add EPISTEMIC_EXPANDED_SPEC with 6 steps (early_processor, early_suppressor, mid_composer, mid_suppressor, late_router, late_suppressor), 5 predictions, and 2 negative controls. Mark superposition risk as HIGH (32 heads = 26% of all GPT-2 heads).

Include a prediction that ablating late_suppressor INCREASES output (removes inhibition), testing the inhibitory pathway hypothesis.

## Key finding

Cross-circuit analysis shows:
- epistemic_tight and epistemic_expanded share 13 heads (J=0.406)
- epistemic_framing and epistemic_tight share 0 heads (J=0.000)
- No universal heads across all 4 variants
- Total unique heads across all variants: 44 (36% of GPT-2's 144 heads)

This is a strong case for V4 (Mechanism Adjudication) — the same task produces radically different circuits depending on discovery method.
