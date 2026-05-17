---
title: "Audit a Claim"
---

# A — How to Audit a Claim: The 5-Pass Procedure

Use this procedure to evaluate any MI claim — your own or a published result. Run all five passes in order. A claim cannot pass a later pass if it has failed an earlier one.

---

## Pass 1: Mode-tag check (V1 — Level Declaration)

**Question:** Does the claim state an explicit description-level mode tag?

**Action:**
1. Find the verdict or conclusion sentence(s).
2. Check for one of the five tags verbatim: `[functional]`, `[representational]`, `[causal-mechanistic]`, `[structural-mechanistic]`, `[transportable]`.
3. If absent, infer the implicit tag from claim language.

**Pass condition:** A tag is stated or unambiguously inferred.
**Fail action:** Not auditable until mode tag is declared.

---

## Pass 2: Evidence–tag fit check (V2 — Level–Evidence Match)

**Question:** Is the evidence sufficient to license the declared mode tag?

**Action:**
1. Look up required criteria for the declared mode tag (from `../03_criteria/index.md`).
2. For each, identify whether the paper reports a satisfying result.
3. Mark each: ✓ / ◑ / ✗.

**Pass condition:** All required criteria for the declared tag are at least ◑.
**Fail action:** Downgrade mode tag to the highest level whose criteria are all satisfied.

| Claimed tag | What's missing | Downgrade to |
|---|---|---|
| `[causal-mechanistic]` | No circuit-only forward pass (I2) | `[representational]` or lower |
| `[causal-mechanistic]` | Only one ablation method (I1 partial) | `[causal-mechanistic]` with ◑ note |
| `[representational]` | No baseline separation (M3 absent) | `[functional]` |
| `[transportable]` | No cross-prompt or cross-model test | `[causal-mechanistic]` at best |

---

## Pass 3: Narrative coherence check (V3)

**Question:** Is the prose consistent with the verified mode tag?

**Action:**
1. Read every claim sentence in Results and Discussion.
2. For each, identify the implied mode tag from language.
3. Flag sentences where implied tag is stronger than the verified tag.

**Pass condition:** No sentence implies a stronger mode than evidence licenses.
**Fail action:** Revise flagged sentences using permitted language for the verified tag.

Permitted language per tag:
- `[functional]`: "predicts," "correlates with," "is associated with"
- `[representational]`: "encodes," "represents," "contains information about"
- `[causal-mechanistic]`: "causally implements," "is necessary and sufficient for," "mediates"
- Never: "implements" or "is responsible for" at `[representational]` tier

---

## Pass 4: Alternative exclusion check (V4)

**Question:** Have competing mechanism descriptions been considered?

**Action:**
For each standard alternative, check whether it is addressed or acknowledged:
1. Hub alternative — component is general-purpose, not task-specific (addressed by I3, C3)
2. Correlation alternative — component correlates but doesn't cause (addressed by I2, I5)
3. Architectural prior — finding reflects architecture, not learned representation (addressed by M3 untrained baseline)
4. Prompt-artifact — finding reflects prompt distribution regularities (addressed by E5)
5. Jaccard-disagreement — weight-circuit and EAP-circuit disagree (addressed by discriminating experiment)

**Pass condition:** All standard alternatives addressed or acknowledged as unresolved.
**Fail action:** Add explicit section noting unresolved alternatives.

---

## Pass 5: Scope check (V5)

**Question:** Does the verdict scope statement correctly restrict the claim to the conditions tested?

**Action:**
Verify the verdict includes:
- Model name + size
- Task + prompt distribution
- Ablation method(s) + hook point
- n prompts + seeds + checkpoint
- Named dimensions not yet tested

**Pass condition:** All five scope dimensions present; untested dimensions named.
**Fail action:** Add missing scope restrictions; revise conclusions accordingly.

---

## Summary

| Pass | Criterion | Question | Fail action |
|---|---|---|---|
| 1 | V1 Level Declaration | Is a mode tag declared? | Declare or infer before proceeding |
| 2 | V2 Evidence–tag fit | Does evidence license the tag? | Downgrade tag |
| 3 | V3 Narrative coherence | Does prose match the tag? | Revise language |
| 4 | V4 Alternative exclusion | Are alternatives addressed? | Add alternative discussion |
| 5 | V5 Scope honesty | Is scope correctly restricted? | Add scope restrictions |
