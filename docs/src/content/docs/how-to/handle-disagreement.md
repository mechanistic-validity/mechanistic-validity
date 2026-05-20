---
title: "Handle Disagreement"
---

# G — How to Handle Metric Disagreement

Disagreement between metrics is not a problem to minimize. It is the most informative signal the framework produces. This guide provides the protocol for characterizing, interpreting, and resolving metric disagreement.

---

## Step 1: Characterize the disagreement

First, determine which type of disagreement you have.

### Type A: Metrics from the same evidence family disagree

Example: Zero ablation and resample ablation give opposite results (one shows Δ > 0, the other Δ ≈ 0).

**Most likely cause:** Mean-field confound. Mean ablation disrupts downstream computation through the mean-field signal, independent of the target component's specific contribution. If mean ablation shows degradation but resample ablation does not, the effect is a confound, not a real mechanism.

**Resolution:** Use the resample ablation result as ground truth for I1 (necessity). Downgrade or remove the mean ablation result from the necessity claim. Report the discrepancy explicitly.

### Type B: Causal metrics agree, structural metric disagrees (Jaccard ≈ 0)

Example: Zero ablation + EAP agree on components {A, B, C}. Weight classifier nominates {D, E, F}. Jaccard ≈ 0.

**Three interpretations:**
1. **Weight classifier is wrong:** nominates structurally similar but causally inactive components.
2. **EAP is wrong:** misses structurally plausible but low-gradient components.
3. **Both are right about different things:** weight classifier finds structurally plausible components; EAP finds causally active components; these are different populations.

### Type C: Representational metric disagrees with causal metric

Example: DAS-IIA at component X is high (0.48), but ablating X shows no degradation (Δ ≈ 0).

**Most likely cause:** The component represents the relevant variable but is not the causal route. The computation may be implemented by a parallel route.

**Resolution:** This is a genuine finding — the component has representational content without being causally necessary. It should be reported as `[representational]` *Causally suggestive* rather than `[causal-mechanistic]`. The causal mechanism is elsewhere.

### Type D: Causal metric shows necessity but not sufficiency

Example: Ablation degrades behavior, but complement ablation (circuit only) cannot reproduce behavior.

**Most likely cause:** The circuit is incomplete. Additional necessary components exist outside the proposed circuit.

**Resolution:** Expand the circuit search. Add the most likely missing components (based on weight classifier or EAP nominations not currently in the circuit) and re-run complement ablation.

---

## Step 2: Run the discriminating experiment

For **Type B** (the project's primary disagreement):

| Interpretation | Discriminating experiment | Expected result if true |
|---|---|---|
| Weight classifier wrong | Run DAS-IIA on weight-circuit components and EAP-circuit components separately | EAP components have higher causal-axis IIA; weight-circuit components have IIA ≈ random-vector baseline |
| EAP wrong | Run activation patching on weight-circuit components | Weight-circuit components show high patching effect (Δ ≥ 0.10 × full model) |
| Both right, different things | Run both IIA and patching on both sets | Weight-circuit: high IIA, low patching effect; EAP-circuit: high patching effect, lower IIA |

**The minimum discriminating experiment for this project:**
Run DAS-IIA (causal-axis) on the weight-circuit component set and the EAP-circuit component set separately. Compare:
- `IIA(weight-circuit components)` vs. `IIA(EAP-circuit components)`
- `IIA(weight-circuit components)` vs. `IIA(random-vector baseline)`

If `IIA(weight-circuit) ≈ IIA(random-vector)` and `IIA(EAP-circuit) >> IIA(random-vector)`, interpretation 1 (weight classifier wrong) is confirmed.

---

## Step 3: Update the verdict

| Discriminating result | Updated verdict |
|---|---|
| Interpretation 1 confirmed (weight classifier wrong) | Downgrade weight-circuit claim to *Proposed*; upgrade EAP-circuit claim to *Mechanistically supported* + *Underdetermined* resolved |
| Interpretation 2 confirmed (EAP wrong) | EAP-circuit claim downgraded; weight-circuit claim upgraded; convergent validity partially satisfied |
| Interpretation 3 confirmed (complementary) | Both claims are valid at different levels; publish as complementary evidence; structural and causal circuits are distinct |
| Inconclusive | Remain at *Underdetermined*; report what was run and what the result was; state next discriminating experiment |

---

## Common disagreement patterns and their meanings

| Pattern | Meaning | Verdict impact |
|---|---|---|
| Necessity without sufficiency | Circuit is incomplete | Remain at *Causally suggestive*; expand circuit |
| Sufficiency without specificity | Component is a hub, not task-specific | `[causal-mechanistic]` downgraded; C3 fails |
| Convergence on one seed, robustness failure | Finding is locally real, not general | *Mechanistically supported* with narrow scope; E5 fail |
| High IIA, baseline separation near zero | IIA is noise, not signal | M3 fails; `[representational]` downgraded to `[functional]` |
| Structural alignment, ablation no effect | Represents but doesn't implement | `[representational]` *Proposed* only |
| All metrics agree, control task also affected | C3 (task specificity) fails; component is a hub | Add hub-vs-specific analysis; I3 (specificity) may also fail |

---

## The Jaccard ≈ 0 report block

For the project's primary disagreement, the report block should read:

```
## Metric Disagreement: Weight-circuit vs. EAP-circuit (SVA, GPT-2 Small)

Disagreement type: Type B (causal and structural metrics disagree; Jaccard ≈ 0)

Weight-circuit nominations (structural): {[components], n=[k]}
EAP-circuit nominations (causal):        {[components], n=[k]}
Jaccard:                                  [J] ≈ 0

Three interpretations:
  1. Weight classifier wrong: nominates structurally similar but causally inactive components.
  2. EAP wrong: misses low-gradient structurally plausible components.
  3. Complementary: structural and causal circuits are different real populations.

Discriminating experiment planned:
  Run DAS-IIA (causal-axis) on both component sets.
  Threshold: if IIA(weight-circuit) − IIA(random-vector) < 0.05, interpretation 1 confirmed.

Current verdict: Underdetermined [representational] pending discriminating experiment.
```
