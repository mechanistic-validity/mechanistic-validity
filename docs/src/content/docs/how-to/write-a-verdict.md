---
title: "Write a Verdict"
---

# B — How to Write a Verdict Statement

A verdict statement has six required components. Missing any one makes the verdict incomplete.

---

## The full template

```
**Verdict:** [Tier] `[mode tag]` for [component(s)] as [functional description]
             in [model] on [task] / [prompt distribution].

**Satisfied:**
- [Criterion ID] ([Name]): [metric], [value vs. threshold], [n], [seeds] ✓
  (repeat for each satisfied criterion)

**Partial:**
- [Criterion ID] ([Name]): [what was done], [what is missing to reach ✓] ◑
  (omit section if none)

**Open:**
- [Criterion ID] ([Name]): [reason not run, or "not run"] ✗
  (list ALL required criteria for the declared tag that are open)

**Upgrade path:**
- → [Next tier]: [specific criteria to satisfy], [specific metrics to run]
  (one upgrade path per tier level, up to Triangulated minimum)

**Scope:** [model + size]. [Task] on [distribution] ([n], [seeds], [checkpoint]).
           [Ablation method(s)] at [hook point(s)].
           Not tested: [untested dimensions].
```

---

## Worked example 1: L8.MLP IIA = 0.48

```
Verdict: Causally suggestive (pending baselines) `[representational]`
         for L8.MLP as SVA-associated subspace in GPT-2 Small
         on Linzen et al. (2016) held-out split.

Satisfied:
- M5 (Calibration): IIA = 0.48, within published transcoder range (0.40–0.60) ◑
  [M5 partial because M3 baselines not computed]

Partial:
- M3 (Baseline separation): IIA = 0.48; random-vector and untrained-model
  baselines NOT YET COMPUTED. ◑

Open:
- I1 (Necessity): ablation on L8.MLP not run ✗
- I2 (Sufficiency): complement ablation not run ✗
- I3 (Specificity): control-axis IIA not computed ✗
- I4 (Consistency): sigma-ablation not run ✗
- I5 (Confound control): component-specific ablation not run ✗
- E1–E6: all open ✗
- C1–C5: all open ✗

Upgrade path:
→ Causally suggestive (confirmed):
    compute random-vector + untrained-model baselines (M3)
    run I1 (zero + resample ablation, n ≥ 100, 3 seeds)
→ Mechanistically supported:
    add I2 (complement ablation, threshold ≥ 70% recovery)
→ Triangulated:
    I3 (control-axis IIA) + I4 (sigma-ablation, 3 seeds) + I5 (component-specific)
    + E1 (activation delta) + C2 (structural plausibility)

Scope: GPT-2 Small (124M, 12L). SVA on Linzen (2016) held-out (n=200, 3 seeds, final ckpt).
       DAS-IIA at blocks.8.mlp.hook_post.
       Not tested: causal necessity, specificity, cross-scale, cross-prompt-family.
```

---

## Worked example 2: SAE direction aligned with IOI head

```
Verdict: Proposed `[structural-mechanistic]`
         for learned direction as candidate IOI name-copying component aligned
         with L8H6 W_OV direction in GPT-2 Small.

Satisfied:
- C2 (Structural plausibility): cos(direction, W_OV[L8H6]) = 0.82 > 0.70 ✓
- C2 (additional): cos(direction_2, W_OV[L3H0]) = 0.72 > 0.70 ✓

Open:
- I1 (Necessity): ablation of learned direction not run ✗
- M3 (Baseline separation): IIA not yet computed ✗
- All other criteria ✗

Upgrade path:
→ Causally suggestive:
    run I1 (zero ablation on the direction, measure IOI logit diff, n ≥ 100, 3 seeds)
    compute M3 (random-vector + untrained-model IIA)

Scope: GPT-2 Small (124M). IOI on Wang et al. (2022) 15-template set.
       W_OV SVD at L8H6 and L3H0. No causal evidence.
       Not tested: causal necessity, sufficiency, cross-scale, cross-prompt.
```

---

## Quick-reference: tier by evidence pattern

| Evidence in hand | Write |
|---|---|
| Structural alignment only (cos score) | *Proposed* `[structural-mechanistic]` |
| Ablation degrades behavior (necessity only) | *Causally suggestive* `[causal-mechanistic]` |
| Necessity + complement ablation (sufficiency) | *Mechanistically supported* `[causal-mechanistic]` |
| I1–I5 + ≥1 external + ≥1 construct | *Triangulated* `[causal-mechanistic]` |
| IIA above baseline, no ablation | *Causally suggestive* `[representational]` |
| Metrics disagree (Jaccard ≈ 0) | *Underdetermined* regardless of tier |
| Falsification condition triggered | *Disconfirmed* |
