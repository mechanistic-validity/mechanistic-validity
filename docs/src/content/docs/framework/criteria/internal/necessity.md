---
title: "Necessity"
validity_type: "Internal"
criterion_id: "I1"
---

# Criterion I1 — Necessity

| | |
|---|---|
| Validity type | Internal |
| Pass condition | Ablating the proposed component reliably degrades the target behavior, across ≥2 ablation methods |
| Evidence family | Causal |
| Minimum reporting | Ablation method(s), metric, delta value(s), number of prompt samples |
| Common failure mode | Reporting only one ablation method; not comparing zero, resample, and mean ablation |

## What this criterion requires

Necessity establishes that the proposed component is causally required for the behavior: when it is removed or disrupted, the behavior degrades.

Satisfied when:

1. **Ablating the component degrades the target behavior**, measured on the target metric.
2. **The result holds across ≥2 ablation methods.** Zero, resample, and mean ablation each have different failure modes. Consistent degradation across all three is robustly necessary.
3. **Result is computed over ≥30–50 prompts** with variance estimated.

## Ablation methods and failure modes

| Method | What it does | Failure mode |
|---|---|---|
| Zero ablation | Sets activation to zero | Unnatural zero can trigger unexpected downstream effects |
| Resample ablation | Replaces with activation from shuffled prompt | May understate effects for correlated heads |
| Mean ablation | Replaces with dataset mean | Mean carries task-relevant information; can overstate effects via mean-field disruption |
| Patch ablation | Replaces with corrupted-input run activation | Most controlled; establishes necessity under specific counterfactual |

Using all three unconditional methods together and showing consistent degradation across all three is the gold standard.

## Necessity vs. sufficiency

Necessity establishes the component is required. It does not establish that restoring it alone recovers the behavior. Necessity alone licenses *Causally suggestive* tier at best. *Mechanistically supported* requires sufficiency (I2) as well.

**This is the most common understated result in MI:** showing ablation degrades behavior, calling the component "a core circuit member," and implicitly claiming more than necessity licenses.

## Minimum reporting rule

- Ablation method(s), metric, delta for each method.
- Number of prompts and whether variance was estimated.
- If only one ablation method used: flag necessity as partially satisfied (method variance untested).
