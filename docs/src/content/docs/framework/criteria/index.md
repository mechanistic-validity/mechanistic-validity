---
title: "Criteria"
description: "All ~27 Layer C criteria, grouped by validity type, with pass conditions and minimum-reporting rules."
---

# Criteria — Layer C

Layer C is where the framework becomes operational. Each criterion is a specific, falsifiable condition that must be met for a validity type to be satisfied. A validity type is satisfied only when *all* its criteria are met. Partial satisfaction is reported explicitly.

## Criteria by validity type

### Construct validity — Is the theoretical entity coherent and well-defined?

| # | Criterion | One-line pass condition | Page |
|---|---|---|---|
| C1 | Falsifiability | A named result stated in advance would disconfirm the claim | [construct/A_falsifiability.md](construct/A_falsifiability.md) |
| C2 | Structural plausibility | Components at predicted layers/positions with consistent weight-space signatures | [construct/B_structural-plausibility.md](construct/B_structural-plausibility.md) |
| C3 | Task specificity | Circuit does not score highly on unrelated tasks under same instrument | [construct/C_task-specificity.md](construct/C_task-specificity.md) |
| C4 | Minimality | No redundant members; removing any member degrades performance | [construct/D_minimality.md](construct/D_minimality.md) |
| C5 | Convergent validity | Multiple independent instruments nominate the same components | [construct/E_convergent-validity.md](construct/E_convergent-validity.md) |

### Internal validity — Did the manipulation cause the effect?

| # | Criterion | One-line pass condition | Page |
|---|---|---|---|
| I1 | Necessity | Ablating the component reliably degrades the behavior across ≥2 methods | [internal/A_necessity.md](internal/A_necessity.md) |
| I2 | Sufficiency | Isolating/restoring the component reproduces the behavior | [internal/B_sufficiency.md](internal/B_sufficiency.md) |
| I3 | Specificity | Effect is selective; control-axis IIA ≈ 0 while causal-axis IIA is high | [internal/C_specificity.md](internal/C_specificity.md) |
| I4 | Consistency | Finding holds across prompt samples, ablation methods, and random seeds | [internal/D_consistency.md](internal/D_consistency.md) |
| I5 | Confound control | Effect not explained by collateral disruption to non-circuit components | [internal/E_confound-control.md](internal/E_confound-control.md) |

### External validity — Does the claim generalize?

| # | Criterion | One-line pass condition | Page |
|---|---|---|---|
| E1 | Intervention reach | Activation delta at hook point is in predicted direction and non-trivial | [external/A_intervention-reach.md](external/A_intervention-reach.md) |
| E2 | Graded response | Effect scales monotonically with intervention strength; threshold and plateau visible | [external/B_graded-response.md](external/B_graded-response.md) |
| E3 | Selectivity | On-task effect exceeds off-task effect at the same intervention strength | [external/C_selectivity.md](external/C_selectivity.md) |
| E4 | Effect magnitude | Absolute effect large enough to support the computational story | [external/D_effect-magnitude.md](external/D_effect-magnitude.md) |
| E5 | Robustness | Claim survives prompt paraphrase, cross-scale transfer, held-out generalization | [external/E_robustness.md](external/E_robustness.md) |
| E6 | Cross-architecture generalization | Mechanism appears in at least one other model family | [external/F_cross-architecture.md](external/F_cross-architecture.md) |

### Measurement validity — Is the instrument trustworthy?

| # | Criterion | One-line pass condition | Page |
|---|---|---|---|
| M1 | Reliability | Scores stable across prompt splits, seeds, and checkpoints | [measurement/A_reliability.md](measurement/A_reliability.md) |
| M2 | Invariance | Instrument gives comparable results across model sizes and families | [measurement/B_invariance.md](measurement/B_invariance.md) |
| M3 | Baseline separation | Score exceeds random-vector AND untrained-model baselines by meaningful margin | [measurement/C_baseline-separation.md](measurement/C_baseline-separation.md) |
| M4 | Sensitivity | Detects real circuits at acceptable hit rates (AUROC ≥ 0.85) without excess false positives | [measurement/D_sensitivity.md](measurement/D_sensitivity.md) |
| M5 | Calibration | Raw scores interpretable relative to known reference points | [measurement/E_calibration.md](measurement/E_calibration.md) |
| M6 | Construct coverage | Instrument measures its nominal target, not a correlated proxy | [measurement/F_construct-coverage.md](measurement/F_construct-coverage.md) |

### Interpretive validity — Does the verdict match the evidence?

| # | Criterion | One-line pass condition | Page |
|---|---|---|---|
| V1 | Level declaration | A specific description-mode tag is stated explicitly in the verdict | [interpretive/A_level-declaration.md](interpretive/A_level-declaration.md) |
| V2 | Level–evidence match | Evidence collected is sufficient to license the declared mode tag | [interpretive/B_level-evidence-match.md](interpretive/B_level-evidence-match.md) |
| V3 | Narrative coherence | Prose description is consistent with and entailed by the mode-tagged claim | [interpretive/C_narrative-coherence.md](interpretive/C_narrative-coherence.md) |
| V4 | Alternative exclusion | Competing mechanism descriptions have been considered and addressed | [interpretive/D_alternative-exclusion.md](interpretive/D_alternative-exclusion.md) |
| V5 | Scope honesty | Verdict does not silently generalize beyond the evidence scope | [interpretive/E_scope-honesty.md](interpretive/E_scope-honesty.md) |

## How to use this index

**Building a claim (bottom-up):** Identify instruments run (Layer A). For each, locate the criteria it addresses from the mapping table in [../taxonomy/](../taxonomy/). Check each criterion's pass condition. Assemble the verdict from satisfied and unsatisfied criteria.

**Auditing a claim (top-down):** Start with the verdict tier. All criteria in the required validity types must be satisfied. Check each criterion page against reported evidence. Note gaps.

**Minimum-reporting rule:** Every published claim must report, for each satisfied criterion, which instrument satisfied it and what value was obtained.
