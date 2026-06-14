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
| C1 | Falsifiability | A named result stated in advance would disconfirm the claim | [falsifiability](construct/falsifiability/) |
| C2 | Structural plausibility | Components at predicted layers/positions with consistent weight-space signatures | [structural-plausibility](construct/structural-plausibility/) |
| C3 | Task specificity | Circuit does not score highly on unrelated tasks under same metric | [task-specificity](construct/task-specificity/) |
| C4 | Minimality | No redundant members; removing any member degrades performance | [minimality](construct/minimality/) |
| C5 | Convergent validity | Multiple independent metrics nominate the same components | [convergent-validity](construct/convergent-validity/) |

### Internal validity — Did the manipulation cause the effect?

| # | Criterion | One-line pass condition | Page |
|---|---|---|---|
| I1 | Necessity | Ablating the component reliably degrades the behavior across ≥2 methods | [necessity](internal/necessity/) |
| I2 | Sufficiency | Isolating/restoring the component reproduces the behavior | [sufficiency](internal/sufficiency/) |
| I3 | Specificity | Effect is selective; control-axis IIA ≈ 0 while causal-axis IIA is high | [specificity](internal/specificity/) |
| I4 | Consistency | Finding holds across prompt samples, ablation methods, and random seeds | [consistency](internal/consistency/) |
| I5 | Confound control | Effect not explained by collateral disruption to non-circuit components | [confound-control](internal/confound-control/) |
| I6 | Rival mechanism exclusion | No alternative component set achieves comparable faithfulness, or rivals declared and claim scoped | [rival-mechanism-exclusion](internal/rival-mechanism-exclusion/) |

### External validity — Does the claim generalize?

| # | Criterion | One-line pass condition | Page |
|---|---|---|---|
| E1 | Intervention reach | Activation delta at hook point is in predicted direction and non-trivial | [intervention-reach](external/intervention-reach/) |
| E2 | Graded response | Effect scales monotonically with intervention strength; threshold and plateau visible | [graded-response](external/graded-response/) |
| E3 | Selectivity | On-task effect exceeds off-task effect at the same intervention strength | [selectivity](external/selectivity/) |
| E4 | Effect magnitude | Absolute effect large enough to support the computational story | [effect-magnitude](external/effect-magnitude/) |
| E5 | Robustness | Claim survives prompt paraphrase, cross-scale transfer, held-out generalization | [robustness](external/robustness/) |
| E6 | Cross-architecture generalization | Mechanism appears in at least one other model family | [cross-architecture](external/cross-architecture/) |

### Measurement validity — Is the metric trustworthy?

| # | Criterion | One-line pass condition | Page |
|---|---|---|---|
| M1 | Reliability | Scores stable across prompt splits, seeds, and checkpoints | [reliability](measurement/reliability/) |
| M2 | Invariance | Metric gives comparable results across model sizes and families | [invariance](measurement/invariance/) |
| M3 | Baseline separation | Score exceeds random-vector AND untrained-model baselines by meaningful margin | [baseline-separation](measurement/baseline-separation/) |
| M4 | Sensitivity | Detects real circuits at acceptable hit rates (AUROC ≥ 0.85) without excess false positives | [sensitivity](measurement/sensitivity/) |
| M5 | Calibration | Raw scores interpretable relative to known reference points | [calibration](measurement/calibration/) |
| M6 | Construct coverage | Metric measures its nominal target, not a correlated proxy | [construct-coverage](measurement/construct-coverage/) |

### Interpretive validity — Does the verdict match the evidence?

| # | Criterion | One-line pass condition | Page |
|---|---|---|---|
| V1 | Level declaration | A specific description-mode tag is stated explicitly in the verdict | [level-declaration](interpretive/level-declaration/) |
| V2 | Level–evidence match | Evidence collected is sufficient to license the declared mode tag | [level-evidence-match](interpretive/level-evidence-match/) |
| V3 | Narrative coherence | Prose description is consistent with and entailed by the mode-tagged claim | [narrative-coherence](interpretive/narrative-coherence/) |
| V4 | Alternative exclusion | Competing mechanism descriptions have been considered and addressed | [alternative-exclusion](interpretive/alternative-exclusion/) |
| V5 | Scope honesty | Verdict does not silently generalize beyond the evidence scope | [scope-honesty](interpretive/scope-honesty/) |

## How to use this index

**Building a claim (bottom-up):** Identify metrics run (Layer A). For each, locate the criteria it addresses from the mapping table in [../taxonomy/](../taxonomy/). Check each criterion's pass condition. Assemble the verdict from satisfied and unsatisfied criteria.

**Auditing a claim (top-down):** Start with the verdict tier. All criteria in the required validity types must be satisfied. Check each criterion page against reported evidence. Note gaps.

**Minimum-reporting rule:** Every published claim must report, for each satisfied criterion, which metric satisfied it and what value was obtained.
