# Criteria Quick-Reference Checklist

Use this when auditing a circuit claim or designing experiments.

## Construct Validity — Is the claim well-defined?

- [ ] **C1 Falsifiability** — Named disconfirmation condition with threshold stated in advance
- [ ] **C2 Structural Plausibility** — Components at predicted layers, weight-space signatures consistent
- [ ] **C3 Task Specificity** — On-task:off-task ratio >= 2:1
- [ ] **C4 Minimality** — No redundant members (removing any degrades performance)
- [ ] **C5 Convergent Validity** — Multiple instruments from different evidence families agree (Jaccard >= 0.5)

## Internal Validity — Did the manipulation cause the effect?

- [ ] **I1 Necessity** — Ablating component degrades behavior across >= 2 ablation methods
- [ ] **I2 Sufficiency** — Isolating/restoring component reproduces behavior (>= 70% recovery)
- [ ] **I3 Specificity** — Effect selective; control-axis IIA ~ 0 while causal-axis IIA high
- [ ] **I4 Consistency** — Finding holds across prompt samples, methods, seeds (sigma <= 0.05)
- [ ] **I5 Confound Control** — Effect not explained by collateral disruption

## External Validity — Does the claim generalize?

- [ ] **E1 Intervention Reach** — Activation delta at hook point non-trivial (delta > 0.01)
- [ ] **E2 Graded Response** — Monotonic dose-response with threshold and plateau (>= 7 values)
- [ ] **E3 Selectivity** — On-task > off-task (selectivity ratio >= 2.0)
- [ ] **E4 Effect Magnitude** — Recovery fraction >= 0.10 for primary claims
- [ ] **E5 Robustness** — Survives prompt paraphrase, cross-scale, or held-out generalization
- [ ] **E6 Cross-Architecture** — Mechanism in >= 1 other model family

## Measurement Validity — Is the instrument trustworthy?

- [ ] **M1 Reliability** — Stable across splits (CI width <= 0.05), seeds (SD <= 0.02)
- [ ] **M2 Invariance** — Comparable results across model sizes/families
- [ ] **M3 Baseline Separation** — Exceeds random-vector AND untrained-model baselines by >= 0.10
- [ ] **M4 Sensitivity** — AUROC >= 0.85; false positives controlled (AUPRC above random)
- [ ] **M5 Calibration** — Raw scores interpretable relative to published baselines
- [ ] **M6 Construct Coverage** — Instrument measures target, not proxy

## Interpretive Validity — Does the verdict match the evidence?

- [ ] **V1 Level Declaration** — Description-mode tag stated explicitly
- [ ] **V2 Level-Evidence Match** — Evidence sufficient to license declared mode tag
- [ ] **V3 Narrative Coherence** — Prose consistent with mode-tagged claim
- [ ] **V4 Alternative Exclusion** — Competing mechanisms considered and addressed
- [ ] **V5 Scope Honesty** — Doesn't silently generalize beyond evidence scope
