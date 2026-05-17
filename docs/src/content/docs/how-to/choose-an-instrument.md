---
title: "Choose an Instrument"
---

# D — How to Choose an Instrument: Criterion → Instrument Lookup

---

## Internal validity instruments

| Criterion | Primary instrument(s) | Evidence family | Notes |
|---|---|---|---|
| I1 Necessity | Zero ablation, resample ablation, mean ablation | Causal | Run ≥2 methods; if all three agree, I5 partially addressed |
| I2 Sufficiency | Complement ablation (ablate all except circuit) | Causal | Alternative: activation patching into corrupted run |
| I3 Specificity | Control-axis DAS-IIA | Representational | Causal axis (target task) vs. control axis (unrelated); ratio = specificity score |
| I3 Specificity (alt) | Cross-task ablation effect | Causal | Ablate; measure target AND control task degradation; ratio = selectivity |
| I4 Consistency | Sigma-ablation (`c03sigmaablation.py`) | Measurement | 8 ablation variants; sigma = SD of metric |
| I4 Consistency (alt) | Bootstrap prompt subsampling | Measurement | 100 subsamples of 50% prompt set; CI on metric |
| I5 Confound control | Component-specific ablation | Causal | Zero only target head; compare to full-circuit ablation |
| I5 Confound control (alt) | Zero vs. resample vs. mean comparison | Causal | If all three give similar Δ, mean-field confound ruled out |

---

## External validity instruments

| Criterion | Primary instrument(s) | Evidence family | Notes |
|---|---|---|---|
| E1 Intervention reach | Activation delta logging | Causal | Measure `act[after] − act[before]` at hook; confirm direction + magnitude |
| E2 Graded response | Steering multiplier sweep (`steering multiplier sweep`) | Causal | 7+ values 0–20; plot metric vs. multiplier; identify threshold and plateau |
| E3 Selectivity | Cross-task metric ratio | Behavioral | Same intervention; on-task vs. off-task metric; ratio ≥ 2.0 |
| E4 Effect magnitude | Recovery fraction | Behavioral | `(metric_circuit − metric_zero) / (metric_full − metric_zero)` |
| E5 Robustness | Cross-prompt-family transfer | Behavioral | New prompt distribution; report IIA or faithfulness |
| E5 Robustness (alt) | Cross-scale weight transfer F1 | Structural | Weight classifier on Pythia-160M or GPT-2 Medium |
| E6 Cross-architecture | `dictionary_alignment()` | Structural | `mean_max_cos` between W_dec of circuit heads across model families |

---

## Construct validity instruments

| Criterion | Primary instrument(s) | Evidence family | Notes |
|---|---|---|---|
| C1 Falsifiability | Pre-registered threshold statement | N/A | Not an instrument — a claim structure requirement; must precede data collection |
| C2 Structural plausibility | Weight classifier, SVD of W_OV/W_QK | Structural | Cos alignment to known role-direction (e.g., low-rank copying for name-movers) |
| C3 Task specificity | Cross-task weight classifier F1 | Structural | F1 on target vs. F1 on control; target should dominate |
| C3 Task specificity (alt) | Cross-task DAS-IIA ratio | Representational | IIA on target vs. IIA on control; ratio ≥ 2.0 |
| C4 Minimality | Per-head ablation pruning | Causal | Remove one head at a time; exclude heads whose removal has no effect |
| C5 Convergent validity | Jaccard(weight-circuit, EAP-circuit) | Measurement | ≥2 instruments, different evidence families; Jaccard ≥ 0.5 = strong convergence |

---

## Measurement validity instruments

| Criterion | Primary instrument(s) | Evidence family | Notes |
|---|---|---|---|
| M1 Reliability | Bootstrap CI, seed SD | Measurement | 100 subsamples; 3 seeds; report CI and SD |
| M2 Invariance | Cross-scale F1 transfer | Structural | Classifier trained on GPT-2 Small; evaluated on Pythia-160M |
| M3 Baseline separation | Random-vector IIA, untrained-model IIA | Measurement | Both baselines required; separation = IIA_circuit − IIA_random |
| M4 Sensitivity | AUROC, AUPRC for circuit membership | Measurement | Binary: circuit head vs. non-circuit head |
| M5 Calibration | Published SOTA comparison | Measurement | See `task_reference_baselines.py`; transcoder range 0.40–0.60 for GPT-2 Small SVA |
| M6 Construct coverage | Constrained vs. unconstrained IIA | Representational | Linear constraint vs. none; compare on OOD prompts |

---

## Instrument selection by phase

| Phase | Minimum instruments | Additional if time |
|---|---|---|
| Initial sweep | Weight classifier (C2), DAS-IIA (M3, M5) | Activation patching (I1 partial) |
| First publication | I1 + I2 + M3 | I3, E1, C2 |
| Full publication | I1–I5 + M1–M3 + ≥1 external + ≥1 construct | All remaining criteria |
