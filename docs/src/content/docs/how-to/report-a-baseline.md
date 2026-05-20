---
title: "Report a Baseline"
---

# E — How to Report a Baseline

Three baselines are required for any IIA or faithfulness score to be interpretable.

---

## The three required baselines

### 1. Random-vector baseline

**What:** Run DAS-IIA with the same architecture but replace circuit factor activations with random unit vectors from the same space.

**Why non-negotiable:** In high-dimensional spaces, random vectors can produce surprisingly high IIA because the alignment map has enough degrees of freedom to fit noise. IIA(circuit) = 0.48 vs. IIA(random) = 0.44 is not a finding; the separation of 0.04 is within noise.

**How to compute:**
1. Draw 100 random unit vectors from ℝ^d_model (uniform on sphere).
2. Run DAS-IIA with each as the "circuit subspace."
3. Report: mean, SD, and 95th percentile.

**Reporting format:**
```
IIA(circuit)          = 0.48
IIA(random, mean)     = [X]   (SD = [Y], 95th pct = [Z], n = 100)
Separation            = 0.48 − [X] = [Δ]
```

---

### 2. Untrained-model baseline

**What:** Run DAS-IIA on a model with the same architecture but randomly initialized weights (no training).

**Why it matters:** Separates signal from architectural priors. If the untrained model produces IIA = 0.30, your trained model's IIA = 0.48 has a learning-attributable separation of 0.18.

**How to compute:**
1. Initialize model with same architecture + hyperparameters.
2. Run DAS-IIA on 3 random initializations with the same prompt distribution.
3. Report: mean and SD.

**Reporting format:**
```
IIA(circuit, trained)     = 0.48
IIA(untrained, mean)      = [X]   (SD = [Y], n = 3 random inits)
Learning contribution     = 0.48 − [X] = [Δ]
```

---

### 3. Published SOTA comparison

**What:** Compare to the best published baseline for the same task and model.

**Reference table (from `task_reference_baselines.py`):**

| Task | Metric | Full Model | Best Circuit | Recovery | Source |
|---|---|---|---|---|---|
| IOI | logit diff | 3.56 | 3.10 | 87% | Wang et al. 2022 |
| Greater-Than | prob diff | 81.7% | 72.7% | 89.5% | Hanna et al. 2023 |
| SVA (base) | logit diff | 0.70 | 0.65 | 93% | Lazo et al. 2025 |
| Gendered pronoun | logit diff | — | ≥ full model | 100% | Mathwin 2023 |

**IIA-specific reference values:**

| Metric | Task | GPT-2 Small range | Source |
|---|---|---|---|
| Transcoder IIA | SVA | 0.40–0.60 | Published (multiple) |
| DAS IIA | IOI | 0.86–0.95 | MIB benchmark (Mueller et al.) |
| Raw neuron IIA | IOI | 0.60–0.75 | MIB (SAE features < raw neurons) |

---

## The baseline report block

Every published IIA score must include:

```
## Baseline Report: [metric] at [component] on [task]

Observed score:             [X]
Random-vector (mean):       [Y]  (SD = [z], n = 100)
Untrained-model (mean):     [W]  (SD = [v], n = 3 inits)
Published SOTA:             [range or value] ([source])

Separation from random:     [X − Y] = [Δ_r]
Separation from untrained:  [X − W] = [Δ_u]
Relative to SOTA:           [X] is [above/within/below] the [source] range of [range]

Interpretation: [X] is [signal/noise/competitive/below SOTA]
                because Δ_r = [Δ_r] and Δ_u = [Δ_u].
```

---

## Project-specific baseline status

| Component | IIA | Random-vector | Untrained | SOTA range | Status |
|---|---|---|---|---|---|
| L8.MLP (SVA) | 0.48 | NOT YET COMPUTED | NOT YET COMPUTED | 0.40–0.60 | M3 partial — run both baselines |
