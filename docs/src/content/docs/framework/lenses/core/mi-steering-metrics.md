---
title: "MI Steering Metrics"
description: "Steering and representation editing metrics: CAA, LEACE, RepE, and cross-model transfer."
---

# MI Steering Metrics

These metrics evaluate whether artifact directions (from SAEs, transcoders, or factor banks) can steer model behavior, erase concept representations, or transfer across models. They combine causal interventions (activation addition, subspace erasure) with behavioral measurement (logit-difference shift, dose-response) to test whether learned representations are causally load-bearing. All are implemented in `mechval_v2.core.mechanistic_interpretability.methods`.

---

### C09 -- Contrastive Activation Addition (`93_caa.py`)

**What it computes.** Implements CAA (Panickssery et al., ACL 2024) as a validation metric. For each artifact feature direction, computes a steering vector and adds it to the residual stream at inference time across a range of coefficients ($-2, -1, -0.5, 0.5, 1, 2$). Measures the resulting logit-difference shift relative to baseline. Reports steerability (magnitude of behavioral shift relative to baseline), dose-response linearity (Pearson correlation between coefficient and effect), and the fraction of steerable features.

$$
\text{steerability}_f = \frac{\max_{\alpha > 0} |\text{LD}(x + \alpha \cdot d_f) - \text{LD}(x)|}{|\text{LD}(x)| + \epsilon}
$$

**Evidence family.** Causal (intervention via activation addition).

**Key metrics.**
| Metric | Description | Pass threshold |
|---|---|---|
| `steerable_fraction` | Fraction of tested directions with steerability $> 0.3$ | $> 0.20$ |
| `mean_steerability` | Mean steerability across tested features | reported |
| `dose_response_r` | Mean absolute Pearson $r$ between coefficient and behavioral shift | reported |

**What it establishes.** Artifact directions actually control model behavior when added as steering vectors. If a substantial fraction of feature directions produce graded, dose-responsive behavioral shifts, the artifact encodes causally relevant structure -- not mere correlation.

**What it does not establish.** Specificity. A direction that steers behavior may also disrupt unrelated computations. The metric does not test whether steering affects only the target behavior. Combine with concept erasure (C15) or selectivity metrics for specificity evidence.

**Usage.**
```bash
uv run python 93_caa.py --tasks ioi --n-prompts 40
```

---

### C15 -- Concept Erasure / LEACE (`99_concept_erasure.py`)

**What it computes.** Implements LEACE (Least-Squares Concept Erasure; Belrose et al., NeurIPS 2023) as a dissociation test. Given an artifact adapter's top-$k$ feature directions as a concept subspace, computes the orthogonal projection matrix via SVD:

$$
P = V_r^T V_r, \quad X_{\text{erased}} = X - X \cdot P
$$

where $V_r$ is the right singular vectors of the concept directions with non-negligible singular values. The model is re-run with erased activations via hooks, and the behavioral change is measured as the normalized reduction in logit difference.

**Evidence family.** Causal (subspace erasure intervention).

**Key metrics.**
| Metric | Description | Pass threshold |
|---|---|---|
| `dissociation_strength` | $\|LD_{\text{clean}} - LD_{\text{erased}}\| / \|LD_{\text{clean}}\|$ | $> 0.3$ |
| `erasure_kl` | KL divergence between clean and erased output distributions | reported |

**What it establishes.** The concept subspace defined by the artifact's top feature directions is load-bearing: erasing it from the residual stream destroys task performance. This is a necessity test for the artifact's feature directions -- they encode information that the model actually uses.

**What it does not establish.** That the erased directions are the unique encoding of the concept. Other directions may also encode the same information redundantly. LEACE erases a subspace, not individual features, so the dissociation may reflect removal of multiple distinct computations that happen to share direction.

**Usage.**
```bash
uv run python 99_concept_erasure.py --tasks ioi --n-prompts 40
```

---

### C16 -- Representation Engineering / RepE (`100_representation_engineering.py`)

**What it computes.** Implements RepE (Zou et al., 2023), generalizing CAA by using PCA on the difference of positive/negative activation distributions to discover multi-component concept directions. For each task: collects residual-stream activations, splits prompts into positive/negative by median logit-diff, computes a paired contrast matrix, and applies PCA. Then steers the model with cumulative PCA components (1, then 1+2, ...) and measures the logit-diff shift at each level.

**Evidence family.** Causal (PCA-based concept direction discovery + steering).

**Key metrics.**
| Metric | Description | Pass threshold |
|---|---|---|
| `concept_dimensionality` | Number of PCA components to reach 90% of max cumulative steering effect | $\leq 5$ |
| `steerability` | Max cumulative steering effect relative to baseline logit-diff | reported |
| `artifact_cosine_similarity` | Max absolute cosine similarity between PC1 and artifact directions | reported (if artifact provided) |

**What it establishes.** Task-relevant concepts occupy low-dimensional subspaces in the residual stream. Low concept dimensionality ($\leq 5$) indicates the concept is compactly represented. When an artifact adapter is provided, the cosine similarity between the discovered PCA directions and artifact directions provides convergent validity.

**What it does not establish.** That the discovered directions are causally specific to the target concept. PCA captures the largest variance direction in the contrast, which may conflate the target concept with confounded features. Unlike DAS (which optimizes for causal intervention accuracy), RepE is purely observational at the discovery stage.

**Usage.**
```bash
uv run python 100_representation_engineering.py --tasks ioi --n-prompts 40
```

---

### B21 -- Steering-Bench Reliability (`102_steering_reliability.py`)

**What it computes.** Implements the Steering-Bench decomposition (Tan et al., NeurIPS 2024). The key insight is that raw steerability conflates baseline model propensity with genuine causal effect. Decomposes steering evaluation into: (1) propensity -- $P(\text{correct})$ without steering; (2) raw steerability -- change in $P(\text{correct})$ when adding the artifact direction; (3) propensity-corrected steerability:

$$
\text{corrected} = \frac{\text{raw steerability}}{1 - \text{propensity}}
$$

which corrects for ceiling effects. Also measures dose-response linearity ($R^2$ of coefficient vs effect).

**Evidence family.** Behavioral (propensity-corrected steering).

**Key metrics.**
| Metric | Description | Pass threshold |
|---|---|---|
| `corrected_steerability` | Propensity-corrected steerability at optimal coefficient | $> 0.15$ |
| `dose_response_r2` | $R^2$ of linear fit between steering coefficient and behavioral effect | reported |
| `propensity` | Baseline $P(\text{correct})$ without steering | reported |

**What it establishes.** The artifact direction produces genuine behavioral change beyond what the model's baseline propensity would predict. Propensity correction prevents the artifact from receiving credit for behavior the model already exhibits without intervention.

**What it does not establish.** That the steering direction is the correct causal variable. A direction that passes propensity correction may still be a confounded proxy for the true mechanism. The metric quantifies the magnitude of causal effect but not its specificity.

**Usage.**
```bash
uv run python 102_steering_reliability.py --tasks ioi --n-prompts 30
```

---

### EX15 -- Cross-Model Steering Transfer (`111_cross_model_transfer.py`)

**What it computes.** Implements cross-model steering transfer (Oozeer et al., ICML 2025). For a source model A and target model B: (1) collects paired activations from both models on the same prompts at corresponding layers; (2) learns a linear mapping $M: \mathbb{R}^{d_A} \to \mathbb{R}^{d_B}$ via least-squares regression; (3) extracts a steering vector $v_A$ from model A via mean-difference between positive/negative concept prompts; (4) transfers it: $v_B = M \cdot v_A$; (5) measures whether the transferred vector produces behavioral effects correlated with the native vector's effects across multiple steering coefficients.

**Evidence family.** External validity (cross-architecture generalization).

**Key metrics.**
| Metric | Description | Pass threshold |
|---|---|---|
| `transfer_fidelity` | Pearson correlation between transferred and native steering effects across coefficients | $> 0.3$ |
| `cosine_similarity` | Cosine similarity between transferred and natively-extracted steering vectors | reported |
| `mapping_r2` | $R^2$ of the learned linear mapping on held-out activations | reported |

**What it establishes.** The concept encoded by the steering vector is genuinely represented in both models' activation spaces -- not a model-specific artifact. If a steering vector transfers across architectures via a simple linear map, the underlying concept representation is shared and likely reflects a general computational strategy.

**What it does not establish.** That the transferred vector produces the same mechanistic effect. Two models may represent the same concept but implement it through different circuits. Transfer fidelity measures behavioral agreement, not mechanistic agreement.

**Usage.**
```bash
uv run python 111_cross_model_transfer.py --source-model gpt2 --target-model gpt2-medium --tasks ioi
```
