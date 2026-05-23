---
title: "Measurement Theory -- Metrics & Protocols"
description: "Reference for measurement theory metrics and protocols: reliability, SAE-specific diagnostics, psychometric extensions, safety subspace analysis, and faithfulness curves."
---

# Measurement Theory -- Metrics & Protocols

This page documents the extended metrics under the [Measurement Theory lens](/framework/lenses/core/measurement-theory). These metrics go beyond the original F01--F08 suite (documented at their [existing pages](/framework/metrics/measurement/f01-bootstrap-stability)) to cover SAE-specific validity diagnostics, psychometric extensions from cognitive science, safety representation analysis, and benchmark meta-diagnostics.

All metrics in this page follow the same principle as the core measurement theory lens: **is the metric that produced the number trustworthy?** Some operate at the decomposition level (is the SAE itself a valid instrument?), some at the evaluation level (are our benchmarks reliable?), and some import constructs from psychophysics and psychometrics to formalize properties that MI evaluates informally.

---

## Reliability and Stability Metrics

These metrics test whether the decomposition and the evaluation pipeline produce stable, reproducible results.

### M09 -- DMSAE Core Stability

**Source:** Martin-Linares & Ling (2025). arXiv:2512.24975.

**Criteria:** M1 Reliability

**What it establishes:** Which SAE features are reliably recovered across iterative distillation cycles. Only a small fraction of features form a stable "core" -- the paper found 197 out of 65,000 features in a 65k SAE are stable. This provides a reliability diagnostic for the decomposition itself: features outside the core may be fitting noise or arbitrary local optima rather than stable structure.

**What it does not establish:** Whether the stable core features are interpretable, causally important, or correspond to "true" features. Core membership is a necessary condition for reliability but not sufficient for validity.

**Method:**

1. Train a small SAE on activations at a hook point.
2. Identify high gradient-times-activation features, mark as "core" (top fraction by importance).
3. Reinitialize non-core features, retrain.
4. After $n$ cycles, record which features converged into the stable core.
5. Core membership rate = reliability score.

**Key quantities:**

- `core_fraction` -- fraction of features in the stable core after all cycles
- `mean_overlap` -- mean Jaccard overlap of core sets between consecutive cycles
- `is_stable` -- whether mean overlap exceeds the stability threshold (default 0.8)

**Pass condition:** Report-only (diagnostic). Any nonzero core fraction is informative.

**Usage:**

```bash
uv run python 115_core_stability.py --model gpt2 --device cpu
uv run python 115_core_stability.py --hook blocks.6.hook_resid_pre --n-cycles 5
```

**Reading the scores:**

| Pattern | What it means |
|---|---|
| core_fraction > 0.05 | Reasonable core -- most features are unstable but a meaningful subset persists |
| core_fraction < 0.01 | Very small core -- the decomposition is highly sensitive to initialization |
| mean_overlap > 0.8 | Core membership converges -- distillation is reaching a fixed point |
| mean_overlap < 0.5 | Core membership drifts -- even "important" features change across cycles |

---

### EX25 -- Reproducibility Check

**Source:** Bai, Baumgartner, Sun, Holtzman, Tan (2026). "The Story is Not the Science: Execution-Grounded Evaluation of Mechanistic Interpretability Research." arXiv:2602.18458.

**Criteria:** M1 Reliability (test-retest)

**What it establishes:** Whether a metric computation pipeline produces reproducible results across runs with different random seeds. Inspired by MechEvalAgent's finding that 93% of MI research outputs fail reproducibility when code is actually executed.

**What it does not establish:** Whether the metric measures the right thing -- only that it measures the same thing each time. A perfectly reproducible metric can still be invalid if it measures a confound.

**Method:**

1. Select a base metric (logit-diff, probe accuracy, or ablation recovery).
2. Run it $N$ times on the same model and prompts with different random seeds (controlling subsample selection and ordering).
3. Compute:
   - `deviation_rate` -- fraction of run pairs differing by more than the max-deviation threshold
   - `max_deviation` -- largest relative deviation from the mean across runs
   - `coherence_score` -- mean pairwise Spearman rank correlation between per-prompt rankings across runs

**Pass condition:**

- `deviation_rate` < 0.05
- `max_deviation` < 0.08
- `coherence_score` > 0.9

**Usage:**

```bash
uv run python 132_reproducibility.py --model gpt2 --device cpu
uv run python 132_reproducibility.py --n-runs 10 --n-prompts 50
```

**Reading the scores:**

| Pattern | What it means |
|---|---|
| Low deviation, high coherence | Pipeline is reproducible; results can be trusted |
| High deviation, high coherence | Rankings are stable but absolute values shift -- report rankings, not point estimates |
| High deviation, low coherence | Pipeline is unreliable; results should not be interpreted |

---

### EX24 -- SAEBench Reliability Audit

**Source:** Anonymous (2026). "Are Sparse Autoencoder Benchmarks Reliable?" arXiv:2605.18229.

**Criteria:** M1 Reliability, M2 Measurement Invariance

**What it establishes:** Whether evaluation metrics used for SAE comparison are themselves reliable (low reseed noise) and discriminative (can distinguish meaningfully different SAEs). The SAEBench audit independently found that TPP and SCR fail comprehensively (CV of 16--39%) while sae-probes is most reliable.

**What it does not establish:** Whether any particular SAE is good -- only whether the metrics used to evaluate SAEs produce stable, discriminating numbers.

**Method:**

1. Select an evaluation metric (e.g., probe accuracy, logit-diff recovery).
2. Run the metric $N$ times on the same model and prompts with different random seeds.
3. Compute coefficient of variation: $\text{CV} = \sigma / |\mu|$.
4. Compute discriminability: run metric on two configurations differing by a known quality dimension, compute Cohen's $d$.

**Pass condition:**

- CV < 0.05
- Discriminability $d$ > 0.8

**Usage:**

```bash
uv run python 131_saebench_audit.py --model gpt2 --device cpu
uv run python 131_saebench_audit.py --n-reseeds 10 --n-prompts 50
```

**Reading the scores:**

| Pattern | What it means |
|---|---|
| Low CV, high discriminability | Metric is both stable and sensitive -- suitable for SAE comparison |
| Low CV, low discriminability | Metric is stable but cannot distinguish quality differences -- not useful for comparison |
| High CV, any discriminability | Metric is noisy -- differences between SAEs may reflect measurement noise |

---

## SAE-Specific Validity Diagnostics

These metrics test whether the SAE decomposition itself is a valid measurement instrument, independent of any downstream circuit claim.

### M07 -- Architecture Duality

**Source:** Lindsey et al. (2025). NeurIPS 2025.

**Criteria:** M2 Hyperparameter Sensitivity, M6 Artifact Quality

**What it establishes:** Whether two different SAE architectures trained on the same model and hook point agree on what features exist. This is a construct validity test for the decomposition method itself: if TopK-SAE and JumpReLU-SAE discover completely different features, the "features" are partly determined by the architecture rather than being properties of the model.

**What it does not establish:** Which architecture's features are "correct" -- the metric measures agreement, not accuracy. High agreement is necessary for construct validity but two architectures could agree on an artifact.

**Method:**

1. Collect activations at a shared hook point from the model.
2. Encode activations through both artifact adapters.
3. Compute `feature_overlap`: Jaccard similarity of active feature sets at a threshold.
4. Compute `direction_agreement`: mean max cosine similarity between encoder directions of the two artifacts (symmetric: A-to-B and B-to-A averaged).
5. `architecture_agreement` = mean(feature_overlap, direction_agreement).

**Pass condition:** `architecture_agreement` > 0.3

**Usage:**

```bash
uv run python 110_architecture_duality.py \
    --artifact-a-path <release_a> --artifact-b-path <release_b>
uv run python 110_architecture_duality.py --device cpu
```

**Reading the scores:**

| Pattern | What it means |
|---|---|
| Agreement > 0.5 | Architectures substantially agree -- features reflect model structure more than architecture choice |
| Agreement 0.3--0.5 | Partial agreement -- some features are robust but many are architecture-dependent |
| Agreement < 0.3 | Low agreement -- the decomposition is largely determined by architecture, not model structure |

---

### M08 -- WeightLens Convergence

**Source:** Golimblevskaia, Jain, Puri, Ibrahim, Samek, Lapuschkin (2026). ICLR 2026. arXiv:2510.14936.

**Criteria:** C5 Convergent Validity

**What it establishes:** Whether weight-based and activation-based feature descriptions agree. A feature's structural identity (what it promotes in logit space via `W_dec @ W_U`) should match its functional identity (what inputs it fires on). Divergence means the feature's "meaning" depends on whether you look at its weights or its activations -- a construct validity failure.

**What it does not establish:** Whether either description is "correct" in isolation. The metric tests convergence between two independent characterization methods, not ground truth.

**Method:**

1. Compute weight-based descriptions: for each feature, project its decoder direction through the model's unembedding (`W_dec @ W_U`) to get top-$k$ promoted tokens.
2. Compute activation-based descriptions: run prompts through the model, encode at the hook point, and for each feature track which tokens produce the highest activations.
3. Measure agreement: Jaccard overlap of the two top-$k$ token sets, averaged over features.

**Pass condition:** `weight_activation_agreement` > 0.3

**Usage:**

```bash
uv run python 114_weightlens.py --artifact-path <release> --sae-id <id>
uv run python 114_weightlens.py --device cpu --top-k 50
```

**Reading the scores:**

| Pattern | What it means |
|---|---|
| Agreement > 0.5 | Strong weight-activation convergence -- feature identity is robust to description method |
| Agreement 0.3--0.5 | Moderate convergence -- some features have consistent identity, others diverge |
| Agreement < 0.3 | Low convergence -- weight-based and activation-based descriptions measure different constructs |
| High frac_above_threshold | Most active features individually converge, even if the mean is pulled down by dead features |

---

### M10 -- PRISM Polysemanticity Score

**Source:** Kopf, Feldhus, Bykov, Bommer, Hedstrom, Hohne, Eberle (2025). NeurIPS 2025. arXiv:2506.15538.

**Criteria:** M6 Artifact Quality, E1 Predictive Validity

**What it establishes:** What fraction of SAE features are polysemantic -- activating on multiple semantically distinct clusters of contexts. Standard autointerp pipelines are architecturally incapable of reliably describing polysemantic features (they assign a single label), so the polysemanticity rate directly bounds the fraction of features whose automated descriptions can be trusted.

**What it does not establish:** Whether polysemantic features are "bad" -- some may represent genuine multifaceted concepts. The metric quantifies polysemanticity, not whether it is a problem.

**Method:**

1. Collect feature activations across prompts via the artifact adapter.
2. For each sampled feature, find the top-activating contexts.
3. Embed those contexts using the model's residual stream (mean-pooled token embeddings).
4. Compute pairwise cosine similarity among context embeddings.
5. Apply agglomerative clustering with a cosine distance threshold (default 0.5).
6. A feature is polysemantic if it has > 1 cluster.
7. `polysemanticity_rate` = fraction of sampled alive features that are polysemantic.

**Pass condition:** Report-only (diagnostic). `polysemanticity_rate >= 0` trivially passes.

**Usage:**

```bash
uv run python 117_prism.py --artifact-path <release> --sae-id <id>
uv run python 117_prism.py --device cpu --n-features 100 --cluster-threshold 0.5
```

**Reading the scores:**

| Pattern | What it means |
|---|---|
| Rate < 0.1 | Most features are monosemantic -- autointerp descriptions likely reliable |
| Rate 0.1--0.4 | Moderate polysemanticity -- autointerp descriptions should be cross-checked |
| Rate > 0.4 | High polysemanticity -- single-label descriptions unreliable for most features |
| Many dead features | The SAE has unused capacity; polysemanticity rate computed only over alive features |

---

### M11 -- Matryoshka Cross-Scale Consistency

**Source:** arXiv:2503.17547 (NeurIPS 2025).

**Criteria:** M1 Reliability, M2 Hyperparameter Sensitivity

**What it establishes:** Whether features at SAE dictionary width $k$ correspond to coherent feature clusters at width $2k$. This is a measurement consistency check across SAE scales: a feature that exists at width 16k should either remain as-is or cleanly split into semantically related sub-features at width 32k. Incoherent splitting or many-to-one absorption are reliability failures.

**What it does not establish:** The "correct" dictionary width. The metric tests consistency between scales, not which scale is optimal.

**Method:**

1. Collect activations at a shared hook point from the model.
2. Encode through both artifact adapters (small and large dictionary).
3. Compute per-feature correspondence via activation correlation.
4. `splitting_rate`: fraction of small features whose top-$k$ correlated large features have low pairwise cosine similarity (incoherent cluster).
5. `absorption_rate`: fraction of large features that are the top match for multiple small features (many-to-one collapse).
6. `cross_scale_consistency` = $1 - (\text{splitting\_rate} + \text{absorption\_rate}) / 2$

**Pass condition:** `cross_scale_consistency` > 0.7

**Usage:**

```bash
uv run python 118_matryoshka.py \
    --artifact-small-path <release_small> --artifact-large-path <release_large>
uv run python 118_matryoshka.py --device cpu --top-k 20
```

**Reading the scores:**

| Pattern | What it means |
|---|---|
| Consistency > 0.7 | Features are stable across scales -- dictionary width is not distorting the decomposition |
| High splitting, low absorption | Small features break into unrelated pieces at larger width -- small dictionary over-compresses |
| Low splitting, high absorption | Large dictionary collapses distinct small features -- large dictionary under-differentiates |
| Both rates high | Decomposition is fundamentally unstable across scales |

---

### M12 -- Adaptive Sparsity Diagnostic

**Source:** Convergent evidence from three papers: Bussmann, Leask, Nanda (NeurIPS 2024, BatchTopK); Yao & Du (arXiv:2508.17320, AdaptiveK); SoftSAE (arXiv:2605.06610).

**Criteria:** E1 Content Validity, M6 Artifact Quality

**What it establishes:** Whether fixed-$k$ SAE sparsity matches input complexity. Fixed-$k$ architectures activate exactly $k$ features per input regardless of the input's actual complexity. For simple inputs, this means spurious features are activated to fill the quota; for complex inputs, real concepts are truncated. Three independent papers converge on this as a systematic content validity failure.

**What it does not establish:** Whether adaptive-$k$ architectures solve the problem -- only that fixed-$k$ exhibits systematic mismatch. The metric diagnoses the problem without prescribing a solution.

**Method:**

1. Collect activations at the hook point from the model.
2. Encode through the artifact adapter to get active feature counts per position.
3. Estimate input complexity via residual stream embedding norm (L2 norm as proxy for information content).
4. Fit a linear relationship: $\text{expected\_k} \sim \text{complexity}$.
5. Flag examples where $|\text{k\_active} - \text{k\_expected}| / \text{k\_expected} > \text{threshold}$.
6. `k_mismatch_rate` = fraction of flagged examples.

**Key quantities:**

- `k_mismatch_rate` -- fraction of inputs where active count deviates from expected
- `complexity_k_correlation` -- Pearson correlation between input complexity and active feature count (high = SAE adapts naturally; low = fixed behavior)
- `over_activation_rate` -- fraction with spurious features
- `under_activation_rate` -- fraction with truncated concepts

**Pass condition:** `k_mismatch_rate` < 0.2

**Usage:**

```bash
uv run python 120_adaptive_sparsity.py --artifact-path <release> --sae-id <id>
uv run python 120_adaptive_sparsity.py --device cpu --mismatch-threshold 2.0
```

---

### M13 -- Superposition Regime Diagnostic

**Source:** Liu, Liu, Gore (2025). "Superposition Yields Robust Neural Scaling." NeurIPS 2025 Oral, Best Paper Runner-Up. arXiv:2505.10465.

**Criteria:** M6 Construct Coverage

**What it establishes:** Whether a model layer operates in the weak or strong superposition regime. In the strong regime (packing ratio >> 1), models pack more features than dimensions with irreducible interference, meaning no decomposition -- SAE or otherwise -- can recover unique "true features." They are one of many valid decompositions. In the weak regime, feature recovery is feasible.

**What it does not establish:** Whether any particular SAE's features are valid -- only the theoretical upper bound on what recovery is possible. A model in strong superposition may still have useful (but non-unique) decompositions.

**Method:**

1. Run model on diverse text, capturing residual stream at each layer.
2. Compute effective rank via participation ratio of singular values:
$$\text{PR} = \frac{(\sum s_i^2)^2}{\sum s_i^4}$$
3. Packing ratio = effective_rank / $d_{\text{model}}$. Values >> 1 indicate strong superposition.
4. Interference score = mean absolute pairwise cosine similarity between top-$k$ principal components.
5. Classify regime per layer: weak (packing $\leq$ 0.8, low interference), transition (0.8 < packing $\leq$ 1.2), strong (packing > 1.2 or high interference).

**Pass condition:** Diagnostic (no pass/fail). Reports regime classification and quantitative indicators per layer plus aggregate.

**Usage:**

```bash
uv run python 126_superposition_regime.py --model gpt2 --device cpu
uv run python 126_superposition_regime.py --n-samples 200
```

**Reading the scores:**

| Pattern | What it means |
|---|---|
| All layers weak | Feature recovery is feasible -- SAE decomposition can in principle find unique features |
| Mixed weak/strong | Early layers typically weak, later layers stronger -- validity claims should be qualified by layer |
| All layers strong | Model packs more features than dimensions -- any decomposition is one of many valid ones; claims about "the true features" are not licensed |

---

### EX34 -- NLA-SAE Convergent Validity

**Source:** Derived from Anthropic (2026). "Natural Language Autoencoders." transformer-circuits.pub/2026/nla/, cross-referenced with SAE-based feature descriptions via SAELens.

**Criteria:** C5 Convergent Validity

**What it establishes:** Whether two independent feature description methods -- NLA-style (activation-based pattern reconstruction via PCA) and SAE-style (weight-based decoder direction projected through unembedding) -- converge on the same feature characterization. This is a multitrait-multimethod (MTMM) test: agreement between two independent methods is C5 Convergent Validity evidence; divergence indicates the feature's meaning is method-dependent.

**What it does not establish:** Which method is "correct." Like all convergent validity tests, it measures inter-method agreement, not ground truth.

**Method:**

1. For each feature direction at a hook point, compute two independent characterizations:
   - **Activation-based (NLA proxy):** identify top-$k$ activating tokens, compute PCA direction from their activation patterns.
   - **Weight-based (SAE proxy):** project the feature direction through the unembedding matrix to get top promoted/suppressed tokens.
2. Compute agreement:
   - `token_overlap`: Jaccard similarity of top promoted tokens.
   - `direction_cosine`: cosine similarity between the PCA-reconstructed direction and the original feature direction.

**Pass condition:** `mean_token_overlap` > 0.3; `mean_direction_cosine` > 0.5

**Usage:**

```bash
uv run python 133_nla_sae_convergence.py --model gpt2 --device cpu
uv run python 133_nla_sae_convergence.py --n-features 30 --top-k 20
```

---

## Psychometric Extensions

These metrics import established constructs from psychophysics and psychometrics to formalize properties that MI evaluates informally.

### EX1 -- d-prime (Signal Detection Theory)

**Source:** Green & Swets (1966), "Signal Detection Theory and Psychophysics"; Macmillan & Creelman (2005), "Detection Theory: A User's Guide."

**Criteria:** Signal Detection, Causal

**What it establishes:** Separates a circuit's sensitivity ($d'$) from its criterion ($\beta$). Standard circuit evaluations (ablation accuracy, logit-diff) conflate these: a circuit might have high sensitivity but conservative criterion (it CAN detect the pattern but only fires when very confident), or vice versa. $d'$ isolates pure discriminability.

**What it does not establish:** Whether the circuit is the unique mechanism for the task. High $d'$ means the circuit discriminates signal from noise; it does not mean other components cannot also discriminate.

**Method:**

1. Run model on task prompts with full circuit: count hits (correct predictions where logit_diff > 0).
2. Mean-ablate all circuit heads: count "false alarms" (still correct despite circuit removal).
3. Compute:

$$d' = Z(\text{hit\_rate}) - Z(\text{false\_alarm\_rate})$$

where $Z$ is the inverse normal CDF.

4. Compute criterion: $\beta = -0.5 \times (Z(\text{hit\_rate}) + Z(\text{false\_alarm\_rate}))$
5. Compute AUC from an ROC curve by sweeping the logit-diff threshold.

**Pass condition:** $d'$ > 1.0 (meaningful discrimination above chance) AND AUC > 0.7.

**Usage:**

```bash
uv run python EX1_dprime.py --tasks ioi --n-prompts 40
uv run python EX1_dprime.py --device cpu
```

**Reading the scores:**

| Pattern | What it means |
|---|---|
| $d'$ > 2.0, high AUC | Strong discriminability -- the circuit is a reliable detector |
| $d'$ 1.0--2.0 | Moderate discriminability -- circuit contributes but does not dominate |
| $d'$ < 1.0 | Weak discriminability -- circuit barely distinguishes signal from noise |
| High $d'$, negative $\beta$ | Sensitive but liberal criterion -- circuit fires broadly |
| High $d'$, positive $\beta$ | Sensitive but conservative criterion -- circuit fires selectively |

---

### EX2 -- Differential Item Functioning (DIF)

**Source:** Holland & Wainer (1993), "Differential Item Functioning"; Zumbo (1999), "A Handbook on the Theory and Methods of DIF."

**Criteria:** Behavioral, Measurement Equivalence

**What it establishes:** Whether the circuit performs equivalently across different name types (common, uncommon, diverse-origin names), controlling for overall circuit ability. If the IOI circuit performs differently on "John and Mary" versus "Hiroshi and Priya" at matched model confidence, the measurement is confounded with token frequency or cultural associations -- a measurement bias, not a circuit property.

**What it does not establish:** Whether the bias is "in the circuit" or "in the model." DIF detects measurement non-equivalence; disentangling the source requires further intervention.

**Method:**

1. Generate prompts with three name categories:
   - Common English names (John, Mary, James, ...)
   - Less common names (Nigel, Mabel, Rupert, ...)
   - Names from different linguistic origins (Hiroshi, Priya, Oluwaseun, ...)
2. Run the circuit on each category, compute logit-diff for each prompt.
3. For each category pair, compute Cohen's $d$:

$$d = \frac{\bar{X}_A - \bar{X}_B}{s_{\text{pooled}}}$$

4. DIF magnitude = max $|d|$ across all group pairs.

**Pass condition:** Cohen's $d$ < 0.5 across all group pairs.

**Usage:**

```bash
uv run python EX2_dif.py --tasks ioi --n-prompts 40
uv run python EX2_dif.py --device cpu
```

**Reading the scores:**

| Pattern | What it means |
|---|---|
| Max $d$ < 0.2 | Negligible DIF -- circuit measures syntax, not token frequency |
| Max $d$ 0.2--0.5 | Small-to-medium DIF -- some confound with name type |
| Max $d$ > 0.5 | Large DIF -- circuit performance is substantially confounded with name familiarity |

---

### EX11 -- Weber-Fechner / JND (Just-Noticeable Difference)

**Source:** Weber (1834); Fechner (1860), "Elemente der Psychophysik"; Gescheider (1997), "Psychophysics: The Fundamentals."

**Criteria:** Behavioral, Construct Validity

**What it establishes:** Whether circuit heads follow Weber's law: the just-noticeable difference (JND) in output is proportional to the stimulus intensity, yielding a constant Weber fraction. This tests whether the circuit's response follows a principled input-output relationship (logarithmic scaling) rather than arbitrary nonlinearities.

**What it does not establish:** Whether Weber's law is the "correct" response function -- only whether the circuit's behavior is consistent with this well-characterized psychophysical pattern.

**Method:**

1. For each circuit head, scale its output by factors [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 1.0].
2. Find the JND: smallest scale change from 1.0 that produces a detectable output change (logit-diff drops by > 5% of baseline).
3. Test at two baseline levels (full and reduced by 0.5).
4. Weber fraction = JND / baseline_scale.
5. Weber consistency = $1 - \sigma(\text{weber\_fractions}) / \mu(\text{weber\_fractions})$.

**Pass condition:** All circuit heads have detectable JND (all heads contribute measurably).

**Usage:**

```bash
uv run python EX11_weber_fechner.py --tasks ioi --n-prompts 40
uv run python EX11_weber_fechner.py --device cpu
```

**Reading the scores:**

| Pattern | What it means |
|---|---|
| Weber consistency > 0.8 | Circuit follows Weber's law -- response scales predictably with stimulus |
| Weber consistency 0.5--0.8 | Partial Weber compliance -- some heads follow the law, others do not |
| Not all heads detectable | Some circuit heads have no measurable contribution -- they may be false positives in the circuit definition |
| Different JNDs across baselines | Head sensitivity changes nonlinearly with overall activation level |

---

## Safety Metrics

### M14 -- Safety Singular Value Entropy

**Source:** Anonymous (2026). "Safety Alignment for Large Language Models through Low-Rank Safety Subspace Fusion." arXiv:2602.00038.

**Criteria:** M1 Reliability, M6 Construct Coverage

**What it establishes:** How densely safety information is packed across a model's layers. Low SVE means safety occupies a compact, low-rank subspace; high SVE means safety information is diffusely spread. The LSSF paper shows safety subspaces are stable under fine-tuning, providing M1 Reliability evidence for safety-related representations.

**What it does not establish:** Whether the safety subspace is "correct" or complete. The metric measures compactness and stability, not the content of the safety representation.

**Method:**

1. Compute safety contrast directions at each layer: mean(safe prompts) - mean(contrast prompts) in residual stream space.
2. Stack direction vectors across layers into a matrix of shape (n_layers, d_model).
3. SVD to get singular values $s_i$.
4. Compute SVE:

$$\text{SVE} = -\sum_i p_i \log p_i \quad \text{where} \quad p_i = \frac{s_i^2}{\sum_j s_j^2}$$

5. Stability test: repeat with perturbed prompt subsets and check SVE consistency.
6. Effective rank: number of singular values needed to capture 90% of variance.

**Pass condition:** `safety_sve` < 2.0; `stability` > 0.8

**Usage:**

```bash
uv run python 136_safety_sve.py --model gpt2 --device cpu
uv run python 136_safety_sve.py --n-prompts 30 --n-stability-runs 5
```

**Reading the scores:**

| Pattern | What it means |
|---|---|
| Low SVE, high stability | Safety is compactly represented and stable -- amenable to subspace-based interventions |
| High SVE, high stability | Safety is diffusely represented but consistently so -- no compact safety subspace exists |
| Low SVE, low stability | Compact representation exists but it is prompt-sensitive -- findings may not generalize |
| Low effective rank (1--3) | Safety information concentrates in very few directions -- potentially easy to attack or defend |

---

## Faithfulness Curve Metrics

### MIB -- Faithfulness Curve (CPR/CMD)

**Source:** Mueller et al. (2025). "MIB." ICML 2025.

**Criteria:** Multi-threshold faithfulness

**What it establishes:** Circuit quality via the area under the faithfulness curve across edge-count thresholds. Rather than evaluating a circuit at a single threshold, this sweeps across thresholds from 0.1% to 100% of edges and measures faithfulness at each. CPR (Cumulative Performance Recovery) is the area under this curve; CMD (Cumulative Metric Deficit) is the area between the curve and perfect faithfulness.

**What it does not establish:** Whether the circuit is causally necessary or sufficient at any particular threshold -- only the aggregate faithfulness profile across all thresholds.

**Method:**

For each threshold $t$ in [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]:

1. $n = \max(1, \lfloor t \times \text{total\_edges} \rfloor)$
2. Keep top-$n$ edges (by layer order).
3. Convert kept edges to heads, compute faithfulness (logit-diff recovery under mean ablation of non-circuit heads).
4. Record faithfulness at threshold $t$.

Compute:

$$\text{CPR} = \int_0^1 f(t) \, dt \quad \text{(trapezoidal rule)}$$
$$\text{CMD} = \int_0^1 (1 - f(t)) \, dt$$

**Pass condition:** CPR > 0.5

**Usage:**

```bash
uv run python MIB_faithfulness_curve.py --tasks ioi --n-prompts 40
uv run python MIB_faithfulness_curve.py --device cpu
```

**Reading the scores:**

| Pattern | What it means |
|---|---|
| CPR > 0.7 | Strong circuit -- maintains faithfulness even at aggressive pruning thresholds |
| CPR 0.5--0.7 | Moderate circuit -- faithfulness degrades substantially at low thresholds |
| CPR < 0.5 | Weak circuit -- most edges are needed; the circuit is not well-separated from the full model |
| Flat curve near 1.0 | Nearly all edges contribute -- the circuit is the whole model |
| Sharp elbow | Clear separation between essential and non-essential edges |

---

## Summary Table

| Metric ID | Name | Criteria | Requires Artifact | Pass Condition |
|---|---|---|---|---|
| M07 | Architecture Duality | M2, M6 | Two SAE adapters | agreement > 0.3 |
| M08 | WeightLens Convergence | C5 | One SAE adapter | agreement > 0.3 |
| M09 | DMSAE Core Stability | M1 | Model + hook | Diagnostic (report-only) |
| M10 | PRISM Polysemanticity | M6, E1 | One SAE adapter | Diagnostic (report-only) |
| M11 | Matryoshka Cross-Scale | M1, M2 | Two SAE adapters (small/large) | consistency > 0.7 |
| M12 | Adaptive Sparsity | E1, M6 | One SAE adapter | mismatch_rate < 0.2 |
| M13 | Superposition Regime | M6 | Model only | Diagnostic (report-only) |
| M14 | Safety SVE | M1, M6 | Model only | SVE < 2.0, stability > 0.8 |
| EX1 | d-prime (SDT) | Causal | Model + circuit | $d'$ > 1.0, AUC > 0.7 |
| EX2 | DIF | Behavioral | Model + circuit | Cohen's $d$ < 0.5 |
| EX11 | Weber-Fechner / JND | Behavioral | Model + circuit | All heads detectable |
| EX24 | SAEBench Audit | M1, M2 | Model only | CV < 0.05, $d$ > 0.8 |
| EX25 | Reproducibility Check | M1 | Model only | dev_rate < 0.05 |
| EX34 | NLA-SAE Convergence | C5 | Model only | overlap > 0.3, cosine > 0.5 |
| MIB | Faithfulness Curve | Faithfulness | Model + circuit | CPR > 0.5 |

---

## Connection to Original Metrics

The original F01--F08 metrics are documented at their existing pages:

- [F01 -- Bootstrap Stability](/framework/metrics/measurement/f01-bootstrap-stability)
- [F02 -- Seed Variance](/framework/metrics/measurement/f02-seed-variance)
- [F03 -- Convergent Validity](/framework/metrics/measurement/f03-convergent-validity)
- [F04 -- Discriminant Validity](/framework/metrics/measurement/f04-discriminant-validity)
- [F05 -- Internal Consistency](/framework/metrics/measurement/f05-internal-consistency)
- [F06 -- Inter-Rater](/framework/metrics/measurement/f06-inter-rater)
- [F07 -- Measurement Invariance](/framework/metrics/measurement/f07-measurement-invariance)
- [F08 -- Incremental Validity](/framework/metrics/measurement/f08-incremental-validity)

The extended metrics on this page complement F01--F08 by:

- **Deepening reliability testing** (M09 Core Stability, EX24 SAEBench Audit, EX25 Reproducibility) -- going beyond prompt-level resampling to test decomposition stability, benchmark reliability, and pipeline reproducibility.
- **Adding SAE-specific construct validity** (M07 Architecture Duality, M08 WeightLens, M10 PRISM, M11 Matryoshka, M12 Adaptive Sparsity, M13 Superposition Regime) -- testing whether the decomposition instrument itself is valid before interpreting its outputs.
- **Importing psychometric rigor** (EX1 d-prime, EX2 DIF, EX11 Weber-Fechner) -- formalizing sensitivity, measurement bias, and scaling behavior using established frameworks from cognitive science.
- **Extending to safety** (M14 Safety SVE) -- applying measurement theory to safety-relevant representations.
- **Multi-threshold evaluation** (MIB Faithfulness Curve) -- replacing single-threshold faithfulness with curve-based analysis.
