---
title: "Philosophy of Science -- Metrics & Protocols"
description: "Reference for philosophy-of-science-lens metrics and protocols: sigma ablation, CATE, weight-space analysis, logic gates, intervention specificity, resample complement, misalignment, INUS conditions, actual causation, operation specification, held-out prediction, replacement test, procedure specification, composition test, minimality classification, and protocols for MDL/SLT, transportability, and causal discovery."
---

# Philosophy of Science -- Metrics & Protocols

This page documents the metrics and protocols under the [Philosophy of Science lens](/framework/lenses/core/philosophy-of-science). These metrics formalize criteria from the philosophy of science -- mechanistic explanation, causal discovery, minimality, transportability, and model complexity -- to evaluate whether circuit claims meet the evidential standards that philosophers of science use to assess explanatory adequacy.

All metrics on this page address the question: **does the circuit qualify as a scientific explanation of the model's behavior?** Some test structural properties (weight-space analysis, logic gates), some test causal robustness (sigma ablation, resample complement, misalignment), some test explanatory adequacy (operation specification, held-out prediction, replacement test), and some test the circuit's place in a broader theory (minimality, transportability, causal discovery).

---

## Causal Robustness Metrics

These metrics test whether causal claims about the circuit are robust across methods, contexts, and intervention types.

### C3 -- Sigma Ablation

**Source:** Woodward (2003), "Making Things Happen."

**Criteria:** I1 Necessity (Woodward Interventionism / A04)

**What it establishes:** Whether the circuit's causal role is robust across different ablation methods. Measures the coefficient of variation (CV) of faithfulness scores across 8 ablation techniques: zero, mean, resample, noise, causal_resample, soft, attn_knockout, and mean_last. Low CV means the finding is not an artifact of a particular ablation choice.

**What it does not establish:** Which ablation method is "correct." The metric tests consistency, not correctness. A circuit could produce consistent results across all methods while still being mischaracterized.

**Method:**

1. For each of 8 ablation methods, ablate all circuit heads and measure faithfulness (logit-diff recovery).
2. Compute the coefficient of variation: $\text{CV} = \sigma / |\mu|$ across the 8 faithfulness scores.
3. Low CV indicates method-robust findings; high CV indicates method-dependent findings.

**Key quantities:**

- `cv` -- coefficient of variation of faithfulness across ablation methods
- `faithfulness_per_method` -- individual faithfulness scores for each of 8 methods
- `mean_faithfulness` -- mean across all methods

**Pass condition:** Report-only. Lower CV is better; CV < 0.2 indicates strong robustness.

**Usage:**

```bash
uv run python 03_sigma_ablation.py --model gpt2 --device cpu
uv run python 03_sigma_ablation.py --tasks ioi --n-prompts 50
```

**Reading the scores:**

| Pattern | What it means |
|---|---|
| CV < 0.1 | Very robust -- faithfulness is nearly identical across all ablation methods |
| CV 0.1--0.2 | Robust -- minor variation, results are trustworthy |
| CV 0.2--0.5 | Moderate variation -- some methods give substantially different results |
| CV > 0.5 | High variation -- the causal claim depends heavily on which ablation method is used |
| One method outlier | A specific ablation type interacts unusually with the circuit -- investigate why |

---

### C35 -- Resample Complement

**Source:** Woodward (2003), "Making Things Happen"; Craver & Bechtel (2007).

**Criteria:** I1 Necessity (Woodward Interventionism / A04)

**What it establishes:** Whether non-circuit heads are genuinely uninvolved in the task. Replaces non-circuit head activations with activations from a different prompt (resampling) and measures whether the model's task performance is preserved. If replacing non-circuit components with activations from unrelated prompts does not degrade performance, they are genuinely irrelevant.

**What it does not establish:** Why non-circuit heads are irrelevant. The metric confirms they can be replaced without consequence but does not explain what they are doing instead.

**Method:**

1. Select a set of different-prompt activations for each non-circuit head.
2. Replace non-circuit head activations with the resampled activations.
3. Measure faithfulness (logit-diff recovery) under resampling.
4. Compare to mean ablation faithfulness as a reference.

**Key quantities:**

- `resample_faithfulness` -- faithfulness when non-circuit heads are resampled
- `mean_ablation_faithfulness` -- faithfulness when non-circuit heads are mean-ablated (reference)

**Pass condition:** Report-only. `resample_faithfulness` close to 1.0 indicates genuine circuit-complement independence.

**Usage:**

```bash
uv run python 35_resample_complement.py --model gpt2 --device cpu
uv run python 35_resample_complement.py --tasks ioi --n-prompts 50
```

**Reading the scores:**

| Pattern | What it means |
|---|---|
| resample_faithfulness > 0.9 | Non-circuit heads are genuinely uninvolved -- strong complement independence |
| resample_faithfulness 0.7--0.9 | Moderate complement independence -- some non-circuit heads carry residual task information |
| resample_faithfulness < 0.7 | Weak complement independence -- "non-circuit" heads contribute substantially |
| resample >> mean_ablation | Mean ablation is destructive in ways that resampling is not -- mean ablation overstates circuit importance |

---

### C37 -- Misalignment Score

**Source:** Woodward (2003), "Making Things Happen."

**Criteria:** I1 Necessity (Woodward Interventionism / A04)

**What it establishes:** Whether the circuit passes both noising (necessity) and denoising (sufficiency) tests consistently. Misalignment = |noising_necessity - denoising_sufficiency| per head. A head that is necessary but not sufficient (or vice versa) has high misalignment, indicating that the causal claim is incomplete -- the head's role is more nuanced than simple "in the circuit / not in the circuit."

**What it does not establish:** Which direction of misalignment is worse. A head that is necessary but not sufficient may be part of a distributed mechanism; a head that is sufficient but not necessary may have backup mechanisms. Both are informative.

**Method:**

1. For each head, compute noising necessity: logit-diff drop when the head is corrupted.
2. For each head, compute denoising sufficiency: logit-diff recovery when only this head is restored.
3. Misalignment = |necessity - sufficiency| per head.
4. Severity flag: misalignment > 0.3 for any head.

**Key quantities:**

- `misalignment_per_head` -- per-head |necessity - sufficiency|
- `mean_misalignment` -- mean across all circuit heads
- `max_misalignment` -- maximum misalignment (worst-case head)
- `n_severe` -- number of heads exceeding severity threshold (0.3)

**Pass condition:** misalignment < 0.3 for all heads.

**Usage:**

```bash
uv run python 37_misalignment_score.py --model gpt2 --device cpu
uv run python 37_misalignment_score.py --tasks ioi --n-prompts 50
```

**Reading the scores:**

| Pattern | What it means |
|---|---|
| All heads < 0.15 | Low misalignment -- necessity and sufficiency agree; clean causal roles |
| Some heads 0.15--0.3 | Moderate misalignment -- nuanced causal roles worth investigating |
| Any head > 0.3 | Severe misalignment -- the head's role cannot be captured by simple necessity/sufficiency |
| High necessity, low sufficiency | Head is needed but not enough alone -- part of a distributed mechanism |
| Low necessity, high sufficiency | Head is enough alone but has backups -- redundant mechanism |

---

### C25 -- Intervention Specificity

**Source:** Rubin (1974); Imbens & Rubin (2015).

**Criteria:** I3 Specificity (Rubin CATE / A03)

**What it establishes:** Whether the circuit's causal effect is specific to the target task. Computes the ratio of the circuit's effect on the target task to its mean effect on non-target tasks. High specificity means the circuit selectively affects the intended task.

**What it does not establish:** That the circuit has zero effect on other tasks. Some overlap is expected (shared representations); the metric quantifies the degree of selectivity.

**Method:**

1. For the target task and several non-target tasks, ablate the circuit and measure the effect.
2. Compute specificity = target_effect / mean(nontarget_effects).

**Key quantities:**

- `specificity_ratio` -- target effect / mean nontarget effect
- `target_effect` -- effect magnitude on the target task
- `mean_nontarget_effect` -- mean effect magnitude on non-target tasks

**Pass condition:** Report-only. Higher ratio indicates more specific intervention.

**Usage:**

```bash
uv run python 25_intervention_specificity.py --model gpt2 --device cpu
uv run python 25_intervention_specificity.py --tasks ioi --n-prompts 50
```

**Reading the scores:**

| Pattern | What it means |
|---|---|
| Specificity > 5.0 | Highly specific -- circuit selectively affects the target task |
| Specificity 2.0--5.0 | Moderately specific -- target task affected more than others |
| Specificity < 2.0 | Low specificity -- the circuit affects many tasks similarly |
| Specificity near 1.0 | No specificity -- the circuit is a general-purpose component, not task-specific |

---

### C6 -- CATE (Conditional Average Treatment Effect)

**Source:** Rubin (1974); Holland (1986); Imbens & Rubin (2015).

**Criteria:** I3 Specificity (Rubin CATE / A03)

**What it establishes:** Whether the circuit's causal effect is heterogeneous across syntactic contexts. Computes the average treatment effect (ATE) of circuit ablation and decomposes it into subgroup effects using Cohen's d to measure heterogeneity. If the circuit's effect varies substantially across contexts (e.g., different sentence structures), the causal claim needs qualification.

**What it does not establish:** Why the effect varies. CATE identifies heterogeneity but not the moderating mechanism.

**Method:**

1. Ablate the circuit across multiple syntactic subgroups.
2. Compute ATE within each subgroup.
3. Compute Cohen's d between subgroups to quantify heterogeneity.
4. Report overall ATE and heterogeneity measures.

**Key quantities:**

- `ate` -- overall average treatment effect
- `subgroup_ates` -- per-subgroup treatment effects
- `heterogeneity_d` -- Cohen's d between subgroups with largest effect difference

**Pass condition:** Report-only. Low heterogeneity indicates a robust, context-independent effect.

**Usage:**

```bash
uv run python 06_cate.py --model gpt2 --device cpu
uv run python 06_cate.py --tasks ioi --n-prompts 50
```

---

## Mechanistic Explanation Metrics

These metrics test whether the circuit qualifies as a mechanistic explanation under the MDC/Glennan framework -- organized entities performing specifiable operations to produce the phenomenon.

### C18 -- Weight-Space Analysis

**Source:** Machamer, Darden & Craver (2000); Glennan (2017).

**Criteria:** Structural (MDC/Glennan / A05)

**What it establishes:** Whether the circuit's weight matrices have structural properties consistent with its proposed function. Three sub-metrics:

- **Effective rank** (`C18.wqk_effective_rank`): exponential of the entropy of normalized singular values of $W_Q W_K^T$. Higher rank means the attention pattern uses more dimensions.
- **Cosine alignment** (`C18.cosine_alignment`): maximum cosine similarity between the top-3 SVD directions of $W_{OV}$ projected through $W_U$. High alignment means the head's output is directionally consistent with specific tokens.
- **Spectral norm ratio** (`C18.spectral_norm_ratio`): ratio of circuit to non-circuit spectral norms. Higher ratio means circuit heads have more "capacity" in weight space.

**What it does not establish:** That the weight structure is used at runtime. These are static, weight-space diagnostics that reveal structural capacity but not functional behavior.

**Method:**

1. For each circuit head, compute $W_Q W_K^T$ and its SVD.
2. Effective rank = $\exp(-\sum p_i \log p_i)$ where $p_i = s_i^2 / \sum s_j^2$.
3. Compute $W_{OV} = W_V W_O$ and its top-3 SVD directions; project through $W_U$ (unembedding); cosine similarity with top promoted tokens.
4. Spectral norm ratio = $\|W_{\text{circuit}}\|_2 / \|W_{\text{non-circuit}}\|_2$.

**Key quantities:**

- `wqk_effective_rank` -- effective dimensionality of the QK circuit
- `cosine_alignment` -- directional consistency of OV output with token embeddings
- `spectral_norm_ratio` -- relative weight magnitude of circuit vs non-circuit

**Pass condition:** Report-only (CPU-only weight-space diagnostics).

**Usage:**

```bash
uv run python 18_weight_extended.py --model gpt2 --device cpu
```

**Reading the scores:**

| Pattern | What it means |
|---|---|
| High effective rank | Head uses many dimensions for attention -- complex query-key interaction |
| Low effective rank | Head attends to a low-dimensional subspace -- simple, interpretable attention pattern |
| High cosine alignment | OV circuit strongly promotes specific tokens -- consistent with a "lookup" operation |
| High spectral norm ratio | Circuit heads have disproportionately large weights -- capacity argument for importance |

---

### C19 -- Logic Gates

**Source:** Machamer, Darden & Craver (2000); Glennan (2017).

**Criteria:** C2 Structural Plausibility (MDC/Glennan / A05)

**What it establishes:** How pairs of circuit heads interact: AND (superadditive -- both needed), OR (redundant -- either sufficient), NOT (inhibitory -- one suppresses the other), or ADDITIVE (independent contributions). This reveals the circuit's computational structure -- whether it operates as a serial pipeline, parallel redundant system, or mixed architecture.

**What it does not establish:** Whether the interaction classification is complete. Pairwise analysis misses higher-order interactions among three or more heads.

**Method:**

1. For each pair of circuit heads, measure logit-diff under four conditions:
   - Both active (clean)
   - Only head A ablated
   - Only head B ablated
   - Both ablated
2. Classify the interaction:
   - **AND/superadditive**: joint effect > sum of individual effects
   - **OR/redundant**: joint effect < sum of individual effects
   - **NOT/inhibitory**: one head's effect reverses when the other is present
   - **ADDITIVE**: joint effect $\approx$ sum of individual effects
3. Report noising vs denoising completeness delta (consistency between the two test directions).

**Key quantities:**

- `interaction_counts` -- number of head pairs classified as AND, OR, NOT, ADDITIVE
- `completeness_delta` -- |noising_completeness - denoising_completeness|

**Pass condition:** Report-only. Low completeness_delta indicates consistent classification.

**Usage:**

```bash
uv run python 19_logic_gates.py --model gpt2 --device cpu
uv run python 19_logic_gates.py --tasks ioi --n-prompts 50
```

**Reading the scores:**

| Pattern | What it means |
|---|---|
| Mostly AND | Heads work synergistically -- removing any one breaks the circuit |
| Mostly OR | Heads are redundant -- the circuit is robust to single-head failures |
| Mixed AND/OR | Multi-path architecture with some critical nodes and some backup paths |
| High completeness_delta | Noising and denoising give different interaction patterns -- results are fragile |

---

### F1 -- Operation Specification

**Source:** Machamer, Darden & Craver (2000); Glennan (2017).

**Criteria:** MDC/Glennan / A05

**What it establishes:** Whether each circuit head performs a specifiable, consistent operation. Measures two aspects: (1) output consistency -- first principal component variance ratio across prompts (does the head always produce output in the same direction?), and (2) attention-weighted OV prediction -- R-squared of predicting head output from its attention-weighted OV circuit (is the head's operation well-described by its attention pattern + OV matrix?).

**What it does not establish:** What the operation is in human-interpretable terms. The metric tests consistency and predictability, not interpretability.

**Method:**

1. Run the model on task prompts, collecting each circuit head's output.
2. Stack outputs across prompts, compute PCA. First PC variance ratio = output consistency.
3. Predict head output from attention_pattern @ V @ W_O. R-squared = OV prediction quality.
4. Pass if circuit heads have higher consistency and prediction than non-circuit heads.

**Key quantities:**

- `output_consistency` -- first PC variance ratio (higher = more consistent operation)
- `ov_prediction_r2` -- R-squared of attention-weighted OV prediction

**Pass condition:** Circuit > non-circuit baseline on both measures.

**Usage:**

```bash
uv run python 70_operation_specification.py --model gpt2 --device cpu
```

---

### F2 -- Held-Out Prediction

**Source:** Machamer, Darden & Craver (2000); Glennan (2017).

**Criteria:** MDC/Glennan / A05

**What it establishes:** Whether the circuit's operations generalize to held-out data. Trains a characterization of each head's operation on a training split (principal direction of activation) and tests whether that characterization predicts behavior on a test split.

**What it does not establish:** Whether the generalization extends to out-of-distribution inputs. The metric tests held-out generalization within the same distribution.

**Method:**

1. Split prompts into train and test sets.
2. On train: compute principal direction of each head's activation magnitude.
3. On test: predict activation magnitude from train-derived principal direction.
4. Report Pearson r between predicted and actual magnitudes.

**Key quantities:**

- `pearson_r` -- correlation between predicted and actual activation magnitudes on held-out data

**Pass condition:** Report-only. Higher Pearson r indicates better generalization.

**Usage:**

```bash
uv run python 71_held_out_prediction.py --model gpt2 --device cpu
```

---

### F3 -- Replacement Test

**Source:** Machamer, Darden & Craver (2000); Glennan (2017).

**Criteria:** MDC/Glennan / A05

**What it establishes:** Whether circuit heads can be replaced by simplified approximations without losing task performance. Two variants:

- **Constant replacement** (`F3.replacement_constant`): replace head output with its mean activation across prompts. If recovery remains high, the head contributes only a constant bias, not prompt-specific computation.
- **Linear OV replacement** (`F3.replacement_linear_ov`): replace with resid_pre @ W_V (ignoring the attention pattern). If recovery remains high, the attention pattern is not doing useful work.

**What it does not establish:** Whether the replacement captures the same computation. Low recovery under replacement means the head's full computation is needed; high recovery means a simpler model suffices (possibly the circuit is overspecified).

**Method:**

1. For each circuit head, replace its output with the simplified version.
2. Measure recovery = replaced_logit_diff / clean_logit_diff.
3. Compare circuit heads vs non-circuit heads.

**Key quantities:**

- `constant_recovery` -- logit-diff recovery under mean-activation replacement
- `linear_ov_recovery` -- logit-diff recovery under attention-free OV replacement

**Pass condition:** Report-only. Low recovery means the head's full computation is needed (good for the circuit claim).

**Usage:**

```bash
uv run python 72_replacement_test.py --model gpt2 --device cpu
```

**Reading the scores:**

| Pattern | What it means |
|---|---|
| Low constant_recovery | Head contributes prompt-specific information -- not just a bias |
| High constant_recovery | Head contributes a near-constant output -- may not need to be in the circuit |
| Low linear_ov_recovery | Attention pattern matters -- the head is doing nontrivial routing |
| High linear_ov_recovery | Attention pattern is irrelevant -- head output is determined by OV alone |

---

### A1 -- Procedure Specification

**Source:** Machamer, Darden & Craver (2000); Glennan (2017).

**Criteria:** MDC/Glennan / A05

**What it establishes:** Whether information gain is monotonically ordered along pathway chains in the circuit. If the circuit is a genuine procedure, information should flow in a specifiable order: earlier components should contribute less to the final answer than later components, producing monotonic information gain along the computational pathway.

**What it does not establish:** That the ordering is the unique correct one. Multiple valid orderings may exist if the circuit has parallel paths.

**Method:**

1. Identify pathway chains through the circuit (sequences of heads connected by residual stream).
2. For each chain, measure cumulative information gain (logit-diff recovery) at each step.
3. Compute ordering_score = fraction of chains showing monotonic information gain.

**Key quantities:**

- `ordering_score` -- fraction of pathway chains with monotonic information gain

**Pass condition:** ordering_score > 0.7.

**Usage:**

```bash
uv run python 77_procedure_specification.py --model gpt2 --device cpu
```

---

### A2 -- Composition Test

**Source:** Machamer, Darden & Craver (2000); Glennan (2017).

**Criteria:** MDC/Glennan / A05

**What it establishes:** Whether the circuit's pathways compose into a functioning whole. Tests pathway-level complement ablation: ablating the complement of each pathway and measuring whether the pathway alone produces meaningful output.

**What it does not establish:** That the pathways are independent. Composition testing confirms that pathways contribute, not that they operate without interaction.

**Method:**

1. Identify pathways through the circuit.
2. For each pathway, ablate everything outside the pathway and measure logit-diff recovery.
3. Report full_circuit recovery and max_single_pathway recovery.

**Key quantities:**

- `full_circuit_recovery` -- logit-diff recovery with the full circuit active
- `max_single_pathway` -- best single-pathway recovery

**Pass condition:** full_circuit > 0.30 OR max_single_pathway > 0.20.

**Usage:**

```bash
uv run python 78_composition_test.py --model gpt2 --device cpu
```

---

## Minimality and INUS Metrics

These metrics test whether the circuit is minimal -- whether all its components are genuinely needed -- and how each component relates to sufficient conditions for the task.

### C4b -- Minimality Classification

**Source:** Hadad, Katz & Bassan (ICLR 2026).

**Criteria:** Minimality

**What it establishes:** The minimality class of the circuit: how close it is to containing only necessary components.

- **Subset minimal** (1.0): no proper subset is sufficient -- every head is strictly needed.
- **Locally minimal** (0.75): no single head can be removed without breaking sufficiency -- but subsets of 2+ might be removable.
- **Quasi minimal** (0.5): removing some heads preserves sufficiency, but not many.
- **Not minimal** (0.0): many heads can be removed -- the circuit is substantially overspecified.

**What it does not establish:** The unique minimal subset. Multiple minimal subcircuits may exist (the circuit may have redundant pathways that are each individually minimal).

**Method:**

1. Test all single-head removals: if any preserves sufficiency, the circuit is not locally minimal.
2. If locally minimal, test all pairwise removals: if any preserves sufficiency, the circuit is not subset minimal.
3. Classify accordingly.

**Key quantities:**

- `minimality_class` -- one of: subset_minimal (1.0), locally_minimal (0.75), quasi_minimal (0.5), not_minimal (0.0)

**Pass condition:** At least locally_minimal (>= 0.75).

**Usage:**

```bash
uv run python C4b_minimality_class.py --model gpt2 --device cpu
uv run python C4b_minimality_class.py --tasks ioi --n-prompts 50
```

**Reading the scores:**

| Pattern | What it means |
|---|---|
| Subset minimal (1.0) | Every head is strictly needed -- tightest possible circuit |
| Locally minimal (0.75) | No single head is redundant, but some subsets might be |
| Quasi minimal (0.5) | Some redundancy exists but circuit is not grossly overspecified |
| Not minimal (0.0) | Circuit contains many unnecessary heads -- needs pruning |

---

### C39 -- INUS Conditions

**Source:** Mackie (1965), "Causes and Conditions."

**Criteria:** Regularity / INUS (A10)

**What it establishes:** Whether each head is an INUS condition -- Insufficient but Necessary part of an Unnecessary but Sufficient condition. Finds minimal sufficient subcircuits and classifies each head's relationship to them:

- **Necessary**: present in all sufficient subcircuits.
- **INUS**: present in some but not all sufficient subcircuits.
- **Non-redundant necessary**: necessary within its subcircuit but the subcircuit itself is not unique.
- **Redundant**: can be removed from all sufficient subcircuits without breaking sufficiency.

**What it does not establish:** Whether the sufficient subcircuits are complete. INUS analysis is relative to the heads in the defined circuit; heads outside the circuit are not considered.

**Method:**

1. Enumerate candidate subcircuits (subsets of circuit heads).
2. Test each for sufficiency (logit-diff recovery above threshold, default 0.7).
3. Find minimal sufficient subcircuits (no proper subset is also sufficient).
4. Classify each head based on its membership pattern across minimal sufficient sets.

**Key quantities:**

- `head_classifications` -- per-head INUS classification
- `n_minimal_sufficient` -- number of distinct minimal sufficient subcircuits found

**Pass condition:** Report-only.

**Usage:**

```bash
uv run python 39_inus_conditions.py --model gpt2 --device cpu
uv run python 39_inus_conditions.py --tasks ioi --sufficiency-threshold 0.7
```

---

### C40 -- Actual Causation (Halpern-Pearl)

**Source:** Halpern & Pearl (2005), "Causes and Explanations: A Structural-Model Approach."

**Criteria:** Actual Causation (A11)

**What it establishes:** Whether each head is an actual cause of the task behavior under the Halpern-Pearl definition (AC1--AC3). Unlike standard counterfactual tests, actual causation handles preemption (a backup mechanism would have produced the same result) and overdetermination (multiple sufficient causes). Detects heads that are actual causes but not standard-necessary (indicating backup mechanisms exist).

**What it does not establish:** A complete causal model of the task. Actual causation is context-specific -- a head may be an actual cause in one context but not another.

**Method:**

1. For each head, test AC1 (the head's value and the outcome both occurred).
2. Test AC2 (there exists a "witness set" of other variables such that changing the head's value, while holding the witness set fixed, changes the outcome).
3. Test AC3 (minimality -- no proper subset of the head satisfies AC1-AC2).
4. Report which heads satisfy all three conditions (actual causes) and which satisfy AC2 but not standard necessity (backup-protected causes).

**Key quantities:**

- `n_actual_causes` -- number of heads satisfying AC1-AC3
- `n_backup_protected` -- number of heads that are actual causes but not standard-necessary
- `witness_sets` -- the witness sets that establish AC2 for each actual cause

**Pass condition:** Report-only.

**Usage:**

```bash
uv run python 40_actual_causation.py --model gpt2 --device cpu
uv run python 40_actual_causation.py --tasks ioi --n-prompts 50
```

**Reading the scores:**

| Pattern | What it means |
|---|---|
| All circuit heads are actual causes | Strong support -- every head is genuinely causal |
| backup_protected > 0 | Some heads have backup mechanisms -- standard ablation would miss their importance |
| Few actual causes | Most circuit heads are not actual causes in the HP sense -- circuit may be overspecified |

---

## Protocols

### Protocol A09 -- MDL/SLT

**Source:** Rissanen (1978), "Modeling by shortest data description"; Watanabe (2009), "Algebraic Geometry and Statistical Learning Theory."

**Framework:** Minimum Description Length and Singular Learning Theory. Tests whether the circuit is a parsimonious description of the model's behavior (MDL) and whether its learning coefficient indicates regular or singular learning dynamics (SLT).

**Metrics and thresholds:**

| Metric | Threshold |
|---|---|
| `hyperparam_sensitivity` | < 0.2 |
| `llc` | > 0.0 |

**What it establishes:** Whether the circuit represents a compressed, stable description of the model's task behavior. Low hyperparameter sensitivity means the circuit's quality does not depend critically on evaluation parameters. Positive LLC (local learning coefficient from SLT) indicates the circuit has learnable structure.

**What it does not establish:** Whether the circuit is the minimum-length description. MDL provides a relative comparison (is this description shorter than alternatives?), not an absolute optimality guarantee.

---

### Protocol A12 -- Causal Transportability

**Source:** Pearl & Bareinboim (2011), "Transportability of Causal and Statistical Relations."

**Framework:** Tests whether circuit findings transport across models. A circuit claim is more credible if the same circuit structure (or its analogue) appears in independently trained models.

**Metrics and thresholds:**

| Metric | Threshold |
|---|---|
| `cross_model_invariance` | > 0.5 |

**What it establishes:** Whether the circuit is a property of the task rather than an idiosyncrasy of a particular trained model. High cross-model invariance means the circuit structure is transportable -- it generalizes beyond the specific model it was discovered in.

**What it does not establish:** That the circuit is "fundamental" to the task. The same task might be solved by different circuits in different model families (e.g., attention-based vs MLP-based solutions).

---

### Protocol A13 -- Causal Discovery

**Source:** Zheng et al. (2018), "DAGs with NO TEARS"; Spirtes, Glymour & Scheines (2000).

**Framework:** Automated causal structure learning. Uses continuous optimization methods (NOTEARS) to discover causal relationships between circuit components from observational data.

**Metrics and thresholds:**

| Metric | Threshold |
|---|---|
| `notears` | > 0.5 |

**What it establishes:** Whether automated causal discovery algorithms recover a DAG structure consistent with the proposed circuit. If NOTEARS independently identifies the same edges, this is convergent evidence for the circuit's causal structure.

**What it does not establish:** That the discovered DAG is the true causal graph. NOTEARS recovers structure from statistical dependencies, which may not correspond to genuine causal relationships (faithfulness assumption violations, latent confounders).

---

## Summary Table

| Metric ID | Name | Criteria | Evidence Family | Pass Condition |
|---|---|---|---|---|
| C3 | Sigma Ablation | I1 Necessity | Causal | Report-only (CV < 0.2 preferred) |
| C6 | CATE | I3 Specificity | Causal | Report-only |
| C18.wqk | WQK Effective Rank | Structural | Weight-space | Report-only |
| C18.cos | Cosine Alignment | Structural | Weight-space | Report-only |
| C18.snr | Spectral Norm Ratio | Structural | Weight-space | Report-only |
| C19 | Logic Gates | C2 Structural Plausibility | Structural | Report-only |
| C25 | Intervention Specificity | I3 Specificity | Causal | Report-only |
| C35 | Resample Complement | I1 Necessity | Causal | Report-only |
| C37 | Misalignment Score | I1 Necessity | Causal | misalignment < 0.3 |
| C39 | INUS Conditions | INUS | Structural-Causal | Report-only |
| C40 | Actual Causation | Actual Causation | Causal | Report-only |
| C4b | Minimality Classification | Minimality | Structural | >= locally_minimal |
| F1 | Operation Specification | MDC/Glennan | Mechanistic | Circuit > baseline |
| F2 | Held-Out Prediction | MDC/Glennan | Mechanistic | Report-only |
| F3.const | Replacement (Constant) | MDC/Glennan | Mechanistic | Report-only |
| F3.ov | Replacement (Linear OV) | MDC/Glennan | Mechanistic | Report-only |
| A1 | Procedure Specification | MDC/Glennan | Mechanistic | ordering_score > 0.7 |
| A2 | Composition Test | MDC/Glennan | Mechanistic | full_circuit > 0.30 or pathway > 0.20 |
| p_a09 | MDL/SLT | Complexity | Protocol | sensitivity < 0.2, LLC > 0.0 |
| p_a12 | Transportability | Cross-model | Protocol | invariance > 0.5 |
| p_a13 | Causal Discovery | NOTEARS | Protocol | notears > 0.5 |

---

## Connection to Philosophy of Science Lens

The philosophy of science lens is documented at the [Philosophy of Science lens page](/framework/lenses/core/philosophy-of-science). The core insight is that circuit claims are scientific explanations, and scientific explanations must meet standards developed over centuries of philosophical analysis.

The metrics on this page operationalize those standards:

- **Causal robustness metrics** (C3, C25, C35, C37) implement Woodward's interventionist criteria -- stability across methods, proportionality of interventions, and invariance across contexts.
- **Mechanistic explanation metrics** (C18, C19, F1, F2, F3, A1, A2) implement MDC/Glennan's framework -- specifiable operations, predictive power, non-trivial mechanisms, ordered procedures, and compositional structure.
- **Minimality and INUS metrics** (C4b, C39, C40) formalize what it means for a circuit to contain "just the right components" -- drawing on Mackie's INUS conditions and Halpern-Pearl actual causation.
- **Protocols** (A09, A12, A13) test the circuit against broader explanatory desiderata -- parsimony (MDL/SLT), transportability (Pearl-Bareinboim), and independent causal discovery (NOTEARS).
