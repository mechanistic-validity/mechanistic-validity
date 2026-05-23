---
title: "MI Causal Metrics"
description: "Causal metrics from the mechanistic interpretability lens: ablation, patching, scrubbing, and causal discovery methods."
---

# MI Causal Metrics

The [mechanistic interpretability lens](/framework/lenses/core/mechanistic-interpretability) contributes 21 causal metrics spanning ablation, interchange intervention, causal scrubbing, causal discovery, and cross-model transportability. All belong to **evidence family A (causal)** --- they establish claims through interventions on model internals or through observational causal inference on activation data.

This page documents each metric: what it computes, what it establishes, and how to interpret the results. The metrics are organized into five groups by methodology.

---

## 1 --- Interchange Intervention and DAS

These metrics train or apply learned rotations of residual-stream subspaces to test whether causal variables are linearly encoded.

### C1 --- DAS-IIA (Constrained Distributed Alignment Search)

**Metric ID:** `C1.das_iia`
**Instrument:** A02 --- Counterfactual DAS/IIA
**Validity layer:** Internal + Representational
**Criteria:** I2 Sufficiency

#### What it computes

Trains a linear rotation \( R \in \mathbb{R}^{d_{\text{head}} \times k} \) of a head's \( z \)-space onto a binary causal variable using full-forward-pass CE loss with SVD warm-start. Computes interchange intervention accuracy (IIA) on held-out counterfactual pairs: swap the rotated subspace from a source prompt into a base prompt and check whether the model's output flips to the source's target.

#### Key metrics and thresholds

| Metric | Threshold | Interpretation |
|---|---|---|
| `das_iia_k{d}` | IIA > 0.70 (pass) | The learned \( k \)-dimensional subspace captures the causal variable |
| `das_iia_random_k{d}` | baseline | Haar-random rotation IIA (expected ~0.50) |
| `das_iia_untrained_k{d}` | baseline | IIA on a randomly initialized model |

Sweeps subspace dimensions \( k \in \{1, 2, 4\} \) by default.

**What it establishes:**
- A low-dimensional linear subspace of a circuit head's activation space encodes the task's causal variable.
- The learned rotation generalizes to held-out counterfactual pairs, not just the training distribution.

**What it does not establish:**
- That this subspace is the *only* encoding of the variable (other heads may encode it redundantly).
- That the causal variable is correctly identified --- IIA tests the rotation, not the variable definition.

**Usage:**
```bash
uv run python 01_das_iia.py --tasks ioi sva --n-prompts 40
uv run python 01_das_iia.py --mode random    # Haar baseline only
uv run python 01_das_iia.py --mode untrained # random-init model baseline
```

---

### C15 --- IIA Variant Suite (Neuron-Level, IIA@k, Cross-Layer)

**Metric ID:** `C15.neuron_iia`, `C15.iia_at_k{k}`, `C15.cross_layer_iia`
**Instrument:** A02 --- Counterfactual DAS/IIA
**Validity layer:** Internal + Representational
**Criteria:** I2 Sufficiency

#### What it computes

Three IIA variants that probe different intervention granularities:

- **Neuron-level IIA** (Metric #7): Swaps the full `hook_z` vector at a head --- no learned rotation. Tests whether the raw activation space already encodes the causal variable without alignment.
- **IIA@k** (Metric #14): Restricts DAS-IIA to the top-\( k \) heads ranked by individual IIA. Sweeps \( k \in \{1, 2, 4, 8, 15\} \).
- **Cross-layer IIA** (Metric #15): Groups circuit heads by layer and reports per-layer best IIA, identifying which layers carry the strongest causal encoding.

**What it establishes:**
- Whether the causal variable is accessible without a learned rotation (neuron-level IIA).
- How IIA concentrates across the top-\( k \) heads and across layers.

**What it does not establish:**
- That the variable is exclusively represented at one granularity --- all three views may pass simultaneously.

**Usage:**
```bash
uv run python 15_iia_variants.py --tasks ioi sva --n-prompts 40
```

---

### C31 --- Multi-Axis IIA

**Metric ID:** `C31.multi_axis_iia`, `C31.multi_axis_control`
**Instrument:** A02 --- Counterfactual DAS/IIA
**Validity layer:** Internal + Representational
**Criteria:** I2 Sufficiency

#### What it computes

For tasks with multiple causal variables (e.g., IOI has name identity and gender), trains separate DAS rotations per axis and measures joint IIA. The control metric zeros one axis and measures residual IIA on the other, testing axis independence.

- **Metric #10 (joint):** Intervene on all axes simultaneously and measure IIA.
- **Metric #11 (control):** Zero out axis A's subspace, then intervene on axis B. If residual IIA remains high, the axes are encoded independently.

**What it establishes:**
- That multiple causal variables have separable linear encodings in the same head.
- That intervening on one variable does not corrupt the other (axis independence).

**What it does not establish:**
- That the axes correspond to the "correct" causal decomposition --- the decomposition is assumed, not discovered.

**Usage:**
```bash
uv run python 31_multi_axis_iia.py --tasks ioi sva --n-prompts 40
```

---

### C32 --- Cross-Task IIA Transfer

**Metric ID:** `C32.cross_task_iia_transfer`, `C32.self_iia`
**Instrument:** A07 --- Granger/Transfer Entropy
**Validity layer:** Internal
**Criteria:** I4 Consistency

#### What it computes

Trains a DAS rotation on task A, then evaluates IIA using task B's prompts with task A's rotation. Constructs a transfer matrix (rows = train task, columns = test task). Reports specificity = diagonal mean minus off-diagonal mean.

High diagonal + low off-diagonal means representations are task-specific. Low specificity means the same subspace encodes multiple tasks (shared representation).

**What it establishes:**
- Whether learned causal representations are task-specific or shared across tasks.
- A quantitative measure of representational specificity.

**What it does not establish:**
- Whether shared representations indicate a common computational mechanism or merely geometric overlap.

**Usage:**
```bash
uv run python 32_cross_task_iia_transfer.py --tasks ioi sva --n-prompts 40
```

---

## 2 --- Ablation and Patching

Standard do-calculus interventions that corrupt or restore component activations and measure effects on task performance.

### C2 --- Activation Patching

**Metric ID:** `C2.activation_patching`
**Instrument:** A01 --- SCM / Pearl Causal Hierarchy
**Validity layer:** Internal
**Criteria:** I1 Necessity, I2 Sufficiency

#### What it computes

For each component \( c \), patches its `hook_z` activation from a clean run into a corrupted run and measures the fraction of the clean-to-corrupted logit-difference gap restored:

$$
AP(c) = \frac{LD_{\text{patched at } c} - LD_{\text{corrupted}}}{LD_{\text{clean}} - LD_{\text{corrupted}}}
$$

Reports the sum of per-head effects for the circuit, compared against random same-size subsets of heads.

**What it establishes:**
- Necessity-oriented attribution: components with high \( AP(c) \) are causally important for the task.
- Circuit heads collectively restore more logit difference than random same-size sets.

**What it does not establish:**
- Sufficiency of the circuit (that is tested by corrupt-restore).
- Uniqueness --- other head subsets may achieve comparable restoration.

**Usage:**
```bash
uv run python 02_activation_patching.py --tasks ioi sva --n-prompts 40
```

---

### C20 --- Corrupt-Restore Patching

**Metric ID:** `C20.corrupt_restore`
**Instrument:** A02 --- Counterfactual DAS/IIA
**Validity layer:** Internal + Representational
**Criteria:** I2 Sufficiency

#### What it computes

The reverse of standard activation patching. Starts with a fully corrupted run (all heads mean-ablated), then restores specific circuit heads from the clean cache and measures recovery:

$$
\text{restoration rate} = \frac{LD_{\text{restored}} - LD_{\text{corrupt}}}{LD_{\text{clean}} - LD_{\text{corrupt}}}
$$

Also measures per-head restoration contributions and compares against restoring random same-size head sets.

**What it establishes:**
- Sufficiency of the circuit: whether the circuit heads alone can restore task performance from a fully corrupted baseline.
- Per-head contribution to the restoration, identifying the most important components.

**What it does not establish:**
- Necessity (standard patching tests that). A circuit can be sufficient without being necessary if redundant pathways exist.

**Usage:**
```bash
uv run python 20_corrupt_restore.py --tasks ioi sva --n-prompts 40
```

---

### C33 --- Path Patching

**Metric ID:** `C33.path_patching`
**Instrument:** A02 --- Counterfactual DAS/IIA
**Validity layer:** Internal + Representational
**Criteria:** I2 Sufficiency

#### What it computes

Instead of patching a head's entire output, isolates the causal path from head A to head B. Corrupts head A's `hook_z` while freezing all other paths to clean values, then measures the effect at the model output routed through head B only. This quantifies the A \(\to\) B edge contribution:

$$
\text{path effect}(A \to B) = \frac{LD_{\text{clean}} - LD_{\text{path-patched}}}{LD_{\text{clean}}}
$$

Reports per-edge effects, identifies the strongest edges, and compares against random head pairs.

**What it establishes:**
- That specific edges between circuit heads carry causal information.
- The relative importance of different edges in the circuit graph.

**What it does not establish:**
- That the edge list is complete --- paths through MLP layers or multi-hop routes are not captured by pairwise path patching.

**Usage:**
```bash
uv run python 33_path_patching.py --tasks ioi sva --n-prompts 40
```

---

### Role Ablation

**Metric ID:** `role_ablation`
**Instrument:** Track 3 interventional
**Validity layer:** Internal
**Criteria:** I1 Necessity

#### What it computes

Given a mechanistic claim specification, ablates the heads (and optionally MLP layers or individual neurons) assigned to an intervention target role (e.g., "name movers") and measures the normalized effect on a measurement target (another role or the model output). Supports zero-ablation and mean-ablation.

$$
\text{effect} = \frac{LD_{\text{ablated}} - LD_{\text{clean}}}{|LD_{\text{clean}}|}
$$

In full-scan mode, ablates each role independently and reports all pairwise role-to-output effects.

**What it establishes:**
- That specific functional roles in the circuit are necessary for task performance.
- The causal direction and magnitude of each role's contribution.

**What it does not establish:**
- Sufficiency of the role or the completeness of the role assignment. A role may contain heads that contribute through unmeasured pathways.

**Usage:**
```bash
# Typically called by mv.verify(); for standalone:
mv run role_ablation --tasks ioi
```

---

## 3 --- Causal Scrubbing and Consistency

These metrics test circuit hypotheses under stricter criteria than single-component ablation.

### C4 --- Causal Scrubbing

**Metric ID:** `C4.causal_scrubbing`
**Instrument:** A01 --- SCM / Pearl Causal Hierarchy
**Validity layer:** Internal
**Criteria:** I1 Necessity, I2 Sufficiency

#### What it computes

Tests whether a circuit hypothesis explains model behavior under the scrubbing criterion. For each non-circuit head, resamples activations from a randomly chosen compatible prompt (different from the current one), then measures KL divergence between the clean and scrubbed output distributions:

$$
CS(H) = \mathbb{E}\left[ D_{\text{KL}}\left( P_{\text{model}}(\cdot \mid x) \;\|\; P_{\text{scrubbed}}(\cdot \mid x, H) \right) \right]
$$

Also reports logit-difference recovery: the fraction of clean logit difference preserved after scrubbing.

| Outcome | Interpretation |
|---|---|
| KL < 0.5 | Circuit hypothesis is explanatorily complete |
| KL > 2.0 | Significant information leaks through non-circuit heads |
| High recovery, low KL | Circuit captures the full computation |

**What it establishes:**
- Sufficiency of the circuit hypothesis at the causal-variable level (Pearl's Rung 3).
- That non-circuit components do not carry task-relevant information beyond what the hypothesis predicts.

**What it does not establish:**
- Uniqueness of the hypothesis --- multiple circuit specifications may achieve low KL.
- Correctness of the named causal variables.

**Usage:**
```bash
uv run python 04_causal_scrubbing.py --tasks ioi --n-prompts 20
```

---

### C34 --- Counterfactual Consistency

**Metric ID:** `C34.counterfactual_consistency`
**Instrument:** A02 --- Counterfactual DAS/IIA
**Validity layer:** Internal + Representational
**Criteria:** I2 Sufficiency

#### What it computes

Tests whether the circuit gives consistent results across paraphrased prompts. Generates prompts with five different random seeds, computes faithfulness and logit difference on each variant, and reports consistency as:

$$
\text{consistency} = 1 - \text{CV}(\text{faithfulness across seeds})
$$

where CV is the coefficient of variation. The overall consistency score averages faithfulness consistency and logit-difference consistency.

**What it establishes:**
- That the circuit captures an invariant computation, not surface features tied to specific prompt templates.
- That faithfulness measurements are stable across prompt variation.

**What it does not establish:**
- That the circuit generalizes to genuinely out-of-distribution inputs (only template variation is tested).

**Usage:**
```bash
uv run python 34_counterfactual_consistency.py --tasks ioi sva --n-prompts 40
```

---

### C29 --- Hyperparameter Sensitivity

**Metric ID:** `C29.hyperparam_sensitivity`
**Instrument:** A09 --- MDL/SLT
**Validity layer:** Construct
**Criteria:** C4 Minimality

#### What it computes

Tests sensitivity of faithfulness to evaluation hyperparameters. For each task, computes faithfulness under varied settings:

- **Prompt count:** \( n \in \{20, 40, 80\} \)
- **Ablation type:** zero, mean, mean-last-position

Reports the coefficient of variation (CV) across settings. Low CV indicates robust measurement; high CV indicates hyperparameter dependence.

| Outcome | Interpretation |
|---|---|
| CV < 0.10 | Faithfulness is robust to evaluation choices |
| CV > 0.30 | Results are unstable --- conclusions depend on specific hyperparameters |

**What it establishes:**
- Whether the faithfulness claim is robust to reasonable methodological variation.
- Which hyperparameters (prompt count vs. ablation type) most affect the measurement.

**What it does not establish:**
- That the circuit itself is robust --- only the *measurement* of faithfulness is tested.

**Usage:**
```bash
uv run python 29_hyperparam_sensitivity.py --tasks ioi sva
```

---

## 4 --- Causal Discovery

These metrics recover circuit structure from observational or optimization-based methods, without assuming the circuit is known in advance.

### C7 --- Observational Circuit Discovery (oCSE + Stability Selection)

**Metric ID:** `C7.ocse`
**Instrument:** A07 --- Granger/Transfer Entropy
**Validity layer:** Internal
**Criteria:** I4 Consistency

#### What it computes

Two complementary observational discovery methods, both operating on per-head direct logit attribution (DLA) features:

1. **Stability selection:** Bootstrap LassoCV (50 resamples) identifies heads whose DLA coefficients are stably nonzero. Heads selected in > 50% of bootstraps are included.
2. **Greedy oCSE:** Forward selection using Gaussian conditional mutual information with permutation-calibrated thresholds (95th percentile of null).

Compares each method's discovered set against the known circuit via precision, recall, and F1. Also reports the union of both methods.

**What it establishes:**
- Whether the known circuit structure is recoverable from purely observational data (no interventions).
- Convergent evidence: if observational and interventional methods agree, the circuit is robust to methodological choice.

**What it does not establish:**
- Causal direction --- observational methods recover associations, not do-calculus effects. High F1 means the circuit is *identifiable*, not necessarily *causal*.

**Usage:**
```bash
uv run python 07_ocse.py --tasks ioi sva --n-prompts 200
```

---

### C8 --- Partial Information Decomposition (PID)

**Metric ID:** `C8.pid`
**Instrument:** A08 --- Partial Information Decomposition
**Validity layer:** Internal
**Criteria:** I3 Specificity

#### What it computes

Decomposes the mutual information between pairs of circuit heads and the model output into four atoms:

- **Redundancy:** Information that either head alone provides about the output.
- **Unique \( X \) / Unique \( Y \):** Information that only head \( X \) (or \( Y \)) provides.
- **Synergy:** Information available only from the joint observation of both heads.

Uses the BROJA PID decomposition if the `dit` library is available; otherwise falls back to a binned mutual-information approximation with quantile discretization (\( n_{\text{bins}} = 5 \)).

**What it establishes:**
- Whether circuit head pairs carry unique vs. redundant vs. synergistic information about the output.
- High synergy indicates genuine computational interaction that linear attribution methods miss.

**What it does not establish:**
- The *mechanism* of the interaction. PID quantifies the information structure but not the computational algorithm.

**Usage:**
```bash
uv run python 08_pid.py --tasks ioi sva --n-prompts 60
```

---

### C9 --- NOTEARS Structure Learning

**Metric ID:** `C9.notears`
**Instrument:** A13 --- Causal Discovery
**Validity layer:** Internal
**Criteria:** I4 Consistency

#### What it computes

Learns a DAG over component activations using continuous optimization (Zheng et al., NeurIPS 2018). Solves:

$$
\min_{W} \; \frac{1}{2n} \|X - XW\|_F^2 + \lambda_1 \|W\|_1 \quad \text{s.t.} \quad \text{tr}(e^{W \circ W}) - d = 0
$$

where the constraint enforces acyclicity. Identifies which heads are "causal parents" of the logit difference (the last column of the data matrix). Compares the discovered parent set against the known circuit via precision, recall, and F1.

Includes a permutation baseline: shuffles the activation matrix and re-runs NOTEARS to calibrate the expected number of spurious parents.

**What it establishes:**
- Whether continuous DAG optimization recovers the known circuit structure from activation data.
- The Structural Hamming Distance (SHD) between the learned and known graphs.

**What it does not establish:**
- That the learned DAG is the *true* causal graph. NOTEARS assumes linearity and Gaussian noise, which may not hold for transformer activations.

**Usage:**
```bash
uv run python 09_notears.py --tasks ioi sva --n-prompts 80
```

---

### C42 --- PC Algorithm

**Metric ID:** reported in `a13_pc_algorithm.json`
**Instrument:** A13 --- Causal Discovery
**Validity layer:** Internal
**Criteria:** I4 Consistency

#### What it computes

Runs the PC algorithm (Spirtes, Glymour & Scheines, 2000) on per-head DLA data. PC discovers the causal DAG skeleton using conditional independence (CI) tests: it starts with a complete graph and removes edges whenever a partial correlation is non-significant at level \( \alpha \).

Applied to circuits: collects DLA features for circuit heads plus their neighbors, runs PC, and compares the discovered undirected skeleton against known circuit edges via precision, recall, F1, and SHD.

Key insight: PC discovers structure *without* interventions. If its output matches the intervention-based circuit, the causal structure is identifiable from observations alone.

**What it establishes:**
- Whether the circuit's conditional independence structure matches expectations from the known circuit graph.
- Agreement between constraint-based (PC) and optimization-based (NOTEARS) discovery methods provides convergent evidence.

**What it does not establish:**
- Edge directionality --- PC produces a CPDAG (partially directed graph), not a fully directed DAG. Some edges remain undirected.

**Usage:**
```bash
uv run python 42_pc_algorithm.py --tasks ioi sva --n-prompts 100 --alpha 0.05
```

---

### C7 (EAP) --- Edge Attribution Patching

**Metric ID:** `C7.eap_auroc`
**Instrument:** C07 --- Edge Attribution Patching
**Validity layer:** Internal
**Criteria:** C7 Edge Attribution Discrimination

#### What it computes

Computes edge-level attribution scores between all pairs of attention heads using the EAP method (Syed et al., 2023). For each directed edge (sender \(\to\) receiver), the score is the mean dot product between the sender's `hook_z` output and the receiver's `hook_z` gradient at the last token position:

$$
\text{EAP}(s \to r) = \frac{1}{n} \sum_{i=1}^{n} z_s^{(i)} \cdot \nabla_{z_r} \, LD^{(i)}
$$

Circuit edges are treated as positives, all other forward edges as negatives. AUROC measures discrimination.

| Outcome | Interpretation |
|---|---|
| AUROC > 0.70 (pass) | EAP scores reliably distinguish circuit edges from non-circuit edges |
| AUROC < 0.60 | Gradient-based attribution fails to identify the circuit's edge structure |

**What it establishes:**
- That gradient-based edge attribution is consistent with interventional circuit discovery.
- A fast, single-backward-pass approximation to path patching.

**What it does not establish:**
- Causal sufficiency or necessity of individual edges --- EAP is a first-order approximation that misses nonlinear interactions.

**Usage:**
```bash
uv run python 91_eap.py --tasks ioi --n-prompts 40
```

---

## 5 --- Complexity, Transportability, and Structural Metrics

These metrics test whether circuit properties generalize beyond the specific model and task configuration they were discovered in.

### C10 --- Local Learning Coefficient (LLC)

**Metric ID:** `C10.llc`
**Instrument:** A09 --- MDL/SLT
**Validity layer:** Construct
**Criteria:** C4 Minimality

#### What it computes

Measures the effective geometric complexity of each circuit component at the current model weights using Singular Learning Theory. Approximates the LLC via the Fisher information diagonal (variance of per-prompt gradients):

$$
\hat{\lambda} = \frac{\text{effective rank of Fisher}}{2 \ln n}
$$

where effective rank counts eigenvalues above 1% of the maximum. Compares mean LLC for circuit heads against a random sample of non-circuit heads.

| Pattern | Interpretation |
|---|---|
| Circuit LLC < non-circuit LLC | Circuit heads are more specialized (lower degeneracy) |
| Circuit LLC > non-circuit LLC | Circuit heads are more polyfunctional or geometrically complex |
| Ratio near 1.0 | No structural distinction between circuit and non-circuit heads |

**What it establishes:**
- Whether circuit components have measurable geometric specialization compared to non-circuit components.
- A structural (non-interventional) signature of circuit membership.

**What it does not establish:**
- Causal role --- LLC measures local loss-landscape geometry, not functional contribution.

**Usage:**
```bash
uv run python 10_llc.py --tasks ioi sva --n-prompts 40
```

---

### C38 --- Cross-Model Invariance

**Metric ID:** `C38.90_configural_invariance`, `C38.91_metric_invariance`, `C38.92_scalar_invariance`, `C38.93_cross_model_weight_alignment`, `C38.95_scale_invariance`
**Instrument:** A12 --- Transportability
**Validity layer:** External
**Criteria:** E5/E6 Cross-model

#### What it computes

Tests whether circuit properties are invariant across GPT-2 model sizes (small, medium, large, XL). Since circuit definitions exist only for GPT-2 Small, larger models use activation patching to identify top-\( k \) heads as proxy circuits.

Five sub-metrics implement a measurement-invariance hierarchy:

| Sub-metric | Tests |
|---|---|
| **#90 Configural** | Same layers contain circuit heads? (Spearman correlation of layer histograms) |
| **#91 Metric** | Faithfulness ranking preserved across sizes? (Spearman rank correlation) |
| **#92 Scalar** | Absolute faithfulness values comparable? (mean and std across sizes) |
| **#93 Weight alignment** | Cosine similarity of \( W_{OV} \) singular-value distributions across sizes |
| **#95 Scale** | Faithfulness vs. parameter count (slope of log-log regression) |

**What it establishes:**
- Whether the circuit concept generalizes across model scale, not just within a single model.
- Which level of invariance holds (configural > metric > scalar, in increasing strength).

**What it does not establish:**
- That the proxy circuits in larger models are the "true" circuits --- they are identified by activation patching heuristics.

**Usage:**
```bash
uv run python 38_cross_model_invariance.py --tasks ioi sva --device cuda
uv run python 38_cross_model_invariance.py --tasks ioi --skip-large --device cpu
```

---

### C41 --- Transportability

**Metric ID:** reported in `a12_transportability.json`
**Instrument:** A12 --- Transportability
**Validity layer:** External
**Criteria:** E5/E6 Cross-model

#### What it computes

Tests whether causal circuit findings from GPT-2 Small transport to larger models using Pearl and Bareinboim's (2014) transportability framework. Three quantities are assessed:

1. **Structure:** Do the same relative layer depths contain circuit heads? Measured by cosine similarity of thirds-distributions (early/mid/late).
2. **Effect:** Is normalized faithfulness comparable across sizes?
3. **Specificity:** Do the same relative head rankings hold?

Runs activation patching on target models to identify top-\( k \) heads, then computes structural similarity against the source circuit.

**What it establishes:**
- That circuit-level causal findings are not artifacts of a specific model size.
- A quantitative measure of how well structural patterns transport.

**What it does not establish:**
- That the transported circuit performs the same *computation* --- structural similarity does not imply functional equivalence.

**Usage:**
```bash
uv run python 41_transportability.py --tasks ioi sva --target-models gpt2-medium gpt2-large
```

---

### A4 --- Intermediate State Prediction

**Metric ID:** `A4.intermediate_state_prediction`
**Instrument:** A02 --- Counterfactual DAS/IIA
**Validity layer:** Internal
**Criteria:** A4 Intermediate State Prediction

#### What it computes

For each circuit edge (sender \(\to\) receiver), computes scalar logit attributions for both heads across prompts (\( z \cdot W_O \cdot (W_U[\text{correct}] - W_U[\text{incorrect}]) \)), then measures Spearman rank correlation between sender and receiver attributions. Compares mean pathway correlation against a baseline of random non-circuit head pairs.

Pass condition: mean pathway \( \rho > 0.3 \) AND uplift over baseline \( > 0.15 \).

**What it establishes:**
- That sender and receiver heads in a circuit edge co-vary in a predictable, task-aligned way.
- That the pathway structure predicted by the circuit hypothesis is reflected in activation statistics.

**What it does not establish:**
- Causal direction --- correlation between sender and receiver attributions does not prove information flow through the edge.

**Usage:**
```bash
uv run python 79_intermediate_state_prediction.py --tasks ioi sva --n-prompts 60
```

---

## 6 --- Baseline Metrics

### Logit Difference

**Metric ID:** `logit_diff`
**Validity layer:** N/A (baseline)

#### What it computes

Measures the mean logit difference between the correct and incorrect tokens across prompts:

$$
LD = \text{logit}[\text{correct}] - \text{logit}[\text{incorrect}]
$$

This is the standard IOI metric from Wang et al. (2023). It is not a validity metric itself but a prerequisite: if the model does not solve the task (\( LD \leq 0 \)), causal metrics are uninterpretable.

**Usage:**
```bash
mv run logit_diff --tasks ioi sva
```

---

## Reading the scores together

The causal metrics form a hierarchy of evidential strength. The table below shows how patterns across metrics map to circuit validity claims.

| Evidence pattern | What it establishes |
|---|---|
| C2 pass + C20 pass | Circuit heads are both necessary (patching) and sufficient (restore). Strongest single-model causal evidence. |
| C1 pass + C15 neuron-IIA pass | Causal variable is encoded in a learnable subspace AND accessible at the raw neuron level. |
| C4 low KL + C34 high consistency | Circuit hypothesis is explanatorily complete and invariant to prompt paraphrases. |
| C7 oCSE pass + C9 NOTEARS pass | Observational and optimization-based discovery converge on the same structure. |
| C33 high edge effects + EAP AUROC pass | Both interventional and gradient-based methods identify the same circuit edges. |
| C29 low CV + C38 metric invariance | Faithfulness is robust to hyperparameters and invariant across model sizes. |
| C31 joint IIA pass + C32 high specificity | Multi-variable encoding is separable and task-specific. |
| C8 high synergy + C33 strong edges | Information-theoretic and interventional methods agree on head interactions. |
| C10 circuit LLC < non-circuit LLC | Circuit heads are structurally specialized, providing weight-space corroboration of activation-based findings. |
| C20 pass + C2 fail | Circuit is sufficient but not necessary --- redundant pathways exist. Qualifies the circuit claim. |
| C4 high KL + C2 pass | Components matter individually but the circuit hypothesis mis-specifies how they interact. |

---

## Relationship to other lenses

The MI causal metrics establish the interventional core of circuit validity. They interact with other lenses as follows:

- **Neuroscience lens:** Neuroscience metrics (dissociation, specificity, lesion studies) provide the experimental-design framework that causal metrics operationalize. MI causal metrics *are* the neuroscience of transformers --- activation patching is the neural lesion study, causal scrubbing is the double dissociation.
- **Pharmacology lens:** Pharmacology asks "how much intervention is needed to change behavior?" (dose-response, EC50). Causal metrics ask "which components, when intervened on, change behavior?" They are complementary: pharmacology quantifies effect magnitude, causal metrics identify effect location.
- **Philosophy of science lens:** Counterfactual consistency (C34) and hyperparameter sensitivity (C29) operationalize Woodward's interventionist account of causation and Ioannidis's replicability criteria. The causal metrics provide the raw interventional data; the philosophy lens judges whether those data license causal claims.
- **Economics lens:** Pairwise ablation synergy (PAS) and Shapley interactions from the economics lens decompose the second-order structure that first-order causal metrics (C2, C20) cannot capture. PID (C8) provides the information-theoretic complement to PAS.
- **Measurement theory lens:** Cross-model invariance (C38) implements configural/metric/scalar invariance from psychometric measurement theory. The transportability metric (C41) extends this from invariance *testing* to causal *transportability* in Pearl's sense.

---

## 6 --- Circuit Discovery & Attribution

These metrics wrap specific MI discovery and attribution techniques. They produce causal evidence (component importance scores, circuit graphs) but are organized here by the tool used. See the [Methods Index](/framework/evidence/methods-index) for a cross-reference by technique.

### C08 -- Sparse Feature Circuits (`92_sfc.py`)

**What it computes.** For each task, runs the model on prompts while collecting activations at the artifact's hook point(s). Activations are encoded through the artifact adapter (SAE, transcoder, crosscoder, or factor bank) to get per-feature activations. Each feature's causal importance is estimated via gradient-based attribution: feature activation multiplied by the gradient of logit difference with respect to that feature. Features are mapped to layers and heads based on hook point location. The metric then measures whether high-importance features concentrate in circuit heads using AUROC, and computes feature mass concentration within circuit heads.

$$
\text{importance}_f = |a_f \cdot \nabla_{a_f} \text{LD}|
$$

where $a_f$ is feature $f$'s activation and $\text{LD}$ is the logit difference.

**Evidence family.** Causal (gradient-based attribution).

**Key metrics.**
| Metric | Description | Pass threshold |
|---|---|---|
| `sfc_auroc` | AUROC of head-level importance scores discriminating circuit vs non-circuit heads | $> 0.65$ |
| `concentration` | Fraction of total feature attribution mass in circuit heads | reported |

**What it establishes.** Feature-level attribution through a learned dictionary (SAE/transcoder) identifies features concentrated in the claimed circuit heads. This provides convergent validity between feature-level and head-level circuit descriptions -- if the same components emerge from both granularities, the circuit claim is more robust.

**What it does not establish.** That the features themselves are causally necessary. Gradient-based attribution is a first-order approximation; it can miss nonlinear interactions and overweight features that happen to have large gradients but small behavioral effects. Requires an artifact adapter (SAE, transcoder, etc.) -- results depend on the quality of the learned dictionary.

**Usage.**
```bash
uv run python 92_sfc.py --tasks ioi --n-prompts 40
```

---

### C10 -- Automatic Circuit Discovery / ACDC (`94_acdc.py`)

**What it computes.** Implements a simplified version of ACDC (Conmy et al., NeurIPS 2023). Starting from the full computation graph of attention head edges, iteratively prunes edges whose removal causes less than a threshold increase in KL divergence between ablated and clean logits. For each candidate forward edge (sender head to receiver head), ablates the sender's output at its own layer and measures the resulting KL divergence from clean logits. Edges below the threshold are pruned, and the surviving edges form the ACDC-discovered circuit. Agreement with the claimed circuit is measured via Jaccard overlap and AUROC.

**Evidence family.** Causal (iterative KL-divergence edge pruning).

**Key metrics.**
| Metric | Description | Pass threshold |
|---|---|---|
| `jaccard` | Jaccard overlap between ACDC-discovered and claimed circuit edges | $> 0.3$ |
| `auroc` | AUROC of per-edge KL-impact scores discriminating circuit from non-circuit edges | reported |

**What it establishes.** An independent, automated circuit discovery method recovers edges that overlap with the claimed circuit. Since ACDC uses a different algorithmic approach (greedy KL-based pruning) than the original circuit discovery method, agreement provides convergent validity. High Jaccard indicates the circuit is robust to discovery methodology.

**What it does not establish.** That the ACDC-discovered circuit is the unique or optimal circuit. Different thresholds yield different circuits. Low Jaccard may reflect threshold sensitivity rather than a genuine discrepancy between the claimed and true circuit.

**Usage.**
```bash
uv run python 94_acdc.py --tasks ioi --n-prompts 40 --threshold 0.01
```

---

### C11 -- Relevance Patching / LRP (`95_relp.py`)

**What it computes.** Implements Layer-wise Relevance Propagation (LRP) for computing edge-level attribution scores between attention heads. For each prompt, runs a forward pass caching head outputs and attention patterns, then seeds relevance at each head from the gradient of logit-diff with respect to hook_z. Relevance propagates backward layer-by-layer: each receiver head distributes its relevance to sender heads in proportion to $\text{attention\_weight} \times \|z_{\text{sender}}\|$, approximating the LRP z-rule conservation principle. Circuit edges are treated as positives, all other forward edges as negatives.

**Evidence family.** Causal (gradient-based attribution with LRP conservation).

**Key metrics.**
| Metric | Description | Pass threshold |
|---|---|---|
| `auroc` | AUROC of relevance flow scores discriminating circuit from non-circuit edges | $> 0.70$ |
| `pearson_correlation` | Pearson $r$ between binary circuit membership and relevance magnitude | reported |

**What it establishes.** LRP-based relevance scores, which obey conservation (total relevance is preserved across layers), concentrate on circuit edges. Conservation provides a formal guarantee that attribution is not "created" at intermediate layers -- it is merely redistributed. This makes LRP-based agreement stronger than raw gradient-based methods.

**What it does not establish.** That the LRP decomposition reflects the true information flow. The z-rule approximation makes linearity assumptions that may not hold for attention mechanisms with softmax nonlinearities.

**Usage.**
```bash
uv run python 95_relp.py --tasks ioi --n-prompts 40
```

---

### C12 -- Contextual Decomposition (`96_contextual_decomposition.py`)

**What it computes.** Implements Contextual Decomposition for Transformers (CD-T), following Hsu & Yu et al. (ICLR 2025). For each attention head, extracts the head's hook_z output at the last token position, projects it through $W_O$ and the unembedding matrix $W_U$, and measures the signed contribution to the correct-versus-incorrect logit difference. This is a closed-form computation requiring only a single forward pass per prompt -- no gradient computation, no training.

$$
\text{CD}(L, H) = z_{L,H}^{(\text{last})} \cdot W_O^{(L,H)} \cdot W_U \cdot (e_{\text{correct}} - e_{\text{incorrect}})
$$

**Evidence family.** Structural (closed-form decomposition).

**Key metrics.**
| Metric | Description | Pass threshold |
|---|---|---|
| `auroc` | AUROC of absolute contribution scores discriminating circuit from non-circuit heads | $> 0.75$ |

**What it establishes.** A gradient-free, closed-form decomposition identifies the claimed circuit heads as having high direct logit contributions. The higher threshold ($0.75$) reflects that CD-T is more faithful than gradient approximations (Hsu & Yu et al. report substantially higher correlation with activation patching ground truth).

**What it does not establish.** Interactions between heads. CD-T decomposes each head's contribution independently; synergistic effects (where two heads jointly contribute more than the sum of their individual contributions) are not captured.

**Usage.**
```bash
uv run python 96_contextual_decomposition.py --tasks ioi --n-prompts 40
```

---

### C13 -- Information Bottleneck Circuit Discovery (`97_information_bottleneck.py`)

**What it computes.** Implements a simplified IBCircuit approach (Bian et al., ICML 2025). Learns binary edge masks between attention heads via the Hard Concrete distribution (Louizos et al., 2018), optimizing:

$$
\mathcal{L} = D_{\text{KL}}(P_{\text{full}} \| P_{\text{masked}}) - \beta \cdot \frac{\mathbb{E}[\|z\|_0]}{|\text{edges}|}
$$

where the first term measures faithfulness (how well the masked circuit approximates the full model) and the second penalizes circuit size. Edge mask logits are optimized by gradient descent. Discovered edges above a probability threshold ($0.5$) form the IB-discovered circuit.

**Evidence family.** Causal (learned intervention masks).

**Key metrics.**
| Metric | Description | Pass threshold |
|---|---|---|
| `jaccard` | Jaccard overlap between IB-discovered and claimed circuit edges | $> 0.25$ |
| `faithfulness` | Fraction of logit-diff preserved when keeping only discovered edges | reported |
| `precision` / `recall` | Edge-level precision and recall against claimed circuit | reported |

**What it establishes.** An information-theoretic optimization -- explicitly trading faithfulness against compression -- recovers edges that overlap with the claimed circuit. The IB framing provides a principled criterion for circuit size: the discovered circuit is the minimal subset of edges that preserves the model's behavior.

**What it does not establish.** That the IB-optimal circuit is unique. The optimization landscape may have multiple local minima corresponding to different faithful sub-circuits. The lower threshold ($0.25$) reflects that IB circuits tend to be sparser than the claimed circuit.

**Usage.**
```bash
uv run python 97_information_bottleneck.py --tasks ioi --n-prompts 40
```

---

### C14 -- Position-Aware Edge Attribution Patching (`98_position_aware_eap.py`)

**What it computes.** Extends standard EAP (Syed et al., 2023) following Haklay & Belinkov (ACL 2025 Oral) by treating each (head, position) pair as a distinct node. For each prompt, computes hook_z outputs and their gradients at every position. Edge scores between sender (layer $L_s$, head $H_s$, position $p_s$) and receiver (layer $L_r$, head $H_r$, position $p_r$) are computed as the dot product $z_s[p_s] \cdot \nabla_{z_r}[p_r]$. These position-level scores are aggregated to head-level via two methods: position-averaged and max-over-positions.

**Evidence family.** Causal (gradient-based edge attribution).

**Key metrics.**
| Metric | Description | Pass threshold |
|---|---|---|
| `auroc_max` | AUROC using max-over-positions aggregation | $> 0.70$ |
| `auroc_avg` | AUROC using position-averaged aggregation | reported |

**What it establishes.** Position-aware attribution provides finer-grained edge scores than standard EAP. If position-aware scores discriminate circuit edges, it indicates the circuit's information flow is localized at specific token positions -- not diffusely spread across the sequence.

**What it does not establish.** Which positions are critical. The max-over-positions aggregation identifies the existence of a discriminative position but does not characterize it. Inspect the per-position scores in the metadata for position-level analysis.

**Usage.**
```bash
uv run python 98_position_aware_eap.py --tasks ioi --n-prompts 40
```

---

### C18 -- Adversarial Parameter Decomposition / VPD (`105_vpd.py`)

**What it computes.** Implements VPD (Bushnaq, Braun, Sharkey -- Goodfire AI, 2026). For each circuit-relevant layer: extracts Q/K/V/O weight matrices, decomposes each into rank-1 subcomponents via SVD, and for each subcomponent finds the adversarial ablation direction that maximally damages model behavior. A subcomponent counts as important only if behavior degrades even under the adversarially chosen ablation -- this is strictly harder than standard ablation, where a favorable ablation direction might leave behavior intact.

**Evidence family.** Causal (adversarial weight-space ablation).

**Key metrics.**
| Metric | Description | Pass threshold |
|---|---|---|
| `adversarial_faithfulness` | Fraction of adversarially-important subcomponents in circuit heads | $> 0.5$ |
| `auroc` | AUROC of head-level adversarial importance discriminating circuit vs non-circuit | reported |
| `concentration` | Fraction of total importance in circuit heads | reported |

**What it establishes.** Circuit heads contain weight subcomponents that are adversarially necessary -- they cannot be removed by any ablation direction without degrading behavior. This is a stronger claim than standard ablation necessity, which only tests one ablation direction (mean, zero, or resample).

**What it does not establish.** That individual rank-1 subcomponents correspond to interpretable features. SVD produces a mathematically convenient basis but not necessarily a semantically meaningful one. The adversarial search explores a finite number of directions and may miss the true worst case.

**Usage.**
```bash
uv run python 105_vpd.py --tasks ioi --n-prompts 40
```

---

### C19 -- RelP Faithfulness (`106_relp.py`)

**What it computes.** Implements Relevance Patching (RelP) from Jafari, Eberle, Khakzar & Nanda (NeurIPS 2025 Workshop Spotlight). RelP replaces local gradients in standard attribution patching with LRP epsilon-rule coefficients. For each component $c$ with output activation $a_c$ and gradient $g_c$:

$$
R_c = \sum_d \frac{a_{c,d} \cdot g_{c,d}}{|a_{c,d}| + \epsilon}
$$

Both activation patching ground truth and RelP scores are computed, and their Pearson correlation is reported. On GPT-2 Large MLP outputs, attribution patching achieves $r = 0.006$ with activation patching ground truth; RelP achieves $r = 0.956$.

**Evidence family.** Measurement (faithfulness of attribution method).

**Key metrics.**
| Metric | Description | Pass threshold |
|---|---|---|
| `pearson_r` | Pearson correlation between RelP and activation patching ground truth | $> 0.8$ |
| `per_layer` | Per-layer Pearson correlations | reported |

**What it establishes.** The LRP-based attribution method faithfully approximates activation patching ground truth at per-component granularity. This is a measurement validity metric -- it validates the attribution method itself, not the circuit. If RelP correlates highly with activation patching, it can serve as a cheap proxy for expensive per-component patching.

**What it does not establish.** Circuit validity directly. RelP faithfulness is about whether the measurement tool works, not whether the circuit is correct. A faithful attribution method applied to a wrong circuit will faithfully identify the wrong components.

**Usage.**
```bash
uv run python 106_relp.py --tasks ioi --n-prompts 40
```

---

### C20 -- CircuitLens Weight-Based Circuit Recovery (`119_circuitlens.py`)

**What it computes.** Implements the CircuitLens approach (Golimblevskaia et al., ICLR 2026). For each feature at layer $L$: (1) extracts the feature's decoder direction from the artifact (SAE/transcoder); (2) projects that direction through the next layer's MLP input weights to get downstream connectivity -- a weight-derived circuit edge; (3) thresholds connectivity scores to produce a sparse weight-derived circuit. An activation-derived circuit is computed via gradient-based attribution for comparison. Agreement between the two circuits is measured as Jaccard similarity of the top-connected feature sets.

**Evidence family.** Structural (weight-space connectivity) + convergent validity.

**Key metrics.**
| Metric | Description | Pass threshold |
|---|---|---|
| `weight_activation_circuit_overlap` | Mean Jaccard similarity between weight-derived and activation-derived circuits | $> 0.2$ |

**What it establishes.** Weight connectivity alone -- without running any activations -- can recover meaningful circuit structure. Agreement between weight-derived and activation-derived circuits provides structural convergent validity: the circuit is not merely an activation-time phenomenon but is grounded in the model's parameterization.

**What it does not establish.** That weight connectivity predicts activation-time behavior with high fidelity. Weights define the computational capacity of a component, but whether that capacity is exercised depends on the input distribution. The modest threshold ($0.2$) reflects that weight-derived and activation-derived circuits are expected to diverge somewhat.

**Usage.**
```bash
uv run python 119_circuitlens.py --tasks ioi --n-prompts 50
```
