# MI Deep Survey Part IX: Extraction

Extracted from "MI Methods Deep Survey Part IX --- Infrastructure in
Crisis: SAEBench Audit, MechEvalAgent, Atlas-Alignment, MOT, and the
Safety Subspace Cluster" (May 2026).

Covers 8 papers from late 2025 through May 2026, constituting a
meta-level reckoning with MI's own evaluation infrastructure. The
unifying theme: the benchmarks used to evaluate SAE quality are
themselves unreliable, 93% of MI research fails reproducibility, and
safety representations are the most validity-passing feature class in
the literature. Each finding has direct framework implications.

---

## A. New Methods to Implement

### Tier 1: Critical (directly maps to framework criteria or fills gap)

| Method | What | Source | Lens | Metric type |
|---|---|---|---|---|
| SAEBench Reliability Audit | Audits SAE evaluation metrics via reseed noise, ground-truth correlation, and discriminability; TPP and SCR fail comprehensively; sae-probes most reliable | arXiv:2605.18229 (May 2026) | measurement_theory | benchmark |
| MechEvalAgent Reproducibility | Execution-grounded evaluation of MI research: 93% fail reproducibility, 80% fail coherence, 51 issues missed by narrative-only review | Bai, Baumgartner, Sun, Holtzman, Tan; arXiv:2602.18458 (Feb 2026) | measurement_theory | evaluation |

### Tier 2: High (new methods with clear validity framework mapping)

| Method | What | Source | Lens | Metric type |
|---|---|---|---|---|
| Atlas-Alignment | Concept Atlas from one model transfers labels to new models via representational alignment (CKA, RSA, linear regression); solves "transparency tax" | Puri, Berend, Lapuschkin, Samek; ICLR 2026 Re-Align (arXiv:2510.27413) | mechanistic_interpretability | evaluation |
| MOT Cross-Architecture Alignment | Multi-Level Optimal Transport: global alignment score for architecturally different networks; handles depth mismatch via mass distribution | Shah, Khosla; ICLR 2026 Poster (arXiv:2510.01706) | mechanistic_interpretability | evaluation |
| Safety Subspace Identification | SAE features identify safety-relevant directions; low-rank safety subspace constructed via PCA; causal validation confirms safety features are causally responsible | NCSU; arXiv:2512.23260 (Dec 2025) | mechanistic_interpretability | evaluation |
| Safety Singular Value Entropy | Safety subspace stable under fine-tuning; new metric quantifies how densely safety info is packed across layers; enables dynamic rank selection | arXiv:2602.00038 (Jan 2026) | measurement_theory | evaluation |

### Tier 3: Medium (useful reference points, less directly implementable)

| Method | What | Source | Lens | Metric type |
|---|---|---|---|---|
| SaLoRA Orthogonality Constraint | Safety module orthogonal to LoRA task updates; C4 discriminant validity by construction | CISPA; arXiv:2501.01765 | mechanistic_interpretability | steering |
| Safety at One Shot | Single safety example recovers full alignment via low-rank gradient structure; E2 compact causal sufficiency | arXiv:2601.01887 (May 2026) | mechanistic_interpretability | evaluation |
| Human Alignment Interpretability | Aligned DINOv2 more interpretable but less visually rich; first controlled causal evidence that alignment causes interpretability | Colin, Oliver, Serre; ICLR 2026 Re-Align | mechanistic_interpretability | evaluation |

---

## B. New Metrics (suggested implementations)

### B1. SAEBench Reliability Audit

- **metric_id**: `EX24_saebench_reliability_audit`
- **lens**: measurement_theory
- **validity_type**: Measurement (M1 Reliability)
- **criteria**: M1 Reliability; M2 Measurement Invariance
- **what it measures**: For each SAE evaluation metric in the evaluation pipeline, computes three audit diagnostics: (1) reseed coefficient of variation (CV) --- run the metric N times with different random seeds and measure variance, (2) ground-truth correlation --- on synthetic SAEs with known quality, check monotonic correlation with true quality, (3) discriminability --- whether the metric can distinguish SAEs that differ by other validated criteria.
- **pass condition**: CV < 5%; ground_truth_correlation > 0.5; discriminability_auc > 0.7
- **implementation notes**: The reseed CV is the most actionable diagnostic. For each metric in the suite, run it 5+ times with different seeds and compute CV. Metrics with CV > 10% (like TPP at 16--39%) should be flagged as unreliable. This is a meta-metric: it evaluates other metrics.
- **why critical**: The SAEBench audit paper is the strongest single-paper argument for the validity framework. TPP and SCR --- used by dozens of papers --- fail all three audit lenses. This metric operationalizes the audit as a reusable diagnostic.

### B2. Reproducibility Check (MechEvalAgent-inspired)

- **metric_id**: `EX25_reproducibility_check`
- **lens**: measurement_theory
- **validity_type**: Measurement (M1 Reliability)
- **criteria**: M1 Reliability (test-retest)
- **what it measures**: For a given metric computation pipeline, runs it twice on the same inputs with different random seeds and measures output deviation. Reports: (1) deviation_rate --- fraction of outputs that differ by more than a threshold, (2) max_deviation --- largest single-output deviation, (3) coherence_score --- whether the ranking of features/components is preserved across runs.
- **pass condition**: deviation_rate < 0.05; max_deviation < 0.08; coherence_score > 0.9
- **implementation notes**: This is a lightweight version of MechEvalAgent's execution-grounded check. Rather than evaluating a full paper, it evaluates a single metric computation for reproducibility. The coherence score (rank correlation between two runs) is the most informative diagnostic: even if absolute values differ, preserved rankings indicate reliable relative assessments.
- **why critical**: 93% of MI research fails reproducibility. This metric operationalizes reproducibility as a first-class validity criterion that can be applied to any metric in the framework.

### B3. Atlas-Alignment Cross-Model Convergence

- **metric_id**: `EX26_atlas_alignment`
- **lens**: mechanistic_interpretability
- **validity_type**: Construct (C5 Convergent Validity)
- **criteria**: C5 Convergent Validity (cross-model)
- **what it measures**: For pairs of models, computes representational alignment at each layer using CKA on shared input activations. The atlas_alignment_score is the maximum CKA across layer pairs. The feature_transfer_rate measures what fraction of interpretable directions in the reference model have a high-CKA correspondent in the target model.
- **pass condition**: atlas_alignment_score > 0.3; feature_transfer_rate > 0.5
- **implementation notes**: Uses existing CKA infrastructure. The key innovation over standard CKA is the asymmetric "reference atlas" framing: one model is the reference with labeled features, and alignment measures whether those labels transfer. For the minimal implementation, uses layer-wise CKA on shared prompts between two instances of the same architecture (different seeds) or two architectures.
- **why critical**: Atlas-Alignment operationalizes C5 as a production tool. If a feature in a new model aligns to a Concept Atlas concept, that alignment is cross-model convergent evidence.

### B4. MOT Global Alignment Score

- **metric_id**: `EX27_mot_alignment`
- **lens**: mechanistic_interpretability
- **validity_type**: Construct (C5 Convergent Validity)
- **criteria**: C5 Convergent Validity (cross-architecture)
- **what it measures**: Computes a global alignment score between two networks using optimal transport over their layer-wise representations. Unlike greedy layer-wise CKA, MOT distributes representation mass across multiple target layers, yielding a single global score and soft layer-to-layer couplings. Handles depth mismatches natively.
- **pass condition**: mot_global_score > 0.3 (two independently trained networks of the same architecture show representational convergence)
- **implementation notes**: Requires solving an optimal transport problem (Sinkhorn iterations) over CKA similarity matrices. Each source layer distributes its mass across target layers weighted by CKA. The transport cost is the global alignment score. Computationally more expensive than CKA alone but provides a single, globally consistent comparison.
- **why critical**: The most powerful C5 upgrade for cross-architecture comparison. MOT handles depth mismatches that CKA cannot.

### B5. Safety Subspace Causal Validation

- **metric_id**: `EX28_safety_subspace`
- **lens**: mechanistic_interpretability
- **validity_type**: Internal (I1 Necessity, I2 Sufficiency)
- **criteria**: E2 Causal Sufficiency; I1 Component Necessity
- **what it measures**: Identifies safety-relevant directions in the residual stream by contrasting activations on safe vs. unsafe prompts. Constructs a low-rank subspace via PCA of the contrast directions. Tests causal sufficiency by projecting activations onto the safety subspace and measuring whether the projection alone predicts safe/unsafe behavior. Tests necessity by ablating the safety subspace and measuring safety degradation.
- **pass condition**: causal_sufficiency > 0.6 (safety subspace projection predicts safety with >60% accuracy); ablation_deficit > 0.3 (removing the safety subspace degrades safety by >30%)
- **implementation notes**: Requires paired safe/unsafe prompts (e.g., "How do I bake a cake?" vs. harmful requests with refusal). The subspace is constructed from mean difference directions at each layer, then PCA-reduced. The causal tests use activation patching: replace activations in the safety subspace and measure behavior change.
- **why critical**: Safety representations pass E2, M1, and C4 criteria better than any other feature class in the literature. This metric verifies this claim for any given model.

### B6. Safety Singular Value Entropy

- **metric_id**: `M14_safety_sve`
- **lens**: measurement_theory
- **validity_type**: Measurement (M1 Reliability)
- **criteria**: M1 Reliability; M6 Construct Coverage
- **what it measures**: For the safety subspace identified via contrast directions, computes singular value entropy across layers. Low entropy means safety information is concentrated in few dimensions (compact); high entropy means it is spread across many dimensions (diffuse). Also measures stability: repeats the analysis after fine-tuning on non-harmful data and checks whether the safety subspace principal components remain stable.
- **pass condition**: safety_sve < 2.0 (safety is compact, not diffuse); stability_correlation > 0.8 (safety subspace is preserved after fine-tuning)
- **implementation notes**: Computes SVD of the safety direction matrix across layers. Entropy = -sum(p_i * log(p_i)) where p_i = s_i^2 / sum(s_j^2). Stability test requires a fine-tuned variant of the model (or simulated perturbation).
- **why critical**: New metric from the LSSF paper. Quantifies how densely safety information is packed, enabling dynamic rank selection for safety-preserving interventions.

### B7. Single-Shot Safety Recovery

- **metric_id**: `EX29_safety_one_shot`
- **lens**: mechanistic_interpretability
- **validity_type**: Internal (I2 Sufficiency)
- **criteria**: E2 Causal Sufficiency (compact)
- **what it measures**: Tests whether a single safety example can recover safety alignment after perturbation. Perturbs the model's safety subspace (via random direction injection), then computes the gradient of a single safety example. Measures (1) gradient alignment with the safety subspace (cosine similarity), and (2) recovery rate (fraction of safety behavior restored by a single gradient step in the safety direction).
- **pass condition**: gradient_alignment > 0.5; recovery_rate > 0.3
- **implementation notes**: A lightweight proxy for the full "Safety at One Shot" procedure. Rather than full fine-tuning, uses activation-level perturbation and single-step gradient correction. The gradient alignment diagnostic is the key insight: if a single safety example's gradient aligns with the safety subspace, safety is a compact construct.
- **why critical**: Tests the compactness of safety representations. If a single example can recover safety, the safety construct has high validity by the E2 parsimony criterion.

### B8. Alignment-Interpretability Trade-off

- **metric_id**: `EX30_alignment_interpretability`
- **lens**: mechanistic_interpretability
- **validity_type**: Construct (C4 Discriminant Validity)
- **criteria**: C4 Discriminant Validity
- **what it measures**: For a model and an aligned variant (e.g., RLHF-tuned), compares feature interpretability (monosemanticity of top activating examples) and feature richness (effective rank of activation patterns). Reports (1) interpretability_delta (aligned - base), (2) richness_delta (aligned - base), and (3) trade-off_ratio (interpretability gain per unit of richness loss).
- **pass condition**: This is a diagnostic, not pass/fail. Reports whether alignment improves interpretability at the cost of representational richness.
- **implementation notes**: Interpretability is estimated via activation pattern consistency (top-k examples for each feature should cluster tightly; high silhouette score = high interpretability). Richness is estimated via effective rank of the feature activation matrix. The trade-off ratio quantifies the cost of alignment for interpretability.
- **why critical**: First controlled evidence that alignment causes interpretability. The trade-off reveals a C4 discriminant validity issue: interpretability and richness are different constructs.

---

## C. New Adapter Types Needed

Part IX papers primarily operate on standard model representations and
do not require new adapter types. The key infrastructure needs are
computational, not architectural.

| Adapter | For | Why it's different | Priority |
|---|---|---|---|
| `SafetySubspaceAdapter` (method on existing adapter) | Safety subspace extraction from contrast directions | Requires paired safe/unsafe prompts and PCA on contrast vectors; adds safety subspace projection as a reusable method | MEDIUM |
| `OptimalTransportAligner` (utility, not adapter) | MOT computation between two models | Solves Sinkhorn OT over CKA similarity matrices; reusable across any cross-model comparison | MEDIUM |

Note: Atlas-Alignment's CKA-based alignment uses the same CKA
infrastructure already present in the framework (see `92_cka.py` and
`E6b_cka_cross_arch.py`). No new adapter class is needed.

---

## D. Key Paper Framings (strongest arguments from Part IX)

### D1. The benchmarks themselves fail (SAEBench Audit)

> "TPP has a coefficient of variation of 16--39% across random seeds.
> A difference smaller than the noise floor cannot be detected.
> Single-seed TPP comparisons are 'largely unreliable noise.'"

The SAEBench audit independently derived the framework's M-frame
validity criteria (reliability, content validity, discriminant
validity) and applied them to SAEBench. The result: most SAEBench
metrics fail. The audit's five desiderata (tracks ground-truth,
increases throughout training, low reseed noise, discriminative,
internally consistent) are exactly the framework's M1/M2/M4/M5
criteria.

**Use in paper**: Lead with this as the motivating crisis. The field's
own benchmark-of-benchmarks has been audited and failed. Every paper
since 2024 that compared SAE architectures using TPP or SCR has drawn
conclusions from unreliable metrics.

### D2. 93% reproducibility failure (MechEvalAgent)

> "93% of evaluated MI research outputs have reproducibility failures
> detectable only by running the code. 80% fail coherence. 51 issues
> surfaced that human reviewers examining only the narrative missed."

MechEvalAgent's framework (coherence + reproducibility +
generalizability) is a natural prerequisite check for the full
validity framework. Before applying the 5-lens validity protocol to
a claimed result, the result must be reproducible.

**Use in paper**: Make explicit: "validity testing presupposes that the
result being validated is reproducible." The 93% failure rate is the
strongest possible evidence for systematic validity problems at the
most basic level.

### D3. Safety representations are the most validity-passing feature class

> "Safety representations are low-rank, stable under fine-tuning,
> causally sufficient, and compact (recoverable from a single
> example's gradient). They pass E2, M1, and C4 criteria by
> construction."

Four independent papers converge on the same empirical finding:
safety-relevant information occupies a low-rank, stable subspace.
This is the strongest positive existence proof for the framework:
MI-based safety alignment works precisely when the constructs have
high validity by the framework's criteria.

**Use in paper**: Positive example in the framework application
section. Safety is what high-validity interpretability looks like.

### D4. The transparency tax (Atlas-Alignment)

> "Every time a new model is released, interpretability research
> starts from scratch. The 'transparency tax' scales with the pace
> of model development, making interpretability perpetually behind."

Atlas-Alignment's "invest once, transfer many times" logic is the
practical implementation of the framework's goal: establish validity
criteria once, then apply them scalably.

**Use in paper**: The framework is the measurement-theoretic
counterpart to Atlas-Alignment's engineering solution.

### D5. Global alignment handles architectural diversity (MOT)

> "Standard RSA/CKA methods fail when models have different depths.
> MOT handles depth mismatches natively via optimal transport, producing
> a single global alignment score."

MOT is the correct mathematical tool for C5 Convergent Validity
across architecturally different models. The global alignment score
is a directly interpretable C5 metric.

**Use in paper**: Technical upgrade for C5 comparisons. CKA is
limited to same-architecture; MOT generalizes.

### D6. Human alignment causes interpretability (ELLIS)

> "The aligned variant is 'significantly more interpretable than both
> non-aligned counterparts' but qualitatively 'less visually rich.'
> More interpretable in the sparse/monosemantic sense but capturing
> fewer complex visual phenomena."

A representation optimized for human interpretability is not the same
construct as one optimized for richness. They are discriminant, and
interpreting one as the other is a validity failure.

**Use in paper**: Concrete C4 example. The alignment-interpretability
trade-off shows that single-number validity scores are insufficient.

---

## E. Gaps Identified

### E1. No meta-metric for evaluating evaluation metrics

The SAEBench audit shows that evaluation metrics themselves need
evaluation. The framework should include a meta-level tier: metrics
that assess the reliability and validity of other metrics. The B1
metric (SAEBench Reliability Audit) is the first step.

### E2. No execution-grounded reproducibility criterion

MechEvalAgent's 93% failure rate shows that narrative-level validity
is insufficient. The framework should include an explicit
reproducibility prerequisite: results must be reproducible before
they can be validated. The B2 metric (Reproducibility Check)
operationalizes this.

### E3. No framework criterion for construct compactness

The safety subspace cluster reveals that validity-passing constructs
are compact (low-rank, single-example recoverable). The framework
should formalize compactness as a desirable property: constructs
that require many dimensions or many examples to specify are less
valid (harder to falsify, harder to intervene on).

### E4. No cross-model feature transfer metric

Atlas-Alignment enables transferring feature labels across models,
but the framework has no criterion for evaluating the quality of
such transfers. The B3 metric (Atlas-Alignment Convergence) fills
this gap, but a more general "feature transfer fidelity" criterion
should be formalized.

### E5. No alignment-interpretability trade-off quantification

The ELLIS finding shows that alignment improves interpretability at
the cost of richness. The framework needs criteria for when
interpretability gains from alignment are worth the richness cost,
and when they indicate a construct validity failure (the aligned
model is interpretable but in a less meaningful way).
