# MI Deep Survey Part VII: Extraction

Extracted from "The SAE Architecture Explosion: PRISM, WeightLens, Matryoshka,
GradSAE, Output-Centric Descriptions, Adaptive Sparsity, and Scaling Sparse
Feature Circuits" (May 2026).

Covers 10 new methods / architectural variants from 8 papers (2024-2026),
all targeting specific failure modes of vanilla SAEs. The unifying theme:
every known SAE failure mode now has at least one remediation paper, but
none of these remediations have been validated under a shared framework.

---

## A. New Methods to Implement

### Tier 1: Critical (directly maps to existing mechval criteria gaps)

| Method | What | Source | Lens | Metric type |
|---|---|---|---|---|
| GradSAE | Activation x gradient = causal influence score per latent; activation magnitude alone is poor proxy for causal effect | Shu, Wu, Zhao, Du, Liu; EMNLP 2025 (arXiv:2505.08080) | mechanistic_interpretability | evaluation |
| WeightLens | Weight-based feature descriptions from encoder/decoder vectors; zero inference passes; context-independent | Golimblevskaia, Jain, Puri, Ibrahim, Samek, Lapuschkin; ICLR 2026 (arXiv:2510.14936) | mechanistic_interpretability | evaluation |
| DMSAE (Distilled Matryoshka SAE) | Iterative distillation identifies stable core features (197/65k survive 7 cycles); direct M1 operationalization | Martin-Linares, Ling; arXiv:2512.24975 | measurement_theory | evaluation |
| Output-centric descriptions (VocabProj + TokenChange) | Feature descriptions from output effects (decoder -> unembed projection, or stimulation -> token probability change); captures causal output behavior that input-centric methods miss | Gur-Arieh, Mayan, Agassy, Geiger, Geva; ACL 2025 (arXiv:2501.08319) | mechanistic_interpretability | evaluation |

### Tier 2: High (new methods with clear validity framework mapping)

| Method | What | Source | Lens | Metric type |
|---|---|---|---|---|
| PRISM | Multi-concept feature descriptions; clusters activation examples by semantic similarity, generates one label per cluster, scores polysemanticity | Kopf, Feldhus, Bykov, Bommer, Hedstrom, Hohne, Eberle; NeurIPS 2025 (arXiv:2506.15538) | measurement_theory | evaluation |
| Matryoshka SAE | Nested dictionaries at increasing widths; hierarchy consistent by construction; prevents splitting and absorption | Multiple authors; NeurIPS 2025 (arXiv:2503.17547) | measurement_theory | evaluation |
| CircuitLens | Extension of WeightLens to circuit-level; traces feature propagation through weight-determined connectivity across layers | Golimblevskaia et al.; ICLR 2026 | mechanistic_interpretability | discovery |
| Scaling Sparse Feature Circuits (ICL at 2B) | Task vectors decompose into SAE latents; two-stage circuit (detecting -> executing) at Gemma-1 2B scale | Kharlapenko, Shabalin, Barez, Conmy, Nanda; ICML 2025 (arXiv:2504.13756) | mechanistic_interpretability | discovery |

### Tier 3: Medium (architectural variants, useful as comparison points)

| Method | What | Source | Lens | Metric type |
|---|---|---|---|---|
| BatchTopK | Batch-level sparsity; variable per-sample k with fixed average | Bussmann, Leask, Nanda; NeurIPS 2024 | measurement_theory | architecture |
| AdaptiveK | Complexity-driven sparsity via linear probes; adjusts k per input | Yao, Du; arXiv:2508.17320 (ICLR 2026 workshop, withdrawn) | measurement_theory | architecture |
| SoftSAE | Differentiable Soft Top-K; learns k from data | arXiv:2605.06610; May 2026 | measurement_theory | architecture |

---

## B. New Metrics (suggested implementations)

### B1. GradSAE Causal Influence Score

- **metric_id**: `EX12_gradsae_causal_influence`
- **lens**: mechanistic_interpretability
- **validity_type**: Internal (I1 Necessity, I2 Sufficiency)
- **criteria**: E2 Causal Sufficiency (direct proxy); I1 Necessity
- **what it measures**: For each active SAE latent, computes activation x gradient (gradient of output probability w.r.t. latent activation). Produces per-feature causal influence scores. Ranks features by causal influence rather than activation magnitude.
- **pass condition**: Features claimed as causally important for a task should rank in top-k by GradSAE score; features with high activation but low GradSAE score are correlational-only.
- **implementation notes**: Same cost as standard SAE forward + one backward pass. Could be added as a column to existing feature-level evaluation outputs. The activation/gradient dissociation ratio (features with high activation but low gradient vs. vice versa) is itself a diagnostic for whether the artifact's features are causally grounded.
- **why critical**: The survey reports that highest-activation features are often NOT the most causally influential. This means any metric that selects features by activation magnitude (which is nearly all existing metrics) has a systematic bias toward correlational features. GradSAE is a corrective lens.

### B2. Weight-Based Description Agreement (WeightLens convergence)

- **metric_id**: `EX13_weightlens_convergence`
- **lens**: measurement_theory
- **validity_type**: Construct (C5 Convergent Validity)
- **criteria**: C5 Convergent Validity
- **what it measures**: For each feature, generates (a) weight-based description via decoder/encoder weight vector projection through unembedding (WeightLens/VocabProj), and (b) activation-based description via standard max-activating examples. Measures agreement between the two descriptions (cosine similarity of top-k promoted tokens, or semantic similarity of generated descriptions).
- **pass condition**: Agreement >= 0.5 (arbitrary threshold, needs calibration). Features where weight-based and activation-based descriptions diverge have low construct validity -- the "meaning" depends on whether you ask via weights or activations.
- **implementation notes**: Weight-based description is dataset-free (single matrix multiply). This directly addresses the Apollo open problem: "verify features based on model weights; show that features are a model-property and not (only) a dataset property." Code available: github.com/egolimblevskaia/WeightLens.
- **why critical**: This is the most accessible C5 test for SAE features. Two completely different evidence sources (weights vs. activations) pointing to the same description is strong convergent validity.

### B3. DMSAE Core Stability Score

- **metric_id**: `M7_core_stability`
- **lens**: measurement_theory
- **validity_type**: Measurement (M1 Reliability)
- **criteria**: M1 Reliability (test-retest)
- **what it measures**: Runs n training cycles of iterative distillation (train SAE, identify high grad x activation features, freeze them as "core", reinitialize rest, retrain). After n cycles, records which features converged into the stable core. Core membership rate = reliability score.
- **pass condition**: Features used in validity claims must be core members. If a feature is not in the core after 7 cycles, its M1 reliability is effectively zero.
- **implementation notes**: Computationally expensive (7+ full SAE training runs). But the result is devastating: only 197/65,000 features in a 65k SAE are stable across runs. This means ~99.7% of SAE features have near-zero test-retest reliability. This is the strongest quantitative evidence for the SAE reliability crisis. Even if not run as a metric (too expensive), the finding should be cited as calibration for M1 claims.
- **why critical**: Directly operationalizes M1 as an iterative procedure. The 197/65k number is the single most important quantitative finding in this survey for the mechval paper.

### B4. PRISM Polysemanticity Score

- **metric_id**: `EX14_prism_polysemanticity`
- **lens**: measurement_theory
- **validity_type**: Measurement (M6 Construct Coverage)
- **criteria**: M6 Construct Coverage; E1 Content Validity
- **what it measures**: Clusters a feature's activation examples by semantic similarity. If clusters form (>1 distinct semantic group), the feature is polysemantic. Score measures how well a multi-concept description captures all clusters vs. a single-label description.
- **pass condition**: For features claimed as monosemantic, polysemanticity score should be below threshold. For features where polysemanticity is detected, downstream validity claims must be qualified per-concept.
- **implementation notes**: Extends existing `52_polysemanticity.py` (which uses structural proxies: effective rank, participation ratio, fan-out) with an activation-based polysemanticity detection method. PRISM is complementary -- it tests polysemanticity from the activation side while 52 tests it from the weight side. Agreement between the two is itself a C5 convergent validity signal.
- **relationship to existing**: Extends `52_polysemanticity` with activation-space clustering. Also relevant to `EX4_feature_absorption` -- absorption is one mechanism that creates apparent polysemanticity.

### B5. Output-Centric Description Score

- **metric_id**: `EX15_output_centric_description`
- **lens**: mechanistic_interpretability
- **validity_type**: Internal (I2 Sufficiency); Construct (C5 Convergent)
- **criteria**: E2 Causal Sufficiency; C5 Convergent Validity
- **what it measures**: For each feature, computes output-centric description via VocabProj (decoder -> unembed projection) and/or TokenChange (stimulation -> probability change). Compares against input-centric description (max-activating examples). Measures agreement and identifies features where input/output descriptions diverge.
- **pass condition**: Features with high encoding validity (input-centric) but low execution validity (output-centric) are correlational but not causal -- flagged as I2 failures. Features with both high are genuinely valid.
- **implementation notes**: VocabProj is weight-only (decoder direction projected through W_U). TokenChange requires forward passes with feature stimulation. Both are efficient. The input/output agreement is the most accessible convergent validity test for individual features.
- **why critical**: Decomposes feature validity into two independent dimensions (encoding vs. execution). A feature can be "valid" on one and "invalid" on the other. This distinction is currently missing from the framework.

### B6. Adaptive Sparsity Diagnostic

- **metric_id**: `EX16_adaptive_sparsity_diagnostic`
- **lens**: measurement_theory
- **validity_type**: Measurement (M6 Construct Coverage)
- **criteria**: E1 Content Validity
- **what it measures**: For each evaluation example, estimates true concept count (using AdaptiveK's complexity probe or SoftSAE's learned k), compares to the fixed k used in the SAE. Flags examples where mismatch exceeds threshold (k_fixed >> k_true: spurious features included; k_fixed << k_true: real concepts truncated).
- **pass condition**: Fraction of examples with k-mismatch > threshold should be below 20%. If above, E1 validity of the entire SAE on that corpus is compromised.
- **implementation notes**: Requires either (a) training the AdaptiveK complexity probe (cheap linear probe), or (b) using SoftSAE's learned k as reference. Could also use a simpler proxy: input perplexity as complexity measure, then correlate with SAE active feature count.
- **why critical**: Fixed-k SAEs have a systematic E1 validity failure: for simple inputs they include spurious features, for complex inputs they truncate real features. This diagnostic quantifies the severity.

### B7. Matryoshka Cross-Scale Consistency

- **metric_id**: `EX17_matryoshka_consistency`
- **lens**: measurement_theory
- **validity_type**: Measurement (M1 Reliability)
- **criteria**: M1 Reliability (cross-scale); M2 Invariance
- **what it measures**: For Matryoshka SAEs (or comparing SAEs at different dictionary widths), measures whether features at width k correspond to feature clusters at width 2k. Tracks feature splitting rate and absorption rate across scales.
- **pass condition**: Features at width k should map to coherent clusters at width 2k (not split into unrelated sub-features). Splitting rate below 0.1, absorption rate below 0.15.
- **implementation notes**: Extends `EX4_feature_absorption` to the cross-scale setting. SAEBench already has an absorption metric -- this diagnostic would add the cross-scale dimension. Matryoshka SAEs enforce this by construction; the diagnostic is for comparing non-Matryoshka SAEs.
- **relationship to existing**: Extends `EX4_feature_absorption` with cross-scale tracking.

### B8. CircuitLens Weight-Based Circuit Recovery

- **metric_id**: `107b_circuitlens_weight_circuits`
- **lens**: mechanistic_interpretability
- **validity_type**: Construct (C2 Structural Plausibility)
- **criteria**: C2 Structural Plausibility; C5 Convergent Validity
- **what it measures**: Recovers feature-level circuits from weight connectivity alone (no activations). Compares weight-derived circuit to activation-derived circuit (e.g., from sparse feature circuits or activation patching). Agreement between weight-circuit and activation-circuit is structural convergent validity.
- **pass condition**: Jaccard overlap between weight-circuit and activation-circuit edges >= 0.3 (lower threshold than head-level because feature-level circuits are larger).
- **implementation notes**: CircuitLens code available: github.com/egolimblevskaia/CircuitLens. The key advantage: weight circuits distinguish "this feature fires because an earlier feature caused it" from "this feature fires because the input directly activated it" -- a distinction that activation-only methods cannot make.
- **why critical**: Direct analog to the weight vs. activation framing in the factorized circuits project. If weight-circuit and activation-circuit agree, the circuit is a model property. If they disagree, the circuit may be a dataset artifact.

---

## C. New Adapter Types Needed

| Adapter | For | Why it's different | Priority |
|---|---|---|---|
| `MatryoshkaSAEAdapter` | Matryoshka SAEs with nested dictionaries | Must expose multiple dictionary widths simultaneously; cross-scale feature correspondence maps; feature hierarchy | HIGH |
| `TranscoderAdapter` | Transcoders (used by WeightLens/CircuitLens) | Input-invariant and input-dependent components separable; encoder and decoder weights have different interpretive roles | HIGH |
| `GradSAEWrapper` (not a new adapter, but an adapter method) | Any existing SAE adapter | Adds `causal_influence(model, tokens, hook_name)` method that returns activation x gradient scores alongside standard activations | MEDIUM (method on existing adapter) |

Note: `WeightArtifactAdapter` (already exists in `mechval/lib/artifacts/adapter.py`) covers the WeightLens use case -- WeightLens produces directions from weights, which is exactly what `WeightArtifactAdapter` does. No new base class needed, but a `WeightLensAdapter` subclass that implements the specific encoder/decoder weight decomposition would be useful.

---

## D. Key Paper Framings (strongest arguments from Part VII)

### D1. The crisis of proliferation

> "At least eight new architectures or analysis methods were proposed in 2025-2026, each targeting a specific failure mode of vanilla SAEs... no principled framework governs which remediation to use, when, or how to evaluate whether it actually fixed the problem."

This is the Part VII version of C9 (from Parts IV-VI: "15+ methods, no shared validity standard"). Now the count is higher and the pattern is clearer: each failure mode has generated its own remediation, but no remediation has been validated for whether it actually fixes the validity problem it targets. Matryoshka SAE reduces absorption -- but does reduced absorption mean features are now valid descriptions? No paper tests this.

**Use in paper**: This is the strongest version of the "methods zoo needs a measurement framework" argument. Each new method implicitly defines a validity criterion; the framework makes these criteria explicit and testable.

### D2. Activation magnitude is not causal influence (GradSAE)

> "A feature can fire strongly but have zero impact on the output (high activation, zero gradient), or fire weakly but critically determine the output (low activation, high gradient)."

This directly undermines the standard interpretability workflow, which equates "what fires" with "what matters." Nearly all existing metrics (activation patching, feature visualization, max-activating examples) are built on the activation magnitude proxy. GradSAE shows this proxy is unfaithful.

**Use in paper**: This is a concrete example of an M6 Construct Coverage failure -- the instrument (activation magnitude) measures the wrong construct (correlation, not causation). The framework's E2 Causal Sufficiency criterion catches this.

### D3. Only 197/65,000 features are stable (DMSAE)

> "An SAE trained to find 65,000 features reliably identifies only 197 of them across training runs. The M1 reliability rate for the remaining 64,803 is essentially zero."

This is the most devastating quantitative finding for SAE validity. It means ~99.7% of SAE features fail M1 Reliability. Any validity claim built on features outside the stable core is built on noise.

**Use in paper**: Lead with this number when motivating M1 Reliability as a prerequisite. The framework doesn't just add nice-to-have checks -- it catches catastrophic failures that the field is currently ignoring.

### D4. Weight-based vs. activation-based descriptions measure different things (WeightLens)

> "A feature has context-independent validity (what the feature does unconditionally, readable from weights) and context-dependent validity (what the feature does in specific contexts, requiring activation analysis). A feature can be valid at one level and invalid at the other."

This creates a new diagnostic category: two distinct validity modes for every feature. Agreement between them is C5 convergent validity. Disagreement is a construct validity failure that needs to be characterized (not just flagged).

**Use in paper**: Directly addresses the Apollo open problem: "verify features based on model weights; show that features are a model-property and not (only) a dataset property." WeightLens answers the open question; the framework provides the evaluation protocol.

### D5. Input-centric vs. output-centric descriptions (encoding vs. execution validity)

> "Input-centric and output-centric descriptions are measuring different constructs. A feature has an encoding description (what inputs cause it to fire) and an execution description (what outputs it causes). These can diverge."

Four possible validity states:
1. High encoding + high execution = genuinely valid
2. High encoding + low execution = correlational but not causal (I2 failure)
3. Low encoding + high execution = causal but unreliable trigger (M1 failure)
4. Both low = invalid

**Use in paper**: This 2x2 matrix is a simple, visual way to explain why single-number validity scores are insufficient. The framework's multi-criteria structure is the only way to capture all four states.

### D6. Monosemanticity assumption is architecturally baked in (PRISM)

> "Standard automated interpretability pipelines are architecturally incapable of producing reliable descriptions for polysemantic features. The reliability failure isn't a bug -- it's a consequence of the monosemanticity assumption built into the pipeline design."

This means the standard interp pipeline has a systematic M6 Construct Coverage failure: the instrument can only measure monosemantic features, but the target population includes polysemantic features. PRISM shows the population is substantially polysemantic.

**Use in paper**: Example of how the framework reveals instrument limitations that are invisible from within the instrument's own evaluation methodology.

### D7. "Dead" features are not dead (PRISM + VocabProj)

> "PRISM can find inputs that activate features previously labeled 'dead' (zero activation on all standard benchmark inputs), suggesting that dead features are not truly inactive but are simply not activated by the particular examples in the evaluation set."

VocabProj independently recovers "dead" feature meanings from weights alone. Both methods show that the "dead feature" concept is a dataset artifact, not a model property.

**Use in paper**: Another C5 convergent validity example -- two independent methods (activation clustering, weight projection) agree that "dead" features are alive, providing a concrete case where convergent evidence overturns a widely-held field assumption.

### D8. Fixed k is a systematic E1 Content Validity failure

> "For inputs where the true number of active concepts is below k, the top-k features include spurious activations; for inputs above k, real concepts are truncated. Both conditions violate E1 Content Validity."

Three independent papers (BatchTopK, AdaptiveK, SoftSAE) arrived at the same conclusion from different directions. AdaptiveK adds the finding that context complexity is linearly encoded in LLM representations -- meaning the model itself knows how many concepts are active.

**Use in paper**: The convergence of three independent papers on the same conclusion is itself a C5 signal. The framework's E1 criterion catches what all three papers independently identified.

### D9. Sparse feature circuits do scale (ICL at 2B)

> "The ICL mechanism found in Gemma 2B is structurally consistent with (and extends) what earlier work found in GPT-2 Small. This is evidence for C5 Convergent Validity at the architectural level."

The task vector decomposition into SAE latents provides a reproducible benchmark. The two-stage circuit (detecting -> executing) with cross-layer causal chains is the most complex sparse feature circuit identified at this scale.

**Use in paper**: Positive result in a crisis-heavy landscape. The framework's criteria can distinguish between scaled methods that maintain validity (this one) and scaled methods that don't.

### D10. Remediation-validity gap

> "All 12 major failure modes now have empirical evidence, and 10 have at least partial remediations. None of the remediations are unified under a single framework. None of the remediations have themselves been validated for whether they actually fix the validity problem they target."

This is the meta-finding of Part VII: the field has moved from "identifying problems" to "proposing fixes" without an evaluation framework for the fixes themselves. Matryoshka SAE reduces absorption, but does reduced absorption mean valid features? GradSAE identifies causal features, but are those features reproducible? WeightLens removes dataset dependence, but does dataset independence guarantee validity?

**Use in paper**: This is the central argument for why the framework matters NOW, not later. The remediation wave has arrived, and without a shared standard, the field will proliferate fixes without knowing which ones work.

---

## E. Gaps Identified (potential contributions)

### From Part VII specifically:

1. **No validation of remediations** -- Each SAE variant (Matryoshka, DMSAE, GradSAE, etc.) claims to fix a specific failure mode. No paper validates whether the fix actually restores the validity criterion the failure mode violated. The framework can provide this by running each variant through the relevant criterion protocol and comparing.

2. **GradSAE not integrated into any existing benchmark** -- SAEBench does not include gradient-weighted feature ranking. Causal influence scoring should be a standard column in SAE evaluation. Currently, every SAE evaluation uses activation magnitude as the selection criterion.

3. **No cross-method comparison on validity criteria** -- Do GradSAE's causal features overlap with DMSAE's stable features? If yes, stability and causality are convergent indicators. If no, they measure orthogonal properties. This cross-method comparison is a new contribution the framework enables.

4. **Weight/activation description agreement not benchmarked** -- WeightLens and VocabProj both produce weight-based descriptions. Standard autointerp produces activation-based descriptions. No paper measures the agreement rate across a large feature population and correlates it with downstream validity metrics.

5. **Polysemanticity prevalence not characterized per SAE variant** -- PRISM shows many features are polysemantic, but only for standard SAEs. Are Matryoshka SAE features less polysemantic? Are DMSAE core features monosemantic? These cross-variant comparisons are missing.

6. **Adaptive sparsity not tested for validity improvement** -- BatchTopK/AdaptiveK/SoftSAE show reconstruction improvements. None test whether the features found under adaptive sparsity are more valid (more reliable, more causal, more convergent) than fixed-k features.

7. **CircuitLens weight-circuits not compared to activation-circuits** -- CircuitLens identifies circuits from weights. The comparison to standard sparse feature circuits (from activations) would directly test whether circuits are model properties or dataset artifacts. This comparison is proposed but not performed in the paper.

8. **Encoding/execution description decomposition not formalized** -- The input/output description distinction (from output-centric paper) creates a 2x2 validity space that the framework should formalize as two independent validity scores per feature. Currently, the framework treats feature validity as unidimensional.

9. **ICL benchmark not standardized** -- The scaling sparse feature circuits paper provides a reproducible ICL task. This should be added to the mechval benchmark suite as a standard sparse feature circuit recovery test at the 2B scale.

10. **Dead feature resurrection rate** -- Both PRISM and VocabProj can "resurrect" dead features. The resurrection rate (fraction of dead features that have non-trivial weight-based descriptions) is an unstudied diagnostic for SAE quality. High resurrection rate = the SAE is discarding valid features due to dataset coverage.

---

## F. Cross-References to Parts IV-VI

| Part VII finding | Parts IV-VI connection | Implication |
|---|---|---|
| GradSAE (activation != causation) | C2 (attribution patching broken, r=0.006) | GradSAE is a lightweight fix for the attribution faithfulness problem -- it's a first-order approximation to activation patching that doesn't require corrupted inputs |
| WeightLens (weight-based descriptions) | C3 (architecture determines ontology) | Weight-based descriptions are architecture-invariant in a way activation-based descriptions are not, providing a possible resolution to the architecture-dependence problem |
| DMSAE (197/65k stable) | C4 (three-layer validation failure) | DMSAE quantifies the Layer 1 failure: the object-level method (SAE) has ~99.7% unreliable features. This is the number that makes the three-layer argument concrete |
| PRISM (polysemanticity) | C5 ("one head = one behavior" is false) | PRISM extends VPD's finding from heads to features: "one feature = one concept" is also false for a substantial fraction of features |
| Output-centric (encoding vs. execution) | C6 (CoT unfaithfulness is E2 failure) | The encoding/execution distinction is the feature-level analog of CoT faithfulness: a feature that "says" one thing (encoding) but "does" another (execution) is unfaithful at the feature level |
| Adaptive sparsity (fixed k is wrong) | C9 (methods proliferation) | Three independent papers converging on the same conclusion is itself a C5 signal, and the convergence is more informative than any individual paper |
| Scaling sparse feature circuits | C10 (platonic convergence as C5 foundation) | ICL circuits generalizing across model scale (GPT-2 Small -> Gemma 2B) is direct empirical evidence for the platonic convergence thesis at the circuit level |
| Remediation-validity gap | C4 (three-layer validation failure) | The remediation wave is a Layer 2 failure: the fixes themselves are unvalidated. The framework prevents this by requiring validation of the validation methods |

---

## G. Unified Failure Mode Table (Parts I-VII cumulative)

The survey provides a complete table of 12 SAE failure modes, their best
remediations, and framework criteria. Reproduced here with mechval metric
mappings:

| Failure Mode | Best Remediation (2025-2026) | Framework Criterion | Suggested mechval metric |
|---|---|---|---|
| Features unstable across training runs | DMSAE (197 stable core) | M1 Reliability | `M7_core_stability` |
| Monosemanticity assumption false | PRISM (multi-concept descriptions) | E1 Content Validity, M6 Construct Coverage | `EX14_prism_polysemanticity` |
| Activation magnitude != causal influence | GradSAE (gradient-weighted) | E2 Causal Sufficiency (direct proxy) | `EX12_gradsae_causal_influence` |
| Input-centric descriptions miss output effects | VocabProj/TokenChange (ACL 2025) | E2 Causal Sufficiency, C5 Convergent | `EX15_output_centric_description` |
| Fixed k clips or pads real feature counts | SoftSAE / AdaptiveK / BatchTopK | E1 Content Validity | `EX16_adaptive_sparsity_diagnostic` |
| Feature splitting/absorption across scales | Matryoshka SAE | M1 Reliability, E1 Content | `EX17_matryoshka_consistency` |
| Activation-based descriptions are dataset-dependent | WeightLens (weight-based) | C5 Convergent Validity | `EX13_weightlens_convergence` |
| Attribution patching unfaithful for MLPs | RelP (r=0.956 vs r=0.006) | M1 Reliability for circuits | `106_relp` (already in Parts IV-VI) |
| SAE architecture determines what can be seen | Architecture duality (NeurIPS 2025) | E1 Content Validity (construct) | SAE architecture duality (already in Parts IV-VI) |
| SAE geometry inconsistent with superposition | Geometry crisis (Sharkey 2024) | Theoretical (construct) | N/A (theoretical) |
| SAEs not competitive with simple baselines | AxBench (Stanford NLP) | E-frame (downstream) | `101_axbench` (already implemented) |
| 9% feature recovery at 71% explained variance | SynthSAEBench | E2 Causal Sufficiency | `EX9_saebench` (already implemented) |

---

## H. Strategic Implications for the Paper

### H1. The remediation wave argument

Parts IV-VI established that the field has diagnosed its own problems (Nanda's
pivot, TMLR open problems, Apollo's list). Part VII shows the field has now
entered the remediation phase -- each failure mode has generated a fix. The
framework is needed NOW because the fix wave needs a shared evaluation standard.
Without it, the field will have 12 remediations and no way to compare them.

### H2. The 197/65k number

This is the single most citable statistic from the entire 7-part survey. An SAE
with 65,000 features has only 197 stable ones. Use this in the abstract or
introduction to motivate why validity measurement is urgent. It quantifies the
M1 crisis in a way that no other number in the survey does.

### H3. The encoding/execution decomposition

The input/output description distinction from the output-centric paper provides
a simple framework extension: every feature has two validity scores (encoding
and execution), and they can diverge. This maps cleanly to the framework's
existing multi-criteria structure and provides a concrete worked example.

### H4. The convergent evidence argument gets stronger

Part VII adds three new C5 Convergent Validity tests:
1. Weight-based vs. activation-based descriptions (WeightLens)
2. Input-centric vs. output-centric descriptions (VocabProj/TokenChange)
3. Cross-scale feature consistency (Matryoshka SAE)

These join Parts IV-VI's existing C5 tests (cross-architecture agreement,
cross-model alignment, cross-method circuit comparison). The framework now
has at least 6 independent operationalizations of C5, making convergent
validity the single most well-supported criterion.

### H5. Factorized circuits connection

WeightLens and CircuitLens directly validate the factorized circuits approach:
both are weight-based interpretability methods that analyze features through
their weight-space decomposition rather than activations. The factorized
transformer's factor bank is already a weight decomposition; WeightLens provides
independent evidence that weight-based feature analysis is a viable and
complementary approach to activation-based analysis. CircuitLens's weight-based
circuit recovery is analogous to the factorized circuits project's weight-space
circuit scan (stage 30).
