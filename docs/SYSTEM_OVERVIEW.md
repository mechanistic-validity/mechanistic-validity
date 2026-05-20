# Mechanistic Validity: Complete System Overview

This document provides a self-contained description of the Mechanistic Validity framework as of May 2026. It covers the theoretical foundations, the benchmark implementation, the current state of the codebase, and the relationship to the Factorized Circuits project.

---

## 1. What is Mechanistic Validity?

Mechanistic Validity is a theoretical framework for evaluating claims in mechanistic interpretability (mech-interp). When researchers say "GPT-2 uses a 6-role circuit to solve indirect object identification," that is a claim about the model's internal mechanism. The question is: how well-supported is that claim?

The framework draws on established validation methodology from five fields:

- **Philosophy of science** -- What does it mean for a mechanism to be "real"? When is a hypothesis sufficiently tested? (Popper, Mayo, Glennan)
- **Neuroscience** -- How do neuroscientists validate claims about neural circuits? (lesion studies, optogenetics, circuit dissection)
- **Psychometrics** -- How do measurement scientists ensure their instruments are reliable, valid, and not measuring something else? (construct validity, factor analysis, measurement invariance)
- **Pharmacology** -- How do drug developers prove that a drug works through a specific mechanism? (dose-response, proof of mechanism, target characterization)
- **Causal inference** -- How do statisticians reason about cause and effect from observational and interventional data? (Pearl, Spirtes-Glymour-Scheines, Rubin, Bareinboim, Geiger)

The framework is applied to transformer circuits, specifically GPT-2 small, but the methodology is model-agnostic.

**Repository**: `mechanistic-validity/mechanistic-validity` on GitHub. The primary output is markdown documentation (an Astro/Starlight docs site) plus Python example implementations. The contribution is purely theoretical -- no novel experiment results or methods.

---

## 2. The Theoretical Framework

The framework has five interconnected components: validity types, description modes, evidence families, criteria, and verdict tiers.

### 2a. Five Validity Types (Dependency Order)

Validity types form a dependency chain. You cannot establish a later type without first satisfying earlier ones.

```
Construct --> Measurement --> Internal --> External --> Interpretive
```

**Construct Validity** (C1-C5): Is the thing being measured well-defined?
- C1: Falsifiability -- can the claim be refuted?
- C2: Structural plausibility -- is the proposed mechanism physically possible in the architecture?
- C3: Convergent validity -- do multiple independent methods agree?
- C4: Discriminant validity -- does the measure distinguish this circuit from others?
- C5: Nomological validity -- does the circuit fit into a broader theory?

**Measurement Validity** (M1-M6): Are the instruments trustworthy?
- M1: Reliability -- do repeated measurements give the same answer?
- M2: Baseline separation -- is the score distinguishable from random/untrained baselines?
- M3: Stability -- is the classification robust to perturbation?
- M4: Calibration -- are the numbers meaningful (not just artifacts of hyperparameters)?
- M5: Sensitivity -- can the instrument detect known-true effects?
- M6: Invariance -- does the metric behave consistently across conditions?

**Internal Validity** (I1-I5): Does the evidence support the causal claim?
- I1: Necessity -- is the circuit required for the behavior?
- I2: Sufficiency -- is the circuit enough to produce the behavior?
- I3: Specificity -- does the circuit do this task and not everything?
- I4: Double dissociation -- can you separate this circuit from other circuits?
- I5: Confound control -- are alternative explanations ruled out?

**External Validity** (E1-E6): Does the mechanism generalize?
- E1: Intervention reach -- do different intervention methods agree?
- E2: Prompt generalization -- does it work on diverse prompts, not just the test set?
- E3: Cross-task generalization -- does the mechanism transfer to related tasks?
- E4: Cross-model generalization -- does the mechanism appear in other models?
- E5: Graded response -- does partial ablation produce partial effects?
- E6: Novel prediction -- does the mechanism predict new, untested behaviors?

**Interpretive Validity** (V1-V5): Is the interpretation of the mechanism correct?
- V1: Level declaration -- at what description mode is the claim being made?
- V2: Level-evidence match -- does the evidence actually support claims at that level?
- V3: Alternative level -- could the evidence be explained at a different level?
- V4: Anthropomorphism check -- is the interpretation projecting human concepts?
- V5: Scope declaration -- what does the claim explicitly NOT cover?

Total: **27 operational criteria** across 5 validity types.

### 2b. Seven Description Modes (Partial Order)

Description modes answer "at what level of detail is the claim being made?" They form a partial order by commitment -- higher modes require all evidence from lower modes plus additional bridging evidence.

```
Computational > Algorithmic > Representational > Impl-Functional > Impl-Connectomic > Impl-Topographic
                                                                              (Impl-Statistical is orthogonal)
```

| Mode | Question it answers | Example |
|------|-------------------|---------|
| **Computational** | What function does the system compute, and why? | "The model computes argmax P(next_token given context)" |
| **Algorithmic** | What procedure does it execute? | "It runs a 6-step lookup-suppress-copy algorithm" |
| **Representational** | What information is encoded, and in what geometry? | "Subject number is linearly encoded in layer 8 residual stream" |
| **Impl-Functional** | What input-output transformation does each component perform? | "Head 9.9 copies the IO name to output logits via OV circuit" |
| **Impl-Connectomic** | How are components wired? | "DTH -> IND -> S-Inh -> NM via residual stream composition" |
| **Impl-Topographic** | Which components are involved? | "The circuit uses heads (0,1), (3,0), (5,5), (6,9), ..." |
| **Impl-Statistical** | What do activations look like? (orthogonal to others) | "Head 5.5 has bimodal activation distribution with 23% sparsity" |

Mode tags are applied to verdicts, not to metrics or evidence. A single experiment can provide evidence for multiple modes simultaneously.

### 2c. Six Evidence Families

Evidence families categorize what kind of evidence a metric produces. Each metric belongs to exactly one family.

| Family | Prefix | What it measures | Example metrics |
|--------|--------|-----------------|-----------------|
| **Causal** | A | Effects of interventions (ablation, patching, clamping) | Activation patching, DAS-IIA, CATE, mediation, sigma ablation |
| **Structural** | B | Weight-space properties (no forward pass needed) | SVD, effective rank, OV/QK norms, K-composition, template distance |
| **Information-theoretic** | C | Information flow and dependencies | Mutual information, PID, OCSE, transfer entropy, NOTEARS |
| **Behavioral** | D | Input-output behavior under manipulation | Faithfulness, logit diff, KL divergence, CE delta, dose-response |
| **Representational** | E | Geometry and content of internal representations | CKA, RSA, linear probes, DAS-IIA, attention entropy |
| **Measurement** | F | Meta-properties of other metrics (calibrations) | Bootstrap stability, convergent validity, discriminant validity |

### 2d. Verdict Tiers

Verdict tiers express how well-supported a mechanistic claim is. They form a strict hierarchy.

| Tier | Meaning | Requirements |
|------|---------|-------------|
| **Proposed** | No intervention evidence. Structural or representational only. | Construct validity criteria met |
| **Causally Suggestive** | Necessity shown, but sufficiency not established. | + Internal validity I1 (necessity) |
| **Mechanistically Supported** | Necessity + sufficiency, at least 2 ablation variants. | + I1+I2, intervention reach (E1) |
| **Triangulated** | All internal criteria met, plus external and construct criteria. | + Full internal + external + construct |
| **Validated** | All five validity types addressed with explicit baselines. | + Measurement + interpretive validity |
| **Underdetermined** | Evidence consistent with multiple mechanisms. | (Cannot resolve between rival specs) |
| **Disconfirmed** | Fails decisively on a key criterion. | (Negative result) |

The 13 worked case studies apply these verdicts to published circuits:

| Study | Verdict |
|-------|---------|
| IOI Circuit (Wang et al. 2023) | Triangulated |
| Othello World Model (Li et al. 2023) | Triangulated |
| Induction Heads (Olsson et al. 2022) | Mechanistically Supported |
| Greater-Than (Hanna et al. 2023) | Mechanistically Supported |
| Copy Suppression (McDougall et al. 2023) | Mechanistically Supported |
| Grokking (Nanda et al. 2023) | Causally Suggestive |
| Successor Heads (Gould et al. 2023) | Causally Suggestive |
| Docstring Circuit (Heimersheim and Janiak 2023) | Causally Suggestive |
| SAE Features (Cunningham et al. 2023) | Causally Suggestive |
| Knowledge Neurons (Dai et al. 2022) | Proposed |
| Superposition (Elhage et al. 2022) | Proposed |
| Probing Classifiers (Belinkov 2022) | Proposed |
| Gender Bias Circuits (Vig et al. 2020) | Proposed |

---

## 3. The Benchmark Implementation

The benchmark operationalizes the theoretical framework into runnable code. It has three tracks, four views, and four gates.

### 3a. Three Tracks

Tracks are competitive benchmarks where methods compete on standardized inputs/outputs.

#### Track 1: Circuit Localization (Causal Discovery)

- **Question**: What causal structure/components explain the behavior?
- **Input**: model + task
- **Output**: edge set / circuit
- **Score**: CMD (Circuit Model Distance), faithfulness
- **Methods**: EAP, ACDC, path patching, NOTEARS, subnetwork probing
- **Cross-field mapping**:
  - Causal inference: Causal discovery (Spirtes, Glymour, Scheines)
  - Neuroscience: Circuit mapping / connectomics
  - Psychometrics: Exploratory factor analysis (EFA)
  - Pharmacology: Target discovery
  - Philosophy of science: Abduction / exploratory inference

#### Track 2: Causal Variable Localization (Causal Abstraction)

- **Question**: Where is the high-level causal variable represented in the model?
- **Input**: model + task + variable
- **Output**: aligned subspace / features
- **Score**: IIA (Interchange Intervention Accuracy)
- **Methods**: DAS (Distributed Alignment Search), boundless DAS, probing, CCA/CKA alignment
- **Cross-field mapping**:
  - Causal inference: Causal abstraction (Geiger, Potts et al.)
  - Neuroscience: Functional localization
  - Psychometrics: Construct operationalization
  - Pharmacology: Target characterization / mechanism-of-action
  - Philosophy of science: Natural kind identification

#### Track 3: Causal Model Testing (Novel Contribution)

Track 3 is the framework's original contribution. It is theory-driven hypothesis testing, not blind discovery.

- **Question**: Is a proposed mechanistic model consistent with the model's interventional behavior?
- **Input**: model + task + `MechanisticClaimSpec` (a pre-registered hypothesis)
- **Output**: verdict profile + claim ceiling
- **Score**: confirmation rate, negative control rate, claim ceiling
- **Methods**: Pre-registered predictions executed via existing metrics (activation patching, DAS-IIA, mediation, ablation, etc.)
- **Cross-field mapping**:
  - Causal inference: Model testing / specification testing (SGS, Pearl Ch. 2)
  - Neuroscience: Causal circuit dissection
  - Psychometrics: Confirmatory factor analysis (CFA)
  - Pharmacology: Proof of mechanism
  - Philosophy of science: Hypothetico-deductive testing / severe testing (Mayo)

**Key distinction from Track 1**: Track 1 is unsupervised -- discover whatever circuit exists. Track 3 is supervised -- test whether a specific proposed mechanism is actually implemented.

### 3b. Four Views (Scoring Aggregations)

Views are not new measurement infrastructure. They aggregate existing metrics into causal-inference-grounded lenses.

#### V1: Causal Effect Estimation
- **Question**: How large is each component/edge's causal contribution?
- **Aggregates**: mediation, mediation_v2, cate, effect_size, dose_response, pse, intervention_specificity
- **Maps to**: Effect estimation (Pearl/Rubin -- ATE, CATE, NDE/NIE), dose-response (pharmacology), tuning curves (neuroscience)

#### V2: Causal Transportability
- **Question**: Does the mechanism generalize across prompts, tasks, and models?
- **Aggregates**: cross_task_generalization, cross_model_invariance, generalization_gap, measurement_invariance
- **Maps to**: Transportability (Pearl/Bareinboim), cross-condition generalization (neuroscience), measurement invariance (psychometrics)

#### V3: Counterfactual Verification
- **Question**: Do counterfactual interventions produce the expected results?
- **Aggregates**: das_iia, iia_variants, counterfactual_consistency, corrupt_restore, multi_axis_iia
- **Maps to**: Pearl's rung-3 counterfactuals, optogenetic interrogation (neuroscience)

#### V4: Mechanism Adjudication
- **Question**: When multiple mechanisms could explain the data, which is better supported?
- **Aggregates**: discriminant_validity, rival spec comparison, alternative exclusion scoring
- **Maps to**: SEM equivalent-models testing, crucial experiment (Bacon, philosophy of science)

### 3c. Four Gates (Preconditions)

Gates must pass before tracks and views produce meaningful results. They are preconditions, not scores.

#### G0: Construct Operationalization
- **Checks**: Is the construct (the thing being measured) defined independently of the instruments used to measure it?
- **Maps to**: Criterion C1 (falsifiability)
- **Why it matters**: A circuit named for what instruments found ("the patching-relevant heads") is circular. The construct must be defined before measurement begins.

#### G1: Measurement Calibration
- **Checks**: Are the metric outputs trustworthy -- stable across seeds, separable from random baselines, and replicable?
- **Maps to**: Bootstrap stability (F01), random baseline (F09), untrained model baseline (F10)
- **Why it matters**: An IIA score of 0.48 without knowing the random-vector baseline is not evidence.

#### G2: Causal Identifiability
- **Checks**: Can the causal effects specified in the `MechanisticClaimSpec` be estimated with the available interventions?
- **Maps to**: Metadata on the spec (`identifiability` field)
- **Why it matters**: In Pearl's framework, a causal effect must be identifiable (expressible in terms of available data/interventions) before it can be estimated.

#### G3: Confound/Superposition Risk
- **Checks**: Are ablation-based interventions confounded by polysemantic collateral damage?
- **Maps to**: Metadata on the spec (`superposition_risk` field)
- **Why it matters**: Ablating a "duplicate token detector" head that is also polysemantically involved in other computations introduces confounds. The intervention is available (G2 passes) but not specific (G3 flags risk).

### 3d. Cross-field Mapping (Complete)

| Component | Causal Inference | Neuroscience | Psychometrics | Pharmacology | Phil of Science |
|-----------|-----------------|--------------|---------------|--------------|----------------|
| Track 1: Causal Discovery | Causal discovery (SGS) | Circuit mapping | Exploratory factor analysis | Target discovery | Abduction |
| Track 2: Causal Abstraction | Causal abstraction (Geiger) | Functional localization | Construct operationalization | Target characterization | Natural kind identification |
| Track 3: Causal Model Testing | Model testing (SGS, Pearl) | Circuit dissection | Confirmatory factor analysis | Proof of mechanism | Hypothetico-deductive testing |
| V1: Effect Estimation | Effect estimation (Pearl/Rubin) | Perturbation quantification | Effect size estimation | Dose-response | Parameter estimation |
| V2: Transportability | Transportability (Bareinboim) | Cross-condition generalization | Measurement invariance | Phase 3 generalization | Novel prediction |
| V3: Counterfactual | Rung-3 counterfactuals (Pearl) | Optogenetic interrogation | -- | -- | Counterfactual testing |
| V4: Adjudication | Equivalent-models testing (SEM) | -- | Model comparison | -- | Crucial experiment |
| G0: Construct | -- | -- | Construct validity | -- | Operationalism |
| G1: Calibration | -- | Test-retest reliability | Measurement reliability | Assay validation | -- |
| G2: Identifiability | Identifiability (Pearl) | -- | -- | -- | -- |
| G3: Superposition | -- | Collateral damage in lesions | Discriminant validity | Off-target effects | Confound control |

---

## 4. MechanisticClaimSpec -- Track 3's Core Data Structure

The `MechanisticClaimSpec` is a Pydantic v2 model that formalizes a pre-registered mechanistic hypothesis. It is the required input artifact for Track 3.

### Structure

```
MechanisticClaimSpec
  task_id: str                                    # which task (e.g. "ioi")
  model_family: str                               # which model (default "gpt2")
  linguistic_claim: str                           # natural language description

  steps: list[ComputationalStep]                  # nodes in the mechanism DAG
    name: str                                     #   e.g. "duplicate_token_detection"
    category: str                                 #   e.g. "detection", "movement", "output"
    description: str                              #   what this step does
    input_type, output_type, position: str        #   I/O typing
    maps_to_role: str | None                      #   circuit role name (e.g. "DTH")
    maps_to_heads: list[tuple[int, int]]          #   attention heads
    maps_to_mlps: list[int]                       #   MLP layers
    maps_to_neurons: list[tuple[int, int]]        #   individual neurons
    maps_to_features: list[tuple[int, str, int]]  #   SAE/factor features
    description_mode: DescriptionMode             #   what level this step is described at
    discovery_status: str                         #   "hypothesized" | "confirmed" | ...

  edges: list[ComputationalEdge]                  # directed edges between steps
    source, target: str                           #   step names
    mechanism: str                                #   "residual_stream" | "attention_composition" | ...
    description: str
    source_component, target_component: tuple | None  # optional specific head-to-head

  predictions: list[CausalPrediction]             # things that SHOULD happen
    name: str                                     #   e.g. "ablate_dth_reduces_output"
    claim: str                                    #   natural language prediction
    intervention: InterventionType                #   ablate | patch | clamp | resample
    intervention_target: str                      #   which step to intervene on
    measurement_target: str                       #   where to measure the effect
    expected_direction: PredictionDirection        #   decrease | increase | invariant
    expected_metric: str                          #   which metric to use
    expected_threshold: float | None              #   minimum effect size

  negative_controls: list[CausalPrediction]       # things that should NOT happen
    is_negative_control: bool = True              #   marks as control
    # (same fields as predictions)

  rival_specs: list[str]                          # IDs of competing mechanisms
  identifiability: IdentifiabilityGate            # G2 metadata
  superposition_risk: SuperpositionGate           # G3 metadata
  description_mode: DescriptionMode               # overall mode
  paper_ref: str | None                           # citation
  author: str                                     # who wrote this spec
```

### Computed Properties

- `confirmation_rate` -- fraction of positive predictions that passed (0.0 to 1.0)
- `negative_control_rate` -- fraction of negative controls that passed
- `untested_predictions()` -- predictions not yet evaluated
- `all_predictions()` -- positive + negative combined

### SpecVerificationResult -- Track 3 Output

When `mv.verify(spec)` runs, it produces a `SpecVerificationResult`:

```
SpecVerificationResult
  spec_id, task_id, model_family: str
  prediction_results: list[PredictionResult]      # per-prediction verdicts
    prediction: CausalPrediction                  #   the original prediction
    measured_value: float                         #   what was actually observed
    verdict: PredictionVerdict                    #   pass | partial | fail | gap
    metric_used: str                              #   which metric ran
  mode_verdicts: list[ModeVerdict]                # aggregated by description mode
    mode: DescriptionMode
    predictions_tested, predictions_passed: int
    negative_controls_tested, negative_controls_passed: int
    verdict: PredictionVerdict
  claim_ceiling: DescriptionMode | None           # highest mode with all predictions passing
  verdict_tier: VerdictTier                       # proposed -> validated
  gates_passed: dict[str, bool]                   # G0-G3 status
  effect_estimation_score: float | None           # V1
  transportability_score: float | None            # V2
  counterfactual_score: float | None              # V3
  adjudication_score: float | None                # V4
  confirmation_rate: float                        # computed property
```

### How verify() Works

For each prediction in the spec:
1. If the prediction has an `intervention_target` and `measurement_target`, route to the `role_ablation` metric, which ablates all heads in the intervention role and measures the effect on the measurement role.
2. Otherwise, fall back to running the raw `expected_metric`.
3. Compare the measured value against `expected_direction` and `expected_threshold`.
4. Aggregate results by description mode to compute the claim ceiling.

### Example: IOI Claim Spec

The IOI spec defines a 6-step computational DAG:

```
duplicate_token_detection (DTH: heads 0.1, 3.0)
    --> induction (IND: heads 5.5, 6.9)
    --> s_inhibition (S-Inh: heads 7.3, 7.9, 8.6, 8.10)
previous_token_tracking (PTH: heads 2.2, 4.11)
    --> induction
induction --> s_inhibition
s_inhibition --> name_mover (NM: heads 9.9, 9.6, 10.0)
s_inhibition --> negative_name_mover (NegNM: heads 10.7, 11.10)
```

**Positive predictions** (7 total):
- "Ablating DTH reduces output" (threshold >= 0.2 decrease)
- "Ablating induction heads substantially reduces logit diff" (threshold >= 0.5)
- "Ablating S-Inh kills output" (threshold >= 0.8)
- "Ablating S-Inh reduces name mover activation" (threshold >= 0.2)
- "Ablating NegNM increases logit diff" (removes opposing signal)
- "Ablating DTH reduces induction head activation" (tests DTH->IND edge)
- "Ablating PTH reduces induction head activation" (tests PTH->IND edge)

**Negative controls** (5 total):
- "Ablating NM does not affect upstream S-Inh"
- "Ablating NegNM does not affect upstream S-Inh"
- "Ablating PTH alone does not destroy circuit output" (threshold < 0.3)
- "Ablating S-Inh does not affect upstream DTH"
- "Ablating NM does not affect NegNM" (parallel roles, not connected)

---

## 5. Current State (May 2026)

### Codebase Numbers

| Category | Count | Details |
|----------|-------|---------|
| **Tasks** | 54 total | 12 full_circuit, 10 proxy_circuit, 25 generator_only, 7 planned |
| **Metrics** | 84 | Across 5 evidence families (causal, structural, behavioral, representational, info-theoretic) |
| **Calibrations** | 14 | Bootstrap, seed variance, convergent/discriminant validity, measurement invariance, etc. |
| **Claim Specs** | 12 | Pre-registered mechanistic hypotheses for Track 3 |
| **Linguistic Domains** | 10 | agreement, binding, coreference, morphology, phonology, pragmatics, semantics, syntax, math, patterns |
| **Worked Case Studies** | 13 | Published circuits evaluated against the framework |
| **Evidence Families** | 5 (+ measurement) | Causal (A), Structural (B), Information (C), Behavioral (D), Representational (E) |
| **ADRs** | 14 | Architecture decision records documenting design evolution |

### The 54 Tasks by Status

**Full Circuit (12)** -- have a defined circuit with ROLES, BANDS, PATHWAYS:
- Published (7): `ioi`, `greater_than`, `induction`, `sva`, `gendered_pronoun`, `acronym`, `copy_suppression`
- Ours (1): `rti`
- Experimental (4): `epistemic_framing`, `epistemic_tight`, `epistemic_eap`, `epistemic_expanded`

**Proxy Circuit (10)** -- inherit a circuit from a related task without independent verification:
- RTI family (4): `rti_pattern`, `token_flood`, `buffalo`, `mib_rti`
- IOI family (3): `centering_theory`, `resumptive`, `self_allo`
- Induction family (3): `sequence_internal`, `alternating_pair`, `novel_song`

**Generator Only (25)** -- have prompt generators but no circuit:
- Linguistic probes (7): `reflexive_anaphora`, `filler_gap`, `negation`, `conditional`, `ellipsis`, `definiteness`, `but_reversal`
- BLiMP categories (12): `blimp_anaphor_agreement`, `blimp_argument_structure`, `blimp_binding`, `blimp_control_raising`, `blimp_determiner_noun`, `blimp_ellipsis`, `blimp_filler_gap`, `blimp_irregular_forms`, `blimp_island_effects`, `blimp_npi_licensing`, `blimp_quantifiers`, `blimp_subject_verb`
- Phonetic composition (6): `phonetic_composition`, `hypocorism`, `phonetic_sequential`, `double_shortening`, `homophone_recognition`, `reverse_decomposition`

**Planned (7)** -- catalog entries with no implementation:
- `less_than`, `sva_pp`, `colored_objects`, `docstring`, `bracket_matching`, `npi_licensing`, `sentiment`

### The 12 Claim Specs

Each claim spec defines a computational DAG (steps + edges), testable predictions ("ablating X should decrease Y"), and negative controls ("ablating downstream Z should not affect upstream W").

| Task | Steps | Heads | Predictions | Neg Controls | Source |
|------|-------|-------|-------------|--------------|--------|
| `ioi` | 6 | 15 | 7 | 5 | Wang et al. 2023 |
| `greater_than` | 3 | 7 | 4 | 1 | Hanna et al. 2023 |
| `induction` | 2 | 7 | 3 | 1 | Olsson et al. 2022 |
| `sva` | 4 | 12 | 4 | 2 | Lazo et al. 2025 |
| `rti` | 4 | 15 | 6 | 3 | Tower et al. (weight-space) |
| `gendered_pronoun` | 3 | 5 | 4 | 2 | Mathwin 2023 |
| `copy_suppression` | 3 | 7 | 3 | 2 | McDougall et al. 2023 |
| `acronym` | 3 | 8 | 3 | 1 | Garcia-Carrasco et al. 2024 |
| `epistemic_framing` | 3 | 4 | 4 | 2 | Tower 2026 (core, manual) |
| `epistemic_tight` | 5 | 13 | 4 | 1 | Tower 2026 (activation patching) |
| `epistemic_eap` | 4 | 15 | 4 | 2 | Tower 2026 (EAP) |
| `epistemic_expanded` | 6 | 32 | 5 | 2 | Tower 2026 (broad scan) |

### The 84 Metrics by Evidence Family

**Causal (32 metrics)** -- intervention-based:
- SCM/Pearl: activation_patching, logit_diff, role_ablation, causal_scrubbing
- Woodward interventionism: sigma_ablation, resample_complement, misalignment
- Counterfactual/DAS: das_iia, iia_variants, path_patching, counterfactual_consistency, multi_axis_iia, corrupt_restore
- Mediation: mediation, mediation_v2, pse
- Rubin/CATE: cate, intervention_specificity
- EAP: eap, atp_star
- Hedonic/PAS: pairwise_synergy, shapley_interactions
- MDC/Glennan: replacement_test, composition_test, operation_specification, held_out_prediction, procedure_specification, logic_gates
- Transportability: cross_task_transfer, cross_model_invariance
- Other: minimality_class, intermediate_state_prediction, hyperparam_sensitivity

**Structural (18 metrics)** -- weight-space, no forward pass:
- Composition: k_composition, copying_score, qk_norms
- Template distance: cmd, edge_jaccard, weight_eap_jaccard
- Graph topology: network_motifs, motif_enrichment, attention_clustering
- Effective rank/spectral: capacity_utilization, k_alignment, weight_extended, spectral_svd
- Edge analysis: path_identification, edge_necessity, path_specificity, compositional_sufficiency, graph_minimality
- SLT: llc

**Information-theoretic (9 metrics)**:
- pid, ocse, notears, mutual_information, conditional_mi, granger_causality, info_bottleneck, o_information, transfer_entropy

**Representational (5 metrics)**:
- attention_entropy, cka, cka_cross_arch, probe_decodability, causal_representation

**Behavioral (20 metrics)**:
- Effect measurement: effect_size, dose_response, ce_delta, per_token_nll, calibration
- Generalization: generalization_gap, mdl_compression, subnetwork_probe
- Output analysis: output_variants, output_variants_kl, output_variants_topk, corrupt_restore_behavioral, mean_centered_logit
- Boundary and normative: normative_account, error_boundary, boundary_sweep
- Specialized: epistemic_gradient, cross_task_generalization

### The 14 Calibrations

| ID | Name | What it checks |
|----|------|---------------|
| F01 | bootstrap | Score stable under resampling? |
| F02 | seed_variance | Stable across random seeds? |
| F03 | convergent_validity | Do different metrics agree? |
| F04 | discriminant_validity | Do different constructs disagree? |
| F05 | internal_consistency | Multi-item agreement? |
| F06 | reliability_suite | Split-half and test-retest reliability |
| F07 | measurement_invariance | Metric x condition interaction? |
| F09 | distributional_characterization | Score distribution shape |
| F10 | distributional_stability | Distribution stable under perturbation? |
| F11 | nomological_validity | Fits within broader theory? |
| F12 | incremental_validity | Adds value beyond existing metrics? |
| F13 | ablation_invariance | Stable across ablation methods? |
| F14 | method_invariance | Stable across methods within same family? |
| F15 | certified_stability | Certified stable classification? |

Hard gates (must pass before results are meaningful): F01 bootstrap stability, F09 random baseline, F10 untrained baseline.

---

## 6. The Proxy Circuit Problem

### What It Is

Ten tasks have `circuit_status="proxy_circuit"`. They inherit a circuit from a related task without independent verification that the circuit is actually correct for their specific prompt type.

### The Proxy Families

**RTI family (5 tasks)**: `rti` (full_circuit), `rti_pattern`, `token_flood`, `buffalo`, `mib_rti` -- all share the same 15-head, 4-role circuit (backbone, detector, copier, readout). The assumption is that all repetition-detection tasks use the same circuit.

**IOI family (4 tasks)**: `ioi` (full_circuit), `centering_theory`, `resumptive`, `self_allo` -- all share IOI's 15-head, 6-role circuit (DTH, PTH, IND, S-Inh, NM, NegNM). The assumption is that discourse/coreference variants use the same circuit as IOI.

**Induction family (4 tasks)**: `induction` (full_circuit), `sequence_internal`, `alternating_pair`, `novel_song` -- all share induction's 7-head, 2-role circuit (PTH, IND). The assumption is that all sequence-continuation tasks use the same circuit.

### Why It Matters

- Proxy circuits are hypotheses, not verified facts.
- Different prompt types might recruit different heads or additional heads.
- V2 (Transportability) tests would verify or refute these assumptions.
- Some proxy tasks might need their own circuits discovered via Track 1.
- The framework explicitly tracks this uncertainty via the `circuit_status` field.

### What Cross-Circuit Analysis Has Revealed

**Hub heads** -- some heads appear in many circuits with different roles:
- **(4,11)**: 6 circuits -- PTH, detector, mid_hub roles
- **(10,0)**: 5 circuits -- NM, output, late_router roles
- **(0,8)**: 4 circuits -- backbone, early_processor, embed roles
- **(2,2)**: 4 circuits -- PTH roles only
- **(5,5)**, **(6,9)**: 4 circuits each -- IND, late_gt roles

**Agreement circuits have zero overlap**: Gendered pronoun circuit vs SVA circuit have Jaccard similarity = 0.000 despite both being "agreement" tasks. They use completely different sets of heads.

**Epistemic variants have no universal heads**: Four discovery methods applied to the same task (epistemic framing) found four different circuits with zero heads appearing in all four.

---

## 7. Circuit Fingerprinting

The framework includes a circuit fingerprinting system that uses approximately 26 structural/behavioral/causal instruments as a feature vector per circuit to create a "fingerprint" that uniquely identifies circuit architecture types.

### Fingerprint Features (26 dimensions)

The feature vector includes: OV norm ratio, polysemanticity fan-out, effective rank, dose-response monotonicity, dose-response selectivity, number of synergistic/redundant/independent pairs, mean absolute PAS, CKA cross-layer, attention cluster silhouette, G7 strict motif pass, path specificity, error boundary alignment, NMF sharing ratio, NMF components for 90% variance, OV/QK composition ratio, normative max separation, polysemanticity participation ratio, QK effective rank, OV cosine alignment, spectral ratio, cross-task selectivity, and others.

### Emergent Circuit Taxonomy (from 7 published circuits)

```
Cluster A -- Redundant Specialized
  Induction, Copy Suppression
  Rich synergy/redundancy, low fan-out, high attention clustering,
  non-monotonic dose-response, shared heads (Jaccard 0.44)

Cluster B -- Independent Parallel
  Greater Than, Gendered Pronoun
  Zero pairwise interactions, independent heads, simple architecture,
  monotonic (or flat) dose-response

Outlier -- Complex Compositional
  IOI
  Unique on strict motifs, highest selectivity, highest path specificity,
  monotonic dose-response despite complex structure

Outlier -- Distributed Structural
  SVA
  High OV norms but low selectivity, non-monotonic

Outlier -- Selective Module
  Acronym
  Extreme dose-response selectivity (49x), moderate on other axes
```

### Key Structural Finding

Circuits with non-monotonic dose-response are exactly the ones with rich pairwise synergy/redundancy. Redundant heads compensate when one is removed, breaking monotonicity. Independent heads each contribute proportionally, producing monotonic dose-response.

### Minimum Discriminating Battery

Only 4 instruments suffice to uniquely identify all 7 circuit types (minimum pairwise separation = 2.28 in z-scored space):

1. OV norm ratio
2. NMF components (90% variance)
3. OV cosine alignment
4. Cross-task selectivity

### Head-Level Fingerprinting

Per-head structural features can identify functional roles without task-specific prompts:
- **Previous Token Heads** (PTH) have extremely low QK effective rank (z=-3.19) -- they attend to a single position, so their QK matrices are near rank-1.
- **Name Movers** have high polysemanticity fan-out (z=+1.43).
- **S-Inhibition** and **Induction** heads are structurally similar (centroid distance = 0.56), consistent with S-inhibition being a downstream consumer of induction signals.

### Pairwise Ablation Synergy (PAS) Findings

| Circuit | Synergistic Pairs | Redundant Pairs | Independent Pairs | Max PAS |
|---------|-------------------|-----------------|-------------------|---------|
| Induction | 7 | 8 | 6 | L5H5-L6H9: -0.15 |
| Copy Suppression | 3 | 4 | 14 | **L10H7-L11H10: +0.72** |
| Greater Than | 0 | 0 | 21 | 0.015 |
| Gendered Pronoun | 0 | 0 | 10 | 0.003 |

The L10H7-L11H10 synergy in Copy Suppression (PAS=+0.72) is the strongest second-order interaction measured across any circuit. Greater Than and Gendered Pronoun have zero interactions -- all pairs are truly independent.

---

## 8. API and Usage

### Python API

```python
import mechanistic_validity as mv

# Configure output
mv.set_output_dir("./results")

# Load a task (gymnasium-style)
task = mv.load_task("ioi")
circuit = task.get_circuit()          # CircuitSpec with roles, bands, pathways
prompts = task.get_prompts(tokenizer) # list of TaskPrompt
baselines = task.get_baselines()      # published reference numbers

# Run a metric
results = mv.run("k_composition", tasks=["ioi"])
results = mv.run("activation_patching", tasks=["ioi", "sva"], device="cpu")

# Run a calibration
results = mv.calibrate("bootstrap", tasks=["ioi"])

# Track 3: Causal Model Testing
spec = mv.load_task("ioi").get_claim_spec()   # MechanisticClaimSpec
result = mv.verify(spec, device="cpu")        # SpecVerificationResult
result.confirmation_rate                       # 0.0 to 1.0
result.claim_ceiling                           # DescriptionMode or None
result.verdict_tier                            # VerdictTier enum

# Views (scoring aggregations)
mv.run_view("effect_estimation", tasks=["ioi"])
mv.run_view("transportability", tasks=["ioi", "sva"])

# Gates (precondition checks)
mv.check_gate("construct_operationalization", task="ioi")
mv.check_gate("identifiability", spec=spec)

# Discovery
mv.list_tasks()                              # all 54 tasks
mv.list_tasks(has_circuit=True)              # 22 tasks with circuits
mv.list_tasks(circuit_status="full_circuit") # 12 tasks with full circuits
mv.list_families()                           # ['behavioral', 'causal', ...]
mv.list_metrics()                            # all 84 metrics
mv.list_metrics(family="structural")         # just structural metrics
mv.list_calibrations()                       # all 14 calibrations
mv.list_domains()                            # 10 linguistic domains
mv.list_experiment_groups()                  # experiment-based grouping
mv.status()                                  # what results exist on disk

# Optional Weave tracing
mv.init_tracing("my-project")

@mv.op
def my_metric(model, task):
    ...
```

### CLI

```bash
# Run metric scripts directly
uv run python src/mechanistic_validity/metrics/causal/woodward/03_sigma_ablation.py \
    --tasks ioi sva --device cpu

# Standard CLI args: --model (default gpt2), --device, --tasks, --n-prompts, --out
```

### Key Infrastructure in common.py

- `load_model(name, device)` -- cached HookedTransformer loading
- `get_circuit(task)` -- returns ROLES, BANDS, PATHWAYS dict
- `generate_prompts(task, tokenizer, n_prompts)` -- task-specific prompt objects
- `calibrate_mean_z(model, prompts)` -- per-(layer,head) mean activations for mean ablation
- `make_ablation_hook(heads, mean_z, type)` -- hook function for zero/mean/resample ablation
- `compute_faithfulness(...)` / `compute_completeness(...)` -- core circuit metrics
- `EvalResult` dataclass with `save_results()` for JSON serialization

### CircuitSpec -- Universal Circuit Schema

```python
@dataclass(frozen=True)
class CircuitSpec:
    roles: dict[str, list[tuple[int, int]]]  # role_name -> list of (layer, head)
    bands: dict[str, tuple]                   # band_name -> (layer_range, role_names)
    pathways: list[tuple[str, str]]           # directed edges between roles
    mlp_nodes: dict[str, list[int]]           # role_name -> list of MLP layers
    features: dict[str, list[tuple[int, str, int]]]  # SAE/factor features
    source: str                               # "published" | "ours" | "experimental"
    model_family: str                         # "gpt2" | "attn_only_4L" | ...

    def get_all_heads() -> set[tuple[int, int]]
    def get_all_edges() -> set[tuple[int, int, int, int]]
    def to_dict() -> dict
    def to_mib_edges() -> dict           # MIB benchmark format
    def to_interpbench_edges() -> list   # InterpBench format
```

### ArtifactAdapter -- Extension Point for SAEs/Factors

The `ArtifactAdapter` pattern allows the framework to evaluate learned representation artifacts (SAEs, factor banks, transcoders, crosscoders) without knowing their internals:

```python
class ArtifactAdapter:
    def load() -> None
    def directions(layer=None)              # feature direction vectors
    def activations(model, tokens, hook)    # feature activations
    def ablate(model, tokens, units, site)  # feature-level ablation
    def metadata() -> dict
```

Concrete implementations exist for factor banks (`factor_bank.py`) and SAEs (`sae.py`).

---

## 9. Architecture Decision Records (ADRs)

14 ADRs in `docs/decisions/` document the evolution of the benchmark infrastructure:

| # | Decision | Status |
|---|----------|--------|
| 001 | Pydantic v2 for all models | Implemented |
| 002 | 3 Tracks + 4 Views + 4 Gates architecture | Implemented |
| 003 | Track 3 as pre-registered hypothesis testing | Implemented |
| 004 | role_ablation as the core Track 3 metric | Implemented |
| 005 | Linguistics-based task taxonomy | Implemented |
| 006 | Mixed component support (heads + MLPs + neurons) | Implemented |
| 007 | Edge-level circuit flexibility | Implemented |
| 008 | IOI prediction calibration from real measurements | Implemented |
| 009 | Views as aggregations, not new metrics | Implemented |
| 010 | Gates as preconditions, not scores | Implemented |
| 011 | Artifact adapter pattern for SAEs/factors | Implemented |
| 012 | Weave tracing as optional zero-cost shim | Implemented |
| 013 | Epistemic framing claim spec | Implemented |
| 014 | Coverage and comparison analysis scripts | Implemented |

Key decisions explained:
- **ADR-002**: The 3T+4V+4G architecture avoids conflating discovery (Track 1), variable identification (Track 2), and hypothesis testing (Track 3). Views aggregate, they do not introduce new measurement infrastructure. Gates prevent garbage-in-garbage-out.
- **ADR-003**: Track 3 requires pre-registration via MechanisticClaimSpec to prevent post-hoc rationalization. This is the confirmatory factor analysis analog.
- **ADR-004**: `role_ablation` is the workhorse metric for Track 3. It ablates all heads in one role and measures the effect on another role (or output), enabling directed causal claims.
- **ADR-009**: Views were originally separate metrics but were redesigned as pure aggregations to avoid measurement proliferation.
- **ADR-010**: Gates were originally scored but were redesigned as binary preconditions -- either the precondition is met or it is not.
- **ADR-011**: ArtifactAdapters allow the framework to evaluate SAEs/factor banks without coupling to any specific implementation.

---

## 10. Relationship to the Factorized Circuits Project

The mechanistic-validity framework and the factorized circuits project (SAELensCircuitPort, in the `factorization-circuits` repository) are complementary:

**Factorized Circuits** provides the circuits to evaluate:
- `FactorizedHookedTransformer` decomposes a pretrained model's weight matrices into a shared `FactorBank` + per-projection `Selector` matrices.
- This factorization reveals which "factors" (directions in the residual stream) are used by which projections (Q, K, V, O, W_in, W_out).
- The RTI circuit was discovered via weight-space factorization and is one of the 12 claim specs in mechanistic-validity.
- The analysis pipeline in `factorization-circuits` produces per-factor characterization artifacts that can be evaluated using the mechanistic-validity framework.

**Mechanistic Validity** provides the evaluation methodology:
- How to evaluate whether a discovered circuit is real (Track 3 verification).
- How to compare circuits discovered by different methods (V4 adjudication).
- How to assess whether a circuit generalizes (V2 transportability).
- How to check if the measurements are trustworthy (G1 calibration).
- How to handle polysemantic confounds in ablation experiments (G3 superposition risk).

The `ArtifactAdapter` system in mechanistic-validity can wrap a `FactorBank` from the factorization project, enabling the framework's metrics to run on factor-level circuits in addition to head-level circuits.

---

## 11. Potential Extensions and Improvements

### Near-term (CPU, no GPU needed)

1. **Proxy circuit validation protocol** -- formalize when a proxy circuit is accepted vs needs its own circuit discovery.
2. **Hub head analysis** -- characterize the "infrastructure heads" (4,11), (10,0), (0,8) that appear across many circuits.
3. **More claim specs** -- tasks with full circuits but no specs could be given spec coverage.
4. **Circuit fingerprinting expansion** -- run PAS synergy on IOI, SVA, Acronym (3 circuits still missing synergy data).
5. **Coverage gap filling** -- some specs have steps that have never been tested as ablation targets or measurement targets.

### Medium-term (needs GPU)

1. **Run verify() on all 12 specs** -- get actual confirmation rates and claim ceilings.
2. **Proxy circuit testing** -- run activation patching per variant to verify circuit transfer.
3. **Epistemic rivalry resolution** -- run all 4 epistemic specs through Track 3 to see which has the highest confirmation rate.
4. **Cross-model transportability** -- test circuits on GPT-2 medium/large.
5. **MLP integration** -- the greater_than spec includes MLPs; need MLP ablation support in verify().

### Longer-term

1. **MIB integration** -- submit to Mechanistic Interpretability Benchmark leaderboard (arXiv:2504.13151).
2. **Multi-model support** -- extend beyond GPT-2 small.
3. **SAE/crosscoder integration** -- use ArtifactAdapter system for feature-level circuits.
4. **FastMCP server** -- expose verify/run/calibrate as MCP tools.
5. **Interactive dashboard** -- W&B Weave or local dev server for results visualization.
6. **Automated circuit discovery** -- use weight-space factorization to find circuits programmatically.
7. **Discovery by fingerprint** -- flip the pipeline: given a target fingerprint (e.g. the induction profile), find matching head subsets in a new model without task-specific prompts.

### Open Questions

1. Are proxy circuits generally valid, or does each prompt variant need its own circuit?
2. Is (4,11) truly a general-purpose head, or is it an artifact of circuit discovery methods?
3. Can mechanism adjudication (V4) resolve the epistemic rivalry, or are all 4 circuits "correct" at different description levels?
4. What is the right threshold for "proxy circuit accepted" in V2 transportability?
5. Should generator-only tasks (25 of 54) be prioritized for circuit discovery, or is the taxonomy sufficient?
6. Can structure predict function? If circuit fingerprints encode computational strategy, this would be evidence that circuit architecture is not arbitrary.

---

## 12. Repository Structure

```
docs/                               Astro/Starlight documentation site
  src/content/docs/
    framework/                      THE FRAMEWORK (the core contribution)
      taxonomy/index.md             Five-layer hierarchy overview
      modes_v3/                     7 description modes
      validity-types_v4/            5 validity types
      criteria/                     27 operational criteria
        construct/                  C1-C5
        internal/                   I1-I5
        external/                   E1-E6
        measurement/                M1-M6
        interpretive/               V1-V5
      evidence-families_v3/         6 evidence families
      metrics/                      Metric documentation (by family)
      verdicts_v3/                  Verdict tiers
      lenses_v6/                    13 worked case studies
    how-to/                         Practical guides
    reference/                      Bibliography
    start/                          Quickstart, glossary
  decisions/                        14 ADRs

src/mechanistic_validity/
  __init__.py                       Top-level API (84 metrics, 14 calibrations)
  models.py                         Pydantic v2 enums and I/O models
  spec.py                           MechanisticClaimSpec and verification types
  registry.py                       Gymnasium-style task registry
  views.py                          V1-V4 scoring aggregations
  gates.py                          G0-G3 precondition checks
  tracing.py                        Weave tracing shim
  metrics/                          84 metric scripts (by evidence family)
    causal/                         32 metrics (A-family)
    structural/                     18 metrics (B-family)
    behavioral/                     20 metrics (D-family)
    representational/               5 metrics (E-family)
    information/                    9 metrics (C-family)
    common.py                       Shared CLI, model loading, ablation hooks
  calibrations/                     14 calibration scripts
  methods/                          Composed procedures
  lib/
    tasks/                          54 task definitions
      _builtins.py                  All task class registrations
      task.py                       CircuitTask base class
      spec.py                       CircuitSpec schema
      prompts.py                    TaskPrompt type
      ioi/                          IOI: circuit.py, claim_spec.py, prompts.py
      greater_than/                 Greater-Than: circuit, spec, prompts
      induction/                    Induction: circuit, spec, prompts
      sva/                          SVA: circuit, spec, prompts
      gendered_pronoun/             Gendered Pronoun: circuit, spec, prompts
      acronym/                      Acronym: circuit, spec, prompts
      copy_suppression/             Copy Suppression: circuit, spec, prompts
      rti/                          RTI: circuit, spec, prompts
      epistemic_framing/            Epistemic: 4 circuits, 4 specs, prompts
    artifacts/                      ArtifactAdapter pattern
      adapter.py                    Base ArtifactAdapter class
      factor_bank.py                FactorBank adapter
      sae.py                        SAE adapter
    features/                       Feature-level utilities
    utils/                          Shared helpers

circuit_fingerprints/               Fingerprint analysis scripts and results
specs/                              Exported spec JSON files
scripts/                            Build/utility scripts
tests/                              Test suite
```

---

## 13. Glossary

| Term | Definition |
|------|-----------|
| **Circuit** | A subset of model components (attention heads, MLPs, neurons) that implement a specific computation |
| **Claim spec** | A `MechanisticClaimSpec` -- a pre-registered mechanistic hypothesis with testable predictions |
| **Confirmation rate** | Fraction of a spec's positive predictions that passed verification |
| **Claim ceiling** | The highest description mode at which all predictions pass |
| **Proxy circuit** | A circuit inherited from a related task without independent verification |
| **Hub head** | An attention head that appears in multiple circuits with different functional roles |
| **PAS** | Pairwise Ablation Synergy -- measures second-order interaction between two heads |
| **CMD** | Circuit Model Distance -- a scoring metric for Track 1 (MIB-compatible) |
| **IIA** | Interchange Intervention Accuracy -- the primary score for Track 2 |
| **EAP** | Edge Attribution Patching -- a circuit discovery method |
| **DAS** | Distributed Alignment Search -- finds where a causal variable is represented |
| **CATE** | Conditional Average Treatment Effect -- causal effect estimation |
| **Role ablation** | Ablate all heads in a functional role and measure the effect |
| **Evidence family** | One of 5 categories of evidence: Causal, Structural, Behavioral, Representational, Information-theoretic |
| **View** | A scoring aggregation over metrics from a causal-inference perspective (V1-V4) |
| **Gate** | A binary precondition that must pass before measurement is meaningful (G0-G3) |
| **ROLES** | Dict mapping role names to lists of (layer, head) tuples |
| **BANDS** | Dict mapping temporal bands to (layer_range, role_names) |
| **PATHWAYS** | List of (source_role, target_role) directed edges |
| **HookedTransformer** | TransformerLens model class with hook points for interventions |
| **MIB** | Mechanistic Interpretability Benchmark (Mueller et al. 2025) |
| **ACDC** | Automatic Circuit DisCovery -- an automated circuit discovery method |

---

## 14. References

Core theory:
- Spirtes, Glymour, Scheines. *Causation, Prediction, and Search.* MIT Press, 2000.
- Pearl. *Causality: Models, Reasoning, and Inference.* Cambridge, 2009.
- Mayo. *Statistical Inference as Severe Testing.* Cambridge, 2018.
- Glennan. *The New Mechanical Philosophy.* Oxford, 2017.
- Woodward. *Making Things Happen.* Oxford, 2003.

Causal abstraction:
- Geiger et al. "Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability." JMLR, 2023.
- Pearl, Bareinboim. "Transportability of Causal and Statistical Relations." AAAI, 2011.

Circuits evaluated:
- Wang et al. "Interpretability in the Wild: A Circuit for Indirect Object Identification in GPT-2 Small." ICLR 2023.
- Olsson et al. "In-context Learning and Induction Heads." arXiv:2209.11895, 2022.
- Hanna et al. "How Does GPT-2 Compute Greater-Than?" NeurIPS 2023.
- McDougall et al. "Copy Suppression: Comprehensively Understanding an Attention Head." arXiv:2310.04625, 2023.
- Lazo et al. "Subject-Verb Agreement Circuits in GPT-2." arXiv:2506.22105, 2025.
- Garcia-Carrasco et al. "Acronym Prediction with ACDC." AISTATS 2024.

Benchmarks:
- Mueller et al. "MIB: Mechanistic Interpretability Benchmark." arXiv:2504.13151, 2025.
