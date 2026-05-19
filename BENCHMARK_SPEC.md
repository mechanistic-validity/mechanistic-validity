# Mechanistic Validity Benchmark Specification

## Overview

The Mechanistic Validity benchmark evaluates circuit claims in mechanistic interpretability through a unified causal framework: **3 Tracks + 4 Views + 4 Gates**.

- **Tracks** are competitive benchmarks where methods compete on standardized inputs/outputs.
- **Views** are scoring aggregations over existing metrics — different lenses on the same results.
- **Gates** are precondition checks that must pass before tracks/views are meaningful.

The framework maps each component to established terminology from causal inference (Pearl, SGS, Rubin, Bareinboim, Geiger), neuroscience, psychometrics, pharmacology, and philosophy of science.

---

## Tracks

### Track 1: Causal Discovery

**MIB-compatible name:** Circuit Localization

**Question:** What causal structure/components explain the behavior?

| Field | Value |
|-------|-------|
| Input | model + task |
| Output | edge set / circuit |
| Score | CMD (Circuit Model Distance), faithfulness |
| Causal inference | Causal discovery (Spirtes, Glymour, Scheines) |
| Neuroscience | Circuit mapping / connectomics |
| Psychometrics | Exploratory factor analysis (EFA) |
| Pharmacology | Target discovery |
| Phil of science | Abduction / exploratory inference |

**Methods:** EAP, ACDC, path patching, NOTEARS, subnetwork probing.

**What it does:** Given a model and a task, discover which components (heads, MLPs, edges) are causally relevant to the task behavior. No prior theory required — purely data-driven.

---

### Track 2: Causal Abstraction

**MIB-compatible name:** Causal Variable Localization

**Question:** Where is the high-level causal variable represented in the model?

| Field | Value |
|-------|-------|
| Input | model + task + variable |
| Output | aligned subspace / features |
| Score | IIA (Interchange Intervention Accuracy) |
| Causal inference | Causal abstraction (Geiger, Potts et al.) |
| Neuroscience | Functional localization |
| Psychometrics | Construct operationalization |
| Pharmacology | Target characterization / mechanism-of-action |
| Phil of science | Natural kind identification |

**Methods:** DAS (Distributed Alignment Search), boundless DAS, probing, CCA/CKA alignment.

**What it does:** Given a model, task, and a hypothesized high-level causal variable (e.g., "subject number"), find where and how that variable is represented in the model's internal activations.

**Note:** "Causal abstraction" originates from Rubenstein et al. (2017) and was developed for mechanistic interpretability by Geiger et al. (2021, 2023). It extends classical causal inference rather than being a classical pillar.

---

### Track 3: Causal Model Testing

**Question:** Is a proposed mechanistic model consistent with the model's interventional behavior?

| Field | Value |
|-------|-------|
| Input | model + task + `MechanisticClaimSpec` |
| Output | verdict profile + claim ceiling |
| Score | confirmation rate, negative control rate, claim ceiling |
| Causal inference | Model testing / specification testing (SGS, Pearl Ch. 2) |
| Neuroscience | Causal circuit dissection |
| Psychometrics | Confirmatory factor analysis (CFA) |
| Pharmacology | Proof of mechanism |
| Phil of science | Hypothetico-deductive testing / severe testing (Mayo) |

**Methods:** Pre-registered predictions executed via existing metrics (activation patching, DAS-IIA, mediation, ablation, etc.).

**What it does:** Given a model, task, and a pre-registered mechanistic hypothesis (`MechanisticClaimSpec`), test whether the model's interventional behavior matches the spec's predictions. This is not blind discovery (Track 1) — it is theory-driven hypothesis testing.

**Key distinction from Track 1:**
- Track 1 is **unsupervised** — find whatever circuit exists (exploratory factor analysis).
- Track 3 is **supervised** — test whether a specific proposed mechanism is implemented (confirmatory factor analysis).

**Why it's a track, not just validation:** It changes the input contract (requires a `MechanisticClaimSpec`), the output contract (verdicts over predictions, not an edge set), and the success condition (confirmation rate, not faithfulness). Methods compete on how well their candidate circuits satisfy pre-registered mechanistic predictions.

**Naming note:** "Causal model testing" is descriptively correct in causal inference terms but is not a named subfield the way "causal discovery" is. The activity is well-established in SGS and Pearl (testing testable implications of a DAG against data). The compound phrase is our naming.

---

## Views

Views aggregate existing metrics into scoring lenses. They do not introduce new measurement infrastructure.

### V1: Causal Effect Estimation

**Question:** How large is each component/edge's causal contribution?

| Field | Value |
|-------|-------|
| Aggregates | mediation, mediation_v2, cate, effect_size, dose_response, pse, intervention_specificity |
| Causal inference | Effect estimation (Pearl/Rubin — ATE, CATE, NDE/NIE) |
| Neuroscience | Quantitative neurophysiology / tuning curves |
| Psychometrics | Effect size estimation / reliability |
| Pharmacology | Dose-response / PK-PD modeling |

### V2: Causal Transportability

**Question:** Does the mechanism generalize across prompts, tasks, and models?

| Field | Value |
|-------|-------|
| Aggregates | cross_task_generalization, cross_model_invariance, generalization_gap, measurement_invariance |
| Causal inference | Transportability (Pearl/Bareinboim) |
| Neuroscience | Cross-condition / cross-species generalization |
| Psychometrics | Measurement invariance / external validity |
| Pharmacology | Phase 3 generalization / translation |

### V3: Counterfactual Verification

**Question:** Do counterfactual interventions produce the expected results?

| Field | Value |
|-------|-------|
| Aggregates | das_iia, iia_variants, counterfactual_consistency, corrupt_restore, multi_axis_iia |
| Causal inference | Pearl's rung-3 counterfactuals |
| Neuroscience | Optogenetic interrogation |

### V4: Mechanism Adjudication

**Question:** When multiple mechanisms could explain the data, which is better supported?

| Field | Value |
|-------|-------|
| Aggregates | discriminant_validity, rival spec comparison, alternative exclusion scoring |
| Causal inference | SEM equivalent-models testing |
| Phil of science | Crucial experiment (Bacon) |

---

## Gates

Gates are preconditions. They must pass before tracks and views produce meaningful results.

### G0: Construct Operationalization

**What it checks:** Is the construct (the thing being measured) defined independently of the instruments used to measure it?

**Maps to:** Criterion C1 (falsifiability) in the framework's 27 criteria.

**Why it matters:** A circuit named for what instruments found ("the patching-relevant heads") is circular. The construct must be defined before measurement begins.

### G1: Measurement Calibration

**What it checks:** Are the metric outputs trustworthy — stable across seeds, separable from random baselines, and replicable?

**Maps to:** Existing calibrations — `bootstrap`, `seed_variance`, `distributional_stability`, random baseline (F09), untrained model baseline (F10).

**Why it matters:** An IIA score of 0.48 without knowing the random-vector baseline is not evidence. Calibration determines whether a number is signal or noise.

### G2: Causal Identifiability

**What it checks:** Can the causal effects specified in the `MechanisticClaimSpec` be estimated with the available interventions?

**Maps to:** Metadata on the spec (`identifiability` field).

**Why it matters:** In Pearl's framework, a causal effect must be identifiable (expressible in terms of available data/interventions) before it can be estimated. If the effect requires an intervention type that isn't available (e.g., clamping a specific feature in a polysemantic neuron), the estimate is not trustworthy.

**Note:** Causal identifiability is a precondition (gate), not a competition (track). It asks "CAN this effect be measured?" not "HOW WELL can you measure it?"

### G3: Confound/Superposition Risk

**What it checks:** Are ablation-based interventions confounded by polysemantic collateral damage?

**Maps to:** Metadata on the spec (`superposition_risk` field) and internal validity criterion "confound control."

**Why it matters:** MI-specific concern. Ablating a "duplicate token detector" head that is also polysemantically involved in other computations introduces confounds. The intervention is available (G2 passes) but not specific (G3 flags risk). This is distinct from identifiability.

---

## MechanisticClaimSpec — the Track 3 artifact

The `MechanisticClaimSpec` is a Pydantic v2 model that formalizes a pre-registered mechanistic hypothesis. It is the input artifact for Track 3.

### Structure

```
MechanisticClaimSpec
├── task_id: str
├── model_family: str
├── linguistic_claim: str                    # natural language claim
├── steps: list[ComputationalStep]           # nodes in the mechanism DAG
│   ├── name, category, description
│   ├── input_type, output_type, position
│   ├── maps_to_role, maps_to_heads
│   └── description_mode, discovery_status
├── edges: list[ComputationalEdge]           # directed edges between steps
│   ├── source, target
│   └── mechanism, description
├── predictions: list[CausalPrediction]      # things that SHOULD happen
│   ├── name, claim
│   ├── intervention, intervention_target
│   ├── measurement_target
│   ├── expected_direction, expected_metric
│   ├── expected_threshold
│   └── description_mode
├── negative_controls: list[CausalPrediction]  # things that should NOT happen
├── rival_specs: list[str]                   # IDs of competing mechanisms
├── identifiability: IdentifiabilityGate     # G2
├── superposition_risk: SuperpositionGate    # G3
├── description_mode: DescriptionMode
├── paper_ref: str | None
└── author: str
```

### Computed properties

- `confirmation_rate` — fraction of positive predictions that passed
- `negative_control_rate` — fraction of negative controls that passed
- `untested_predictions()` — predictions not yet evaluated
- `all_predictions()` — positive + negative combined

### SpecVerificationResult — Track 3 output

```
SpecVerificationResult
├── prediction_results: list[PredictionResult]    # per-prediction verdicts
├── mode_verdicts: list[ModeVerdict]              # aggregated by description mode
├── claim_ceiling: DescriptionMode | None         # highest mode with all predictions passing
├── verdict_tier: VerdictTier                     # proposed → validated
├── gates_passed: dict[str, bool]                 # G0-G3 status
├── effect_estimation_score: float | None         # V1
├── transportability_score: float | None          # V2
├── counterfactual_score: float | None            # V3
├── adjudication_score: float | None              # V4
└── confirmation_rate: float                      # computed
```

### Verdict tiers (from the framework)

| Tier | What it means |
|------|---------------|
| **Proposed** | No intervention evidence — structural or representational only |
| **Causally suggestive** | Necessity shown, sufficiency not yet established |
| **Mechanistically supported** | Necessity + sufficiency, at least 2 ablation variants |
| **Triangulated** | All internal criteria met, plus external and construct criteria |
| **Validated** | All five validity types addressed with explicit baselines |
| **Underdetermined** | Evidence consistent with multiple mechanisms |
| **Disconfirmed** | Fails decisively on a key criterion |

### Description modes

| Mode | What the evidence establishes |
|------|------|
| `computational` | What function the system computes and why |
| `algorithmic` | What procedure the system executes |
| `representational` | What information is encoded and in what geometry |
| `implementational-topographic` | Which components are involved |
| `implementational-connectomic` | How components are wired |
| `implementational-statistical` | Distributional properties of activations |
| `implementational-functional` | What input-output transformation a component performs |

---

## Cross-field mapping (complete)

| Component | Causal inference | Neuroscience | Psychometrics | Pharmacology | Phil of science |
|-----------|-----------------|--------------|---------------|--------------|----------------|
| Track 1: Causal Discovery | Causal discovery (SGS) | Circuit mapping | Exploratory factor analysis | Target discovery | Abduction |
| Track 2: Causal Abstraction | Causal abstraction (Geiger) | Functional localization | Construct operationalization | Target characterization | Natural kind identification |
| Track 3: Causal Model Testing | Model testing (SGS, Pearl) | Circuit dissection | Confirmatory factor analysis | Proof of mechanism | Hypothetico-deductive testing |
| V1: Effect Estimation | Effect estimation (Pearl/Rubin) | Perturbation quantification | Effect size estimation | Dose-response | Parameter estimation |
| V2: Transportability | Transportability (Bareinboim) | Cross-condition generalization | Measurement invariance | Phase 3 generalization | Novel prediction |
| V3: Counterfactual Verification | Rung-3 counterfactuals (Pearl) | Optogenetic interrogation | — | — | Counterfactual testing |
| V4: Mechanism Adjudication | Equivalent-models testing (SEM) | — | Model comparison | — | Crucial experiment |
| G0: Construct Operationalization | — | — | Construct validity | — | Operationalism |
| G1: Measurement Calibration | — | Test-retest reliability | Measurement reliability | Assay validation | — |
| G2: Causal Identifiability | Identifiability (Pearl) | — | — | — | — |
| G3: Confound/Superposition | — | Collateral damage in lesions | Discriminant validity | Off-target effects | Confound control |

---

## API

```python
import mechanistic_validity as mv

# Track 3: Causal Model Testing
spec = mv.load_task("ioi").get_claim_spec()
result = mv.verify(spec, device="cuda")
result.confirmation_rate          # 0.8
result.claim_ceiling              # DescriptionMode.impl_functional
result.verdict_tier               # VerdictTier.mechanistically_supported

# Views
mv.run_view("effect_estimation", tasks=["ioi"])
mv.run_view("transportability", tasks=["ioi", "sva"])

# Gates
mv.check_gate("measurement_calibration", task="ioi")
mv.check_gate("identifiability", spec=spec)

# CLI
# mv verify specs/ioi_spec.json --device cuda
# mv view effect_estimation --tasks ioi
# mv gate measurement_calibration --task ioi
```

---

## References

- Spirtes, Glymour, Scheines. *Causation, Prediction, and Search.* MIT Press, 2000.
- Pearl. *Causality: Models, Reasoning, and Inference.* Cambridge, 2009.
- Geiger et al. "Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability." JMLR, 2023.
- Pearl, Bareinboim. "Transportability of Causal and Statistical Relations." AAAI, 2011.
- Mayo. *Statistical Inference as Severe Testing.* Cambridge, 2018.
- Guo et al. "A Survey on Causal Inference." ACM TKDD, 2020.
- MIB: Mechanistic Interpretability Benchmark. arXiv:2504.13151, 2025.
