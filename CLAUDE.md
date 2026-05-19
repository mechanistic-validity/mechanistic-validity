# CLAUDE.md

This file provides guidance to Claude Code when working in this repository.

## Project overview

Mechanistic Validity is a theoretical framework for evaluating claims in
mechanistic interpretability. It provides validation methodology drawn from
philosophy of science, neuroscience, pharmacology, and measurement theory.

This repo contains NO experiment results or novel methods — the contribution
is purely theoretical. Scripts under `src/metrics/` are example
implementations only.

## Repository structure

```
docs/                           Astro-based documentation site
  src/content/docs/
    framework/                  THE FRAMEWORK (the core contribution)
      taxonomy/index.md         Five-layer hierarchy overview + design principles
      modes_v3/                 7 description modes (partial order by commitment)
        index.md                Overview + decision procedure + upgrade paths
        computational.md        "What function, and why?"
        algorithmic.md          "What procedure does it execute?"
        representational.md     "What information is encoded?"
        implementational-functional.md   "What does each component do?"
        implementational-connectomic.md  "How are components wired?"
        implementational-topographic.md  "Which components are involved?"
        implementational-activation.md   "What do activations look like?"
      validity-types_v4/        5 validity types (Construct, Internal, External,
                                Measurement, Interpretive)
      criteria/                 ~27 operational criteria organized by validity type
        construct/              C1-C5 (falsifiability, structural plausibility, etc.)
        internal/               I1-I5 (necessity, sufficiency, specificity, etc.)
        external/               E1-E6 (intervention reach, graded response, etc.)
        measurement/            M1-M6 (reliability, invariance, baseline sep, etc.)
        interpretive/           V1-V5 (level declaration, level-evidence match, etc.)
      evidence-families_v3/     6 evidence families (causal, structural, etc.)
      instruments_v2/           Docs for metrics (legacy name in docs site)
        causal/                 a01-a13 (SCM, DAS-IIA, CATE, mediation, etc.)
        structural/             b01-b09 (SVD, effective rank, OV/QK, etc.)
        representational/       e01-e10 (DAS-IIA, linear probe, RSA, etc.)
        behavioral/             d01-d09 (faithfulness, logit diff, KL, etc.)
        information/            c01-c09 (MI, transfer entropy, PID, OCSE, etc.)
      verdicts_v3/              Verdict tiers (Proposed → Validated + Disconfirmed)
      lenses_v6/                Disciplinary lenses + 13 worked case studies
        examples/               IOI, induction, greater-than, grokking, etc.
    how-to/                     Practical guides (audit a claim, pick a mode tag, etc.)
    reference/                  Bibliography, known circuits
    start/                      Quickstart, glossary, overview
  public/diagrams/              Framework diagrams
  public/figures/               Figures for README and docs

src/                            Example implementations (Python)
  mechanistic_validity/         Installable package — programmatic API
    __init__.py                 Public API: run(), verify(), calibrate(), etc.
    models.py                   Pydantic v2 enums and I/O models
    spec.py                     MechanisticClaimSpec + Track 3 models
    views.py                    View aggregations V1-V4
    gates.py                    Gate checks G0-G3
    cli.py                      Cyclopts CLI (`mv` command)
    config.py                   pydantic-settings (MV_* env vars, mv_config.yaml)
    tracing.py                  W&B Weave shim (@mv.op decorator)
    registry.py                 Gymnasium-style load_task() / list_tasks() API
    metrics/                    Atomic measurements organized by evidence family
      causal/                   A-family: causal probes (ablation, patching, DAS, etc.)
      structural/               B-family: weight-space measurements (SVD, norms, etc.)
      behavioral/               D-family: behavioral tests (faithfulness, KL, etc.)
      representational/         E-family: representation analysis (CKA, RSA, probes, etc.)
      information/              C-family: information-theoretic (MI, PID, etc.)
      common.py                 Shared CLI, EvalResult, load_model, task utilities
    calibrations/               Trustworthiness gates (run on metric outputs)
      bootstrap_stability/      F01: score stability under resampling
      convergent_validity/      F03: do different metrics agree?
      discriminant_validity/    F04: do different constructs disagree?
      internal_consistency/     F05: multi-item consistency
      inter_rater/              F06: agreement across runs
      measurement_invariance/   F07: metric × condition stability
      sensitivity/              F13: detection power (AUROC)
      ablation_invariance/      Ablation method invariance
      certified_stability/      Certified stable classification
      multiple_comparisons/     Multiple comparison correction
      test_retest/              Split-half reliability
    methods/                    Composed procedures (orchestrate metrics + calibrations)
      mib_faithfulness/         MIB Track 1/2 faithfulness curves
    lib/                        Shared utilities, task definitions, features
      tasks/                    Per-task modules (circuit, prompts, claim_spec)
        ioi/                    IOI circuit + prompts + claim spec
        greater_than/           Greater-than circuit + prompts + claim spec
        induction/              Induction circuit + prompts + claim spec
        sva/                    SVA circuit + prompts + claim spec
        rti/                    RTI circuit + prompts + claim spec
        gendered_pronoun/       Gendered pronoun circuit + prompts + claim spec
        copy_suppression/       Copy suppression circuit + prompts + claim spec
        acronym/                Acronym circuit + prompts + claim spec
        epistemic_framing/      Epistemic circuit + prompts + 4 rival claim specs
        _builtins.py            All 54 task class registrations

scripts/                        Build/utility scripts
```

## Key concepts

### Three-tier taxonomy

```
Metrics       — atomic measurements on a circuit/model, output one number
                (5 families: Causal, Structural, Behavioral, Representational, Info-theoretic)
Calibrations  — trustworthiness gates on metric outputs, determine if numbers are meaningful
                (bootstrap, baselines, convergent/discriminant validity)
Methods       — composed procedures that orchestrate metrics + calibrations
                (Track 1: circuit discovery, Track 2: causal variable ID, Track 3: characterization)
```

### Three tracks

```
Track 1: Circuit Discovery    — "Which components are in the circuit?"
         (EAP, ACDC, HISP, weight-space discovery)
Track 2: Causal Variable ID   — "Which representations encode task variables?"
         (DAS, IIA, steering vectors)
Track 3: Causal Model Testing — "Does the proposed mechanism actually work this way?"
         (MechanisticClaimSpec, role_ablation, pre-registered predictions)
         FULLY IMPLEMENTED — see "Benchmark infrastructure" section below
```

### Seven description modes (partial order)

```
Computational > Algorithmic > Representational > Impl-Functional > Impl-Connectomic > Impl-Topographic
                                                                              (Impl-Statistical is orthogonal)
```

Each higher mode requires all evidence from lower modes plus bridging evidence.
Mode tags are applied to verdicts (Layer E), not to metrics or evidence.

### Verdict tiers

Proposed < Causally Suggestive < Mechanistically Supported < Triangulated < Validated

### Validity type dependency order

Construct → Measurement → Internal → External → Interpretive

## How to find things

- **"What are the criteria?"** → `docs/src/content/docs/framework/criteria/`
- **"What are the description modes?"** → `docs/src/content/docs/framework/modes_v3/`
- **"How do I evaluate a specific circuit?"** → `docs/src/content/docs/how-to/audit-a-claim.md`
- **"What metrics exist?"** → `docs/src/content/docs/framework/instruments_v2/`
- **"How was IOI/induction/etc. evaluated?"** → `docs/src/content/docs/framework/lenses_v6/examples/`
- **"What verdict tier is appropriate?"** → `docs/src/content/docs/framework/verdicts_v3/`
- **"Full taxonomy overview?"** → `docs/src/content/docs/framework/taxonomy/index.md`

## Metric scripts (`src/metrics/`)

84 registered metrics organized by evidence family. Each script has
a standardized docstring header:

```
Metric:         A04 — Woodward Interventionism
Categories:     causal
Validity layer: Internal
Criteria:       I1 Necessity
```

### Shared infrastructure (`src/metrics/common.py`)

All metrics share a common CLI and utilities:

```bash
uv run python src/metrics/causal/woodward/03_sigma_ablation.py --tasks ioi sva --device cpu
```

Standard CLI args: `--model` (default gpt2), `--device`, `--tasks`, `--n-prompts`, `--out`

Key functions:
- `load_model(name, device)` — cached HookedTransformer loading
- `get_circuit(task)` → ROLES, BANDS, PATHWAYS
- `generate_prompts(task, tokenizer, n_prompts)` → prompt objects
- `calibrate_mean_z(model, prompts)` → per-(layer,head) mean activations
- `make_ablation_hook(heads, mean_z, type)` → hook function
- `compute_faithfulness(...)` / `compute_completeness(...)` — core metrics
- `EvalResult` dataclass → `save_results()` → JSON

### Registered tasks (54 total)

54 tasks across 4 circuit statuses:

| Status | Count | Examples |
|--------|-------|---------|
| full_circuit | 12 | ioi, greater_than, induction, sva, rti, epistemic_framing (4 variants) |
| proxy_circuit | 10 | rti_pattern, token_flood, buffalo, centering_theory, resumptive, self_allo |
| generator_only | 25 | 7 linguistic probes, 12 BLiMP categories, 6 phonetic composition |
| planned | 7 | less_than, sva_pp, docstring, bracket_matching, sentiment |

10 domains: patterns, linguistics_coreference, linguistics_agreement,
linguistics_pragmatics, linguistics_syntax, linguistics_semantics,
linguistics_binding, linguistics_morphology, linguistics_phonology, math

9 experiment groups: published, rti_discovery, ioi_ablations,
repetition_taxonomy, epistemic_study, linguistic_probes, blimp,
phonetic_composition, roadmap

Each task has a circuit module with ROLES/BANDS/PATHWAYS and a prompt builder in TASK_REGISTRY.

### Adding a new task

1. Create `src/mechanistic_validity/lib/tasks/<task_name>/circuit.py` with ROLES, BANDS, PATHWAYS
2. Create `src/mechanistic_validity/lib/tasks/<task_name>/prompts.py` with a prompt builder function
3. Add a `CircuitTask` subclass in `src/mechanistic_validity/lib/tasks/_builtins.py`
   with `task_id`, `source`, `domain`, `experiment_group`, `circuit_status`
4. Add the class to `BUILTIN_TASK_CLASSES` at the bottom of `_builtins.py`
5. Optionally add a `claim_spec.py` for Track 3 support (see "Adding a claim spec" below)

### Metric → Criteria mapping (by evidence family)

| Family | Metric IDs | Primary criteria |
|--------|-----------|-----------------|
| Causal (a01-a13) | SCM, DAS-IIA, CATE, Woodward, MDC, Mediation, Granger, PID, MDL, INUS, Actual Cause, Transport, Discovery | I1-I5, C2, C4, E5, E6 |
| Structural (b01-b09) | SVD, Effective Rank, OV/QK, Weight Alignment, Norm, Template Distance, Polysemanticity, ICA/NMF, LLC | C2-C5 |
| Behavioral (d01-d09) | Faithfulness, Logit Diff, KL, CE Delta, Top-K, Calibration, Probe, MDL Compression, Generalization | I1, I2, M5, C4, E1 |
| Representational (e01-e10) | DAS-IIA, Linear Probe, RSA, CKA, Subspace, PCA, Intrinsic Dim, Participation, Persistence, Cross-Task | I2, I4, C2, C5, E5 |
| Information (c01-c09) | MI, Conditional MI, Transfer Entropy, PID, Info Bottleneck, O-Info, Granger, OCSE, NOTEARS | I1, I3, I4, C4 |

### Calibrations (`src/calibrations/`)

Trustworthiness gates — run on metric outputs before criteria evaluation.
Hard gates (must pass): F01 bootstrap stability, F09 random baseline, F10 untrained baseline.

| ID | Calibration | Checks |
|----|-------------|--------|
| F01 | Bootstrap stability | Score stable under resampling? |
| F03 | Convergent validity | Do different metrics agree? |
| F04 | Discriminant validity | Do different constructs disagree? |
| F05 | Internal consistency | Multi-item agreement? |
| F06 | Inter-rater agreement | Across independent runs? |
| F07 | Measurement invariance | Metric × condition interaction? |
| F13 | Sensitivity / AUROC | Can we detect known-true heads? |

## Benchmark infrastructure (Track 3)

### Programmatic API

The repo ships an installable Python package (`mechanistic_validity`) with a
Gymnasium-style task registry and a programmatic API:

```python
import mechanistic_validity as mv

# Load a task
task = mv.load_task("ioi")
circuit = task.get_circuit()

# Run a metric
results = mv.run("role_ablation", tasks=["ioi"])

# Track 3: Causal Model Testing
spec = mv.load_task("ioi").get_claim_spec()
result = mv.verify(spec, device="cpu")

# Views (scoring aggregations)
mv.run_view("effect_estimation", tasks=["ioi"])

# Gates (precondition checks)
mv.check_gate("measurement_calibration", task="ioi")

# Listing
mv.list_tasks()              # all 54 tasks
mv.list_metrics()            # all 84 metrics
mv.list_calibrations()       # all 14 calibrations
mv.list_domains()            # 10 linguistic domains
mv.list_experiment_groups()  # 9 experiment groups
```

CLI via Cyclopts (`pip install mechanistic-validity[cli]`):

```bash
mv run role_ablation --tasks ioi sva --device cpu
mv verify path/to/claim_spec.json --device cpu
mv view effect_estimation --tasks ioi
mv gate measurement_calibration --task ioi
mv tasks --has-circuit
mv metrics --family causal
```

### Track 3: Causal Model Testing (MechanisticClaimSpec)

Track 3 is the novel contribution: pre-registered mechanistic hypotheses as
structured Pydantic models. A `MechanisticClaimSpec` declares a mechanism DAG
(computational steps + edges), positive predictions, negative controls, rival
specs, and gate metadata. `mv.verify(spec)` runs all predictions and returns a
`SpecVerificationResult` with per-prediction verdicts, mode-level aggregation,
and a claim ceiling.

Core model hierarchy (in `src/mechanistic_validity/spec.py`):

```
MechanisticClaimSpec
  .steps: list[ComputationalStep]      — DAG nodes (name, category, maps_to_heads/mlps)
  .edges: list[ComputationalEdge]      — DAG edges (source, target, mechanism)
  .predictions: list[CausalPrediction] — positive predictions
  .negative_controls: list[CausalPrediction] — should-not-affect controls
  .rival_specs: list[str]              — linked spec IDs for V4 adjudication
  .identifiability: IdentifiabilityGate
  .superposition_risk: SuperpositionGate

SpecVerificationResult
  .prediction_results: list[PredictionResult]
  .mode_verdicts: list[ModeVerdict]    — per-description-mode aggregation
  .claim_ceiling: DescriptionMode      — highest mode all predictions pass
  .verdict_tier: VerdictTier
```

Core metric: `role_ablation` — ablate one role's components, measure the effect
on another role or on output. This is the workhorse for claim spec verification:
each `CausalPrediction` specifies an `intervention_target` (which role to
ablate) and a `measurement_target` (which role or output to measure), and
`mv.verify()` routes to `role_ablation` automatically.

### Claim specs (12 tasks)

Claim specs live in `src/mechanistic_validity/lib/tasks/<task>/claim_spec.py`.
12 tasks currently have specs:

| Task | Spec count | Rival specs |
|------|-----------|-------------|
| ioi | 1 | -- |
| greater_than | 1 | -- |
| induction | 1 | -- |
| sva | 1 | -- |
| rti | 1 | -- |
| gendered_pronoun | 1 | -- |
| copy_suppression | 1 | -- |
| acronym | 1 | -- |
| epistemic_framing | 1 | epistemic_tight, epistemic_eap, epistemic_expanded |
| epistemic_tight | 1 | epistemic_framing, epistemic_eap, epistemic_expanded |
| epistemic_eap | 1 | epistemic_framing, epistemic_tight, epistemic_expanded |
| epistemic_expanded | 1 | epistemic_framing, epistemic_tight, epistemic_eap |

The epistemic study has 4 rival specs linked for V4 adjudication (mechanism
comparison). Other tasks have a single canonical spec.

### Proxy circuits

Tasks with `circuit_status="proxy_circuit"` inherit a circuit from a related
task and hypothesize that the same mechanism applies to a different behavioral
phenomenon. These are hypotheses that need V2 transportability testing to verify.

Examples:
- rti_pattern, token_flood, buffalo all use RTI's 15-head circuit
- centering_theory, resumptive, self_allo all use IOI's 15-head circuit
- sequence_internal, alternating_pair, novel_song all use induction's circuit

### Views (scoring aggregations)

Views group existing metrics into causal-inference-grounded scoring lenses.
They are not separate measurement infrastructure — they reuse registered metrics.

```
V1: Causal Effect Estimation   (Pearl/Rubin)
    Metrics: mediation, mediation_v2, cate, effect_size, dose_response, pse,
             intervention_specificity

V2: Causal Transportability    (Pearl/Bareinboim)
    Metrics: cross_task_generalization, cross_model_invariance, generalization_gap

V3: Counterfactual Verification (Pearl rung-3)
    Metrics: das_iia, iia_variants, counterfactual_consistency, corrupt_restore,
             multi_axis_iia

V4: Mechanism Adjudication     (SEM equivalent-models)
    Metrics: discriminant_validity
```

### Gates (precondition checks)

Gates are pass/fail checks that must hold before tracks and views are meaningful.

```
G0: Construct Operationalization — is the construct defined independently?
    (checks: task has prompts + circuit)
G1: Measurement Calibration — are metrics stable and separable from baselines?
    (delegates to bootstrap/seed_variance calibrations)
G2: Causal Identifiability — can the effects be estimated with available interventions?
    (checks IdentifiabilityGate on the claim spec)
G3: Confound/Superposition Risk — are interventions confounded by polysemanticity?
    (checks SuperpositionGate on the claim spec)
```

G0 and G1 operate on tasks. G2 and G3 require a `MechanisticClaimSpec`.

### Adding a claim spec to a task

1. Create `src/mechanistic_validity/lib/tasks/<task>/claim_spec.py`
2. Define a `MechanisticClaimSpec` with steps, edges, predictions, negative controls
3. Import the spec in `_builtins.py` and wire `get_claim_spec()` on the task class
4. For rival specs, add additional `claim_spec_<variant>.py` files and link via
   `rival_specs` field on each spec

### API counts summary

| Category | Count |
|----------|-------|
| Metrics | 84 |
| Calibrations | 14 |
| Tasks | 54 (12 full_circuit, 10 proxy_circuit, 25 generator_only, 7 planned) |
| Claim specs | 12 |
| Domains | 10 |
| Experiment groups | 9 |
| Views | 4 (V1-V4) |
| Gates | 4 (G0-G3) |

## Plugin / Skills

This repo is registered as a Claude Code plugin. Skills live in `skills/`
and are installable globally via marketplace settings.

### Current skills

- `/mechval` — Load the full framework context (taxonomy, criteria, modes, etc.)

### Installing globally

Add to `~/.claude/settings.json`:
```json
{
  "extraKnownMarketplaces": {
    "mechanistic-validity": {
      "source": { "source": "github", "repo": "mechanistic-validity/mechanistic-validity" }
    }
  },
  "enabledPlugins": {
    "mechval@mechanistic-validity": true
  }
}
```

Then `/mechval` works in any project.

## 13 worked case studies

The framework has been applied to 13 published MI results:

| Study | Verdict |
|-------|---------|
| IOI Circuit (Wang et al.) | Triangulated |
| Induction Heads (Olsson et al.) | Mechanistically Supported |
| Greater-Than (Hanna et al.) | Mechanistically Supported |
| Copy Suppression (McDougall et al.) | Mechanistically Supported |
| Othello World Model (Li et al.) | Triangulated |
| Grokking (Nanda et al.) | Causally Suggestive |
| Successor Heads (Gould et al.) | Causally Suggestive |
| Docstring Circuit (Heimersheim & Janiak) | Causally Suggestive |
| SAE Features (Cunningham et al.) | Causally Suggestive |
| Knowledge Neurons (Dai et al.) | Proposed |
| Superposition (Elhage et al.) | Proposed |
| Probing Classifiers (Belinkov) | Proposed |
| Gender Bias Circuits (Vig et al.) | Proposed |

## Guidelines

- This is a docs site. The primary output is markdown content, not code.
- Do not edit metric implementations without understanding the
  corresponding docs page — they must stay in sync.
- The docs site uses Astro with Starlight. Build with `npm run dev` inside `docs/`.
- Criteria IDs (C1-C5, I1-I5, E1-E6, M1-M6, V1-V5) are stable identifiers
  referenced across the codebase and in external publications. Do not renumber.
- The 7 description modes and 5 validity types are the core ontology.
  Changes to these require updating the taxonomy, criteria, verdicts, how-to
  guides, and worked examples simultaneously.
- Metric IDs (a01, b01, etc.) follow evidence-family prefixes.
  New metrics get the next number in their family.
- Run metric scripts with `uv run python` from the repo root.
- When adding metrics, follow the docstring header format
  (Metric, Categories, Validity layer, Criteria, Establishes, Requires, Doc).
