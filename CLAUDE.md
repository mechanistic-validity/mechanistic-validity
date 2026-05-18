# CLAUDE.md

This file provides guidance to Claude Code when working in this repository.

## Project overview

Mechanistic Validity is a theoretical framework for evaluating claims in
mechanistic interpretability. It provides validation methodology drawn from
philosophy of science, neuroscience, pharmacology, and measurement theory.

This repo contains NO experiment results or novel methods — the contribution
is purely theoretical. Scripts under `src/instruments/` are example
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
      instruments_v2/           Concrete runnable tests organized by evidence family
        causal/                 a01-a13 (SCM, DAS-IIA, CATE, mediation, etc.)
        structural/             b01-b09 (SVD, effective rank, OV/QK, etc.)
        representational/       e01-e10 (DAS-IIA, linear probe, RSA, etc.)
        behavioral/             d01-d09 (faithfulness, logit diff, KL, etc.)
        information/            c01-c09 (MI, transfer entropy, PID, OCSE, etc.)
        measurement/            f01-f08 (bootstrap, convergent validity, etc.)
      verdicts_v3/              Verdict tiers (Proposed → Validated + Disconfirmed)
      lenses_v6/                Disciplinary lenses + 13 worked case studies
        examples/               IOI, induction, greater-than, grokking, etc.
    how-to/                     Practical guides (audit a claim, pick a mode tag, etc.)
    reference/                  Bibliography, known circuits
    start/                      Quickstart, glossary, overview
  public/diagrams/              Framework diagrams
  public/figures/               Figures for README and docs

src/                            Example instrument implementations (Python)
  instruments/                  Organized by evidence family (mirrors docs)
  lib/                          Shared utilities, task definitions, features

scripts/                        Build/utility scripts
```

## Key concepts

### Five-layer hierarchy (bottom-up)

```
Layer A: Instruments     — concrete runnable tests (activation patching, DAS, etc.)
Layer B: Evidence families — 6 kinds of signal (causal, structural, representational, etc.)
Layer C: Criteria         — ~27 falsifiable conditions grouped by validity type
Layer D: Validity types   — 5 abstract questions (construct, internal, external, measurement, interpretive)
Layer E: Verdict          — the claim with scope, tier, and mode tag
```

### Seven description modes (partial order)

```
Computational > Algorithmic > Representational > Impl-Functional > Impl-Connectomic > Impl-Topographic
                                                                              (Impl-Statistical is orthogonal)
```

Each higher mode requires all evidence from lower modes plus bridging evidence.
Mode tags are applied to verdicts (Layer E), not to instruments or evidence.

### Verdict tiers

Proposed < Causally Suggestive < Mechanistically Supported < Triangulated < Validated

### Validity type dependency order

Construct → Measurement → Internal → External → Interpretive

## How to find things

- **"What are the criteria?"** → `docs/src/content/docs/framework/criteria/`
- **"What are the description modes?"** → `docs/src/content/docs/framework/modes_v3/`
- **"How do I evaluate a specific circuit?"** → `docs/src/content/docs/how-to/audit-a-claim.md`
- **"What instruments exist?"** → `docs/src/content/docs/framework/instruments_v2/`
- **"How was IOI/induction/etc. evaluated?"** → `docs/src/content/docs/framework/lenses_v6/examples/`
- **"What verdict tier is appropriate?"** → `docs/src/content/docs/framework/verdicts_v3/`
- **"Full taxonomy overview?"** → `docs/src/content/docs/framework/taxonomy/index.md`

## Instrument scripts (`src/instruments/`)

93 canonical instrument scripts organized by evidence family. Each script has
a standardized docstring header:

```
Instrument:     A04 — Woodward Interventionism
Categories:     causal
Validity layer: Internal
Criteria:       I1 Necessity
```

### Shared infrastructure (`src/instruments/_common.py`)

All instruments share a common CLI and utilities:

```bash
uv run python src/instruments/causal/woodward/03_sigma_ablation.py --tasks ioi sva --device cpu
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

### Registered tasks (8 + 9 aliases)

Base: ioi, greater_than, induction, sva, gendered_pronoun, rti, acronym, copy_suppression
Aliases: rti_pattern, sequence_internal, alternating_pair, novel_song, centering_theory,
         resumptive, self_allo, token_flood, buffalo

Each task has a circuit module with ROLES/BANDS/PATHWAYS and a prompt builder in TASK_REGISTRY.

### Adding a new task

1. Create `src/lib/tasks/<task_name>/circuit.py` with ROLES, BANDS, PATHWAYS
2. Create `src/lib/tasks/<task_name>/prompts.py` registered in TASK_REGISTRY
3. Update `_common.py`: import module, add to `_TASK_TO_MODULE` and `CIRCUIT_TASKS`

### Instrument → Criteria mapping (by evidence family)

| Family | Instrument IDs | Primary criteria |
|--------|---------------|-----------------|
| Causal (a01-a13) | SCM, DAS-IIA, CATE, Woodward, MDC, Mediation, Granger, PID, MDL, INUS, Actual Cause, Transport, Discovery | I1-I5, C2, C4, E5, E6 |
| Structural (b01-b09) | SVD, Effective Rank, OV/QK, Weight Alignment, Norm, Template Distance, Polysemanticity, ICA/NMF, LLC | C2-C5 |
| Behavioral (d01-d09) | Faithfulness, Logit Diff, KL, CE Delta, Top-K, Calibration, Probe, MDL Compression, Generalization | I1, I2, M5, C4, E1 |
| Representational (e01-e10) | DAS-IIA, Linear Probe, RSA, CKA, Subspace, PCA, Intrinsic Dim, Participation, Persistence, Cross-Task | I2, I4, C2, C5, E5 |
| Information (c01-c09) | MI, Conditional MI, Transfer Entropy, PID, Info Bottleneck, O-Info, Granger, OCSE, NOTEARS | I1, I3, I4, C4 |
| Measurement (f01-f08) | Bootstrap, Seed Variance, Convergent, Discriminant, Internal Consistency, Invariance, Sensitivity, Inter-Rater | M1-M4, C5 |

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
- Do not edit instrument implementations without understanding the
  corresponding docs page — they must stay in sync.
- The docs site uses Astro with Starlight. Build with `npm run dev` inside `docs/`.
- Criteria IDs (C1-C5, I1-I5, E1-E6, M1-M6, V1-V5) are stable identifiers
  referenced across the codebase and in external publications. Do not renumber.
- The 7 description modes and 5 validity types are the core ontology.
  Changes to these require updating the taxonomy, criteria, verdicts, how-to
  guides, and worked examples simultaneously.
- Instrument IDs (a01, b01, etc.) follow evidence-family prefixes.
  New instruments get the next number in their family.
- Run instrument scripts with `uv run python` from the repo root.
- When adding instruments, follow the docstring header format
  (Instrument, Categories, Validity layer, Criteria, Establishes, Requires, Doc).
