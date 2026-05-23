---
title: "Protocols"
description: "Curated metric bundles that target specific validity questions with domain-specific interpretation."
---

# Protocols

A protocol bundles a curated set of metrics and calibrations around a specific validity question, then interprets the results through the theoretical lens that motivated the question. Running a protocol is not required to evaluate a claim — metrics and calibrations alone are sufficient — but protocols provide structured depth where criteria are weak.

Each protocol exposes the same interface: a `run_protocol()` function that takes a model and task list, runs its metrics and calibrations, and returns a `ProtocolResult` containing scored measurements with metadata. The measurements feed into the standard criteria-scoring pipeline.

## Protocol families

Protocols are organized by evidence family. The naming convention uses letter prefixes (A = causal, B = structural, etc.) matching the metric families.

### Causal protocols (A01–A13)

These target internal validity criteria (I1–I5) by running causal metrics through specific theoretical frameworks.

| Protocol | Question | Metrics used | Criteria strengthened |
|---|---|---|---|
| A01 Pearl SCM | Is the circuit a valid structural causal model? | logit_diff, role_ablation, activation_patching, causal_scrubbing | I1 Necessity, I2 Sufficiency |
| A02 Counterfactual DAS | Does counterfactual intervention on the circuit's subspace change behavior as predicted? | das_iia, misalignment, cross_task_transfer | I3 Specificity |
| A03 Rubin CATE | What is the average treatment effect of the circuit, with proper potential-outcomes framing? | activation_patching, effect_size, sigma_ablation | I1 Necessity, E4 Effect magnitude |
| A04 Woodward | Does the circuit satisfy invariant difference-making under intervention? | activation_patching, causal_scrubbing, path_patching | I1 Necessity, I4 Consistency |
| A05 MDC/Glennan | Does the circuit qualify as a mechanism under the mechanistic decomposition criteria? | role_ablation, path_patching, cross_task_transfer | I2 Sufficiency, V3 Narrative coherence |
| A06 Mediation | What fraction of the total effect flows through the circuit (direct vs indirect)? | mediation, path_patching, effect_size | I3 Specificity |
| A07 Granger/TE | Does past circuit activity predict future behavior beyond what other components predict? | granger_causality, transfer_entropy, pid | I1 Necessity, C5 Convergent validity |
| A08 PID | How is information about the task variable decomposed across circuit components? | pid, mutual_information, conditional_mi | I3 Specificity, C5 Convergent validity |
| A09 MDL/SLT | Does the circuit provide a minimal-description-length explanation of the behavior? | mdl, llc, effect_size | C4 Minimality |
| A10 Regularity/INUS | Is the circuit an INUS condition — insufficient but necessary part of an unnecessary but sufficient set? | activation_patching, sigma_ablation, effect_size | I1 Necessity, C4 Minimality |
| A11 Actual Cause | Does the circuit meet Halpern-Pearl's definition of actual causation (not just but-for)? | activation_patching, causal_scrubbing, path_patching | I1 Necessity, I5 Confound control |
| A12 Transportability | Does the causal effect transport across domains (prompts, models, tasks)? | cross_task_transfer, cross_model_invariance, generalization_gap | E5 Robustness, E6 Cross-architecture |
| A13 Causal Discovery | Can the circuit's causal graph be recovered from observational + interventional data? | notears, granger_causality, pid | I4 Consistency, C5 Convergent validity |

### Structural protocols (B01–B04)

These target construct and measurement criteria by analyzing weight-space properties without forward passes.

| Protocol | Question | Criteria strengthened |
|---|---|---|
| B01 Spectral/SVD | Does the circuit's weight structure reveal interpretable spectral modes? | C2 Structural plausibility, M4 Sensitivity |
| B02 Composition | Do weight-space composition scores match the claimed information routing? | C2 Structural plausibility, I4 Consistency |
| B03 Graph analysis | Does the circuit's connectivity graph have non-trivial topological properties? | C4 Minimality, C5 Convergent validity |
| B04 Network motifs | Does the circuit contain recognizable computational motifs (copying, inhibition, routing)? | C2 Structural plausibility, V3 Narrative coherence |

### Behavioral protocols (D01–D03)

| Protocol | Question | Criteria strengthened |
|---|---|---|
| D01 Faithfulness | Does the circuit alone reproduce the behavior? | I2 Sufficiency, E4 Effect magnitude |
| D02 Generalization | Does the circuit transfer to held-out prompts and related tasks? | E5 Robustness, C3 Task specificity |
| D03 Probing | Can the circuit's intermediate representations be decoded by a learned classifier? | C2 Structural plausibility, E5 Robustness |

### Information protocols, representational protocols, and additional families

Beyond the core families above, protocols exist for information-theoretic analysis (C01–C03), representational analysis (E01), and extended families drawn from specific scientific traditions:

- **Molecular biology** (16 protocols) — knockout hierarchies, rescue experiments, Mendelian randomization, dose-response, target engagement, sensitivity analysis, and other designs adapted from experimental biology.
- **Cross-discipline** (11 protocols) — control theory (settling depth, stability margin, observability), dynamical systems (Koopman/DMD, renormalization, TDA), economics (arbitrage search, game theory), and geometry (Fisher-Rao, sheaf consistency).
- **Synthesis** (9 protocols) — see [synthesis protocols](/framework/evidence/synthesis-protocols).

## Choosing protocols

Protocols are not a checklist to run exhaustively. They are targeted tools for strengthening specific criteria.

The workflow: run metrics and calibrations. Score criteria. Identify which criteria are weak. Find the protocol tagged to that criterion and run it. Re-score. The protocol inventory is organized by which criteria each protocol strengthens, so the mapping from "weak criterion" to "run this protocol" is direct.

For a typical evaluation, running 3–5 protocols from different families produces substantially richer evidence than running 15 metrics from the same family. Cross-family convergence is the goal — not exhaustive coverage within one family.

## Running protocols

Protocols live in the [mechanistic-validity-experiments](https://github.com/mechanistic-validity/mechanistic-validity-experiments) repository under `experiments/protocols/`. Each can be run standalone or imported:

```bash
uv run python protocols/neuroscience/a01_scm_pearl.py --tasks ioi induction --device cuda
```

```python
from protocols.neuroscience.a01_scm_pearl import run_protocol
result = run_protocol(model, tasks=["ioi"], n_prompts=40)
```

The main `mechval` library provides registry infrastructure (`register_protocol`, `dispatch_protocol`, `list_protocols`) for programmatic protocol management.
