---
title: "Case Study: Docstring Circuit"
description: "The docstring variable-binding circuit (Heimersheim & Janiak 2023) evaluated through the five core lenses."
---

# Case Study: Docstring Circuit

[Heimersheim & Janiak (2023)](https://arxiv.org/abs/2307.13057) identify eight attention heads in a **4-layer attention-only transformer** that predict the next argument name in a Python docstring. Fuzzy previous-token heads and a positional head set up an induction step, and argument movers carry the name from the definition line to the output position. The claimed mechanism tracks which variable names are bound to which argument positions and retrieves the correct name at the appropriate docstring location.

This is interesting as a case study because it operates in a specific domain (code) and raises questions about whether "variable binding" is the right construct or whether the circuit is doing something simpler (positional copying).

## Composite Verdict

> **Verdict (framework paper, Table 6):** Causally Suggestive. **Capped by:** I4 (specificity).


| Lens | Strongest | Weakest | Overall |
|---|---|---|---|
| Construct | C1 Falsifiability | C4/C3 Discriminant + Convergence | Partial |
| Internal | I1 Necessity | I4/I7 | Causally suggestive |
| External | E5 Graded response | E1/E3/E4 Reach, Cross-task, Cross-model | Weak |
| Measurement | M2 Baseline separation | M5 Sensitivity | Weak–Partial |
| Interpretive | V1 Level declaration | V3 Alternative level | Weak–Partial |

**Overall verdict: Causally suggestive.** The docstring circuit has solid necessity evidence but weak interpretive validity — "variable binding" may be overclaiming what is actually "positional copying." This case study illustrates a common pattern: the circuit is real (ablation confirms it matters), but the *label* may not be right. The framework distinguishes between "the circuit exists" (internal validity) and "the circuit does what you named it" (interpretive validity). Here, internal validity is ahead of interpretive validity.

## Metrics used in original work

| Method | Our metric | Family |
|---|---|---|
| Activation patching | [A02 Counterfactual DAS](/mechanistic-validity/framework/metrics/#a02) | Causal |
| Ablation | [A01 Pearl SCM](/mechanistic-validity/framework/metrics/#a01) | Causal |
| Resample ablation under three corruption types | [D01 Faithfulness](/mechanistic-validity/framework/metrics/#d01) | Behavioral |

> To run these metrics yourself, see [Experiment 10: Published Circuit Evaluation](https://github.com/mechanistic-validity/mechanistic-validity-experiments/tree/main/experiments/10_published_circuit_evaluation).

---

## Philosophy of Science Lens — Construct Validity

*Is "variable binding" a coherent construct for this circuit?*

### Criteria

**[C1 — Falsifiability:](/mechanistic-validity/framework/criteria/construct/falsifiability) Pass.** The claim predicts that the circuit should bind variables by position, not by surface features. If you swap parameter names (rename `x` to `y` and `y` to `x`), the circuit should track the *positions*, not the *names*. This generates testable counterfactual predictions.

**[C2 — Structural plausibility:](/mechanistic-validity/framework/criteria/construct/structural-plausibility) Partial.** The circuit involves attention heads that attend from docstring positions to function definition positions. The attention pattern is consistent with positional binding. However, detailed $W_{OV}$ / $W_{QK}$ analysis showing *how* the binding is encoded in weights is limited compared to e.g. the Greater-Than analysis.

**[C4 — Discriminant validity:](/mechanistic-validity/framework/criteria/construct/discriminant-validity) Not tested.** Is this circuit specific to docstring variable binding, or does it also fire on other name-tracking tasks (IOI-like patterns in code, class attribute resolution)? Cross-task evaluation is not reported.

**[I3 — Minimality:](/mechanistic-validity/framework/criteria/internal/minimality) Partial.** The circuit is identified by resample ablation under three corruption types. Whether this is the minimal sufficient set or an over-inclusive one is not systematically tested via leave-one-out.

**[C3 — Convergent validity:](/mechanistic-validity/framework/criteria/construct/convergent-validity) Partial.** The circuit is identified through activation patching (causal method). Weight-space confirmation of the binding mechanism is limited. A fully independent discovery method (e.g., EAP, probing) has not been applied.

### Key Distinctions

- **Operationalism vs realism:** "Variable binding" is a realist label for what the evidence operationally shows (positional copying from function definition to docstring). The operational evidence supports "position-based name retrieval" — whether this constitutes genuine binding (in the programming language theory sense) is a stronger claim that requires additional tests.
- **Confirmation vs corroboration:** The circuit was discovered by activation patching and evaluated by behavioral metrics on the same task family — confirmation within one method. Independent corroboration (weight-space analysis, or testing whether the same circuit handles binding in non-docstring contexts) is absent.
- **Underdetermination:** On the tested templates, "variable binding" and "positional copying" make identical predictions. The behavioral data underdetermines which mechanism the circuit implements — adversarial prompts where the two strategies diverge are needed to break the tie.

### Nomological Network

The docstring circuit connects to:
- **Attention pattern** — heads attend from docstring positions to function definition (observable, confirmed)
- **Behavioral prediction** — ablation degrades binding accuracy on templates (causal, confirmed)
- **Name-swap invariance** — circuit tracks positions not surface names (predicted, partially confirmed)
- **Positional copying distinction** — does the circuit track argument-parameter associations or just ordinal position? (untested)
- **Cross-domain binding** — does the same circuit handle variable references in code bodies, class attributes, or natural-language coreference? (untested)
- **Weight-space mechanism** — $W_{QK}$/$W_{OV}$ structure that implements binding/copying (untested at detail level)

Three nodes confirmed, three unconnected. A thin network — the confirmed nodes establish that the circuit is real, but the unconnected nodes represent exactly the tests needed to distinguish "variable binding" from "positional copying."

---

## Neuroscience Lens — Internal Validity

*Does the evidence establish implementation, not just participation?*

### Criteria

**[I1 — Necessity:](/mechanistic-validity/framework/criteria/internal/necessity) **Partial.** Necessity is tested component by component and the tests bite. Every head in the circuit is knocked out by resampling and the drop in logit difference is read off, under three corruptions that each remove a different piece of information — which is what lets a head be shown necessary for the B_doc value specifically rather than for the task in general. One ablation regime carries all of it: zero and mean ablation are named alongside the method chosen and never run.

**[I2 — Sufficiency:](/mechanistic-validity/framework/criteria/internal/sufficiency) Partial.** Activation patching restores behavior on corrupted inputs. But full circuit isolation (ablate everything outside) is not the primary methodology.

**[I4 — Specificity:](/mechanistic-validity/framework/criteria/internal/specificity) Not tested.** Does ablating the docstring circuit affect other code completion tasks? Other name-tracking tasks? Collateral damage is not measured.

**[I7 — Confound control:](/mechanistic-validity/framework/criteria/internal/confound-control) Partial.** The line-counting and repeat-inhibition algorithms are designed out of the prompt set rather than argued away, at a measured cost to model performance (~75% to 56%). Confounds beyond these two are not addressed.

### Key Distinctions

- **Single vs double dissociation:** Only single dissociation is demonstrated — ablating the circuit impairs docstring binding. Whether ablating a different circuit (e.g., one for general code completion) leaves docstring binding intact is untested. Without the double dissociation, this circuit could be a general code-understanding module rather than a binding-specific mechanism.
- **Lesion vs stimulation:** Only lesion-style evidence (ablation/patching). No stimulation experiment (steering the circuit to produce a specific variable name at a docstring position) is reported. Stimulation would test whether the circuit is genuinely steerable — a stronger indicator of dedicated function.

### Dissociation Matrix

|  | Docstring binding | General code completion | IOI-like name tracking | Class attribute resolution |
|---|---|---|---|---|
| Ablate docstring circuit | **↓ (confirmed)** | ? | ? | ? |
| Ablate general code circuit | ? | ? | ? | ? |
| Ablate IOI circuit | ? | ? | ? | ? |

One cell filled. The diagonal entry confirms necessity, but without off-diagonal measurements we cannot distinguish "binding-specific mechanism" from "general name-tracking infrastructure that happens to include docstring binding."

---

## Pharmacology Lens — External Validity

*Does intervening on the circuit produce expected downstream effects?*

### Criteria

**[E1 — Intervention reach:](/mechanistic-validity/framework/criteria/external/intervention-reach) Partial.** Four do-operators beyond resample ablation are run and they agree with it. No steering experiment is reported.

**[E5 — Graded response:](/mechanistic-validity/framework/criteria/external/graded-response) Partial.** Circuit size tracks performance monotonically across two axes. The graded axis is circuit extent, not intervention strength.

**[E2 — Prompt generalization:](/mechanistic-validity/framework/criteria/external/prompt-generalization) Disconfirmed.** Circuit and model diverge on a subclass of benign inputs. This is tested and failed, not merely unextended.

**[E4 — Cross-model generalization:](/mechanistic-validity/framework/criteria/external/cross-model-recurrence) Not tested.** One released 4-layer toy model; transfer is named as future work with no model, head set or number reported.

### Key Distinctions

- **Affinity vs efficacy:** The circuit demonstrates affinity (it activates on docstring binding prompts) and partial efficacy (ablation degrades the target behavior). But without steering experiments, we cannot confirm full efficacy — the ability to causally drive specific binding outputs by amplifying the circuit.
- **The metric is part of the finding:** The circuit is evaluated on accuracy for the same template format used to discover it. Whether its "efficacy" extends to naturalistic code (where binding is embedded in complex context) remains unknown.

### Dose-Response Curve

The dose-response curve is almost entirely uncharacterized:
- **α = 0** (no intervention): full binding accuracy on templates
- **α = 1** (complete ablation): binding accuracy drops substantially

What's missing:
- **No intermediate ablation strengths** — no sweep between 0 and 1
- **No off-target measurement at any dose** — general code completion not tracked
- **No stimulation curve** — can you enhance binding accuracy by amplifying the circuit?
- **No naturalistic prompts** — all measurements on templates only

We have two endpoints and nothing between. The curve shape (linear degradation? threshold effect? compensatory plateau?) is completely unknown.

---

## Measurement Theory Lens — Measurement Validity

*Are the metrics measuring variable binding reliably?*

### Criteria

**[M1 — Reliability:](/mechanistic-validity/framework/criteria/measurement/reliability) Not reported.** No confidence intervals.

**[M6 — Invariance:](/mechanistic-validity/framework/criteria/measurement/invariance) **Not tested.** The same conclusion is never re-derived under a changed condition and shown to hold. The three corruptions probe different pieces of information, so they are complementary rather than redundant, and no result is reported twice under two settings for comparison. Both patching directions were in fact run, but only the selected ones are reported, which converts a possible invariance check into a presentation choice.

**[M2 — Baseline separation:](/mechanistic-validity/framework/criteria/measurement/baseline-separation) Partial.** The circuit's 42% is set against a 17% chance level with two further reference points. A matched random head set — the control that would show the 42% belongs to *these* heads — is not run.

**[M5 — Sensitivity:](/mechanistic-validity/framework/criteria/measurement/sensitivity) Unknown.** Can the method distinguish "variable binding" from "positional copying"? These produce similar behavioral outputs but imply different mechanisms.

**[M4 — Calibration:](/mechanistic-validity/framework/criteria/measurement/calibration) **Partial.** The metric is well chosen for what is claimed. Logit difference against the highest wrong answer, recomputed each time, cannot be gamed by a model that raises all argument names together, and the success rate is read directly off the task. The two scales are then compared without adjustment: the 42% circuit is set beside a 56% full model that the isolating prompts have already depressed from about 75%, which the authors say themselves.

### Key Distinctions

- **Sensitivity vs specificity (of the measurement):** The measurement metric (activation patching + accuracy) is sensitive to *some* mechanism being important but cannot discriminate between the binding and copying hypotheses. The metric detects that something matters without resolving what that something does — a sensitivity/specificity mismatch.
- **Convergent vs discriminant validity:** No convergent comparison (e.g., activation patching vs. probing vs. weight analysis on the same circuit). No discriminant comparison (e.g., does the method identify different circuits for different code tasks?). The MTMM is empty.

### MTMM Matrix

| | Act. patching (docstring) | Probing (docstring) | Act. patching (IOI) | Act. patching (code completion) |
|---|---|---|---|---|
| **Act. patching (docstring)** | — | ? | ? | ? |
| **Probing (docstring)** | ? | — | ? | ? |
| **Act. patching (IOI)** | ? | ? | — | ? |
| **Act. patching (code completion)** | ? | ? | ? | — |

Entirely unfilled. No convergent cells (different methods on same task) and no discriminant cells (same method on different tasks). The measurement-theoretic foundation for "this circuit does variable binding" is absent — we have one method, one task, and one evaluation.

---

## MI Lens — Interpretive Validity

*Is "variable binding" the right interpretation?*

### Criteria

**[V1 — Level declaration:](/mechanistic-validity/framework/criteria/interpretive/level-declaration) Pass.** Algorithmic — claims the circuit performs variable binding.

**[V2 — Level-evidence match:](/mechanistic-validity/framework/criteria/interpretive/level-evidence-match) Partial.** The evidence is primarily causal (ablation/patching). Structural evidence for *how* binding is implemented in weights is thin. An algorithmic claim ideally needs structural support.

**[V3 — Alternative level:](/mechanistic-validity/framework/criteria/interpretive/alternative-level) Weak.** The positional-copying alternative (simpler mechanism producing the same behavior) is not excluded. On the tested prompts, variable binding and positional copying make the same predictions. Distinguishing them requires adversarial prompts where the two strategies diverge.

**[V5 — Scope declaration:](/mechanistic-validity/framework/criteria/interpretive/scope-declaration) **Confirmed.** The limit is declared before the argument rather than conceded after it. The epistemic header states that the circuit recovers up to half of model performance and that leads remain unfollowed, and it sits above the first result. The exclusions are then named individually: heads known to be essential that the patching method cannot surface, a head whose ablation improves performance, and a list of components the authors cannot account for.

### Key Distinctions

- **Description vs explanation:** "Variable binding" is an explanatory label (it specifies a computational process). But the evidence only supports a descriptive claim (the circuit contributes to correct parameter name prediction in docstrings). The explanation exceeds what the data strictly establishes.
- **Faithfulness vs understanding:** The circuit is faithful to the behavioral data (ablation confirms it matters). But understanding (why it produces this behavior — binding vs. copying) is not established. High faithfulness with uncertain understanding.
- **Component identity vs component role:** The components (specific attention heads) are identified with high confidence. Their *role* (binder vs. copier) is the interpretive question. Identity is established; role is contested.

### Evidence Convergence Map

- **Implementational → Interpretation:** Moderate. Ablation and patching identify specific heads. Attention patterns point to function-definition-to-docstring information flow. But weight-space confirmation of the binding mechanism is absent.
- **Algorithmic → Interpretation:** Weak. "Variable binding" is claimed at the algorithmic level, but the evidence does not distinguish it from positional copying — a simpler algorithm that produces the same outputs.
- **Computational → Interpretation:** Weak. We know the circuit contributes to docstring completion. Whether "variable binding" is the right computational description — or whether "positional name copying" better captures what's computed — is the core unresolved question.

### Intervention-Interpretation Matrix

| | Necessity | Sufficiency | Representational | Algorithmic | Computational |
|---|---|---|---|---|---|
| Ablation | ✓ | — | ∅ | ∅ | ∅ |
| Act. patching | ✓ | Partial | — | ∅ | ∅ |
| Weight analysis | — | — | — | — | — |
| Steering | — | — | — | — | — |
| Adversarial prompts | — | — | — | — | — |

Cells cluster in the necessity column. The algorithmic and computational columns — precisely where the "binding vs. copying" distinction lives — are entirely empty. The matrix makes visible that the interpretive claim (algorithmic-level "variable binding") has no supporting evidence in the algorithmic column.

### Causal Sufficiency Graph

- Attention heads → docstring position: **solid** (attention patterns confirmed)
- Function definition → attention heads: **solid** (patching confirms information source)
- Binding computation → correct name: **dashed** (inferred from behavioral success, mechanism not directly observed)
- Positional-vs-binding resolution: **absent** (no causal test distinguishes the two)

The graph has solid edges for information flow (where attention looks, where information comes from) but only a dashed edge for the *computation* the circuit performs. The interpretive question ("what does it compute?") corresponds to the weakest edge in the causal graph.

---
