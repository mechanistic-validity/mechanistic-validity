---
title: "Case Study: Knowledge Neurons / ROME"
description: "Factual knowledge localization and model editing (Meng et al. 2022) evaluated through all five validity lenses."
---

# Case Study: Knowledge Neurons / ROME

[Meng et al. (2022)](https://arxiv.org/abs/2202.05262) claim that **factual knowledge is localized in specific MLP layers** of large language models — that "The Eiffel Tower is in Paris" is stored in identifiable MLP weight matrices, and that editing these weights (ROME / MEMIT) can change the model's factual beliefs. The claim has two parts: (1) a localization claim (knowledge is in specific MLPs) and (2) a practical application (you can edit it there).

This is among the most commercially impactful MI claims — it led to model editing tools. It is also among the most contested — subsequent work questions whether the edits are stable, generalizable, or actually targeting the right mechanism.

## Composite Verdict

| Lens | Strongest | Weakest | Overall |
|---|---|---|---|
| Construct | C1 Falsifiability | C3 Task specificity | Partial |
| Internal | I1/I2 Necessity + Sufficiency | I3 Specificity | Causally suggestive |
| External | E4 Effect magnitude | E3 Selectivity | Partial |
| Measurement | M1 Reliability | M2/M4 Invariance + Sensitivity | Weak |
| Interpretive | V1 Level declaration | V4 Alternative exclusion | Weak |

**Overall verdict: Causally suggestive, with significant interpretive challenges.** ROME/knowledge neurons have strong necessity (I1) and narrow sufficiency (I2) — causal tracing works and edits succeed on target. But the specificity failure (I3) and the alternative-exclusion failure (V4) together suggest that the interpretive framing ("knowledge is localized in MLPs") may be wrong even though the practical tool (ROME edits) works. This is an instructive case: a tool can work for the wrong reasons. The framework helps distinguish "the edit works" (external validity for the intervention) from "the mechanistic story is correct" (interpretive validity for the localization claim).

## Metrics used in original work

| Method | Our metric | Family |
|---|---|---|
| Causal tracing (activation patching with noise) | [A02 Counterfactual DAS](/framework/metrics/causal/a02-counterfactual-das) | Causal |
| Rank-one model editing (ROME) | [A05 MDC/Glennan](/framework/metrics/causal/a05-mdc-glennan) | Causal |
| MEMIT (multi-layer editing) | [A05 MDC/Glennan](/framework/metrics/causal/a05-mdc-glennan) | Causal |

> To run these metrics yourself, see [Experiment 10: Published Circuit Evaluation](https://github.com/mechanistic-validity/mechanistic-validity-experiments/tree/main/experiments/10_published_circuit_evaluation).

---

## Philosophy of Science Lens — Construct Validity

*Is "knowledge neuron" a coherent construct?*

### Criteria

**[C1 — Falsifiability:](/framework/criteria/construct/falsifiability) Pass.** The claim predicts: (1) causal tracing should show that early-site MLP layers are the critical path for factual recall, (2) rank-one edits to those layers should change the model's factual outputs. Both are testable and concrete.

**[C2 — Structural plausibility:](/framework/criteria/construct/structural-plausibility) Partial.** The "knowledge is in MLP weights" claim is plausible — MLP layers have the capacity to store key-value associations. But whether a single fact corresponds to a localized rank-one update (vs. being distributed across many parameters) is a strong structural assumption that is not independently verified.

**[C3 — Task specificity:](/framework/criteria/construct/task-specificity) Weak.** The critical question: does editing "Eiffel Tower → Rome" affect *only* Eiffel Tower queries, or does it corrupt related knowledge (French landmarks, Paris facts, tower-related queries)? Subsequent work (Hoelscher-Obermaier et al. 2023, Hase et al. 2024) finds that edits often have unintended side effects — the intervention is not as specific as claimed.

**[C4 — Minimality:](/framework/criteria/construct/minimality) Unclear.** Is one MLP layer the minimal locus, or could the fact be edited at multiple locations? MEMIT (the multi-layer extension) suggests the latter — facts may be distributed, and ROME's single-layer assumption may be over-localizing.

**[C5 — Convergent validity:](/framework/criteria/construct/convergent-validity) Partial.** Causal tracing (activation patching) identifies the critical layers. ROME edits at those layers and works. But these two steps are not independent — ROME is designed to edit where causal tracing points. An independent method (probing, weight-space analysis) finding the same localization would be stronger.

### Key Distinctions

- **Observable vs theoretical:** "Knowledge neuron" bridges from an observable (causal tracing identifies MLP layers whose corruption disrupts recall) to a theoretical entity (a neuron that *stores* a fact). The gap is substantial — causal importance during processing does not entail storage. The theoretical label outruns the observable evidence.
- **Underdetermination:** Two theories equally explain the causal tracing results: (1) facts are stored in MLP layers (the authors' interpretation), and (2) MLP layers are processing bottlenecks for entity representations. Both predict that corrupting these layers disrupts recall; both predict that editing there changes outputs. The data underdetermines which interpretation is correct, and the ripple effects arguably favor interpretation (2).
- **Naming requires criteria:** "Knowledge neuron" is a theoretically loaded name applied to an operationally thin finding (causal importance for factual recall). The name asserts storage, but the evidence only shows processing relevance. A more operationally grounded name like "fact-critical MLP unit" would better match the evidence without presupposing a storage mechanism.

### Nomological Network

The knowledge neuron / ROME framework connects to:
- **Causal tracing** — corrupting identified MLP layers degrades factual recall (causal, confirmed)
- **Edit success** — rank-one updates at those layers change target outputs (causal, confirmed)
- **Ripple effects** — edits corrupt related knowledge (observed, documented by follow-up work)
- **Multi-layer distribution** — MEMIT's success suggests facts are not purely localized (structural, partially confirmed)
- **Independent localization** — probing or weight-space analysis finding the same layers without causal tracing guidance (untested)
- **Cross-fact consistency** — does the "storage location" follow a predictable pattern across facts? (partially tested, results variable)
- **Alternative explanation** — MLP layers as entity-processing bottlenecks rather than fact storage (untested as a formal alternative)

Four nodes confirmed/observed, but two of them (ripple effects, multi-layer distribution) actually *undermine* the localization interpretation. The network is unusual: confirmed nodes partly contradict the theoretical framework they were meant to support.

---

## Neuroscience Lens — Internal Validity

*Does the evidence establish localized storage, not just processing involvement?*

### Criteria

**[I1 — Necessity:](/framework/criteria/internal/necessity) Pass.** Causal tracing shows that corrupting the identified MLP layers degrades factual recall for the target fact. The effect is specific to the fact being tested.

**[I2 — Sufficiency:](/framework/criteria/internal/sufficiency) Pass (narrow).** ROME edits at the identified layer successfully change the model's output for the target query. This is a form of sufficiency — intervening at the identified locus is sufficient to change the behavior. But it is narrow sufficiency: the edit works for the specific query template tested, not necessarily for all ways of asking about the same fact.

**[I3 — Specificity:](/framework/criteria/internal/specificity) Weak — the critical gap.** This is where the claim breaks down. Editing "Eiffel Tower is in Paris" to "Eiffel Tower is in Rome" may also change answers to "What country is the Eiffel Tower in?" (should still be France) or corrupt knowledge about Rome. The edit is not specific to the target fact — it bleeds into related knowledge. This is the "ripple effect" problem documented by subsequent work.

**[I4 — Consistency:](/framework/criteria/internal/consistency) Partial.** Works across many facts (ROME is tested on thousands of subject-relation-object triples). But the *quality* of edits varies — some generalize, some don't, and the conditions for success are not fully characterized.

**[I5 — Confound control:](/framework/criteria/internal/confound-control) Weak.** Causal tracing uses a specific corruption method (noise injection). Whether the identified locus is specific to factual recall or is a general bottleneck for any query involving the subject entity is not controlled. A component could be "where the subject is processed" rather than "where the fact is stored."

### Key Distinctions

- **Localization vs distributed:** The entire debate hinges on this distinction. ROME assumes localization (facts stored at specific sites), but the ripple effects and MEMIT's multi-layer approach suggest distribution. The evidence is more consistent with a distributed picture where causal tracing identifies bottlenecks in a distributed process rather than discrete storage locations.
- **Lesion vs stimulation:** Both directions are tested: causal tracing is a lesion study (corrupt and observe degradation), while ROME is a stimulation study (edit and observe changed output). However, the stimulation results are problematically broad — the edit "stimulates" not just the target fact but related knowledge, suggesting the intervention is less precise than lesion studies imply.
- **Single vs double dissociation:** Only single dissociation — corrupting the MLP layer impairs factual recall. Whether corrupting a different layer leaves factual recall intact (while impairing something else) is not systematically tested. The identified layer could be a general processing bottleneck.

### Dissociation Matrix

|  | Target fact recall | Related fact recall | Entity recognition | Unrelated tasks |
|---|---|---|---|---|
| Corrupt target MLP layer | **↓↓ (confirmed)** | **↓ (ripple, documented)** | ? | ? |
| ROME edit at target layer | **Changed (confirmed)** | **Corrupted (documented)** | ? | ? |
| Corrupt different MLP layer | ? | ? | ? | ? |
| Corrupt attention layers | ? | ? | ? | ? |

Four cells filled — but two of them (related fact recall column) document *failures* of specificity rather than successes. The matrix reveals that the intervention is too broad: it changes what it targets AND what it shouldn't. The empty rows (different layers, attention) represent the untested controls needed to establish specificity.

---

## Pharmacology Lens — External Validity

*Does the editing intervention produce clean, predictable effects?*

### Criteria

**[E1 — Intervention reach:](/framework/criteria/external/intervention-reach) Partial.** ROME successfully changes model outputs — the intervention reaches downstream behavior. But the reach is often too broad (changes things it shouldn't).

**[E2 — Graded response:](/framework/criteria/external/graded-response) Not tested.** Can you partially edit a fact (make the model less confident rather than fully switching)? Parametric dose-response is not standard in the ROME framework.

**[E3 — Selectivity:](/framework/criteria/external/selectivity) Weak.** The key failure. Edits produce off-target effects on related knowledge. The intervention is not selective.

**[E4 — Effect magnitude:](/framework/criteria/external/effect-magnitude) Strong on target.** On the specific query template used, the edit success rate is high (>90% for ROME on the tested benchmark).

**[E5 — Robustness:](/framework/criteria/external/robustness) Partial.** Edits work on the target template but may not generalize to paraphrases or related queries. "Robustness" of the edit (does it hold across phrasings?) is partially demonstrated.

**[E6 — Cross-architecture:](/framework/criteria/external/cross-architecture) Partial.** ROME/MEMIT have been applied to multiple model families (GPT-J, GPT-NeoX, LLaMA). The causal tracing localization varies somewhat across architectures.

### Key Distinctions

- **The system compensates:** This is the defining challenge for knowledge neuron claims. Editing one fact destabilizes related facts because the system's representations are entangled — the model compensates (or fails to) through distributed representations that share parameters. The ripple effects are direct evidence that the system is not modular in the way the localization claim assumes.
- **Affinity vs efficacy:** ROME demonstrates efficacy on the target query (the edit succeeds) but lacks selectivity (off-target effects). In pharmacological terms, it has high efficacy but poor therapeutic index — the "drug" works but has unacceptable side effects.
- **The metric is part of the finding:** Edit success is measured by whether the target answer changes on the test template. The metric does not capture whether the model's broader factual network remains coherent. A narrow metric flatters a broad intervention.

### Dose-Response Curve

The ROME dose-response is characterized at only one point:
- **α = 1** (full rank-one edit): target answer changes with >90% success rate; off-target effects documented but not quantified as a function of dose

What's missing:
- **No partial edits** — can you apply a fraction of the rank-one update and get graded confidence change?
- **No off-target dose-response** — at what edit magnitude do ripple effects begin? Is there a threshold below which the target changes but related knowledge is preserved?
- **No therapeutic window** — the gap between "minimum effective dose" (target changes) and "toxic dose" (related knowledge corrupts) is completely uncharacterized

This is the pharmacological core of the ROME critique: without a dose-response curve, we cannot determine whether clean editing is possible at any dose, or whether the "drug" is inherently non-selective.

---

## Measurement Theory Lens — Measurement Validity

*Is causal tracing a reliable metric for localizing knowledge?*

### Criteria

**[M1 — Reliability:](/framework/criteria/measurement/reliability) Partial.** Causal tracing gives consistent results for a given fact. But the localization can differ between related facts, suggesting the measurement is reliable but the underlying phenomenon is complex.

**[M2 — Invariance:](/framework/criteria/measurement/invariance) Weak.** The identified "knowledge location" varies by fact, by query phrasing, and by model. The measurement is not invariant across conditions — different prompts for the same fact may point to different layers.

**[M3 — Baseline separation:](/framework/criteria/measurement/baseline-separation) Partial.** Causal tracing shows clear peaks at specific layers. But whether the baseline (what random layers contribute) is well-characterized is unclear.

**[M4 — Sensitivity:](/framework/criteria/measurement/sensitivity) Unknown.** Can causal tracing distinguish "where the fact is stored" from "where the subject entity is processed"? This is the core sensitivity question and it is not resolved.

**[M5 — Calibration:](/framework/criteria/measurement/calibration) Not reported.** What constitutes a "successful" edit? Success is measured by whether the target answer changes, but whether the model's broader knowledge remains intact is not part of the standard calibration.

**[M6 — Construct coverage:](/framework/criteria/measurement/construct-coverage) Partial.** Measures whether the target answer changes. Does not measure: consistency of related knowledge, model confidence, or downstream reasoning quality.

### Key Distinctions

- **Sensitivity vs specificity (of the metric):** Causal tracing has reasonable sensitivity (it reliably identifies layers that matter for factual recall) but poor specificity (it cannot distinguish "where the fact is stored" from "where the subject entity is processed"). This sensitivity/specificity imbalance is the core measurement problem — the metric detects something real but cannot determine what it is detecting.
- **Convergent vs discriminant validity:** Convergent validity is weak — causal tracing and ROME editing are not truly independent methods (ROME edits where tracing points). Discriminant validity is untested — would causal tracing incorrectly "localize" non-factual knowledge (e.g., syntactic patterns) to the same MLP layers? If so, the method lacks discriminant power for the specific claim of factual storage.

### MTMM Matrix

| | Causal tracing (facts) | ROME edit success (facts) | Causal tracing (syntax) | Probing (facts) |
|---|---|---|---|---|
| **Causal tracing (facts)** | — | High (by design) | ? | ? |
| **ROME edit success (facts)** | High (by design) | — | ? | ? |
| **Causal tracing (syntax)** | ? | ? | — | ? |
| **Probing (facts)** | ? | ? | ? | — |

The one filled convergent cell (causal tracing vs. ROME success) is high by design — ROME edits where tracing points, so agreement is circular. The discriminant cells are unfilled: we do not know whether causal tracing identifies the same layers for non-factual tasks (which would undermine the "fact storage" interpretation). The MTMM reveals that the apparent convergence is methodologically forced rather than independently discovered.

---

## MI Lens — Interpretive Validity

*Is "knowledge is localized in MLPs" the right interpretation?*

### Criteria

**[V1 — Level declaration:](/framework/criteria/interpretive/level-declaration) Pass.** The claim is [implementational](/framework/modes/implementational) — it names where facts are stored and how they can be modified.

**[V2 — Level-evidence match:](/framework/criteria/interpretive/level-evidence-match) Partial.** The evidence (causal tracing) is at the causal/behavioral level. The claim (facts are *stored* in MLPs) is a structural/implementational assertion. There is a gap — causal importance does not establish storage.

**[V3 — Narrative coherence:](/framework/criteria/interpretive/narrative-coherence) Moderate.** "MLPs store key-value associations; subjects are keys, facts are values; editing the value changes the fact." This is a coherent story but may be overly simplified. The narrative works for the edit success cases but not for the ripple-effect failures.

**[V4 — Alternative exclusion:](/framework/criteria/interpretive/alternative-exclusion) Weak.** The key alternative: MLP layers are where *subject entity representations* are processed, not where *facts are stored*. Under this alternative, ROME edits work because they corrupt the entity representation at a processing bottleneck, not because they target fact storage. This alternative explains both the successes and the failures (ripple effects = corrupted entity representation affects all facts about that entity). It has not been excluded.

**[V5 — Scope honesty:](/framework/criteria/interpretive/scope-honesty) Partial.** "Knowledge is localized" is a strong claim. "Causal tracing identifies MLP layers whose corruption disrupts factual recall on template queries" is what is demonstrated. The scope of the evidence is narrower than the scope of the claim.

### Key Distinctions

- **Description vs explanation:** "Knowledge neurons store facts" is an explanation. "MLP layers are causally important for factual recall" is a description. The evidence supports the description; the explanation is one of multiple compatible accounts.
- **Faithfulness vs understanding:** ROME is faithful in a narrow sense (edits change the target output). But the understanding it implies (localized storage) is contested. High narrow faithfulness with questionable broader understanding.
- **Component identity vs component role:** The components (specific MLP layers) are reliably identified by causal tracing. Their role (storage vs. processing bottleneck) is the interpretive dispute. As with the docstring circuit, identity is established but role is contested.

### Evidence Convergence Map

- **Implementational → Interpretation:** Partial. Causal tracing identifies specific layers; ROME edits at those layers succeed. But the evidence is implementational in location only — it does not confirm the storage mechanism.
- **Algorithmic → Interpretation:** Weak. No algorithm for factual recall is specified. The claim jumps from "this layer matters" to "this layer stores facts" without specifying the retrieval algorithm.
- **Computational → Interpretation:** Partial. We know the model performs factual recall. Whether "key-value lookup in MLP" is the right computational description (vs. "distributed association through residual stream") is unresolved.

### Intervention-Interpretation Matrix

| | Necessity | Sufficiency | Representational | Algorithmic | Computational |
|---|---|---|---|---|---|
| Causal tracing (lesion) | ✓ | — | — | ∅ | ∅ |
| ROME edit (stimulation) | — | ✓ (narrow) | — | ∅ | ∅ |
| Probing | — | — | — | — | — |
| Cross-fact comparison | — | — | — | — | — |
| Multi-layer analysis | — | — | — | Partial | — |

Necessity and sufficiency are confirmed but only in the causal column. The representational and algorithmic columns — where the "storage" claim lives — are empty. The matrix reveals the level mismatch: the evidence is causal (something breaks/changes), but the claim is structural (facts are stored here).

### Causal Sufficiency Graph

- Subject token → MLP layer activation: **solid** (causal tracing confirms)
- MLP layer activation → factual output: **solid** (editing changes output)
- MLP weights → fact storage: **dashed** (inferred, not directly demonstrated)
- Edit at MLP → target change: **solid** (ROME succeeds)
- Edit at MLP → related knowledge corruption: **solid** (ripple effects documented)
- Alternative: MLP as entity bottleneck → all entity-related facts: **dashed** (proposed, not causally tested)

The graph has an unusual structure: the solid edges support the *intervention* working but also document its *failures* (ripple effects). The critical "storage" edge is dashed — the central interpretive claim has no solid causal support. The alternative explanation (entity bottleneck) is equally compatible with all solid edges.

---
