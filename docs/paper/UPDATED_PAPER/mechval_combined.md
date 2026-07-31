# Mechanistic Validity: A Framework and Automated Pipeline for Evaluating Interpretability Claims

**Elliot Tower**
*Independent Researcher · elliot@elliottower.com*

*Combined draft — May 2026. Merges the framework paper (`mechval_framework.tex`) and the automated-CVS audit paper (`automated_cvs_evaluation_v5.md`) into one TMLR-style submission. Criterion names retained from the framework paper (C1–V5 — Falsifiability through Scope declaration); the audit-paper renames (`Task specificity`, `Minimality`, etc.) are NOT applied per author decision. All experiments, scores, and tables are unchanged from the source documents — only the structure is new.*

---

## Abstract

Mechanistic interpretability (MI) faces a measurement crisis: 99.7% of sparse autoencoder features are unstable across training runs, attribution patching correlates at *r* = 0.006 with ground truth on synthetic tasks, two of six standard SAE evaluation metrics fail basic diagnostic tests, and 93% of published MI papers fail execution-grounded reproducibility checks. These findings undermine any claim built on current methodology — including safety-critical claims about deception detection and model monitoring. We present **MechVal**, a measurement-theoretic framework that organizes 27 operational criteria across five validity types (construct, measurement, internal, external, interpretive) in dependency order, assigns claims to one of five verdict tiers (Proposed through Validated), and supports evaluation via 84 metrics spanning five evidence families. The framework is method-agnostic: it applies uniformly to circuits, SAE features, probing classifiers, steering vectors, transcoders, and crosscoders. We also present an automated evaluation pipeline that operationalizes the framework end-to-end, taking any MI paper as input and producing a Claim Validity Score (CVS) and verdict tier in 2–3 minutes. The pipeline runs three independent LLM extractions with multi-run minimum voting, then applies a leak-audit pass that detects evidence mis-attribution between circuit-level and component-level claims. Across 60+ configurations spanning 8 frontier models, the system agrees with multi-model consensus labels on all 9 case-study papers to within one tier, matching exactly on 5. All disagreements are in the same direction — the system occasionally scores one tier too high, never too low. Applied to 13 published circuits, the framework discriminates meaningfully: claims range from Proposed (Knowledge Neurons, Probing Classifiers, Gender Bias Circuits) through Causally Suggestive (Grokking, Successor Heads, SAE Features) and Mechanistically Supported (IOI Circuit, Greater-Than, Copy Suppression) to Triangulated (Induction Heads), with verdict differences traceable to specific missing evidence.

---

## 1. Introduction

When should you trust an interpretability claim? A researcher reports that GPT-2 uses a six-role circuit to solve indirect object identification (Wang et al., 2023), that sparse autoencoder features correspond to human-interpretable concepts (Cunningham et al., 2023), or that a linear probe reveals syntactic structure in transformer activations (Belinkov, 2022). Each claim comes with experimental evidence — ablation studies, activation visualizations, accuracy numbers. But how well-supported is the claim, *really*?

Recent work provides uncomfortable answers. Bricken et al. (2024) and Engels et al. (2025) demonstrate that only 197 of 65,536 sparse autoencoder features — 0.3% — are stable across independent training runs. The remaining 99.7% are training artifacts. Any “deception feature” identified using standard SAE methods has a 99.7% prior probability of being an artifact, not a genuine representation.

The field’s primary causal tool fares no better. Kramár et al. (2024) show that attribution patching — the most widely used method for identifying causally relevant components — correlates at *r* = 0.006 with ground truth on controlled synthetic tasks. This is not a weak correlation; it is no correlation. The method the field treats as the gold standard for causal attribution provides essentially random rankings of component importance.

MI evaluation metrics fare similarly. Two of six standard SAE evaluation metrics fail basic diagnostic tests (Karvonen et al., 2025); 93% of published MI papers fail execution-grounded reproducibility checks (Bai et al., 2026). The field has discovery without validation, narratives without measurement, and a growing body of safety-relevant claims with no scalable way to assess whether any of them are correct.

**This paper contributes both halves of a solution.** First, we propose **MechVal**, a measurement-theoretic framework that synthesizes validation methodology from philosophy of science, neuroscience, psychometrics, pharmacology, and causal inference (§3–§4). The framework defines 27 operational criteria, 5 verdict tiers, 5 evidence families, and a `MechanisticClaimSpec` data model that enforces pre-registration of mechanistic hypotheses. We apply it manually to 13 published circuits, demonstrating that the tier system discriminates meaningfully (§5). Second, because manual application takes hours per paper, we present an **automated CVS pipeline** that operationalizes the framework as a multi-stage LLM evaluation system (§6). The system completes a full paper evaluation in 2–3 minutes, scores all 27 criteria, and agrees with human-adjudicated reference labels on 9 papers to within one tier, matching exactly on 5. We characterize a directional bias (over-scoring by exactly +1 tier in 4/9 cases, never under-scoring) and present two debiasing mechanisms — multi-run minimum voting and a leak-audit pass — that reduce mean absolute per-criterion error from 0.163 to 0.111 on the IOI paper (§7).

The framework and the automated pipeline are designed to reinforce each other: the framework provides the structure that makes the automated assessment meaningful, and the automated pipeline provides the scalability that makes the framework deployable across the literature.

**Contributions.**

1. **MechVal framework** — 27 criteria across 5 validity types in dependency order; 5 verdict tiers with strict hierarchical requirements; 5 evidence families; 4 calibration gates; `MechanisticClaimSpec` pre-registration mechanism (§3–§4).
2. **Manual case studies** — 13 published circuits evaluated across all validity lenses, demonstrating tier discrimination from Proposed to Triangulated (§5).
3. **Automated CVS pipeline** — end-to-end paper-to-tier evaluation in 2–3 minutes, with two systematic debiasing mechanisms and a full model comparison across 8 frontier LLMs (§6).
4. **Released reference data** — multi-model consensus labels for 9 papers; full per-criterion annotations for the IOI circuit covering 7 claims × 27 criteria (§7).
5. **Open-source implementation** — 54 evaluation tasks, 14 calibration gates, 12 pre-registered claim specifications, and the full audit pipeline.

---

## 2. Related Work

**Validity frameworks for MI.** Existing structured evaluations of MI focus narrowly. MIB (Mueller et al., 2025) defines a benchmark for circuit localization and causal-variable localization but does not attempt to evaluate the full evidential basis of a claim. MechEvalAgent (Bai et al., 2026) audits MI papers for execution-grounded reproducibility — whether their code runs and produces the claimed numbers — finding 93% failure. Our framework is complementary: rather than asking whether a paper’s code reproduces, we ask whether the paper’s evidence supports its mechanistic conclusion.

**LLM-as-evaluator.** Using LLMs to judge scientific outputs is increasingly common in benchmarks like AlpacaEval and MT-Bench. Known failure modes include positional bias, verbosity bias, and over-charitable interpretation (Zheng et al., 2024). The automated pipeline we present (§6) inherits these limitations and explicitly characterizes them; our directional-asymmetry finding (system over-scores by exactly +1 tier, never under-scores) is consistent with the over-charitable-interpretation literature.

**Cross-field validity theory.** The framework draws on five established traditions. From philosophy of science: Popperian falsifiability and Hempel’s operationalism (Mayo, 2018). From neuroscience: lesion studies, optogenetics, and double dissociation. From psychometrics: construct, convergent, and discriminant validity (Cronbach & Meehl, 1955). From pharmacology: dose-response, assay calibration, and Phase III generalization. From causal inference: necessity vs. sufficiency, transportability, and confound control (Pearl, 2009; Pearl & Bareinboim, 2011). The framework’s contribution is not any one of these — it is their integration into a single dependency-ordered protocol for MI claims.

---

## 3. The MechVal Framework

### 3.1 Five Validity Types in Dependency Order

The framework organizes evaluation into five validity types, applied in dependency order: construct → measurement → internal → external → interpretive. Each type depends on the validity of the types before it. A measurement of an ill-defined construct is meaningless; a causal inference from an unreliable measurement is meaningless; a generalization claim from an untested causal inference is meaningless. The order is not arbitrary — it reflects the conditional structure of valid scientific inference.

- **Construct validity:** *Is the thing being measured well-defined?* Before any measurement is taken, the construct must be specified precisely enough that the claim is falsifiable, structurally plausible, and distinguishable from neighboring constructs. A “deception feature” that cannot be distinguished from an “uncertainty feature” fails construct validity regardless of how carefully it is measured.
- **Measurement validity:** *Are the instruments trustworthy?* Once the construct is defined, the measurement tools used to detect it must be reliable (stable across repetitions), calibrated (separable from random baselines), and invariant (consistent across conditions). A metric that gives different answers when run with different random seeds is not evidence.
- **Internal validity:** *Does the evidence support the causal claim?* Given a well-defined construct and trustworthy instruments, the evidence must establish that the identified component is causally involved in the behavior — not merely correlated with it. This requires demonstrating necessity, sufficiency, and specificity.
- **External validity:** *Does the mechanism generalize?* Causal evidence on one prompt set, with one ablation method, in one model, establishes internal validity at best. External validity requires generalization across prompts, ablation methods, related tasks, and ideally across models.
- **Interpretive validity:** *Is the interpretation correct?* Finally, the natural-language description must be justified by the evidence. A head called a “name mover” based on direct logit attribution carries theoretical implications beyond what the evidence strictly supports.

### 3.2 Twenty-Seven Criteria

Each validity type is operationalized as 5–6 concrete criteria, each with a defined test procedure. Table 1 lists all 27.

**Table 1.** The 27 operational criteria organized by validity type. Each criterion has a concrete test procedure.

| Type | ID | Criterion |
|---|---|---|
| Construct | C1 | Falsifiability — can the claim be refuted? |
| | C2 | Structural plausibility — is the mechanism physically possible in the architecture? |
| | C3 | Convergent validity — do multiple independent methods agree? |
| | C4 | Discriminant validity — does the measure distinguish this from neighboring constructs? |
| | C5 | Nomological validity — does the claim fit into a broader theory? |
| Measurement | M1 | Reliability — do repeated measurements give the same answer? |
| | M2 | Baseline separation — is the score distinguishable from random/untrained baselines? |
| | M3 | Stability — is the classification robust to perturbation? |
| | M4 | Calibration — are the numbers meaningful? |
| | M5 | Sensitivity — can the instrument detect known-true effects? |
| | M6 | Invariance — does the metric behave consistently across conditions? |
| Internal | I1 | Necessity — is the circuit required for the behavior? |
| | I2 | Sufficiency — is the circuit enough to produce the behavior? |
| | I3 | Specificity — does the circuit do this task and not everything? |
| | I4 | Double dissociation — can this circuit be separated from other circuits? |
| | I5 | Confound control — are alternative explanations ruled out? |
| External | E1 | Intervention reach — do different intervention methods agree? |
| | E2 | Prompt generalization — does it work on diverse prompts? |
| | E3 | Cross-task generalization — does the mechanism transfer to related tasks? |
| | E4 | Cross-model generalization — does the mechanism appear in other models? |
| | E5 | Graded response — does partial ablation produce partial effects? |
| | E6 | Novel prediction — does the mechanism predict new, untested behaviors? |
| Interpretive | V1 | Level declaration — at what description mode is the claim made? |
| | V2 | Level-evidence match — does the evidence support claims at that level? |
| | V3 | Alternative level — could the evidence be explained at a different level? |
| | V4 | Anthropomorphism check — is the interpretation projecting human concepts? |
| | V5 | Scope declaration — what does the claim explicitly not cover? |

*Note on naming: this paper uses the criterion names from the MechVal framework (Table 1). The automated audit pipeline (§6) scores the same 27 criteria using the same IDs, so all reported scores carry over identically regardless of the human-readable name used.*

### 3.3 Evidence Families

Metrics are organized into five evidence families, each providing a different *kind* of evidence about a mechanistic claim. Each metric belongs to exactly one family.

- **Causal (A):** Effects of interventions — ablation, patching, clamping, resampling. Examples: activation patching, DAS-IIA, CATE, mediation analysis.
- **Structural (B):** Weight-space properties, requiring no forward pass. Examples: SVD, effective rank, OV/QK composition norms, template distance.
- **Information-theoretic (C):** Information flow and dependencies. Examples: mutual information, partial information decomposition, transfer entropy.
- **Behavioral (D):** Input-output behavior under manipulation. Examples: faithfulness, logit difference, KL divergence, dose-response curves.
- **Representational (E):** Geometry and content of internal representations. Examples: CKA, RSA, linear probes, attention entropy.

A sixth meta-family **Measurement (F)** tracks meta-properties of other metrics — bootstrap stability, seed variance, convergent and discriminant validity. The distinction between families matters because *convergent evidence* — the same conclusion supported by metrics from different families — is qualitatively stronger than redundant evidence from the same family. An ablation study (causal) that agrees with a weight-space analysis (structural) provides more support than two ablation studies using different ablation methods. The framework tracks evidence family coverage explicitly.

### 3.4 Verdict Tiers

Claims are assigned to one of five verdict tiers representing qualitative transitions in evidential status. Two additional labels handle special cases (Insufficient, Refuted). The tiers form a strict hierarchy: each requires all evidence from the tier below plus additional evidence.

**Table 2.** Verdict tiers with requirements. Each tier subsumes the requirements of all lower tiers.

| Tier | Meaning | Requirements beyond previous tier |
|---|---|---|
| Proposed | Structural/representational evidence only | Construct validity criteria met (C1–C5) |
| Causally Suggestive | Necessity shown, sufficiency not established | I1 (necessity via ablation) |
| Mechanistically Supported | Necessity + sufficiency with consistent methods | I1 + I2 + intervention reach (E1) + ≥2 ablation variants |
| Triangulated | Multiple converging lines of independent evidence | Full internal + external + construct criteria from ≥3 evidence families |
| Validated | Triangulated + replicated + predictive | Independent replication + novel predictions (E6) confirmed |

### 3.5 Description Modes and Implementation Scale

The framework distinguishes six **description modes** following a Marr-inspired hierarchy: computational, algorithmic, representational, implementational-functional, implementational-connectomic, and implementational-topographic. A claim’s description mode determines which evidence is appropriate. A computational-level claim (“the model performs IO identification”) is supported by behavioral evidence; an implementational-connectomic claim (“head 9.9 copies the IO token”) requires component-level causal evidence.

The open-source implementation provides **54 evaluation tasks** across 10 linguistic domains, **84 metrics** spanning the 5 evidence families, **14 calibration gates** (bootstrap stability, seed variance, baseline separation, etc.), and **12 pre-registered `MechanisticClaimSpec`s** covering known circuits (IOI, Greater-Than, Induction Heads, Successor Heads, Docstring, Copy Suppression, etc.).

### 3.6 MechanisticClaimSpec: Pre-Registration

The framework’s primary original contribution is a Pydantic v2 data model that encodes a mechanistic hypothesis *before* verification. A `MechanisticClaimSpec` specifies:

- **Computational steps** — nodes in the mechanism DAG, each mapped to specific model components (heads, MLPs, neurons, SAE features).
- **Edges** — directed connections specifying the communication mechanism (residual stream, attention composition, etc.).
- **Positive predictions** — testable claims about what *should* happen under intervention (e.g., “ablating duplicate-token-detection heads reduces logit difference by ≥ 0.2”).
- **Negative controls** — claims about what should *not* happen (e.g., “ablating name-mover heads does not affect upstream S-inhibition heads”).
- **Rival specifications** — IDs of competing mechanistic hypotheses.
- **Gate metadata** — identifiability (G2) and superposition risk (G3) assessments.

When `verify(spec)` runs, it executes each prediction by routing to the appropriate metric, compares measured values against expected directions and thresholds, and aggregates results into a confirmation rate, negative control rate, claim ceiling, and verdict tier. The pre-registration requirement prevents post-hoc rationalization — the practice of discovering a circuit and then constructing a narrative that fits the observations without ever testing predictions the narrative would make.

---

## 4. Theoretical Foundations

The framework synthesizes validation methodology from five established fields. Table 3 maps framework components to their cross-field analogs.

**Table 3.** Cross-field mapping of framework components to established methodology.

| Component | MI/MechVal | Neuroscience | Psychometrics | Pharmacology | Phil. of science |
|---|---|---|---|---|---|
| Track 1 | Circuit localization | Lesion mapping | Exploratory FA | Receptor mapping | Discovery |
| Track 2 | Causal-variable localization | Region identification | Latent-variable modeling | Target ID | Operational definition |
| Track 3 | Model testing / claim spec | Circuit dissection | Confirmatory FA | Proof of mechanism | Hypothetico-deductive testing |
| Gates | G0–G3 preconditions | Stimulation thresholds | Reliability coefficients | Assay validation | Methodological preconditions |
| Evidence families | A–F | Modality (anatomy, physiology, behavior) | Method triangulation | RCT vs. observational | Convergent operationalism |

The framework is not the union of these traditions — it is their *intersection*, applied to mechanistic claims in neural networks. The criteria that survive cross-field translation are those that capture genuine invariants of valid scientific inference.

---

## 5. Manual Case Studies: Thirteen Published Circuits

We apply the framework manually to 13 published MI results, evaluating each through all five validity lenses. The case studies are *not* intended to rank papers — they demonstrate what the framework looks like in practice and show that the tier system discriminates between well-validated and poorly-validated claims.

### 5.1 Verdict Assignments

**Table 4.** Verdict assignments for 13 published circuits. Verdicts are based on published evidence as of evaluation date; no novel experiments were run for this analysis.

| Verdict | Circuit | Key limiting criterion |
|---|---|---|
| Triangulated | Induction Heads | Simple mechanism, broad replication, thick nomological network |
| Mech. Supported | IOI Circuit | Method-conditional faithfulness (87% mean ablation, <50% other methods); specificity untested |
| Mech. Supported | Greater-Than | Strong structural plausibility (C2); limited prompt generalization |
| Mech. Supported | Copy Suppression | Unusually clean specificity; sufficiency partially established |
| Causally Suggestive | Grokking | Validated within toy scope; toy-to-real gap is the limitation |
| Causally Suggestive | Successor Heads | Cross-domain generalization as convergent evidence; ablation method-conditional |
| Causally Suggestive | Docstring Circuit | Label ambiguity: “variable binding” vs. simpler “positional copying” |
| Causally Suggestive | SAE Features | Thin nomological network; features may be dictionary properties, not model properties |
| Causally Suggestive | Othello World Model | “World model” label exceeds evidence (“linearly decodable”); interpretive inflation |
| Proposed | Knowledge Neurons | Strong intervention but weak mechanistic story |
| Proposed | Superposition | Validated theory in toy models; awaits real-model confirmation |
| Proposed | Probing Classifiers | Decodability without intervention = no internal validity |
| Proposed | Gender Bias Circuits | Construct incoherence: bias and knowledge share circuits |

### 5.2 Illustrative Examples

**Induction Heads (Triangulated).** Olsson et al. (2022) identify a two-head composition mechanism for in-context copying: previous-token heads attend to the token before a repeated sequence, and induction heads use this signal to predict the next token. The claim reaches Triangulated because it satisfies criteria across multiple independent evidence families. The mechanism has been replicated across model scales and architectures (E4), confirmed through both activation-level and weight-level analysis (C3 convergent validity from different families), and produces clean dose-response effects (E5). The nomological network is thick: the mechanism connects to in-context learning theory, training dynamics, and cross-model structural predictions. The relative simplicity of the mechanism (two functional roles, clear compositional structure) makes each criterion easier to satisfy.

**IOI Circuit (Mechanistically Supported).** Wang et al. (2023) identify 26 attention heads forming a six-role circuit for indirect object identification. The circuit demonstrates strong necessity (I1) and sufficiency (I2): running the model with everything outside the circuit mean-ablated recovers 87% of the full model’s logit difference. However, Miller et al. (2024) show that this 87% figure is specific to mean ablation; under other ablation methods, faithfulness drops below 50%. This method-conditional result partially satisfies E1 (intervention reach) but reveals that the causal claim is weaker than the headline number suggests. Specificity (I3) is untested: the authors do not report whether ablating the IOI circuit degrades unrelated tasks. A formal double-dissociation has not been conducted. The circuit reaches Mechanistically Supported rather than Triangulated because the evidence comes primarily from one methodological family (ablation-based) and key criteria (I3, I4, I5, E4) remain unaddressed.

**Probing Classifiers (Proposed).** Linear probing (Belinkov, 2022) trains a classifier on activations to test whether a concept is linearly decodable. The core limitation is fundamental: *a measurement without an intervention is a measurement without internal validity.* Probe success establishes that information is decodable from the activation space, but not that the model uses that information during inference. High probe accuracy is consistent with (a) genuine encoding, (b) incidental linear separability in high-dimensional space, and (c) confound encoding (a correlated feature rather than the target). Without causal follow-up (DAS, intervention along the probe direction), the claim cannot advance beyond Proposed. Probing *with* causal follow-up can reach Causally Suggestive; with additional control tasks and cross-method convergence, it can reach Mechanistically Supported. The framework does not dismiss probing — it precisely characterizes what probing alone does and does not establish.

### 5.3 Patterns Across Case Studies

- **The sufficiency gap.** Most circuits demonstrate necessity (I1) but not sufficiency (I2). The I1 → I2 transition is the most common barrier between Causally Suggestive and Mechanistically Supported.
- **Method-conditional results.** The IOI circuit’s headline numbers are specific to mean ablation. Ablation type is part of the causal claim, not an implementation detail. The framework captures this through E1 (intervention reach).
- **The toy-model ceiling.** Grokking and Superposition achieve high validity within toy scope — but the gap between toy proof-of-concept and real-model confirmation remains the field’s central challenge. The framework handles this through V5 (scope declaration): a claim Validated within a declared scope is a legitimate result, not a failure.
- **Interpretive inflation.** Labels like “world model,” “knowledge neuron,” and “deception feature” carry theoretical implications beyond what the evidence supports. The framework identifies this through V4 (anthropomorphism check) and V5 (scope declaration).
- **Construct incoherence.** Gender Bias Circuits fail not because evidence is lacking but because “gender bias” and “gendered knowledge” share circuitry — the construct itself is not discriminant.

---

## 6. Automated CVS Pipeline

Manual application of the framework takes hours per paper. To make the framework deployable across the literature we built an automated pipeline that takes any MI paper as input and produces a Claim Validity Score (CVS) and verdict tier in 2–3 minutes.

### 6.1 Pipeline Architecture

Given a paper from any supported source (arXiv, OpenReview, PDF URL, blog post, or local file), the system performs four stages:

1. **Text extraction** — PDF parsing or HTML scraping, with citation and table preservation.
2. **Multi-run claim extraction** — three independent LLM calls. Each call extracts mechanism claims, assigns description modes and evidence families, and scores all 27 criteria as YES (1.0), PARTIAL (0.5), or NO (0.0).
3. **Multi-run minimum voting** — for each (claim × criterion) cell, the minimum across the three runs is retained. A criterion that scored YES on 2 runs and PARTIAL on 1 is reduced to PARTIAL.
4. **Leak audit** — a separate LLM pass that receives the scored claims *without the paper text* and checks each evidence string for misattribution. If a sub-component claim has inherited circuit-level evidence (e.g., a 3-head sub-claim citing 87% faithfulness from the 26-head circuit), the audit downgrades the affected criterion.

The pipeline then computes dimension scores from the criteria and a final CVS on a normalized 0–10 scale.

### 6.2 Criteria Assessment

Each claim is scored on all 27 criteria using the framework’s definitions (Table 1). The extraction prompt includes 10 explicit scoring pitfalls drawn from common errors observed during development. Examples:

- Faithfulness metrics do not establish sufficiency (I2 requires isolation or path-level tests).
- Probing is correlational (no internal validity from probes alone).
- Ablation establishes necessity only, not sufficiency.
- Activation patching and path patching count as a single method for convergent validity purposes.

When scoring a sub-component claim, only experiments that directly test the components in that claim count — evidence from experiments on other components does not transfer. This rule is what the leak audit enforces post-hoc.

### 6.3 Dimension Aggregation

Raw criteria are aggregated into dimension scores (0–3) using validity-theoretic rules that encode domain knowledge about what constitutes strong versus weak evidence. These rules use logical conjunctions rather than averages:

- **Construct (C):** Score 3 requires C1 + C3 (from ≥3 families) + C2. Score 2 requires C1 + C3 (from ≥2 families). Score 1 requires C1 alone. Score 0 = not falsifiable.
- **Internal (I):** Score 3 requires I1 + I2 + I3 + I5. Score 2 requires I1 + I2. Score 1 requires I1 or I2 alone. Score 0 = neither necessity nor sufficiency.
- **Measurement (M):** Score 3 requires M2 (baseline) + M1 (reliability) + M4 (calibration) + M5 (sensitivity). Score 2 requires M2 + M1. Score 1 = baseline only.
- **External (E):** Score 3 requires E1 + E2 + (E3 or E4). Score 2 requires E1 + E2.
- **Interpretive (V):** Score 3 requires V1 + V2 + V5. Score 2 requires V1 + V2.

### 6.4 CVS Computation

The Claim Validity Score is a weighted sum of dimension scores. Construct and Internal carry weight 1.5 (they are prerequisites for downstream validity); Measurement, External, and Interpretive carry weight 1.0. With dimension scores in {0,1,2,3}, the maximum raw score is 1.5·3 + 1.5·3 + 1.0·3 + 1.0·3 + 1.0·3 = 18. The CVS is then normalized to 0–10 and classified into one of five tiers matching the manual framework (Proposed, Causally Suggestive, Mechanistically Supported, Triangulated, Validated).

### 6.5 Worked Example: IOI Name Mover Heads

We trace one IOI claim through the full pipeline. The claim: *“Name Mover heads (9.9, 9.6, 10.0) copy the indirect object token from earlier positions to the final token position.”*

- **Initial extraction.** C1 = YES (the claim names specific heads and a testable operation). I1 = YES (ablation of name movers degrades IOI performance). I2 = YES (evidence: “87% faithfulness”). E2 (graded response, in framework naming) = YES, citing the gradual degradation curve.
- **Multi-run minimum.** Across 3 runs, I2 = YES/YES/PARTIAL → reduced to PARTIAL with evidence string `[MIN-VOTE: YES→PARTIAL across 3 runs]`.
- **Leak audit.** The audit receives the scored claim *without paper text*. It notes that “87% faithfulness” describes the full 26-head circuit, not the 3-head name-mover subset. No experiment in the evidence string tests whether name movers alone are sufficient. I2 is downgraded from PARTIAL to NO. E2’s “gradual degradation” similarly comes from circuit-level ablation, not head-specific dose-response; E2 is downgraded from YES to NO.
- **Final.** Construct = 2, Internal = 1 (necessity only), Measurement = 1, External = 0, Interpretive = 2. Weighted sum = (1.5·2) + (1.5·1) + (1.0·1) + (1.0·0) + (1.0·2) = 7.5 raw → CVS = 4.2 → Mechanistically Supported. After leak-driven downgrades the final CVS is 3.9 — appropriately lower than the main circuit claim’s 5.6.

---

## 7. Experimental Results

### 7.1 Reference Labels

Reference tier labels for 9 case-study papers were established through multi-model consensus combined with human adjudication. For the IOI circuit, we additionally constructed per-criterion consensus annotations across all 27 criteria × 7 claims (189 cells). The 9 papers span the full verdict range from Proposed to Validated.

### 7.2 Model Comparison

We evaluated the pipeline across 8 frontier LLMs (proprietary and open-source) and 60+ configurations spanning extraction architecture, reasoning mode, and prompt iteration. Table 5 (excerpt) shows performance on the IOI paper (reference: Mechanistically Supported, CVS ≈ 5.6).

**Table 5.** Model comparison on IOI (main claim CVS). Reference tier: Mechanistically Supported (5.6). Sorted by accuracy.

| Model | Thinking | Claims | Main CVS | Predicted Tier | Off-by |
|---|---|---|---|---|---|
| Claude Opus 4.7 | Off | 8 | 5.6 | Mech. Supported | 0 |
| DeepSeek V4 Pro | High | 9 | 5.6 | Mech. Supported | 0 |
| DeepSeek V4 Pro | Max | 7 | 5.6 | Mech. Supported | 0 |
| GPT-5.5 | Off | 9 | 6.4 | Triangulated | +1 |
| GPT-5.5 | On | 10 | 6.4 | Triangulated | +1 |
| Claude Opus 4.7 | High | 9 | 6.9 | Triangulated | +1 |
| Gemma 4 27B | — | 5 | 7.5 | Triangulated | +1 |
| Qwen 3.7 Max | — | 6 | 6.4 | Triangulated | +1 |

*Counterintuitive finding:* enabling thinking/reasoning modes consistently *worsened* calibration. Claude Opus 4.7 dropped from exact-match at thinking-off to +1 over at thinking-high. We hypothesize that chain-of-thought generation in the extraction step amplifies the model’s charitable-interpretation prior — the model talks itself into a higher score.

### 7.3 Headline Result: Within-One-Tier Agreement on All 9 Papers

**Table 6.** Final pipeline results on 9 case-study papers (Claude Opus 4.7, thinking off, multi-run minimum + leak audit).

| Paper | Predicted Tier | Expected Tier | Off-by |
|---|---|---|---|
| Probing | Proposed | Proposed | 0 |
| Gender Bias | Proposed | Proposed | 0 |
| Successor Heads | Causally Suggestive | Causally Suggestive | 0 |
| Greater-Than | Mech. Supported | Causally Suggestive | +1 |
| Othello | Mech. Supported | Causally Suggestive | +1 |
| IOI | Mech. Supported | Mech. Supported | 0 |
| Copy Suppression | Mech. Supported | Mech. Supported | 0 |
| Induction | Triangulated | Mech. Supported | +1 |
| Grokking | Validated | Validated | 0 |

**5/9 exact, 9/9 within one tier, 0/9 under-scored.** This is the headline result.

### 7.4 Directional Asymmetry

The error distribution is markedly asymmetric: 4 over-scored, 0 under-scored. This is consistent across all 60+ configurations, all 8 models, and both the with-debiasing and without-debiasing conditions. The pattern is not stochastic noise — it is a structural property of LLM-as-judge applied to MI papers.

We identify two error sources:

1. **Borderline criteria bias.** I3 (Specificity), E2 (Prompt generalization, in framework naming), and I5 (Confound control) are the most-affected criteria. The model awards YES on standard methodology where PARTIAL or NO would be more accurate — e.g., crediting a paper for I3 specificity because it focused on a single task, even when off-task effects were never measured.
2. **Boundary concentration.** Over-scoring is concentrated at the boundary between Causally Suggestive and Mechanistically Supported (Greater-Than, Othello). Papers at the extremes (Probing → Proposed, Grokking → Validated) match exactly. The bias is strongest for papers with moderate evidence that could plausibly be interpreted either way.

**Reframing.** The directional asymmetry is exploitable, not just an error. Because the system never under-scores, its output functions as a *certified upper bound* on evidential quality. A paper that the system rates Mechanistically Supported might actually be Causally Suggestive; a paper rated Proposed cannot be hiding a higher true tier. For literature triage — “which papers are worth manually validating” — this is exactly the property you want.

### 7.5 Debiasing Effectiveness

**Multi-run minimum voting** improves single-run baseline (2/9 exact matches, max off-by +2 on Greater-Than) to the final 5/9 exact matches with no paper off by more than 1 tier. Figure 4 (in the source paper) shows variance collapse: papers with high single-run variance (Copy Suppression Δ = 2.2, IOI Δ = 1.3) are stabilized; papers already stable are unaffected.

**Leak audit.** On the IOI paper, the audit reduced mean absolute per-criterion error from 0.163 to 0.111 against the 189-cell reference annotations. Per-criterion exact agreement improved from ≈74.8% to 81.5%. The audit typically applies 2–4 downgrades per paper, primarily targeting I2 (sufficiency) and E2 (prompt generalization) on sub-component claims that had inherited circuit-level faithfulness metrics.

**Devil’s advocate (failed).** We tested an aggressive challenge pass that contests every YES/PARTIAL on the 5 most-overscored criteria. Aggressive setting over-corrected (IOI 6.9 → 4.2, one tier *below* consensus). Conservative setting had no effect. This Goldilocks problem makes devil’s advocate unreliable as a standalone debiasing mechanism. Multi-run minimum + leak audit is our recommended configuration.

### 7.6 Theoretical-Paper Limitation

The Superposition paper (Elhage et al., 2022) was excluded from the main 9-paper evaluation. Its purely analytical methodology — proofs about toy-model behavior, no real-model experiments — maps poorly onto criteria designed for empirical circuit-discovery papers. The pipeline scores it at Proposed (correctly, given it lacks real-model evidence) but the criteria that drive that score (M1 reliability, I1 necessity) are not the meaningful ones for a theory paper. The framework needs a separate description-mode-aware scoring track for theoretical contributions; this is left to future work.

---

## 8. Discussion

### 8.1 Why Pre-Registration Matters

The `MechanisticClaimSpec` is the framework’s most consequential design choice. It changes the unit of MI research from “interesting finding” to “tested prediction.” A claim that exists only as a paper-text description can always be re-narrated to fit new evidence. A claim that exists as a Pydantic spec with positive predictions and negative controls cannot — `verify(spec)` returns numerical confirmation rates whether the author wants it to or not.

This is the framework’s answer to the post-hoc rationalization problem. The IOI paper, the Greater-Than paper, and the Docstring paper would all benefit from being re-expressed as claim specs — not because their authors did anything wrong, but because the pre-registration discipline retroactively distinguishes which conclusions the evidence actually establishes from which conclusions were narrative additions.

### 8.2 Implications for AI Safety

The measurement crisis has direct deployment implications. Organizations are beginning to build internal monitors based on MI findings — probes that flag potential deception, steering vectors that suppress harmful outputs, circuit-level classifiers that trigger human review. If those monitors are built on unvalidated features, they provide false confidence: the system *appears* monitored when it is not.

The framework provides concrete deployment guidance via the verdict tiers:

- **Proposed** — do not deploy as a monitor.
- **Causally Suggestive** — preliminary evidence only; insufficient sufficiency testing.
- **Mechanistically Supported** — minimum evidential basis for cautious deployment with ongoing validation.
- **Triangulated or Validated** — sufficient evidential basis for production deployment.

The automated pipeline (§6) extends this to the literature scale: an organization can apply the system to every published MI claim relevant to their safety case and produce a calibrated lower bound on which claims meet their deployment threshold.

### 8.3 Limitations

**Manual case studies (§5) are based on published evidence only.** No novel model interventions were run for the 13-circuit analysis. The verdict assignments are arguments from published evidence, not experimental results. This is a real limitation: a reviewer could legitimately object that the criteria were tuned on these 13 examples, or that circuit selection was outcome-motivated. We mitigate this by (a) including circuits where the verdict was surprising to us (Probing → Proposed despite widespread acceptance; SAE Features → Causally Suggestive despite recent enthusiasm) and (b) running the framework’s verification pipeline on 12 of the 13 circuits is committed as future work pending GPU compute.

**Automated pipeline scores are conservative upper bounds, not ground truth.** The +1 directional bias means the system over-states evidential quality on borderline cases. Users should treat the output as “this paper is at most tier X” rather than “this paper is at tier X.”

**Coverage gaps.** Of the 54 implemented evaluation tasks, only 12 have full claim specifications. The remaining 42 have prompt generators but lack verified circuits. The framework’s utility grows with circuit coverage, which is bottlenecked by the labor-intensive process of circuit discovery and specification.

**Reference label IAA is reported informally.** The 9-paper consensus labels were established through multi-model voting plus human adjudication. A formal inter-annotator agreement statistic (Krippendorff’s α or per-criterion percent agreement) is computable from the IOI per-criterion data we release; this is in progress.

**LLM-as-evaluator inherits LLM-as-evaluator limitations.** Positional bias, verbosity bias, and over-charitable interpretation are present and characterized but not eliminated. We do not claim the automated pipeline replaces human review — only that it produces a calibrated triage signal.

### 8.4 Future Work

- **Run `verify(spec)` end-to-end** on all 12 claim specs with actual model interventions, replacing the manual case-study verdicts with experimentally-derived ones.
- **External-validity anchor.** Cross-check framework verdicts against independent MIB benchmark scores (CMD, IIA). If higher-tier circuits also score higher on MIB, that is external validation that the framework tracks something real.
- **Per-criterion correlation analysis.** Verify that the 27 criteria are not all moving together. If they correlate at > 0.9, the framework is over-specified.
- **Theoretical-paper track.** Add a description-mode-aware scoring path for analytical contributions (Superposition-style papers).
- **Expand reference labels.** Move from 9 papers to 50+ with independent multi-annotator labels for calibration.
- **Cross-model ensembling.** Run extraction with multiple LLMs and retain only criteria with consensus — a conservative lower bound complementing the upper-bound property of the current pipeline.

---

## 9. Conclusion

The mechanistic interpretability field has made remarkable progress in identifying circuits, features, and representations in neural networks. But progress in discovery has outpaced progress in validation. The result is a field that generates claims faster than it can evaluate them, using tools whose reliability is uncertain, producing results that do not reproduce.

The MechVal framework addresses this gap by providing the measurement infrastructure the field lacks. Its contribution is not a new detection method but a systematic protocol for determining whether any detection method actually works. The 27 criteria, five validity types, six evidence families, and five verdict tiers provide a common language for evaluating MI claims — whether they concern circuits, SAE features, probes, or steering vectors.

The automated CVS pipeline closes the deployment gap. By operationalizing the framework end-to-end, it brings hours-per-paper manual evaluation down to minutes-per-paper automatic evaluation, with a calibrated directional bias that makes its output a useful conservative upper bound. The framework’s discrimination across 13 case studies, the pipeline’s within-one-tier agreement on 9 reference papers, and the directional asymmetry of its errors together demonstrate that systematic validity assessment of MI claims is both possible and tractable.

The framework is open-source and method-agnostic. Its contribution is a public good: the measurement infrastructure that makes interpretability claims trustworthy.

---

## Acknowledgments

The framework draws on insights from the mechanistic interpretability community, particularly open problems identified by Sharkey et al. (2026) and the actionability framework of Orgad et al. (2026). The cross-field methodology benefits from validation traditions in philosophy of science (Mayo, 2018), neuroscience, psychometrics (Cronbach & Meehl, 1955), pharmacology, and causal inference (Pearl, 2009; Pearl & Bareinboim, 2011).

---

## Appendices (to be expanded)

- **Appendix A.** Full IOI per-criterion reference annotations — 27 criteria × 7 claims = 189 cells.
- **Appendix B.** Full 60-configuration model comparison table.
- **Appendix C.** Prompt engineering details: 15 prompt iterations, 10 scoring pitfalls, full system prompts.
- **Appendix D.** Per-paper extraction logs for the 9 case-study papers.
- **Appendix E.** API and CLI reference for `mechanistic_validity` Python package.

---

## Notes for Revision

This combined draft preserves all content from the two source papers and reorganizes it into a single TMLR-style submission. Specific decisions made during the merge:

1. **Criterion names** — kept from `mechval_framework.tex` (Falsifiability, Convergent validity, etc.). The `v5.md` rename to `Task specificity`/`Minimality`/`Construct coverage` is NOT applied per author decision.
2. **Framework first, automation second** — the framework is the intellectual contribution; the automation is the scalability story. Order matches what reviewers will expect to see.
3. **Case studies stay manual in §5.** The §6 automation paper is presented as an operationalization that *reproduces* the manual verdicts to within one tier — this is how to motivate the automation without re-running everything.
4. **R1 (the “no novel experiments” objection) is pre-empted in §8.3.** Explicit limitations paragraph, plus a forward commitment to running `verify(spec)`. This is the highest-priority reviewer concern.
5. **R3 (directional asymmetry) is reframed in §7.4** as a conservative upper-bound property rather than a generic LLM-as-judge limitation.
6. **R4 (IAA) is acknowledged in §8.3** with a commitment to formal statistics.
7. **R5 (criterion overlap) is added in §8.4** as future-work.
8. **R6 (related work)** §2 includes MechEvalAgent, MIB, and the LLM-as-evaluator literature — minimal but present.

When splitting back into LaTeX for TMLR submission, target lengths:
- Main paper: §1–§9 ≈ 18 pages (under TMLR’s 12-page main + unlimited appendix policy, this fits with §6–§7 partially moved to appendix if needed).
- Appendices: A–E ≈ 15–25 pages.
