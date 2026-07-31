# The Validation Layer: Infrastructure, Not Another Detector

**Schmidt Sciences AI Interpretability RFP Response**

---

## 1. The Problem: A Validation Crisis in AI Safety Interpretability

The field of mechanistic interpretability is experiencing a measurement crisis that undermines every claim about detecting deceptive behavior in large language models. The evidence is stark.

**The features are unstable.** Bricken et al. and Engels et al. (NeurIPS 2025) demonstrated that only 197 of 65,536 sparse autoencoder features --- 0.3% --- are stable across independent training runs. The remaining 99.7% are training artifacts. Any "deception feature" identified using standard SAE methods has a 99.7% prior probability of being an artifact, not a genuine representation.

**The research does not reproduce.** MechEvalAgent (Bai, Baumgartner, Sun, Holtzman, and Tan; February 2026) performed execution-grounded evaluation of published MI research: 93% of papers fail reproducibility checks when their code is actually run, and 80% fail coherence checks between claims and evidence. Narrative review --- the kind performed by peer reviewers --- missed 51 issues that execution-grounded evaluation caught. The field lacks the infrastructure to enforce reproducibility.

**The evaluation tools are unreliable.** The SAEBench audit (May 2026) subjected six standard SAE evaluation metrics to three diagnostic tests: reseed stability, ground-truth correlation, and discriminability. Two of six --- Token Prediction Probability (TPP) and Sparse Cross-entropy Recovery (SCR) --- failed all three diagnostics. TPP exhibited reseed coefficients of variation between 16% and 39%, meaning the same SAE evaluated twice gives substantially different scores. These metrics have been used in dozens of papers. Those claims are now in question.

**The default causal tool is nearly uncorrelated with ground truth.** Relevance Propagation (RelP; NeurIPS 2025) demonstrated that attribution patching --- the field's most widely used method for identifying causally relevant components --- correlates at r = 0.006 with ground truth on controlled synthetic tasks. This is not a weak correlation. It is no correlation. The method that the field treats as the gold standard for causal attribution provides essentially random rankings of component importance.

The implication for AI safety is direct: any claim of the form "we found a feature that detects deception" is built on this foundation. If the feature was identified using SAE methods, it is probably a training artifact. If it was validated using attribution patching, the validation is meaningless. If its quality was assessed using TPP or SCR, the assessment is unreliable. If the result was reported in a paper, there is a 93% chance the code does not reproduce.

This is not an abstract methodological concern. It is a concrete threat to AI safety. Organizations are beginning to deploy internal monitors based on mechanistic interpretability findings --- probes that flag potential deception, steering vectors that suppress harmful outputs, circuit-level classifiers that trigger human review. If those monitors are built on unvalidated features, they provide false confidence: the system appears monitored when it is not.

The asymmetry compounds the problem. Behavioral red-teaming --- the alternative to MI-based detection --- has no validation standards either. Red-team evaluations have no stated falsification criteria, no published reproducibility rates, no measurement stability requirements. But MI is held to higher standards, a pattern Neel Nanda (2025) has called "isolated demand for rigor." The solution is not to lower the bar for MI. It is to raise the bar for everything, using explicit validity criteria that apply equally to mechanistic and behavioral approaches.

The Schmidt RFP asks researchers to "move beyond academic benchmarks to address real-world risks." This aspiration presupposes that we know which benchmarks are valid. We do not. Before the field can move beyond benchmarks, it needs to know which benchmarks to move beyond. That is the gap this proposal fills.


## 2. Our Approach: A Measurement-Theoretic Validity Framework

The Mechanistic Validity framework provides the missing evaluation infrastructure. It is not a new detection method. It is a systematic protocol for determining whether any detection method --- existing or future --- actually works.

**Structure.** The framework organizes 172 metrics across five validity types drawn from measurement theory, philosophy of science, and clinical trial methodology:

- **Construct validity** (criteria C1--C5): Is the claimed construct well-defined? C1 requires falsifiability --- stating what evidence would disprove the claim. C3 requires specificity --- the feature must respond to its nominal target and not to correlated confounds. C5 requires convergent validity --- multiple independent methods should identify the same feature.
- **Internal validity** (criteria I1--I5): Is the causal evidence sound? I1 tests necessity via ablation. I2 tests sufficiency via knock-in. I3 tests specificity --- does the intervention affect only the target behavior? I4 tests consistency across ablation methods and prompt sets.
- **External validity** (criteria E1--E6): Does the finding generalize? E1 measures intervention reach. E2 tests graded response --- does increasing the feature's activation produce monotonically more of the target behavior? E5 tests robustness under prompt paraphrase. E6 tests cross-architecture transfer.
- **Measurement validity** (criteria M1--M6): Are the measurements reliable? M1 requires test-retest reliability. M2 requires measurement invariance across conditions. M5 requires calibration against published baselines.
- **Interpretive validity** (criteria V1--V5): Does the interpretation match the evidence? V1 requires that the description level (computational, algorithmic, representational, implementational) is declared. V2 requires that the evidence type matches the declared level.

These 27 criteria are organized into five verdict tiers that represent qualitative transitions in evidential status:

1. **Proposed**: The claim is stated with enough precision to evaluate, but no causal evidence exists.
2. **Causally Suggestive**: At least one well-controlled causal experiment demonstrates necessity.
3. **Mechanistically Supported**: Both necessity and sufficiency are established, with specificity evidence and measurement reliability.
4. **Triangulated**: Multiple converging lines of evidence from different evidence families (causal, structural, behavioral, representational, information-theoretic) support the claim.
5. **Validated**: All criteria are satisfied, including cross-model transfer and adversarial robustness.

The framework is method-agnostic. It evaluates SAE features, neural circuits, probing results, steering vectors, natural language autoencoder descriptions, transcoders, and crosscoders using the same criteria. It does not privilege one detection approach over another. It asks the same questions of each: Is the construct defined? Is the evidence causal? Does the finding generalize? Is the measurement reliable?

**What already exists.** This is not a proposal to build a framework. The framework is built. It includes 84 implemented metric scripts organized by evidence family (causal, structural, behavioral, representational, information-theoretic), 14 calibration gates (bootstrap stability, seed variance, convergent validity, discriminant validity, measurement invariance), 54 evaluation tasks spanning 10 linguistic domains, 12 pre-registered mechanistic claim specifications with causal predictions and negative controls, and 13 worked case studies on published circuits (IOI, Induction Heads, Greater-Than, Copy Suppression, Othello, Grokking, Successor Heads, and others). These case studies assigned verdict tiers ranging from Proposed (Knowledge Neurons, Probing Classifiers) to Triangulated (IOI Circuit, Othello World Model), demonstrating that the tier system discriminates between well-validated and poorly-validated claims.

**Key insight for this proposal.** Deception detection is not a separate research agenda. Every claim of the form "this feature detects deception" is a mechanistic interpretability claim. It has a construct (deception), a measurement (activation magnitude or probe accuracy), a causal hypothesis (the feature is causally involved in deceptive behavior), and generalization requirements (it works across prompts, models, and adversarial conditions). The validity framework already provides the tools to evaluate each of these components. Deception is just an application domain --- and one where the stakes of getting validation wrong are highest.

**Convergent external evidence.** The argument that evaluation criteria are the bottleneck is not ours alone. ICML 2026 accepted "Interpretability Can Be Actionable" (Orgad, Barez, et al.), a 30-author position paper arguing that concreteness and validation --- not new methods --- are the missing ingredients for actionable MI. The 30-author "Open Problems in Mechanistic Interpretability" (Sharkey et al., January 2026) lists validation of mechanistic descriptions as an open problem, noting the absence of standardized evaluation frameworks. Jacob Steinhardt (May 2026) calls for behavioral evaluation over capability evaluation, arguing that the field systematically under-invests in measuring what models actually do versus what they can do. The Mechanistic Validity framework operationalizes all three positions: it defines what "validated" means, provides criteria for evaluation, and includes behavioral prediction as a required validity dimension.


## 3. Application to Deceptive Behavior Detection

Applying the framework to deception detection reveals that the problem is harder than it appears --- and that existing tools already provide partial solutions.

**The construct problem.** "Deception" in LLMs is not a single construct. It is a human-imposed label on a cluster of mechanistically distinct behaviors: strategic lying (asserting known falsehoods), sycophancy (agreeing with the user despite internal disagreement), sandbagging (deliberately underperforming on capability evaluations), goal misgeneralization (pursuing proxy objectives that diverge from intended ones), and faithfulness failures (chain-of-thought that does not reflect actual computation). Each of these behaviors may have a different mechanistic substrate. Treating them as a single construct is a C3 validity failure unless convergent evidence demonstrates a shared mechanism.

The framework's C3 discriminant validity criterion directly addresses this. A candidate "deception feature" must be tested against neighboring constructs: does it fire on deception, or on uncertainty? On deception, or on hedging? On deception, or on politeness? Without these discriminant tests, a "deception detector" may be a "the model is uncertain" detector with a misleading label. The difference matters: intervening on uncertainty to reduce deception would suppress the model's ability to express genuine uncertainty, a harmful side effect.

**Mapping framework criteria to deception detection.** The framework provides specific evaluation requirements for any deception-detection claim:

- **C1 Falsifiability**: What result would disprove the deception feature? If no answer can be stated in advance, the feature label is unfalsifiable and the claim is at most Proposed.
- **C3 Specificity**: Does the feature respond only to deception, or also to uncertainty, hedging, compliance, and other correlated constructs? Discriminant validity testing (calibration F04) provides the methodology.
- **I2 Sufficiency**: Does activating the feature alone produce deceptive outputs? Correlation between feature activation and deceptive behavior is the floor, not the ceiling. A feature that fires during deception but does not cause it when activated is a correlate, not a mechanism.
- **E2 Graded Response**: Does increasing the feature's activation produce monotonically more deceptive behavior? Dose-response testing is the minimum bar for a causal claim about magnitude.
- **M1 Reliability**: Does the detector give consistent results across prompt splits, random seeds, and model checkpoints? The DMSAE finding (99.7% feature instability) is the baseline concern.
- **E5 Robustness**: Does the feature survive adversarial prompt variation? Standard prompt-paraphrase testing (benign) and adversarial ablation (worst-case) provide complementary robustness evidence.

**The safety subspace finding.** The most promising positive result for MI-based safety detection comes from a convergence of four independent papers (late 2025 through early 2026) showing that safety-relevant information in language models occupies a low-rank, stable subspace. This subspace passes several framework criteria by construction: it is stable across fine-tuning (M1), it supports graded intervention (E2 via singular value manipulation), and its causal role has been verified through targeted ablation and restoration experiments (I1, I2). Safety Singular Value Entropy quantifies how densely safety information is packed across layers. The "Safety at One Shot" finding demonstrates that a single safety-relevant example can recover the full alignment subspace via low-rank gradient structure --- the most parsimonious form of causal sufficiency (I2).

This is where a deception-detection effort should begin: not by searching for new features, but by evaluating whether the safety subspace already encodes deception-relevant information, and with what validity.

**Adversarial ablation as the sufficiency test.** Standard circuit ablation tests necessity: removing the circuit disrupts the behavior. The VPD adversarial ablation methodology (Goodfire/Apollo, May 2026) goes further, testing whether the circuit is sufficient under worst-case conditions. Most published circuit claims pass necessity testing but fail adversarial sufficiency testing --- the model can recover the behavior through alternative pathways when the ablation is adversarially chosen. For safety applications, this distinction is critical: a monitor based on a circuit that is necessary under benign conditions but bypassable under adversarial conditions provides false security. The framework's I2 sufficiency criterion, combined with E5 robustness, captures this requirement.


## 4. Proposed Work

### Year 1: Validity Report Cards for Safety Features ($200K)

**Goal.** Produce the first systematic validity assessment of any safety-relevant feature in a language model.

**Target features** (5 evaluations):
1. **Refusal directions** (Arditi et al., 2024) --- the most-cited safety-relevant direction, already used for jailbreaking. Highest causal evidence of any candidate.
2. **Sycophancy directions** (Sharma et al.) --- behavioral tendency to agree with factually incorrect users. Construct validity is uncertain (C3: sycophancy versus politeness versus compliance).
3. **The Assistant Axis** (MATS / Anthropic Fellows, 2026) --- a recently proposed direction encoding "assistant-mode" behavior. Limited causal testing to date.
4. **Safety subspace** (four converging papers, 2025--2026) --- the low-rank subspace identified independently by four groups. The strongest existing candidate.
5. **Sandbagging circuits** (if proposed by the time evaluation begins) --- deliberate capability suppression during evaluations. If no circuit exists, this slot will be replaced with an evaluation of the next most-cited safety feature.

**Target model.** Gemma 2 27B. This model is large enough to exhibit safety-relevant behaviors (sycophancy, refusal, evaluation-aware behavior modification), comprehensively instrumented with Gemma Scope 2 SAE artifacts, and has open weights enabling full intervention access. GPT-2, the default model for MI research, is too small for realistic safety evaluations.

**Protocol per feature** (three phases):
- **Phase I --- Measurement reliability** (8 criteria): Bootstrap stability (F01), seed variance (F02), inter-run agreement (F06), prompt-paraphrase invariance (M2), baseline separation (calibrations F09, F10), construct operationalization (G0), superposition risk assessment (G3).
- **Phase II --- Causal validity** (6 criteria): Necessity via ablation (I1), sufficiency via knock-in (I2), specificity via discriminant testing (I3), consistency across ablation methods (I4), graded response / dose-response (E2), effect magnitude (E4).
- **Phase III --- Generalization** (4 criteria): Cross-prompt generalization (E5), robustness under adversarial prompt variation (E5 adversarial), cross-model transfer to at least one additional architecture (E6), intervention reach (E1).

**Deliverable.** A published validity report card for each feature, stating which criteria pass, which fail, and what verdict tier the feature achieves. This will be the first time any safety feature has been evaluated against explicit, pre-registered validity criteria. The report cards will include full data, code, and negative results. If a feature fails --- for example, if sycophancy directions fail C3 discriminant validity because they cannot be distinguished from politeness --- the failure is itself a contribution. It tells the field exactly where to invest.

### Year 2: Validity Predicts Intervention Effectiveness ($150K)

**Goal.** Test the hypothesis that features passing more validity criteria --- especially I2 sufficiency and E2 graded response --- are more effective steering targets.

**Method.** For each feature evaluated in Year 1:
1. Record the validity profile: number of criteria passed, verdict tier, specific criteria outcomes.
2. Use the feature as a steering vector via activation addition/subtraction at inference.
3. Measure steering effectiveness: behavioral change in the target direction (e.g., sycophancy rate change, refusal rate change).
4. Measure steering specificity: unintended behavioral side effects on non-target tasks (perplexity change on held-out text, accuracy change on unrelated benchmarks).
5. Compute the correlation between validity score and steering effectiveness.

**The key prediction.** A feature with high I2 (causally sufficient) and high E2 (graded, dose-dependent response) should be a more effective and specific steering target than a feature with only high M1 (reliably measurable but not causally tested). If this prediction holds, the framework provides a practical selection criterion: validate first, steer second.

**Counter-prediction to test.** Perhaps any direction that is reliably detectable (high M1) is an adequate steering target regardless of causal status. If this holds, the framework's causal criteria are unnecessarily stringent for steering applications --- a useful finding that would reshape how the field approaches intervention design.

**Deliverable.** Empirical evidence for or against the practical utility of validation. This directly addresses a likely objection to the framework: "validation is theoretically nice but does it help in practice?" If validity predicts steering effectiveness, the answer is yes, with quantified effect sizes.

### Year 3: Adversarial Validation ($150K)

**Goal.** Test whether validated safety monitors are harder to evade than unvalidated ones, using a red-team/blue-team exercise that matches the Schmidt RFP's competition design.

**Method.**
- **Blue team** deploys safety monitors built on features that achieved Mechanistically Supported or higher verdict tiers in Year 1. These monitors use causally validated features with known reliability and specificity.
- **Red team** deploys safety monitors built on features that achieved only Proposed or Causally Suggestive verdict tiers --- features with correlational evidence but without causal validation.
- **Adversary** uses gradient-based adversarial prompt optimization (GCG-style) to craft inputs that evade each monitor class.
- **Measurements**: detection rate, false positive rate, and adversarial evasion rate for each monitor, stratified by the monitor's validity tier.

**The key prediction.** Validated monitors should be harder to evade because they target causally relevant mechanisms, not surface correlates. An adversary can change surface features (prompting style, token distribution, formatting) but cannot easily change the model's internal causal structure without degrading the model's overall performance. Unvalidated monitors, by contrast, may rely on correlates that are trivially manipulable.

**Deliverable.** Adversarial robustness scores stratified by validity tier. If the prediction holds, this provides the strongest possible argument for validation: it is not just methodologically sound, it is a security requirement. An open-source adversarial validation pipeline will be released for other groups to apply to their own safety features.

**Total budget: $500K over 3 years.** This is at the lower end of the RFP's $300K--$1M range, reflecting that the framework infrastructure already exists. The budget covers compute for large-model evaluations (Gemma 2 27B requires substantial GPU resources for intervention experiments), personnel (1--2 postdoctoral researchers with expertise in mechanistic interpretability and measurement methodology), and adversarial ML resources for Year 3.


## 5. Team and Feasibility

The framework is already built. The 84 metrics, 14 calibrations, 54 tasks, and 13 worked case studies are implemented in an open-source Python package with a programmatic API and command-line interface. The package supports Gymnasium-style task registration, pre-registered causal claim specifications with structured predictions and negative controls, and four scoring views grounded in causal inference methodology (Pearl/Rubin effect estimation, Pearl/Bareinboim transportability, Pearl rung-3 counterfactual verification, and SEM-style mechanism adjudication).

The proposal asks for application, not construction. The primary technical risk is not whether the framework works --- it has been applied to 13 published circuits with discriminating results --- but whether the target safety features are mature enough for a full evaluation. This risk is mitigated by the feature selection: refusal directions (Arditi et al., 2024) have the most extensive causal evidence of any safety feature in the literature, and the safety subspace has been independently identified by four groups. Even if some candidate features turn out to be too immature (e.g., sandbagging circuits may not yet exist in published form), the evaluation protocol produces informative results at every stage. A feature that fails at Phase I (measurement reliability) tells the field something different from one that fails at Phase II (causal validity) or Phase III (generalization). Null results are published.

[Team details to be completed with specific PI, institutional affiliation, and collaborator information.]

The framework's method-agnostic design is a key differentiator. This proposal does not compete with groups developing new detection methods (SAE-based, probing-based, steering-based). It complements all of them. Every detection method benefits from knowing whether its outputs are valid. The validation layer makes all detection methods better --- or reveals which ones should not be trusted.


## 6. Broader Impact

There are two possible outcomes, and both are informative.

If safety features pass the validity evaluation, the field gains deployment-ready monitors with known reliability, specificity, causal grounding, and adversarial robustness characteristics. Each validity report card states exactly what the monitor can and cannot do: which types of deception it detects (C3 specificity), how reliably it detects them (M1), whether intervening on it changes behavior (I2), and whether an adversary can evade it (E5). This is the information a deployment team needs to make a principled decision about whether to trust the monitor.

If safety features fail, the field gains equally valuable information. The validity report card states exactly which criteria fail. If refusal directions fail C3 (specificity), the field knows to invest in discriminant validity testing before deploying refusal-direction monitors. If the safety subspace fails E6 (cross-architecture transfer), the field knows that safety features are model-specific and must be re-identified for each architecture. If sycophancy directions fail I2 (sufficiency), the field knows that sycophancy is detectable but not manipulable via those directions --- steering will require a different approach.

Either way, the framework becomes a standard. Clinical trial methodology does not make drugs work. It tells you which drugs work, with quantified confidence. The validation framework serves the same function for AI safety features: it does not make detection methods correct, but it provides the infrastructure for determining which detection methods are correct. The field currently has no such standard. Every safety feature is evaluated ad hoc, with different criteria, different baselines, and different reporting conventions. This proposal provides a common language --- 27 criteria, 5 verdict tiers, explicit thresholds --- for the entire field to use.

The framework is open-source, method-agnostic, and designed for adoption. Its contribution is not a proprietary advantage but a public good: the measurement infrastructure that makes AI safety claims trustworthy.
