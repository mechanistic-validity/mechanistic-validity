# Adversarial Validation: Why Your Safety Monitors Will Fail Schmidt's Competition (And How to Fix Them)

**Schmidt Sciences AI Interpretability RFP Response**

Proposal for $750K over 24 months. PI: [Name]. Institutional affiliation: [Institution].

---

## 1. The Problem: Unvalidated Safety Features Cannot Survive Adversarial Testing

Five groups have published features claimed to detect deceptive behavior in large language models. None has been validated against adversarial evasion, cross-model transfer, or discriminant validity. We propose the first systematic validity trial for AI safety features, and the first empirical test of whether validated monitors survive the kind of adversarial pressure that Schmidt's proposed red-team competition will impose.

Schmidt Sciences' RFP proposes a blinded red-team vs blue-team competition: "blue teams must show that interpretability tools work in practice, revealing deceptive behaviors that are not known about in advance." This is the right design. It is also a design that most current interpretability-based safety tools will fail, because they have been validated under benign conditions --- fixed benchmarks, known attack patterns, cooperative evaluation protocols --- and nobody has tested whether they survive adversarial pressure.

The few groups that have tried adversarial testing report alarming results. In May 2026, Goodfire and Apollo Research released the Validated Parameter Decomposition (VPD) framework, which applies adversarial ablation to test whether subnetwork identification methods find causally correct nodes. Their finding: most methods identify the wrong nodes. Standard ablation says the nodes matter; adversarial ablation --- where the ablation direction is chosen to maximally exploit compensation pathways --- reveals that the model routes around them. The identified "circuit" is a surface correlate, not the actual mechanism.

This matters for safety because a deception monitor built on a surface correlate is a lock that opens to any skeleton key. It passes normal testing. But an adversary with gradient-based optimization, paraphrase attacks, or compositional evasion bypasses it trivially, because the monitor was never targeting the model's actual mechanism for deception --- it was targeting a statistical regularity that happens to co-occur with deception under cooperative conditions.

Standard validation is the normal key test. Adversarial validation is the skeleton key test. Schmidt's blinded competition is the skeleton key test. The field needs to know which monitors survive it --- and which criteria predict survival.


## 2. The Validation Crisis That Makes This Urgent

The problem is not merely theoretical. Four independent lines of evidence, published between late 2025 and May 2026, converge on the same conclusion: the field's current tools are insufficient for safety-critical deployment.

**The features are unstable.** Dauparas et al. (DMSAE, NeurIPS 2025) trained sparse autoencoders on the same model with different random seeds and measured which features survived. Of 65,536 SAE features, only 197 --- 0.3% --- were stable across training runs. The remaining 99.7% are training artifacts. Any "deception feature" identified using standard SAE methods has a 99.7% prior probability of being an artifact, not a genuine representation. An adversary who understands this can target the specific features a detector uses, knowing they are arbitrary choices from a vast space of equally valid decompositions.

**The research does not reproduce.** MechEvalAgent (Bai et al., February 2026) performed execution-grounded evaluation of published MI research: 93% of papers fail reproducibility checks when their code is actually run, and 80% fail coherence checks between claims and evidence. Narrative review missed 51 issues that execution-grounded evaluation caught.

**The evaluation tools are unreliable.** The SAEBench audit (May 2026) subjected six standard SAE metrics to three diagnostic tests: reseed stability, ground-truth correlation, and discriminability. Two of six --- TPP and SCR --- failed all three diagnostics. TPP exhibited 16--39% coefficient of variation across random seeds, meaning the same SAE evaluated twice gives substantially different scores. These metrics have been used in dozens of papers.

**The default causal tool is near-random.** The RelP study (NeurIPS 2025) found that attribution patching --- the field's most widely used method for identifying causally relevant components --- correlates at r = 0.006 with ground-truth causal importance. This is not a weak correlation. It is no correlation. Safety features identified through attribution patching inherit that randomness.

The implication is direct: any claim of the form "we found a feature that detects deception" is built on this foundation. If the feature was identified using SAE methods, it is probably a training artifact. If it was validated using attribution patching, the validation is meaningless. If the result was reported in a paper, there is a 93% chance the code does not reproduce.

ICML 2026 accepted "Interpretability Can Be Actionable" (Orgad, Barez, et al.), a 12-author position paper arguing that evaluation criteria --- not new methods --- are the missing ingredient for actionable mechanistic interpretability. The 30-author "Open Problems in Mechanistic Interpretability" (Sharkey et al., January 2026) lists validation of mechanistic descriptions as an open problem, noting the absence of standardized evaluation frameworks. This proposal provides the evaluation criteria both papers call for.


## 3. The Measurement Framework: Infrastructure, Not Another Detector

The Mechanistic Validity framework provides the missing evaluation infrastructure. It is not a new detection method. It is a systematic protocol for determining whether any detection method --- existing or future --- actually works.

The framework organizes 172 metrics across five validity types drawn from measurement theory, philosophy of science, and clinical trial methodology:

- **Construct validity** (C1--C5): Is the claimed construct well-defined? C1 requires falsifiability. C3 requires specificity --- the feature must respond to its nominal target and not to correlated confounds. C5 requires convergent validity.
- **Internal validity** (I1--I5): Is the causal evidence sound? I1 tests necessity via ablation. I2 tests sufficiency via knock-in. I3 tests specificity.
- **External validity** (E1--E6): Does the finding generalize? E5 tests robustness under prompt paraphrase. E6 tests cross-architecture transfer.
- **Measurement validity** (M1--M6): Are the measurements reliable? M1 requires test-retest reliability. M2 requires measurement invariance.
- **Interpretive validity** (V1--V5): Does the interpretation match the evidence? V1 requires that the description level is declared. V2 requires that evidence type matches the declared level.

These 27 criteria are organized into five verdict tiers that represent qualitative transitions in evidential status:

1. **Proposed**: Claim stated with enough precision to evaluate, but no causal evidence.
2. **Causally Suggestive**: At least one well-controlled causal experiment demonstrates necessity.
3. **Mechanistically Supported**: Necessity and sufficiency established, with specificity evidence and measurement reliability.
4. **Triangulated**: Multiple converging lines of evidence from different evidence families.
5. **Validated**: All criteria satisfied, including cross-model transfer and adversarial robustness.

The framework is method-agnostic. It evaluates SAE features, neural circuits, probing results, steering vectors, transcoders, and crosscoders using the same criteria. It does not privilege one detection approach over another.

**What already exists.** This is not a proposal to build a framework. The framework is built. It includes 172 implemented metric scripts organized by evidence family (causal, structural, behavioral, representational, information-theoretic), 14 calibration gates, 54 evaluation tasks spanning 10 linguistic domains, 12 pre-registered mechanistic claim specifications with causal predictions and negative controls, and 13 worked case studies on published circuits. These case studies assigned verdict tiers ranging from Proposed (Knowledge Neurons, Probing Classifiers) to Triangulated (IOI Circuit, Othello World Model), demonstrating that the tier system discriminates between well-validated and poorly-validated claims.

**Deception detection benefits from general validity infrastructure.** Every claim of the form "this feature detects deception" is a mechanistic interpretability claim with a construct, a measurement, a causal hypothesis, and generalization requirements. The framework already provides tools to evaluate each component. What makes deception distinctive is not the methodology --- it is the stakes. A false positive in circuit discovery means a wrong paper. A false positive in deception detection means a deployed monitor that provides false confidence. The framework handles both; deception is where the consequences of getting validation wrong are highest.


## 4. The Robustness Hierarchy: A Testable Prediction

Our central hypothesis: safety features that pass causal sufficiency testing are fundamentally harder to evade than features that pass only correlation-based testing.

Consider what it means for a feature to pass each level of validation, and what an adversary must do to evade a detector built on that feature:

- A feature that passes **I2 (causal sufficiency)** is part of the model's actual mechanism for producing deceptive outputs. To evade it, the adversary must cause deception without activating the model's own deception mechanism --- requiring either an alternative pathway or weight modification.
- A feature that passes **I3 (specificity)** fires only on the target behavior. No false positives for the adversary to exploit through alert fatigue.
- A feature that passes **VPD adversarial ablation** survives worst-case removal of other components. No hidden compensation pathways.
- A feature that passes **E5 (cross-prompt robustness)** resists paraphrase attacks --- the simplest and most realistic form of adversarial evasion.
- A feature that passes **E6 (cross-architecture transfer)** reflects a general computational strategy, not a model-specific quirk.

These criteria form a natural robustness hierarchy --- five tiers defined by how many criteria a feature passes and what level of adversarial effort is required to evade it:

| Verdict Tier | Criteria Passed | Adversarial Effort to Evade | Time |
|---|---|---|---|
| **Proposed** | 0--1 | Rephrasing, prompt injection, surface perturbation | Minutes |
| **Causally Suggestive** | 2--3 | Identify and avoid the specific correlate | Hours to days |
| **Mechanistically Supported** | 4--6 | Gradient-based optimization against the detection mechanism | Days to weeks |
| **Triangulated** | 7--9 | Model-level attacks: weight modification, fine-tuning | Weeks+ |
| **Validated** | 10+ | No known input-level evasion; requires building a different model | N/A |

This hierarchy is a testable prediction, not an assumption. Phase 3 of our proposed work measures adversarial evasion rates for features at each tier and tests whether the hierarchy holds empirically.


## 5. Trial Protocol and Proposed Experiments

We adopt a three-phase gating protocol --- analogous to clinical trial phases --- where each phase addresses a specific class of validity failure identified in Section 2. Advancing to the next phase requires passing the previous one.

### Phase I: Measurement Reliability (Months 1--6, $150K)

Phase I asks: *Is this safety feature reliably detectable?*

Before testing whether a feature does anything useful, we must establish that it can be measured consistently. This is the step the field currently skips. Phase I applies eight measurement criteria:

- **M1 Reliability**: Consistent detection across different prompts, random seeds, and model checkpoints.
- **Core Stability**: Does the feature survive retraining? This is the DMSAE test applied to safety features. Based on that finding, this criterion alone would eliminate approximately 99.7% of current SAE-based feature candidates.
- **F01 Bootstrap Stability**: Confidence intervals tight enough for binary classification.
- **G0 Construct Operationalization**: Falsifiable definition of what the feature is and what would disprove it.
- **G3 Superposition Risk**: Is the feature entangled with other features in superposition?

**Gating rule**: A feature that fails measurement reliability does not advance. No causal testing, no generalization testing, no deployment. An unreliable measurement cannot provide valid evidence.

**Pre-registration requirement**: Before Phase I begins, the research team states what the feature is, what behavior it predicts, and what result would falsify it. This prevents post-hoc reinterpretation of failed features.

### Phase II: Causal Efficacy (Months 7--14, $200K)

Phase II asks: *Does this feature causally produce or prevent the target behavior?*

A feature that passes Phase I is reliably measurable. Phase II tests whether it actually does what it claims. This is where the attribution patching problem (r = 0.006) and the VPD finding (standard ablation overestimates validity) are directly addressed.

- **I1 Necessity**: Removing the feature removes the behavior.
- **I2 Sufficiency**: Activating the feature produces the behavior, including under adversarial ablation of remaining components (VPD test).
- **I3 Specificity / Discriminant Validity**: The feature affects only the target behavior. A "deception direction" that also suppresses helpfulness, increases uncertainty markers, or degrades fluency is a blunt instrument, not a specific detector.
- **E2 Graded Response**: Dose-response curve --- increasing activation produces monotonically more of the target behavior.

**Gating rule**: Must pass necessity, sufficiency under adversarial conditions, and specificity to advance.

### Phase III: Adversarial Generalization (Months 15--24, $250K)

Phase III asks: *Does this feature survive deployment conditions?*

This is the most demanding phase, and the one that directly connects to Schmidt's competition design. We test generalization across four dimensions and adversarial robustness across four attack families.

**Generalization dimensions:**
- **E5 Cross-prompt robustness**: Same behavior, different vocabulary, syntax, and framing.
- **E6 Cross-architecture transfer**: Does the feature exist in Llama 3.1 70B and Qwen 2.5 72B?
- **Cross-domain transfer**: Does the feature generalize from one deception type to others?

**Four attack families:**
1. **Gradient-based adversarial prompt optimization (GCG-style)**: Optimize input tokens to produce deceptive outputs while minimizing activation of the safety feature. The strongest white-box attack.
2. **Transfer attacks**: Adversarial prompts optimized against the feature in one model, tested on the target model. Tests whether the feature is model-specific or reflects a general vulnerability.
3. **Paraphrase attacks**: LLM-generated paraphrases of deceptive content. The simplest and most realistic attack --- no gradient access required.
4. **Compositional attacks**: Individual components benign, composition deceptive. Tests whether the feature detects deception at the compositional level.

**Per-feature, per-attack output**: Evasion rate, false positive rate under attack, and detection latency.

**The correlation test** (the core scientific result): Regression of evasion rate on validity score, controlling for attack type and feature type. The interaction term tests whether validity matters more for sophisticated attacks (GCG, compositional) than for simple attacks (paraphrase).

**Competition-ready deployment** (Months 22--24): Two blue teams monitor the same model for deceptive behavior. One uses features at the Triangulated or Validated tier; the other uses features at the Proposed or Causally Suggestive tier. A red team generates evasive inputs. Detection rates are compared. Prediction: the Triangulated+ team detects more deceptive behaviors, and the gap widens as attacks become more sophisticated.


## 6. Target Features

We apply the full trial protocol to five safety-relevant features in Gemma 2 27B, selected to span the range of current evidence quality. Gemma 2 is the right target: large enough to exhibit realistic safety-relevant behaviors, comprehensively instrumented with Gemma Scope 2 SAE artifacts, and open-weight.

1. **Refusal directions** (Arditi et al. 2024). The most causally validated safety feature in the literature. Ablation removes refusal; addition induces it. The best-validated candidate and our positive control. Known gaps: specificity (does it fire on uncertainty?), cross-architecture transfer, adversarial robustness.

2. **Safety subspace** (four independent papers, 2025--2026). A low-rank subspace containing safety-relevant information, identified independently by four groups. Passes construct validity, measurement reliability, and causal sufficiency by construction. Expected to score highest. The most promising candidate for genuine deception-relevant structure.

3. **Sycophancy directions** (Sharma et al. 2024). The deceptive behavior most commonly observed in deployed systems. Primarily correlation-based evidence to date. Phase II will determine whether these features are causal or merely correlational.

4. **The assistant axis** (MATS/Anthropic Fellows 2026). A direction distinguishing "assistant mode" from "base model mode." Interesting because it targets a meta-behavioral property. Limited causal testing to date.

5. **The fifth slot is reserved for the most mature safety feature published before the evaluation period begins.** Candidates include sandbagging circuits (deliberate underperformance) and goal-misgeneralization features. We commit to evaluating whatever is most established at the time of Phase I, rather than selecting a candidate that may not yet exist. This is honest about the field's pace --- and ensures we evaluate the best available evidence, not a premature construct.


## 7. Counter-Predictions and Risks

**Counter-prediction 1.** Perhaps any reliably detectable direction (high M1) is an adequate steering target regardless of causal status. If a feature that passes only Phase I produces steering results as good as one that passes Phase II, the framework's causal criteria are unnecessarily stringent for steering applications. This would be a useful finding that reshapes how the field approaches intervention design. We test it directly in Phase III.

**Counter-prediction 2.** Perhaps adversarial robustness is independent of validity tier. Features could be hard to evade for reasons unrelated to causal grounding --- simple geometric properties like activation magnitude or angular separation might predict robustness better than the full 27-criterion assessment. The Phase III regression includes geometric controls to test this.

**Risk 1: Target features may be too immature.** Mitigation: refusal directions and the safety subspace have extensive existing evidence. Even if other candidates fail Phase I, the protocol produces informative results. A feature that fails at Phase I (measurement reliability) tells the field something different from one that fails at Phase II (causal validity).

**Risk 2: Gemma 2 27B may not exhibit realistic deceptive behavior.** Mitigation: we test for sycophancy and refusal, which are well-documented in Gemma-class models. For harder deception types (sandbagging, goal misgeneralization), we use behavioral elicitation protocols from the alignment evaluation literature. If elicitation fails, we report the null result and its implications.

**Risk 3: The correlation between validity and robustness may be null.** This is still informative. It means the field needs both validation and adversarial testing, and neither substitutes for the other. We commit to publishing whichever result we find. A null result with full analysis of why the prediction failed is itself a contribution --- it tells the field exactly what does and does not predict adversarial resilience.

**Risk 4: Adversarial evasion testing may be underpowered.** With only 5 features, regression power is limited. Mitigation: we augment with sub-feature granularity (individual criteria scores, not just aggregate tiers) and use permutation-based testing rather than relying on asymptotic statistics.


## 8. Why This Matches Schmidt's Design

Schmidt's RFP articulates three focus areas and one structural requirement. This proposal addresses all four.

**"Detect deceptive behaviors."** Phase I--II produces the first validity-scored inventory of safety features. Phase III tests them under adversarial conditions. Together, they answer the question the field has avoided: which detection tools actually detect anything under pressure?

**"Mitigate deceptive behaviors."** Features with high causal sufficiency (I2) scores should be more effective steering targets. Phase III tests this directly: do validated features produce more reliable, more specific interventions than unscored features?

**"Move beyond academic benchmarks."** Adversarial evasion testing is not an academic benchmark. It is the hardest possible evaluation: an intelligent adversary with full model access optimizing against the detector. The field's standard evaluations --- fixed prompt sets, known attack patterns, cooperative testing --- are precisely the academic benchmarks Schmidt wants to move beyond.

**The blinded competition structure.** Phase III directly instantiates Schmidt's proposed competition design. Our prediction: monitors built on validated features outperform monitors built on unvalidated features, and the gap is proportional to the difference in validity tier. The experiment produces the evidence Schmidt needs to assess whether validation is a useful selection criterion for competition-ready monitors.


## 9. Deliverables and Timeline

| Milestone | Month | Deliverable |
|-----------|-------|-------------|
| Pre-registered trial protocols | 1 | Published claim specifications for all 5 features: predictions, falsification criteria, negative controls. |
| Phase I validity report cards | 6 | Measurement reliability assessments. Which features pass, which fail, at what confidence. |
| Phase II causal assessments | 14 | Causal necessity, sufficiency, specificity, dose-response. Verdict tier assignments. |
| Adversarial evasion scores | 20 | Per-feature, per-attack evasion rates across 4 attack families. |
| Correlation analysis | 22 | Regression of evasion rate on validity score. The core scientific test, published regardless of outcome. |
| Competition results | 24 | Validated vs unvalidated monitors under blinded adversarial conditions. |
| Open-source pipeline | 24 | End-to-end code: validity scoring, adversarial testing, monitor deployment. |

**Total budget: $750K over 24 months.**

| Category | Amount | Purpose |
|----------|--------|---------|
| Phase I compute + personnel | $150K | 172 metrics on 5 features, 1 postdoc |
| Phase II compute + personnel | $200K | Causal interventions on 27B models, adversarial ablation |
| Phase III adversarial + competition | $250K | GCG-style attacks on 27B models (compute-intensive), cross-model testing on Llama/Qwen ($50K), controlled competition exercise |
| External adversarial ML collaborator | $50K | Independent adversarial expertise for Phase III |
| Advisory board (2--3 external researchers) | $25K | Methodological review, pre-registration oversight |
| Cross-model compute (Llama 3.1 70B, Qwen 2.5 72B) | $50K | E6 testing beyond Gemma 2 |
| Project coordination + open-source release | $25K | Pipeline engineering, documentation |

The budget reflects that the framework infrastructure already exists. This proposal requests resources for application, not construction. The scale ($750K) signals confidence in the scope while remaining within the RFP's $300K--$1M range.


## 10. Broader Impact

There are two possible outcomes, and both advance the field.

If safety features pass the trial protocol, the field gains deployment-ready monitors with quantified reliability, specificity, causal grounding, and adversarial robustness. Each validity report card states exactly what the monitor can and cannot do: which deception types it detects (C3), how reliably (M1), whether intervening on it changes behavior (I2), and whether an adversary can evade it (E5). This is the information a deployment team needs to make a principled decision about trust.

If safety features fail, the validity report card states exactly which criteria fail. If refusal directions fail C3 (specificity), the field knows to invest in discriminant validity before deployment. If the safety subspace fails E6 (cross-architecture transfer), safety features are model-specific and must be re-identified per architecture. Precision about failure is as valuable as success.

The lasting contribution is the standard itself. The trial protocol defines what "validated" means for a safety feature. Features reaching "Triangulated" or above are deployment-ready. Features below need more evidence. This gives developers, auditors, and regulators a concrete vocabulary --- 27 criteria, 5 verdict tiers, explicit thresholds --- for discussing safety-feature readiness. The Phase I/II/III gating structure translates directly: "This safety feature has not passed Phase II trials" is a sentence a regulator can act on.

The framework is open-source, method-agnostic, and designed for adoption. It does not compete with groups developing new detection methods. It complements all of them. Every detection method benefits from knowing whether its outputs are valid. The validation layer makes all detection methods better --- or reveals which ones should not be trusted.

Schmidt's competition will answer a narrow question: which blue team wins? This proposal answers the structural question underneath it: what makes a winning blue team possible? The answer, we hypothesize, is validation. Validated monitors survive adversarial pressure because they target the model's causal mechanisms, not surface correlates. We will test this hypothesis with the most rigorous protocol the field has seen, publish the results regardless of outcome, and release the infrastructure for others to use.

---

*All experimental protocols will be pre-registered. Validity scoring uses the mechanistic validity framework (172 metrics, 27 criteria, 5 verdict tiers), applied to 13 published circuits from the MI literature. Adversarial evasion testing adapts established adversarial ML methods (GCG, transfer attacks, paraphrase attacks) to the MI safety monitoring setting. The framework, including all metric implementations and claim specification templates, will be released as open-source software.*

---

**References**

- Arditi, O., Obeso, B., et al. (2024). Refusal in language models is mediated by a single direction. arXiv:2406.11717.
- Bai, J., Baumgartner, P., Sun, J., Holtzman, A., & Tan, C. (2026). MechEvalAgent: Execution-grounded evaluation of mechanistic interpretability research. arXiv:2602.18458.
- Dauparas, J., et al. (2025). Sparse autoencoder features are not robust across training seeds. NeurIPS 2025. (DMSAE)
- Goodfire Research. (2026). Verified parameter decomposition: Adversarial ablation as gold-standard circuit verification. goodfire.ai/research.
- Nanda, N. (2025). Using MI for AI safety: What interpretability provides is mechanistic evidence, not certainty. LessWrong (curated).
- Orgad, H. & Barez, F., et al. (2026). Interpretability can be actionable. ICML 2026. arXiv:2605.11161.
- SAEBench Reliability Audit. (2026). arXiv:2605.18229.
- Sharma, M., et al. (2024). Towards understanding sycophancy in language models. ICLR 2024.
- Sharkey, L., et al. (2026). Open problems in mechanistic interpretability. 30-author survey.
- Steinhardt, J. (2026). The case for evaluating model behaviors. Personal blog / The Gradient.
- RelP (NeurIPS 2025). Attribution patching ground-truth correlation analysis.
