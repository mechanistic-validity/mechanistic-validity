# From Behavioral Taxonomy to Mechanistic Monitoring: Validated Deception Detection in LLMs

**Schmidt Sciences AI Interpretability RFP Response**

Proposal for $750K over 24 months. PI: Elliot Tower (Independent Researcher). Co-PI: Gagan Bansal (formal XAI, AAAI; SBIR experience). Collaborator: Ivan Googasian (University of Edinburgh, NLP). Advisors: Prof. David Jensen (UMass Amherst, AI/ML), Nathan [Last Name] (philosophy of science and formal epistemology).

---

## 1. We Already Know How Models Deceive. We Don't Know How to Detect It Mechanistically.

Over the past year, our group has conducted the most comprehensive behavioral characterization of deceptive failure modes in frontier LLMs to date. Across six studies, five frontier models (DeepSeek-R1, DeepSeek-V3.2, GPT-5.4, GPT-4.1, Llama-3.3-70B), and four application domains (mathematical reasoning, clinical medicine, code security, multi-domain planning), we have identified and quantified a taxonomy of behaviors in which models possess internal evidence of errors but fail to act on it faithfully.

**The Use-Mention Gap** (Tower, 2026a): Models can *describe* the corrective rule for an error but fail to *apply* it during generation. On GEB-Bench, reasoning-trained models (R1) nearly eliminate this gap, while standard models show systematic knowledge-behavior dissociation. This is not a knowledge deficit --- it is an activation deficit. The model has the information; it does not deploy it.

**Compositional Collapse** (Tower, 2026b): Models solve individual reasoning steps with high accuracy but collapse at composition. The Depth Wall --- a sharp accuracy threshold as a function of reasoning depth --- reveals that failure concentrates at assembly junctions, not at individual sub-tasks. Under complexity pressure, models systematically underperform relative to their demonstrated capabilities.

**Stance-Induced Suppression** (Tower, 2026c): Models asked to verify their own outputs over-confirm errors because their generation history crystallizes context, preventing neutral re-evaluation. Removing the generative trace restores near-baseline verification accuracy --- proving the failure is context-induced, not capability-limited.

**Sunk-Cost Lock-In** (Tower, 2026d; SunkBench): The most directly deceptive behavior. Once a model commits to wrong intermediate steps, it follows that trajectory to completion even when internal monitoring signals (self-correction tokens, entropy collapse, explicit contradiction mentions) indicate the computation has gone wrong. On SunkBench (1,440 tasks, 4 domains, 5 models), we identify three mechanistically distinct failure archetypes:
- *Internal over-committer* (R1): detects the error internally but locks in
- *Silent follower* (GPT-5.4): neither detects nor corrects
- *Surface detector* (GPT-4.1): mentions contradictions at ~100% yet still outputs wrong answers

**The Detection Framing Problem** (Tower, 2026e; clinical companion): Hallucination detectors in clinical settings fail not because the model lacks evidence but because binary framing ("is this correct?") suppresses detection. Asking the same model to "look more carefully" self-corrects 96.7% of missed hallucinations. Then: routing that high-recall frame through a yes/no confirmation step collapses recall back to baseline --- *Confirmation Hysteresis*. The monitor has the evidence and still fails, because of how the monitoring question is posed.

**Binary Framing in Code Security** (Tower, 2026f; vulnerability companion): The same stance-induced suppression pattern in code vulnerability detection. Binary prompting achieves F1 of 40--53 (chance). Diff-format triage achieves F1 88--100. Same models, same knowledge. The framing determines whether the capability deploys.

These six studies establish a consistent picture: **models systematically fail to act on evidence they possess**, and they do so in specific, characterizable, mechanistically distinct ways. This IS the deception monitoring problem. A model that detects its own contradiction and proceeds anyway (sunk-cost lock-in), a model that over-confirms its own errors (stance-induced suppression), a model that knows the rule but doesn't apply it (use-mention gap) --- these are the behaviors Schmidt's competition needs monitors for.

What we do not yet know is **what mechanistic features inside the model correspond to these behavioral failure modes**. The three SunkBench archetypes must have different internal computational signatures --- they produce different behavioral patterns and respond differently to interventions. Can mechanistic interpretability find those signatures? And if it can, should we trust the features it finds?

That is the question this proposal answers.

---

## 2. Why You Can't Trust Current Mechanistic Monitors

The mechanistic interpretability field has a measurement crisis that makes every claim about detecting deceptive behavior suspect.

**99.7% of SAE features are training artifacts** (DMSAE, NeurIPS 2025). Only 197 of 65,536 features survive across independent training runs. A "deception feature" identified using standard SAE methods has a 99.7% prior of being noise.

**93% of MI papers fail reproducibility** (MechEvalAgent, February 2026). Execution-grounded evaluation found that narrative peer review missed 51 issues.

**The default causal tool is near-random.** Attribution patching correlates at r = 0.006 with ground-truth causal importance (RelP, NeurIPS 2025).

**Standard ablation finds causally wrong nodes.** The VPD framework (Goodfire, May 2026) showed that adversarial ablation --- worst-case subset removal --- reveals most circuit claims identify surface correlates, not actual mechanisms.

The implication is direct: if someone publishes a "sycophancy detection feature" or a "deception probe" tomorrow, there is no principled way to know whether to trust it. It might be a training artifact (99.7% chance if SAE-derived). Its causal attribution might be random (r = 0.006). Its ablation evidence might not survive adversarial testing. And --- as our own Paper M demonstrates --- even a monitor with the right evidence can fail if the monitoring protocol itself induces suppression.

The field needs two things it currently lacks:
1. **Behavioral ground truth** for what deception looks like inside models (we have this --- GEB-Bench, SunkBench)
2. **A principled answer to "when should you trust a deception monitor?"** (we have this --- the Mechanistic Validity framework)

This proposal connects them.

---

## 3. The Validity Framework (Condensed)

The Mechanistic Validity framework provides 172 metrics across 5 validity types (construct, internal, external, measurement, interpretive), 27 criteria, and 5 verdict tiers from Proposed to Validated. It is method-agnostic: it evaluates SAE features, circuits, probing results, and steering vectors using the same criteria. Full documentation and metric implementations are available as open-source software (preprint and code: [site URL]; DOI: [Zenodo DOI]).

For this proposal, the key criteria are:

- **C1 Falsifiability**: Can you state what would disprove the deception feature?
- **C3 Specificity**: Does it fire on deception, or on uncertainty/hedging/politeness?
- **I2 Causal Sufficiency**: Does activating the feature *produce* deceptive behavior?
- **M1 Reliability**: Consistent across prompts, seeds, checkpoints?
- **E5 Robustness**: Survives adversarial prompt variation?

The framework has been applied to 13 published circuits with discriminating results (verdict tiers from Proposed to Triangulated). ICML 2026 ("Interpretability Can Be Actionable," Orgad, Barez et al.) and the 30-author Open Problems survey (Sharkey et al., January 2026) independently argue that evaluation criteria --- not new methods --- are the missing ingredient. Our framework operationalizes both positions.

---

## 4. Proposed Work: Connecting Behavioral Ground Truth to Mechanistic Monitors

### Phase I: Mechanistic Signatures of Deceptive Behaviors (Months 1--10, $300K)

**Goal.** Find the internal computational signatures that distinguish our three SunkBench archetypes and validate them using the framework.

We target Gemma 2 27B (open-weight, comprehensively instrumented with Gemma Scope 2 SAEs) and replicate key GEB-Bench and SunkBench experiments to establish behavioral ground truth in this model.

**Experiment 1: Activation cartography of deceptive episodes.** During sunk-cost lock-in episodes (model detects contradiction but proceeds), record full activation traces. Contrast with episodes where the model successfully self-corrects. Use probing, PCA, and SAE feature analysis to identify candidate "deception-in-progress" features --- directions that activate when the model has evidence of error but is not acting on it.

**Experiment 2: Stance-induced suppression signatures.** During confirmation hysteresis episodes (monitor has evidence but binary framing suppresses detection), record activation differences between the suppressed and non-suppressed conditions. The model's internal state should differ even when its outputs are identical --- this is the mechanistic footprint of stance-induced suppression.

**Experiment 3: Use-mention gap internals.** When a model can *describe* the corrective rule but fails to *apply* it, what happens inside? Compare activations during rule-mention (model states the rule correctly) vs. rule-use-failure (model fails to apply it). The knowledge is represented; something blocks deployment. Can we find the gate?

**Experiment 4: Validate candidate features.** Apply the full validity framework to every candidate deception feature identified in Experiments 1--3. Phase I of the trial protocol: measurement reliability (M1, bootstrap stability, seed variance). Phase II: causal testing (I1 necessity, I2 sufficiency, I3 specificity). A feature that fires during sunk-cost lock-in but also fires during genuine uncertainty (C3 failure) is not a deception monitor --- it's an uncertainty monitor with a misleading label.

**Deliverable.** Validity report cards for each candidate feature. The first systematic assessment of whether mechanistic features can distinguish deceptive behaviors from superficially similar non-deceptive ones.

### Phase II: Adversarial Robustness and Monitor Deployment (Months 11--20, $300K)

**Goal.** Test whether validated deception monitors survive adversarial pressure.

**Experiment 5: Adversarial evasion testing.** For each validated feature from Phase I, measure evasion rates under four attack families:
1. *Gradient-based optimization* (GCG-style): optimize inputs that produce deceptive behavior while minimizing feature activation.
2. *Paraphrase attacks*: express the same deceptive intent with different surface forms.
3. *Compositional attacks*: individually benign components that compose into deception.
4. *Cross-model transfer*: adversarial prompts optimized on Gemma 9B, tested on 27B.

**Experiment 6: The robustness hierarchy test.** Our central testable prediction: features that pass more validity criteria are harder to evade. We define five tiers by adversarial effort required:

| Verdict Tier | Adversarial Effort to Evade |
|---|---|
| Proposed (0--1 criteria) | Minutes (rephrasing) |
| Causally Suggestive (2--3) | Hours (identify the correlate) |
| Mechanistically Supported (4--6) | Days (gradient optimization) |
| Triangulated (7--9) | Weeks (model-level attacks) |
| Validated (10+) | Requires a different model |

Regression of evasion rate on validity score, controlling for attack type. The interaction term tests whether validity matters more for sophisticated attacks.

**Experiment 7: Competition-ready deployment.** Two blue teams monitor Gemma 2 27B for deceptive behaviors during GEB-Bench and SunkBench episodes. One uses validated features (Triangulated+); the other uses unvalidated features (Proposed/Causally Suggestive). A red team crafts evasive inputs. Detection rates compared. This directly instantiates Schmidt's proposed competition structure.

**Deliverable.** Adversarial robustness scores for each deception monitor, stratified by validity tier. Empirical test of whether validation predicts adversarial resilience.

### Phase III: Cross-Model Generalization and Open-Source Release (Months 21--24, $150K)

**Goal.** Test whether deception features transfer across architectures and release everything.

**Experiment 8: Cross-architecture transfer.** Do the deceptive behavior signatures found in Gemma 2 27B exist in Llama 3.1 70B? The three SunkBench archetypes appear in different models (R1 = over-committer, GPT-5.4 = silent follower, GPT-4.1 = surface detector). If the mechanistic signatures transfer, we have evidence for universal deception mechanisms. If they don't, we learn that monitoring must be model-specific.

**Deliverables:**
- Validity report cards for all deception features
- Adversarial evasion scores per feature per attack
- Correlation analysis (validity vs. robustness), published regardless of outcome
- GEB-Bench and SunkBench as open-source behavioral benchmarks
- The full validity framework as open-source infrastructure
- Competition results from the blue team exercise

---

## 5. Why This Team, Why This Proposal

**What we bring that others don't:**
1. *Behavioral ground truth.* GEB-Bench and SunkBench provide labeled episodes of deceptive behavior across 5 frontier models and 4 domains. No other group has systematic behavioral benchmarks for LLM deception that can serve as ground truth for mechanistic feature validation.
2. *A validation methodology.* 172 metrics, 27 criteria, 5 verdict tiers, 13 worked case studies. The framework is built, tested, and open-source.
3. *The connection.* This proposal asks a question only someone with both behavioral benchmarks AND a validation framework can ask: "Can mechanistic monitors detect the deceptive behaviors we've already characterized? And when should you trust them?"

**Team.**
- **Elliot Tower** (PI): Author of the six-paper deception taxonomy, developer of the Mechanistic Validity framework. Frontier model behavioral evaluation across clinical, code, and reasoning domains.
- **Gagan Bansal** (Co-PI): Formal explainable AI (AAAI), SBIR experience in applied AI safety.
- **Ivan Googasian** (Collaborator, University of Edinburgh): NLP, co-organizer of mechanistic interpretability workshops.
- **Prof. David Jensen** (Advisor, UMass Amherst): Faculty advisor, AI/ML research and evaluation methodology.
- **Nathan [Last Name]** (Advisor, Philosophy of Science): Formal epistemology and construct validity --- the philosophical foundations underpinning the measurement-theoretic approach.

**Feasibility.** The behavioral benchmarks exist. The validity framework exists. The proposal requests resources for the mechanistic bridge between them: GPU compute for activation analysis on 27B-parameter models, adversarial optimization, and cross-architecture testing. The primary risk is whether deceptive behaviors produce detectable mechanistic signatures at all. If they do, we validate them. If they don't, that null result tells the field something important: behavioral monitoring may be necessary because mechanistic monitoring is insufficient for deception. Either outcome advances the field.

---

## 6. Schmidt Alignment

**"Detect deceptive behaviors."** Phase I produces mechanistic features for six characterized deceptive behaviors, validated against behavioral ground truth.

**"Mitigate deceptive behaviors."** Validated causal features (I2) are steering targets. If activating a feature produces sunk-cost lock-in, subtracting it should reduce it. Phase II tests this.

**"Move beyond academic benchmarks."** GEB-Bench and SunkBench are adversarial behavioral benchmarks tested on frontier models, not fixed prompt sets on GPT-2. Phase II adds adversarial evasion testing --- the hardest possible evaluation.

**"Leverage model internals to outperform black-box baselines."** Our Paper M shows that black-box monitoring (binary framing) fails due to confirmation hysteresis. Mechanistic monitors should bypass this failure mode by reading internal state directly. Phase I tests whether they do.

**The blinded competition.** Phase II directly instantiates Schmidt's red-team vs. blue-team design, with the prediction that validated monitors outperform unvalidated ones.

---

**Budget: $750K over 24 months.**

| Category | Amount |
|---|---|
| Phase I compute + personnel (activation analysis, SAE runs on 27B) | $300K |
| Phase II compute (adversarial optimization, competition exercise) | $300K |
| Phase III cross-model compute (Llama 70B) + open-source release | $100K |
| Advisory board + external adversarial ML collaborator | $50K |

---

*All behavioral benchmarks (GEB-Bench, SunkBench) and the validity framework (172 metrics, 27 criteria) are available as open-source software. Preprints for all six studies are available at [site URL] with DOIs via Zenodo. All experimental protocols will be pre-registered. Results --- including null results --- will be published with full data and code.*
