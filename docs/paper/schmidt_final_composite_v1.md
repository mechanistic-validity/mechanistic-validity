# When Should You Trust a Deception Detector? Validated Whitebox Monitoring for LLM Deceptive Behaviors

**Schmidt Sciences AI Interpretability RFP Response**

Proposal for $750K over 24 months.

PI: Elliot Tower. Co-PI: Gagan Bansal. Collaborator: Ivan Vegner (University of Edinburgh CDT in NLP, supervised by Prof. Alex Doumas and Dr. Siddharth Narayanaswamy).

Fiscal sponsorship: Ashgro (501(c)(3) fiscal sponsor for AI safety research). Compute: RunPod GPU instances + NSF National Deep Inference Fabric (NDIF, Northeastern University) for large-model transparent inference.

---

## 1. The Problem: No One Knows When to Trust a Deception Monitor

Suppose a mechanistic interpretability team hands you a "deception detector" --- a probe, circuit, or SAE feature that fires when a language model is being deceptive. Should you trust it?

Today, there is no principled way to answer this question. You cannot look up the detector's reliability, its false positive rate against neighboring constructs, its causal relationship to the behavior it claims to detect, or whether it survives adversarial evasion. You can only read the paper and hope the evaluation was thorough.

This is not a hypothetical concern. The tools used to build deception monitors are in crisis:

- 99.7% of SAE features are training artifacts, not genuine representations (DMSAE, NeurIPS 2025)
- Attribution patching --- the default causal tool --- correlates at r = 0.006 with ground-truth causal importance (RelP, NeurIPS 2025)
- 93% of MI papers fail execution-grounded reproducibility checks (MechEvalAgent, Feb 2026)
- Most circuit claims identify causally wrong nodes under adversarial ablation (VPD, Goodfire/Apollo, May 2026)

But the problem goes deeper than unreliable tools. Even a monitor built on a genuine feature can fail if the monitoring protocol itself is flawed. We have demonstrated this empirically.


## 2. We Have Characterized How Models Deceive --- and How Monitors Fail

Over the past year, our group has conducted the most comprehensive behavioral characterization of deceptive failure modes in frontier LLMs to date. Across six studies, five frontier models (DeepSeek-R1, DeepSeek-V3.2, GPT-5.4, GPT-4.1, Llama-3.3-70B), and four application domains (mathematical reasoning, clinical medicine, code security, multi-domain planning), we have identified and quantified a taxonomy of behaviors in which models possess internal evidence of errors but fail to act on it faithfully.

**The Sunk-Cost Lock-In** (Tower, 2026d; SunkBench). The most directly deceptive behavior. Once a model commits to wrong intermediate steps, it follows that trajectory to completion even when internal monitoring signals --- self-correction tokens, entropy collapse, explicit contradiction mentions --- indicate the computation has gone wrong. On SunkBench (1,440 tasks, 4 domains, 5 models), we identify three mechanistically distinct failure archetypes:

| Archetype | Internal detection? | Output correction? | Example model |
|---|---|---|---|
| Internal over-committer | Yes (thinking trace) | No | DeepSeek-R1 |
| Silent follower | No | No | GPT-5.4 |
| Surface detector | Yes (output mentions contradiction at ~100%) | No | GPT-4.1 |

The "surface detector" case is the canonical deception monitoring nightmare: the model's output *contains evidence of its own deception*, but it still produces the wrong answer. Detection without resolution is not safety.

**The Detection Framing Problem** (Tower, 2026e; clinical companion). Hallucination detectors in clinical settings fail not because the model lacks evidence but because binary framing ("is this correct?") suppresses detection. Asking the same model to "look more carefully" --- with no new information --- self-corrects 96.7% of missed hallucinations (29/30). Then: routing that high-recall frame through a yes/no confirmation step collapses recall back to baseline. Every gain eliminated. We call this *Confirmation Hysteresis* --- the monitor oscillates between "can detect" and "can't detect" based purely on how the question is asked. This is not a capability failure. It is a protocol failure. And it demonstrates that **black-box behavioral monitoring has structural failure modes that whitebox mechanistic monitoring should be able to bypass** --- by reading internal state directly rather than through a question frame.

**Stance-Induced Suppression** (Tower, 2026c). Models verify third-party outputs far more accurately than their own. Self-generated context crystallizes in the KV-cache, creating an attentional prior that blocks neutral re-evaluation. Removing the generative trace (context surgery) restores near-baseline accuracy. The suppression is purely context-induced, not capability-limited.

**The Use-Mention Gap** (Tower, 2026a; GEB-Bench). Models can *describe* the corrective rule for an error but fail to *apply* it during generation. This is not a knowledge deficit --- it is an activation deficit. The model has the information; it does not deploy it. Reasoning-trained models (R1) nearly eliminate this gap; standard models show systematic knowledge-behavior dissociation.

**Compositional Collapse** (Tower, 2026b). Models solve individual reasoning steps with high accuracy but collapse at composition. The Depth Wall --- a sharp accuracy threshold as a function of reasoning depth --- reveals that failure concentrates at assembly junctions. Under complexity pressure, models systematically underperform relative to their demonstrated capabilities.

**Binary Framing in Code Security** (Tower, 2026f). The same stance-induced suppression pattern in vulnerability detection. Binary prompting achieves F1 of 40--53 (indistinguishable from chance). Diff-format triage achieves F1 88--100. Same models, same knowledge. Domain generality confirmed.

These six studies establish a recurring pattern of **representation-behavior dissociation** (Vegner et al., ACL 2025): models contain or can access the information needed for correct evaluation, yet fail to deploy it under the specific framing, context, or trajectory conditions present at inference time. This IS the deception monitoring problem. And crucially, we have **quantitative behavioral ground truth** --- GEB-Bench and SunkBench provide labeled episodes of deceptive behavior across frontier models. This ground truth is what makes mechanistic monitor validation possible.

What we do not yet know is what mechanistic features inside the model correspond to these representation-behavior dissociations. The three SunkBench archetypes must have different internal computational signatures. Can mechanistic interpretability find them? And if it can --- should we trust the features it finds?


## 3. The Validity Framework: Answering "When Should You Trust It?"

We have built a measurement-theoretic validity framework that answers this question with specific, testable criteria. The framework provides 172 metrics across 5 validity types (construct, internal, external, measurement, interpretive), organized into 27 criteria and 5 verdict tiers from Proposed to Validated. It has been applied to 13 published circuits from the MI literature with discriminating results. Full documentation, metric implementations, and case studies are available as open-source software (preprint: [site URL]; DOI: [Zenodo DOI]).

For deception monitoring, the key criteria are:

- **C3 Specificity**: Does the feature fire on deception, or on uncertainty? A "deception detector" indistinguishable from an "uncertainty detector" is mislabeled.
- **I2 Causal Sufficiency**: Does activating the feature *produce* deceptive behavior? Correlation during deception is the floor, not the ceiling.
- **I3 Discriminant Validity**: Can the monitor distinguish the three SunkBench archetypes? If it can't tell an "internal over-committer" from a "silent follower," it is detecting a surface correlate, not the mechanism.
- **M1 Reliability**: Same result across prompts, seeds, and checkpoints?
- **E5 Adversarial Robustness**: Does the feature survive when an adversary optimizes against it?

These criteria define a robustness hierarchy --- our central testable prediction:

| Verdict Tier | Criteria Passed | Adversarial Effort to Evade |
|---|---|---|
| **Proposed** | 0--1 | Minutes (rephrasing) |
| **Causally Suggestive** | 2--3 | Hours--days (identify the correlate) |
| **Mechanistically Supported** | 4--6 | Days--weeks (gradient optimization) |
| **Triangulated** | 7--9 | Weeks (model-level attacks) |
| **Validated** | 10+ | Requires building a different model |

A Triangulated monitor survives adversarial pressure because it targets the model's actual computational mechanism, not a surface correlate. We test this directly.

ICML 2026 ("Interpretability Can Be Actionable," Orgad, Barez et al.) and the 30-author Open Problems survey (Sharkey et al., 2026) independently argue that evaluation criteria --- not new methods --- are the missing ingredient. Jacob Steinhardt (May 2026) calls for behavioral evaluation over capability evaluation. Our framework operationalizes all three positions, and our six behavioral studies are exactly the kind of "measuring what models do" that Steinhardt advocates.


## 4. Proposed Experiments

### Phase I: Find and Validate Deception Features (Months 1--10, $300K)

We replicate key GEB-Bench and SunkBench experiments in Gemma 2 27B (open-weight, Gemma Scope 2 SAE instrumentation) to establish behavioral ground truth, then search for mechanistic correlates. Primary experiments use Gemma 2 27B; Llama 3.1 70B serves as fallback if the target behaviors are insufficiently exhibited in Gemma 2.

**4.1 Activation cartography of deceptive episodes.** During sunk-cost lock-in (model detects contradiction but proceeds), record full activation traces. Contrast with successful self-correction episodes. Use probing, PCA, and SAE feature analysis to identify candidate "deception-in-progress" features --- directions that activate when the model has evidence of error but is not acting on it.

**4.2 Whitebox vs. blackbox monitoring.** Test whether mechanistic features bypass confirmation hysteresis. During episodes where binary-framed behavioral monitoring fails to detect a hallucination (our Paper M result), does the mechanistic feature still fire? This directly quantifies the information advantage of internal methods over external methods --- the core premise of using model internals for deception monitoring.

**4.3 Stance-induced suppression internals.** During confirmation hysteresis episodes, record activation differences between suppressed and non-suppressed conditions. The model's internal state must differ --- the information is present in one framing and blocked in another. Can we find the gate?

**4.4 Archetype discrimination.** The three SunkBench archetypes must have different mechanistic signatures. Validate that candidate features can discriminate them. A feature that fires for all three is detecting "reasoning failure" generically; a feature that distinguishes them is detecting the *mechanism* of deception.

**4.5 Full validity assessment.** Apply the framework to every candidate feature: measurement reliability (M1, bootstrap stability, seed variance), causal testing (I1 necessity, I2 sufficiency, I3 specificity), construct validity (C3 discriminant testing against uncertainty, hedging, politeness). Assign verdict tiers.

**Deliverable.** Validity report cards for each candidate deception feature --- the first systematic validity assessment of any deception-related mechanistic feature.

### Phase II: Adversarial Testing and Competition (Months 11--20, $300K)

**4.6 Adversarial evasion testing.** For each validated feature, measure evasion rates under four attack families: gradient-based optimization (GCG-style), paraphrase attacks, compositional attacks, and cross-model transfer (optimized on Gemma 9B, tested on 27B).

**4.7 The robustness hierarchy test.** Regression of evasion rate on per-criterion validity scores (27 criteria × N features, avoiding the underpowered aggregate-tier analysis), controlling for attack type. Permutation-based testing. The interaction term tests whether validity matters more for sophisticated attacks (gradient, compositional) than simple ones (paraphrase). This is the core scientific test: does validation predict adversarial robustness?

**4.8 Competition deployment.** Two blue teams monitor the same model during GEB-Bench and SunkBench episodes. One uses Triangulated+ features; the other uses Proposed/Causally Suggestive features. A red team crafts evasive inputs. Detection rates compared. Prediction: validated monitors detect more and false-alarm less, with the gap widening under adversarial pressure. This directly instantiates Schmidt's proposed competition design.

**4.9 Steering test.** Do causally sufficient (I2) deception features make better intervention targets? If activating a "sunk-cost lock-in" feature produces lock-in behavior, subtracting it should reduce it. We test whether validity predicts steering effectiveness --- connecting monitoring to mitigation through a principled selection criterion.

**Deliverable.** Adversarial robustness scores stratified by validity tier. Empirical evidence for or against the hypothesis that validation predicts adversarial resilience.

### Phase III: Generalization and Release (Months 21--24, $150K)

**4.10 Cross-architecture transfer.** Do deception features from Gemma 2 27B exist in Llama 3.1 70B? The behavioral archetypes appear in different models (R1 = over-committer, GPT-5.4 = silent follower, GPT-4.1 = surface detector). If the mechanistic signatures transfer, we have evidence for universal deception mechanisms. If they don't, we learn that monitoring must be model-specific --- itself a high-value finding for deployment.

**4.11 Open-source release.** All benchmarks (GEB-Bench, SunkBench), the full validity framework (172 metrics), all validity report cards, adversarial evasion data, the competition protocol, and pre-registered analysis code. Results published regardless of outcome.


## 5. Three Possible Outcomes (All Informative)

**Outcome 1: Mechanistic features detect deceptive behaviors, and validated ones resist adversarial evasion.** Implication: whitebox monitoring works, and validation is the selection criterion. Deploy validated monitors; distrust unvalidated ones. The validity framework becomes a standard for deception monitor certification.

**Outcome 2: Mechanistic features detect deceptive behaviors, but validity does not predict adversarial robustness.** Implication: validation and adversarial robustness test different properties. The field needs both. We publish which criteria matter for what, enabling targeted investment.

**Outcome 3: Mechanistic features cannot reliably distinguish deceptive episodes from non-deceptive ones.** Implication: behavioral monitoring (our GEB-Bench/SunkBench approach) may be necessary because representation-behavior dissociations do not leave mechanistically detectable traces --- at least not with current MI tools. This null result would be among the most important possible findings for the field, because it redirects the research agenda from mechanistic detection to behavioral detection, with quantified evidence for why.

We commit to publishing whichever outcome we find, with full data and analysis.


## 6. Why This Team, Why This Proposal

**What we bring that no other group has:**

1. *Behavioral ground truth for deception.* Six studies, five frontier models, four domains, two quantitative benchmarks (GEB-Bench, SunkBench). We know what deception looks like behaviorally. No other group has systematic labeled benchmarks for LLM deception that can serve as ground truth for mechanistic feature validation.

2. *A validation methodology.* 172 metrics, 27 criteria, 5 verdict tiers, 13 worked case studies. Built, tested, open-source. We know how to evaluate whether a mechanistic claim is trustworthy.

3. *The connection.* This proposal asks a question only someone with both (1) and (2) can ask: "Can mechanistic monitors detect the deceptive behaviors we've already characterized? And when should you trust them?"

**Team.**

- **Elliot Tower** (PI): Author of the six-paper deception taxonomy across frontier models (GEB-Bench, SunkBench). Developer of the Mechanistic Validity framework (172 metrics, 13 case studies). Active in the mechanistic interpretability research community; presenting at the ICML 2026 Workshop on Interpretability (Seoul), the Machine Consciousness Conference (Berkeley), and the New England Mechanistic Interpretability meetup series.

- **Gagan Bansal** (Co-PI): Formal explainable AI (AAAI), SBIR experience in applied AI safety. Brings formal XAI methodology and government-funded research project management. Invited participant at the Machine Consciousness Conference (2025).

- **Ivan Vegner** (Collaborator, University of Edinburgh CDT in NLP; supervised by Prof. Alex Doumas and Dr. Siddharth Narayanaswamy): ACL 2025 work on behavioral vs. representational systematicity --- the distinction between a model behaving consistently and having the right internal representations. This maps directly onto the proposal's core question: does behavioral deception correspond to a specific internal representation? His expertise in systematic generalization and structured representation connects cognitive science to our measurement-theoretic approach.

**Advisory board.** The project will be advised by Prof. Alex Doumas (University of Edinburgh; computational cognitive science, analogical reasoning) and Dr. Siddharth Narayanaswamy (University of Edinburgh; structured latent representations, generative models), who supervise Vegner's doctoral work and bring expertise in the representation-behavior questions central to this proposal. We are additionally in discussion with Prof. David Jensen (UMass Amherst CICS; AI/ML evaluation methodology) as a senior advisor; pending funding, Jensen's group would host the PI as a visiting researcher at UMass.

**Institutional plan.** Fiscal sponsorship through Ashgro (501(c)(3) fiscal sponsor for AI safety research). Compute via RunPod GPU instances for fine-tuning and controlled experiments, and the NSF National Deep Inference Fabric (NDIF, Northeastern University) for large-model transparent inference. Advisory governance through Doumas, Narayanaswamy, and Jensen. Pending funding, the PI will seek visiting researcher affiliation at UMass Amherst CICS or the University of Edinburgh School of Informatics.


## 7. Schmidt Alignment

**"Detect deceptive behaviors."** Phase I finds mechanistic signatures of six characterized deceptive behaviors, validated against behavioral ground truth. No other proposal will have labeled deception episodes to validate against.

**"Mitigate deceptive behaviors."** Phase II tests validated features as steering targets. If activating a "sunk-cost lock-in" feature produces lock-in, subtracting it should reduce it.

**"Move beyond academic benchmarks."** GEB-Bench and SunkBench test on frontier models (R1, GPT-5.4, GPT-4.1, Llama-3.3-70B), not GPT-2 small. Phase II adds adversarial evasion testing --- the hardest evaluation the field has.

**"Leverage model internals to outperform black-box baselines."** Our Paper M demonstrates that black-box monitoring fails (confirmation hysteresis). Experiment 4.2 directly tests whether whitebox monitoring bypasses this failure mode.

**The blinded competition.** Phase II instantiates Schmidt's competition design. Prediction: validated monitors outperform unvalidated ones, and the gap is proportional to validity tier.

---

**Budget: $750K over 24 months.**

| Category | Amount | Purpose |
|---|---|---|
| Phase I compute + personnel | $300K | Activation analysis on 27B models, SAE runs, behavioral replication, 1 postdoc |
| Phase II compute + personnel | $300K | Adversarial optimization (GCG on 27B is compute-intensive), competition exercise, cross-model testing |
| Phase III compute + release | $100K | Llama 70B cross-architecture experiments, open-source pipeline engineering |
| External adversarial ML collaborator | $25K | Independent adversarial expertise for Phase II evasion testing |
| Advisory board | $25K | Methodological review, pre-registration oversight |

---

*Preprints for all six behavioral studies are available at [site URL] (DOIs via Zenodo). The Mechanistic Validity framework (172 metrics, 27 criteria, 13 case studies) is available as open-source software. All experimental protocols will be pre-registered. All results --- including null results --- will be published with full data and code.*

**References**

- Tower, E. (2026a). The Use-Mention Gap: Knowledge activation failure in single-utterance self-correction. [preprint]
- Tower, E. (2026b). Compositional Collapse: The Depth Wall in multi-step reasoning assembly. [preprint]
- Tower, E. (2026c). Stance-Induced Suppression: Context Lock-Out in self-evaluation. [preprint]
- Tower, E. (2026d). The Sunk-Cost Lock-In: Sunk-Token Depth, Latent Uncertainty, and Generative Entrenchment in LLMs. [preprint / DOI]
- Tower, E. (2026e). The Detection Framing Problem: Stance-Induced Suppression and Confirmation Hysteresis in clinical LLMs. [preprint]
- Tower, E. (2026f). The Wrong Question in Code Security: Binary framing suppresses LLM vulnerability detection. [preprint]
- Dauparas, J., et al. (2025). Sparse autoencoder features are not robust across training seeds. NeurIPS 2025.
- Bai, J., et al. (2026). MechEvalAgent: Execution-grounded evaluation of mechanistic interpretability research. arXiv:2602.18458.
- Goodfire Research. (2026). Verified Parameter Decomposition. goodfire.ai/research.
- Orgad, H. & Barez, F., et al. (2026). Interpretability can be actionable. ICML 2026. arXiv:2605.11161.
- Sharkey, L., et al. (2026). Open problems in mechanistic interpretability. 30-author survey.
- Steinhardt, J. (2026). The case for evaluating model behaviors. The Gradient.
- RelP (NeurIPS 2025). Attribution patching ground-truth correlation analysis.
- Vegner, I., et al. (2025). Behavioural vs. representational systematicity. ACL 2025.
