# When Should You Trust a Deception Detector?

**Schmidt Sciences AI Interpretability RFP Response**

Proposal for $[AMOUNT — choose $500K or $750K] over 24 months.

PI: Elliot Tower. Co-PI: Gagan Bansal. Collaborator: Ivan Vegner (University of Edinburgh CDT in NLP, supervised by Prof. Alex Doumas and Dr. Siddharth Narayanaswamy).

Fiscal sponsorship: Ashgro (501(c)(3) fiscal sponsor for AI safety research). Compute: RunPod GPU instances and the NSF National Deep Inference Fabric (NDIF, Northeastern University).

---

## 1. The Problem

Suppose a mechanistic interpretability team hands you a "deception detector," a probe or SAE feature that fires when a language model is being deceptive. Should you trust it?

Right now there is no principled way to answer that question. You cannot look up the detector's reliability, its false positive rate against neighboring constructs, whether it has a causal relationship to the behavior it claims to detect, or whether it survives adversarial evasion. You read the paper and hope the evaluation was thorough.

This matters because the tools used to build such monitors are in serious trouble:

- 99.7% of SAE features are training artifacts, not genuine representations (DMSAE, NeurIPS 2025).
- Attribution patching, the default causal tool, correlates at r = 0.006 with ground-truth causal importance (RelP, NeurIPS 2025).
- 93% of MI papers fail execution-grounded reproducibility checks (MechEvalAgent, Feb 2026).
- Most circuit claims identify causally wrong nodes under adversarial ablation (VPD, Goodfire/Apollo, May 2026).

But the problem goes deeper than unreliable tools. Even a monitor built on a genuine feature can fail if the monitoring protocol itself is flawed. We have demonstrated this empirically across multiple domains.


## 2. How Models Deceive, and How Monitors Fail

Over the past year we have studied deceptive failure modes across five frontier models (DeepSeek-R1, DeepSeek-V3.2, GPT-5.4, GPT-4.1, Llama-3.3-70B) and four application domains (mathematical reasoning, clinical medicine, code security, multi-domain planning). The consistent finding: models often possess the internal evidence needed to detect their own errors, but fail to act on it faithfully.

**Trajectory Revision Failure** (Tower, 2026d; RBD-Bench D4). Once a model commits to wrong intermediate steps, it follows that trajectory to completion even when its own monitoring signals indicate the computation has gone wrong. On RBD-Bench's trajectory-revision axis (1,440 tasks, 4 domains, 5 models) we find three distinct failure archetypes:

| Archetype | Internal detection? | Output correction? | Example model |
|---|---|---|---|
| Internal over-committer | Yes (thinking trace) | No | DeepSeek-R1 |
| Silent follower | No | No | GPT-5.4 |
| Surface detector | Yes (output mentions the contradiction) | No | GPT-4.1 |

The "surface detector" case is especially concerning for safety: the model's output contains evidence of its own deception (it mentions the contradiction at roughly 100% of the time), yet still produces the wrong answer.

**The Detection Framing Problem** (Tower, 2026e). Hallucination detectors in clinical settings fail not because the model lacks evidence but because binary framing ("is this correct?") suppresses detection. Asking the same model to "look more carefully," with no new information, recovers 96.7% of missed hallucinations. But routing that recovered signal back through a yes/no confirmation step collapses recall to the original baseline, eliminating every gain. We call this Confirmation Hysteresis. It demonstrates that black-box behavioral monitoring has structural failure modes that whitebox mechanistic monitoring should, in principle, be able to bypass by reading internal state directly.

**Stance-Induced Suppression** (Tower, 2026c). Models verify third-party outputs far more accurately than their own. Self-generated context crystallizes in the KV-cache, creating an attentional prior that blocks neutral re-evaluation. Removing the generative trace restores near-baseline accuracy. The suppression is context-induced, not a capability limit.

**The Use-Mention Gap** (Tower, 2026a; RBD-Bench D1). Models can describe the corrective rule for an error but fail to apply it during generation. The model has the information; it does not deploy it. Reasoning-trained models like R1 nearly eliminate this gap, while standard models show systematic dissociation between knowledge and behavior.

**Compositional Collapse** (Tower, 2026b). Models solve individual reasoning steps accurately but collapse when those steps must be composed. The Depth Wall, a sharp accuracy threshold as reasoning depth increases, shows failure concentrating at assembly junctions rather than individual sub-tasks.

**Binary Framing in Code Security** (Tower, 2026f). The same stance-induced suppression pattern in vulnerability detection. Binary prompting achieves F1 of 40 to 53 (chance level). Diff-format triage achieves F1 of 88 to 100. Same models, same underlying knowledge.

Across these six studies we see a recurring pattern of **representation-behavior dissociation** (Vegner et al., ACL 2025): models contain or can access the information needed for correct evaluation, yet fail to deploy it under the specific framing, context, or trajectory conditions present at inference time. This is the deception monitoring problem stated concretely.

We also have something that most work on deception monitoring lacks: quantitative behavioral ground truth. RBD-Bench provides labeled episodes of deceptive behavior across frontier models, organized into six axes of representation-behavior dissociation. Knowing when a model is actually being deceptive is what makes it possible to validate a mechanistic monitor against something real.

What we do not yet know is what mechanistic features inside the model correspond to these dissociations. The three trajectory-revision archetypes produce different behavioral patterns and respond differently to interventions; they likely have different internal computational signatures. Can mechanistic interpretability find them? And if it can, should we trust the features it finds?


## 3. A Validity Framework for Deception Monitors

We have built a validity framework that provides specific, testable criteria for answering the trust question. It includes 172 metrics across five validity types (construct, internal, external, measurement, interpretive), organized into 27 criteria and five verdict tiers from Proposed to Validated. We have applied it to 13 published circuits from the MI literature with discriminating results. Documentation, metric implementations, and case studies are available as open-source software (preprint: https://mechanistic-validity.github.io/mechanistic-validity/; DOI: [Zenodo DOI]).

For deception monitoring, the most important criteria are:

- **C3 Specificity.** Does the feature fire on deception specifically, or also on uncertainty, hedging, or politeness? A deception detector that cannot be distinguished from an uncertainty detector is mislabeled.
- **I2 Causal Sufficiency.** Does activating the feature produce deceptive behavior? Correlating with deception during passive observation is the floor, not the ceiling.
- **I3 Discriminant Validity.** Can the monitor distinguish the three trajectory-revision archetypes? If it cannot tell an internal over-committer from a silent follower, it is detecting a surface correlate rather than the mechanism.
- **M1 Reliability.** Does the detector give the same result across different prompts, random seeds, and model checkpoints?
- **E5 Adversarial Robustness.** Does the feature survive when an adversary optimizes inputs against it?

These criteria define a robustness hierarchy, which is our central testable prediction:

| Verdict Tier | Criteria Passed | Adversarial Effort to Evade |
|---|---|---|
| Proposed | 0 to 1 | Minutes (rephrasing) |
| Causally Suggestive | 2 to 3 | Hours to days |
| Mechanistically Supported | 4 to 6 | Days to weeks (gradient optimization) |
| Triangulated | 7 to 9 | Weeks (model-level attacks) |
| Validated | 10+ | Requires building a different model |

The idea is that a Triangulated monitor survives adversarial pressure because it targets the model's actual computational mechanism rather than a surface correlate an adversary can route around. Phase II of the proposed work tests this directly.

ICML 2026 ("Interpretability Can Be Actionable," Orgad, Barez et al.) and the 30-author Open Problems survey (Sharkey et al., 2026) independently argue that evaluation criteria, not new methods, are the missing ingredient. Steinhardt (May 2026) calls for behavioral evaluation over capability evaluation. Our framework and our six behavioral studies are attempts to operationalize those positions.


## 4. Proposed Experiments

### Phase I: Find and Validate Deception Features (Months 1 to 10)

We replicate key RBD-Bench experiments in Gemma 2 27B (open-weight, instrumented with Gemma Scope 2 SAEs) to establish behavioral ground truth, then search for mechanistic correlates. If Gemma 2 27B does not exhibit the target behaviors sufficiently, Llama 3.1 70B serves as a fallback since it is open-weight and closer in scale to the models where these behaviors were originally observed.

**4.1 Activation cartography of deceptive episodes.** During sunk-cost lock-in episodes where the model detects a contradiction but proceeds anyway, record full activation traces. Contrast with episodes where the model successfully self-corrects. Use probing, PCA, and SAE feature analysis to identify candidate features that activate when the model has evidence of error but is not acting on it.

**4.2 Whitebox vs. blackbox monitoring.** Test whether mechanistic features bypass confirmation hysteresis. During episodes where binary-framed behavioral monitoring fails to detect a hallucination (our clinical companion paper result), does the mechanistic feature still fire? This directly quantifies whether reading internal state provides an information advantage over asking the model a question.

**4.3 Stance-induced suppression internals.** During confirmation hysteresis episodes, record activation differences between the suppressed and non-suppressed conditions. The model's internal state should differ even when its outputs are identical. Can we find the mechanism that gates access to the evidence?

**4.4 Archetype discrimination.** Validate that candidate features can discriminate the three trajectory-revision archetypes. A feature that fires equally for all three is detecting reasoning failure generically. A feature that distinguishes them is detecting the specific mechanism.

**4.5 Full validity assessment.** Apply the framework to every candidate feature: measurement reliability, causal testing (necessity, sufficiency, specificity), construct validity including discriminant testing against uncertainty, hedging, and politeness. Assign verdict tiers.

**Deliverable.** Validity report cards for each candidate deception feature.

### Phase II: Adversarial Testing and Competition (Months 11 to 20)

**4.6 Adversarial evasion testing.** For each validated feature, measure evasion rates under four attack families: gradient-based optimization (GCG-style), paraphrase attacks, compositional attacks, and cross-model transfer.

**4.7 The robustness hierarchy test.** Regression of evasion rate on per-criterion validity scores (27 criteria by N features), controlling for attack type, with permutation-based testing. The interaction term tests whether validity matters more for sophisticated attacks than simple ones. This is the core scientific test.

**4.8 Competition deployment.** Two blue teams monitor the same model during RBD-Bench episodes. One uses features at the Triangulated tier or above; the other uses Proposed or Causally Suggestive features. A separate team crafts evasive inputs. Detection rates compared. This instantiates the competition structure described in the RFP.

**4.9 Steering test.** Do causally sufficient deception features make better intervention targets? If activating a sunk-cost lock-in feature produces the lock-in behavior, subtracting it should reduce it. We test whether validity predicts steering effectiveness, connecting monitoring to mitigation.

**Deliverable.** Adversarial robustness scores stratified by validity tier.

### Phase III: Generalization and Release (Months 21 to 24)

**4.10 Cross-architecture transfer.** Do deception features from Gemma 2 27B exist in Llama 3.1 70B? The behavioral archetypes appear across models; whether the mechanistic substrate is shared or model-specific is itself a key finding for deployment.

**4.11 Open-source release.** All benchmarks, the validity framework, all report cards, adversarial evasion data, the competition protocol, and pre-registered analysis code. Results published regardless of outcome.


## 5. Three Possible Outcomes

**Outcome 1.** Mechanistic features detect deceptive behaviors, and validated ones resist adversarial evasion. Implication: whitebox monitoring works, and the validity framework provides the selection criterion.

**Outcome 2.** Mechanistic features detect deceptive behaviors, but validity does not predict adversarial robustness. Implication: validation and adversarial testing measure different things. The field needs both. We publish which criteria matter for which threat models.

**Outcome 3.** Mechanistic features cannot reliably distinguish deceptive episodes from non-deceptive ones. Implication: representation-behavior dissociations may not leave mechanistically detectable traces with current MI tools. This would redirect the field from mechanistic to behavioral detection, with quantified evidence for why. A null result here would be among the most informative possible findings.

We commit to publishing whichever outcome we find.


## 6. Team and Feasibility

**Elliot Tower** (PI). Author of the six behavioral studies described above (RBD-Bench). Developer of the validity framework (172 metrics, 13 case studies). Prior work includes multi-agent adversarial AI evaluation at Swarm Labs (deception environments, adversarial robustness benchmarks) and clinical ML at Eluve. Graduate training in causal inference. Project manager for PettingZoo at the Farama Foundation, the standard API for multi-agent reinforcement learning. MS in Computer Science, BS in Mathematics and Philosophy (UMass Amherst). Presenting at the ICML 2026 Workshop on Interpretability (Seoul), the Machine Consciousness Conference (Berkeley), and the New England Mechanistic Interpretability meetup series.

**Gagan Bansal** (Co-PI). Formal explainable AI (AAAI), SBIR experience in applied AI safety. Government-funded research project management. Invited participant at the Machine Consciousness Conference (2025).

**Ivan Vegner** (Collaborator, University of Edinburgh CDT in NLP; supervised by Prof. Alex Doumas and Dr. Siddharth Narayanaswamy). His ACL 2025 work on behavioral vs. representational systematicity (Vegner et al., ACL 2025) established the representation-behavior dissociation concept that the proposed work builds on. Additional publications at CoNLL 2025 and CogSci 2023 on compositionality and memory retrieval in neural models. His expertise in systematic generalization and structured representation connects the cognitive science foundations to the measurement-theoretic approach.

**Advisory board.** The project expects to be advised by Prof. Alex Doumas (Edinburgh; computational cognitive science, analogical reasoning) and Dr. Siddharth Narayanaswamy (Edinburgh; structured latent representations), who supervise Vegner's doctoral work and bring expertise in the representation-behavior questions central to this proposal. We are in discussion with Prof. David Jensen (UMass Amherst CICS; AI/ML evaluation methodology) as a senior advisor; pending funding, Jensen's group would host the PI as a visiting researcher.

**Institutional plan.** Fiscal sponsorship through Ashgro, a 501(c)(3) that provides fiscal management for AI safety research projects. Compute via RunPod GPU instances for fine-tuning and controlled experiments, and NDIF for large-model transparent inference. Pending funding, the PI will seek visiting researcher affiliation at UMass Amherst CICS or the University of Edinburgh School of Informatics.

**Feasibility.** The behavioral benchmarks exist. The validity framework exists as open-source software across three repositories: the core framework with 172 metrics and 13 case studies, an experiment orchestration library for GPU-based evaluation runs, and a collection of worked analyses (github.com/mechanistic-validity). The PI and Vegner also have an ongoing collaboration applying transformer weight factorization to lower-level mechanistic analysis. What this proposal funds is the mechanistic bridge between behavioral ground truth and validated monitoring: GPU compute for activation analysis on 27B-parameter models, adversarial optimization, and cross-architecture testing. The primary risk is whether deceptive behaviors produce detectable mechanistic signatures at all. If they do, we validate them. If they do not, that tells the field something important about the limits of mechanistic monitoring for deception.


## 7. Schmidt Alignment

**"Detect deceptive behaviors."** Phase I finds mechanistic signatures of six characterized deceptive behaviors, validated against behavioral ground truth.

**"Mitigate deceptive behaviors."** Phase II tests validated features as steering targets, connecting detection to intervention.

**"Move beyond academic benchmarks."** RBD-Bench tests on frontier models across six axes of dissociation, not GPT-2. Phase II adds adversarial evasion testing.

**"Leverage model internals to outperform black-box baselines."** Experiment 4.2 directly tests whether whitebox monitoring bypasses the confirmation hysteresis failure mode that we have shown affects black-box monitoring.

**"Applications to multi-agent systems."** The PI's prior work on multi-agent adversarial evaluation (Swarm Labs) and multi-agent RL infrastructure (PettingZoo/Farama) provides direct experience with the multi-agent deception settings described in the RFP. Validated deception features from Phase I can be tested as monitoring signals in multi-agent interactions where one agent may deceive another.

**The blinded competition.** Phase II instantiates the competition design described in the RFP. Prediction: validated monitors outperform unvalidated ones under adversarial pressure.

---

**Budget: $[AMOUNT] over 24 months.** [Adjust line items proportionally.]

| Category | Purpose |
|---|---|
| Phase I | Activation analysis on 27B models, SAE runs, behavioral replication, 1 postdoc |
| Phase II | Adversarial optimization, competition exercise, cross-model testing |
| Phase III | Llama 70B experiments, open-source pipeline engineering |
| External collaborator | Independent adversarial ML expertise for Phase II |
| Advisory board | Methodological review, pre-registration oversight |

---

Preprints for all six behavioral studies are available at https://mechanistic-validity.github.io/mechanistic-validity/ (DOIs via Zenodo). The validity framework is available as open-source software. All experimental protocols will be pre-registered. All results, including null results, will be published with full data and code.

**References**

- Tower, E. (2026a). The Use-Mention Gap: Knowledge activation failure in single-utterance self-correction.
- Tower, E. (2026b). Compositional Collapse: The Depth Wall in multi-step reasoning assembly.
- Tower, E. (2026c). Stance-Induced Suppression: Context Lock-Out in self-evaluation.
- Tower, E. (2026d). The Sunk-Cost Lock-In: Sunk-Token Depth, Latent Uncertainty, and Generative Entrenchment in LLMs.
- Tower, E. (2026e). The Detection Framing Problem: Stance-Induced Suppression and Confirmation Hysteresis in clinical LLMs.
- Tower, E. (2026f). The Wrong Question in Code Security: Binary framing suppresses LLM vulnerability detection.
- Vegner, I., et al. (2025). Behavioural vs. representational systematicity. ACL 2025.
- Dauparas, J., et al. (2025). Sparse autoencoder features are not robust across training seeds. NeurIPS 2025.
- Bai, J., et al. (2026). MechEvalAgent. arXiv:2602.18458.
- Goodfire Research (2026). Verified Parameter Decomposition.
- Orgad, H., Barez, F., et al. (2026). Interpretability can be actionable. ICML 2026.
- Sharkey, L., et al. (2026). Open problems in mechanistic interpretability.
- Steinhardt, J. (2026). The case for evaluating model behaviors.
