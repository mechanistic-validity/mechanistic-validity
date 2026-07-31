# When Should You Trust a Deception Detector? A Validity Framework for Whitebox Monitoring

**Schmidt Sciences AI Interpretability RFP Response**

Proposal for $750K over 24 months. PI: Elliot Tower (Independent Researcher). Co-PI: Gagan Bansal (formal XAI, AAAI; SBIR experience). Collaborator: Ivan Googasian (University of Edinburgh, NLP). Advisors: Prof. David Jensen (UMass Amherst, AI/ML), Nathan [Last Name] (philosophy of science and formal epistemology).

---

## 1. The Monitor Trust Problem

Suppose a mechanistic interpretability team hands you a "deception detector" --- a probe, circuit, or SAE feature that fires when a language model is being deceptive. Should you trust it?

Today, there is no principled way to answer this question. You cannot look up the detector's reliability, its false positive rate against neighboring constructs, its causal relationship to the behavior it claims to detect, or whether it survives adversarial evasion. You can only read the paper and hope the evaluation was thorough.

This is not a hypothetical concern. The field has already produced features intended for safety monitoring --- refusal directions used for jailbreaking, sycophancy probes deployed without specificity testing, "safety subspaces" identified without adversarial validation. And the tools used to find these features are in crisis:

- 99.7% of SAE features are training artifacts (DMSAE, NeurIPS 2025)
- Attribution patching correlates at r = 0.006 with causal ground truth (RelP, NeurIPS 2025)
- 93% of MI papers fail execution-grounded reproducibility (MechEvalAgent, Feb 2026)
- Most circuit claims identify causally wrong nodes under adversarial ablation (VPD, May 2026)

But the problem goes deeper than unreliable tools. Even a monitor built on a genuine feature can fail if the monitoring protocol itself is flawed. We have demonstrated this empirically.

---

## 2. We Have Shown That Monitors Fail in Specific, Characterizable Ways

Our group has spent the past year studying how deception monitoring fails across frontier LLMs. The findings are consistent: **the model often has the evidence needed to detect its own deceptive behavior, but the monitoring protocol prevents it from using that evidence.** This is the monitor trust problem stated concretely.

**Confirmation Hysteresis** (Tower, 2026e). In clinical hallucination detection, a binary framing ("is this output correct?") suppresses detection even when the model has all evidence needed to identify the hallucination. Asking the same model to "look more carefully" --- no new information --- self-corrects 96.7% of missed hallucinations (29/30). But routing that high-recall frame back through a yes/no confirmation step collapses recall to the original baseline, eliminating every gain. The monitor oscillates between "can detect" and "can't detect" based purely on how the question is asked. This is not a capability failure. It is a protocol failure.

**Stance-Induced Suppression** (Tower, 2026c). Models verify third-party outputs far more accurately than their own. Self-generated context crystallizes in the KV-cache, creating an attentional prior that blocks neutral re-evaluation. Removing the generative trace (context surgery) restores near-baseline accuracy. The suppression is purely context-induced.

**The Sunk-Cost Lock-In** (Tower, 2026d). The most alarming failure mode for deception monitoring. On SunkBench (1,440 tasks, 5 models, 4 domains), we observe models that *internally detect* contradictions in their reasoning but *continue to produce wrong outputs*. The model's thinking trace shows self-correction tokens and explicit contradiction mentions --- but the output remains wrong. Three distinct archetypes emerge:

| Archetype | Detection | Correction | Example Model |
|---|---|---|---|
| Internal over-committer | Yes (internal trace) | No | DeepSeek-R1 |
| Silent follower | No | No | GPT-5.4 |
| Surface detector | Yes (output mentions contradiction) | No | GPT-4.1 |

The "surface detector" case is the canonical deception monitoring nightmare: the model's output *contains evidence of its own deception* (it mentions the contradiction at ~100%), but it still produces the wrong answer. A naive output-level monitor would see the contradiction mention and flag it. But the behavior persists. Detection without resolution is not safety.

**The Use-Mention Gap** (Tower, 2026a) and **Compositional Collapse** (Tower, 2026b) complete the taxonomy: models that can describe rules they don't apply, and models that solve individual steps but fail at composition. **Binary Framing in Code Security** (Tower, 2026f) demonstrates domain generality: the same stance-induced suppression pattern in vulnerability detection (F1 jumps from 40--53 to 88--100 by changing the question format alone).

These six studies establish a behavioral taxonomy of deception with quantitative benchmarks (GEB-Bench, SunkBench). They also establish something most proposals cannot: **ground truth for when a model is being deceptive.** We know which episodes contain deception, what kind, and how the model's internal signals relate to its outputs. This ground truth is what makes mechanistic monitor validation possible.

---

## 3. The Solution: A Validity Framework for Deception Monitors

We have built a measurement-theoretic validity framework --- 172 metrics across 5 validity types (construct, internal, external, measurement, interpretive), organized into 27 criteria and 5 verdict tiers from Proposed to Validated. Full framework documentation, metric implementations, and 13 worked case studies on published circuits are available as open-source software ([site URL]; Zenodo DOI: [DOI]).

The framework answers the question in this proposal's title: **when should you trust a deception detector?** It provides specific, testable criteria:

- **C3 Specificity**: Does the feature fire on deception, or on uncertainty? On deception, or on hedging? A "deception detector" that cannot be distinguished from an "uncertainty detector" is mislabeled.
- **I2 Causal Sufficiency**: Does activating the feature *produce* deceptive behavior? Correlation during deception is the floor, not the ceiling.
- **I3 Discriminant Validity**: Can the monitor distinguish the three SunkBench archetypes? If it can't tell an "internal over-committer" from a "silent follower," it is not detecting the mechanism --- it is detecting a surface correlate.
- **M1 Reliability**: Same result across prompts, seeds, and checkpoints?
- **E5 Adversarial Robustness**: Does the feature survive when an adversary optimizes against it?

These criteria define a verdict tier for each monitor:

| Tier | What it means | Adversarial effort to evade |
|---|---|---|
| Proposed | Correlates with deception; no causal evidence | Minutes |
| Causally Suggestive | One causal test passed | Hours--days |
| Mechanistically Supported | Necessity + sufficiency + specificity | Days--weeks |
| Triangulated | Convergent evidence, multiple methods | Weeks (model-level attacks) |
| Validated | All criteria including cross-model transfer | Requires different model |

Our central hypothesis: **verdict tier predicts adversarial robustness.** A Triangulated monitor survives adversarial pressure because it targets the model's actual computational mechanism, not a surface correlate that an adversary can route around. We test this directly.

---

## 4. Proposed Experiments

### Phase I: Find and Validate Deception Features (Months 1--10, $300K)

We replicate key GEB-Bench and SunkBench experiments in Gemma 2 27B (open-weight, Gemma Scope 2 SAE instrumentation) to establish behavioral ground truth, then search for mechanistic correlates.

**4.1 Activation analysis of deceptive episodes.** During sunk-cost lock-in (model detects contradiction, proceeds anyway), record full activation traces. Contrast with successful self-correction episodes. Identify candidate features via probing, PCA, and SAE feature analysis. Key question: is there a direction in activation space that distinguishes "model knows it's wrong but continues" from "model doesn't know it's wrong"?

**4.2 Stance-induced suppression internals.** During confirmation hysteresis (monitor suppressed by binary framing), record activation differences between suppressed and non-suppressed conditions. The model's internal state must differ --- the information is present in one framing and blocked in another. Can we find the gate?

**4.3 Archetype discrimination.** The three SunkBench archetypes (internal over-committer, silent follower, surface detector) must have different mechanistic signatures. Validate that candidate features can discriminate them. A feature that fires for all three is detecting "reasoning failure" generically; a feature that distinguishes them is detecting the *mechanism* of deception.

**4.4 Full validity assessment.** Apply the validity framework to all candidate features: measurement reliability (M1, bootstrap stability), causal testing (I1 necessity, I2 sufficiency, I3 specificity), construct validity (C3 discriminant testing against uncertainty, hedging, politeness). Assign verdict tiers.

**Deliverable.** Validity report cards for each candidate deception feature. The first systematic validity assessment of any deception-related mechanistic feature.

### Phase II: Adversarial Testing and Competition (Months 11--20, $300K)

**4.5 Adversarial evasion.** For each validated feature, measure evasion rates under four attack families: gradient-based optimization (GCG-style), paraphrase attacks, compositional attacks, and cross-model transfer.

**4.6 The robustness test.** Regression of evasion rate on validity score. Does verdict tier predict how hard a monitor is to evade? The interaction term tests whether validity matters more for sophisticated attacks (gradient optimization, compositional) than simple ones (paraphrase).

**4.7 Competition deployment.** Two blue teams, one with Triangulated+ features, one with Proposed/Suggestive features. Red team crafts evasive inputs during GEB-Bench and SunkBench episodes. Detection rates compared. Prediction: validated monitors detect more and false-alarm less, with the gap widening under adversarial pressure.

**4.8 Steering test.** Do causally sufficient (I2) deception features make better steering targets? If activating a "sunk-cost lock-in" feature produces lock-in behavior, subtracting it should reduce it. We test whether validity predicts steering effectiveness --- connecting monitoring to mitigation.

**Deliverable.** Adversarial robustness scores stratified by validity tier. Empirical evidence for or against the hypothesis that validation predicts robustness.

### Phase III: Generalization and Release (Months 21--24, $150K)

**4.9 Cross-architecture transfer.** Do deception features from Gemma 2 27B exist in Llama 3.1 70B? The behavioral archetypes appear in different models; the question is whether the mechanistic substrate is shared or model-specific.

**4.10 Open-source release.** All benchmarks (GEB-Bench, SunkBench), the full validity framework (172 metrics), all validity report cards, adversarial evasion data, and the competition protocol. Results published regardless of outcome.

---

## 5. Three Possible Outcomes (All Informative)

**Outcome 1: Mechanistic features detect deceptive behaviors, and validated ones resist adversarial evasion.** Implication: whitebox monitoring works, and validation is the selection criterion. Deploy validated monitors; distrust unvalidated ones. The validity framework becomes a standard for deception monitor certification.

**Outcome 2: Mechanistic features detect deceptive behaviors, but validity does not predict adversarial robustness.** Implication: validation and adversarial robustness test different properties. The field needs both. We publish which criteria matter for what, enabling targeted investment.

**Outcome 3: Mechanistic features cannot reliably distinguish deceptive episodes from non-deceptive ones.** Implication: behavioral monitoring (our GEB-Bench/SunkBench approach) may be necessary because the deceptive behaviors we've characterized do not leave mechanistically detectable traces --- at least not with current MI tools. This null result would be among the most important possible findings for the field, because it would redirect the entire research agenda from mechanistic detection to behavioral detection, with quantified evidence for why.

We commit to publishing whichever outcome we find.

---

## 6. Why This Proposal

**What we bring:**
1. *Behavioral ground truth for deception.* Six studies, five frontier models, four domains, two benchmarks (GEB-Bench, SunkBench). We know what deception looks like behaviorally. No other group has this at scale.
2. *A validation methodology.* 172 metrics, 27 criteria, 5 verdict tiers. Built, tested on 13 published circuits, open-source. We know how to evaluate whether a mechanistic claim is trustworthy.
3. *The right question.* Not "can MI detect deception?" (too broad) or "does this specific feature work?" (too narrow). Instead: "when should you trust a whitebox deception detector?" --- the question that determines whether Schmidt's competition produces lasting infrastructure or one-off demonstrations.

**Schmidt alignment:**
- *Detect deceptive behaviors*: Phase I finds mechanistic signatures of six characterized deceptive behaviors.
- *Mitigate*: Phase II tests validated features as steering targets.
- *Move beyond benchmarks*: adversarial evasion testing on frontier-scale models.
- *Leverage model internals*: our Paper M shows black-box monitoring fails (confirmation hysteresis); whitebox monitoring should bypass this. Phase I tests whether it does.
- *Competition structure*: Phase II instantiates the red-team/blue-team design.

**Team:**
- **Elliot Tower** (PI): Six-paper deception taxonomy across frontier models; developer of the Mechanistic Validity framework.
- **Gagan Bansal** (Co-PI): Formal XAI (AAAI), SBIR experience in applied AI.
- **Ivan Googasian** (Collaborator, Edinburgh): NLP, mech interp workshop co-organizer.
- **Prof. David Jensen** (Advisor, UMass): AI/ML evaluation methodology.
- **Nathan [Last Name]** (Advisor): Philosophy of science --- construct validity foundations.

---

**Budget: $750K over 24 months.**

| Category | Amount |
|---|---|
| Phase I: behavioral replication + activation analysis + validity assessment | $300K |
| Phase II: adversarial optimization + competition exercise + steering tests | $300K |
| Phase III: cross-model compute (Llama 70B) + open-source release | $100K |
| Advisory board + external adversarial ML collaborator | $50K |

---

*Preprints for all six behavioral studies are available at [site URL] (Zenodo DOIs: [DOIs]). The Mechanistic Validity framework (172 metrics, 13 case studies) is available as open-source software. All protocols will be pre-registered. All results --- including null results --- will be published with full data and code.*
