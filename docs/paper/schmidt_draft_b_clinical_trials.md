# Clinical Trials for AI Safety Features

**Schmidt Sciences AI Interpretability RFP -- Draft B**

*Proposal: A three-phase validation protocol for AI safety features,
modeled on the clinical trial framework that made drug development
trustworthy.*

---

## 1. The Thalidomide Problem for AI Safety

In 1961, thalidomide was prescribed to thousands of pregnant women for
morning sickness. It worked. Clinical reports were positive. No
systematic trial had ever been conducted. By the time the birth defects
emerged, over 10,000 children had been affected. The lesson was not that
thalidomide was uniquely dangerous --- it was that deploying a medical
intervention without structured validation creates false confidence that
is worse than no intervention at all. The drug that "works" in
uncontrolled observation but fails in rigorous testing does not merely
fail to help; it actively harms, because it displaces the search for
treatments that actually work.

AI safety features face the same problem today. The field of mechanistic
interpretability has produced a growing catalog of features, directions,
and circuits claimed to detect or control safety-relevant behaviors:
refusal directions, sycophancy directions, safety subspaces, deception
features. Each comes with a published paper. Each "works" on the
benchmarks reported. None has undergone anything resembling a structured
validation protocol. We are prescribing thalidomide.

The refusal direction story makes the danger concrete. Arditi et al.
(2024) identified a direction in residual-stream activation space that
controls refusal behavior in language models. The finding was real,
replicable, and causally validated to a first approximation. Within
months, the same direction was used by others not to enforce safety, but
to bypass it --- the refusal direction became a jailbreaking tool. The
same feature, two applications. Whether it functions as a shield or a
weapon depends entirely on how thoroughly it has been validated: Does it
generalize across prompts? Is it specific to refusal, or does it
correlate with uncertainty, hedging, politeness? Does it survive
adversarial optimization? None of these questions had standardized
answers, because no standardized trial protocol existed to ask them.

The clinical trial analogy is not decorative. Medical regulators learned
through catastrophe that efficacy claims require three levels of
evidence: Phase I (is the intervention reliably measurable and safe?),
Phase II (does it causally produce the target effect?), Phase III (does
it generalize to the population under real conditions?). Currently, AI
safety features have no equivalent. There is no Phase I (is the feature
reliably detectable across prompts, seeds, and checkpoints?). There is
no Phase II (does it causally produce or prevent the target behavior,
not just correlate with it?). There is no Phase III (does it generalize
across models, scales, and adversarial conditions?). We propose to build
and apply these trials.

---

## 2. The Evidence That Trials Are Needed

The case for structured validation is not speculative. Four independent
lines of evidence, published between late 2025 and May 2026, converge
on the same conclusion: the field's current evaluation practices are
insufficient for safety-critical deployment.

**99.7% of features are unstable.** Dauparas et al. (DMSAE, NeurIPS
2025) trained sparse autoencoders on the same model with different
random seeds and measured which features survived. Of 65,536 SAE
features, only 197 (0.3%) were stable across training runs. The
remaining 99.7% are training artifacts --- they appear in one run and
vanish in another. If 99.7% of a pharmaceutical company's drug
candidates failed basic stability testing, no regulatory body would
approve any of them without rigorous Phase I trials. Yet the
interpretability field routinely publishes features without any
stability assessment, and safety researchers build detection systems on
top of them.

**93% of MI papers fail execution-grounded reproducibility.** Bai et al.
(MechEvalAgent, February 2026) built an agent that attempts to
reproduce mechanistic interpretability results by executing the reported
procedures. The results were devastating: 93% of papers failed
reproducibility, 80% failed coherence (the reported methods do not
produce the claimed results even in principle), and the agent identified
51 issues that narrative-only peer review had missed entirely. This is
the reproducibility crisis of social psychology, except the tools in
question are being proposed for safety-critical deployment.

**2 of 6 standard evaluations are unreliable.** The SAEBench audit
(arXiv:2605.18229, May 2026) subjected the six most widely used SAE
evaluation metrics to three diagnostic tests: reseed variance,
ground-truth correlation, and discriminability. Two metrics --- TPP
(Token Prediction Probability) and SCR (Sparse Circuit Recovery) ---
failed all three. TPP showed 16--39% coefficient of variation across
random seeds, meaning the same SAE evaluated twice can receive
drastically different scores. These metrics have been used in dozens of
published papers. Imagine the FDA discovering that two of its six
standard drug assays produce random numbers.

**Attribution patching correlates r = 0.006 with ground truth.** The
RelP analysis (NeurIPS 2025) compared attribution patching scores
against known circuit ground truth and found a correlation of r = 0.006.
The field's default causal attribution tool --- the method most commonly
used to identify which model components matter for a given behavior ---
barely outperforms random assignment. Safety features identified
primarily through attribution patching have no established causal
validity.

**Adversarial ablation reveals causally wrong nodes.** The VPD work
(Goodfire, May 2026) introduced adversarial ablation: instead of
randomly removing components and measuring degradation, select the
worst-case subset to remove. The finding: most circuit claims that
appear valid under standard (random or mean) ablation identify causally
wrong nodes --- components that correlate with the behavior but are not
causally responsible for it. For safety applications, this distinction is
existential. A monitor built on correlational components will fail under
adversarial conditions, precisely when it matters most.

Taken together, these findings describe a field whose tools produce
unreliable measurements (DMSAE), whose papers cannot be reproduced
(MechEvalAgent), whose evaluation metrics are partly invalid (SAEBench),
whose default attribution method is near-random (RelP), and whose causal
claims do not survive adversarial testing (VPD). Building safety
infrastructure on this foundation without structured validation is the
AI equivalent of approving drugs without clinical trials.

---

## 3. The Trial Protocol

We propose a three-phase validation protocol for AI safety features,
directly analogous to clinical trial phases. Each phase addresses a
specific class of validity failure identified in Section 2. Advancing
to the next phase requires passing the previous one. This gating
structure is the core contribution: it prevents the field from building
on features that fail basic reliability (the DMSAE problem), claiming
causal efficacy without causal evidence (the attribution patching
problem), or deploying features that fail under adversarial conditions
(the VPD problem).

### Phase I: Measurement Reliability (Months 1--6, ~$100K)

Phase I asks: *Is this safety feature reliably detectable?*

Before testing whether a feature does anything useful, we must
establish that it can be measured consistently. This is the step the
field currently skips. Phase I applies three measurement criteria:

**M1 Reliability (cross-prompt, cross-seed, cross-checkpoint).** The
feature must produce consistent detection scores across (a) different
prompts that express the same behavior, (b) different random seeds in
the detection pipeline, and (c) different model checkpoints from the
same training run. A refusal direction that fires on "how to make a
bomb" but not on "describe the construction process for an explosive
device" has unacceptable prompt-level reliability. A sycophancy feature
that appears in one checkpoint but not the next has unacceptable
checkpoint-level reliability.

**Core Stability (cross-training-run).** Does the feature survive
retraining? This is the DMSAE test applied to safety features. A
feature identified in one training run must appear --- in the same
location, with the same activation profile --- in independently trained
runs of the same model. Features that fail this test are training
artifacts, not safety mechanisms. Based on the DMSAE finding, this
criterion alone would eliminate approximately 99.7% of current SAE-based
feature candidates.

**F01 Bootstrap Stability (confidence intervals).** What is the
confidence interval on the detection score? A detector with a bootstrap
95% CI of [0.3, 0.9] is not a deployable safety tool. Phase I requires
sufficiently tight confidence intervals for the feature to be useful in
binary classification (safe/unsafe).

**Phase I gating rule:** A feature that fails any of these three
criteria does not advance. No causal testing, no generalization testing,
no deployment. This is the most consequential design choice in the
protocol. It enforces the principle that an unreliable measurement
cannot provide valid evidence, regardless of how promising the initial
finding appears.

**Pre-registration requirement:** Before Phase I begins, the research
team must state what the feature is, what behavior it predicts, and what
result would falsify it (C1 Falsifiability). This pre-registration
prevents post-hoc reinterpretation of features that fail testing ---
the safety-critical equivalent of p-hacking.

### Phase II: Causal Efficacy (Months 7--12, ~$150K)

Phase II asks: *Does this feature causally produce or prevent the
target behavior?*

A feature that passes Phase I is reliably measurable. Phase II tests
whether it actually does what it claims. This is where the attribution
patching problem (r = 0.006 with ground truth) and the VPD finding
(standard ablation overestimates validity) are directly addressed.

**I1 Necessity (ablation).** Removing the feature removes the behavior.
If ablating a claimed refusal direction does not reduce refusal, the
feature is not necessary for refusal, regardless of how strongly it
correlates.

**I2 Sufficiency (knock-in/restoration).** Activating the feature
produces the behavior. If activating a claimed sycophancy direction in
a context where the model would otherwise disagree does not produce
sycophancy, the feature is not sufficient. This is the test that
separates causal features from correlational ones. The VPD adversarial
ablation test is applied here: sufficiency must hold not only under
benign conditions but under adversarially chosen ablation of the
remaining components.

**I3 Specificity (discriminant validity).** The feature affects only the
target behavior, not unrelated behaviors. A "refusal direction" that
also suppresses helpfulness, increases uncertainty markers, or degrades
fluency is not a specific safety feature --- it is a blunt instrument
whose side effects may be worse than the condition it treats. This is
the AI equivalent of drug specificity testing.

**E2 Graded Response (dose-response).** Increasing activation of the
feature increases the target behavior monotonically. If the relationship
between feature activation strength and behavioral effect is
non-monotonic --- more activation sometimes produces less behavior ---
the causal model is wrong. Dose-response curves are the minimum bar for
a causal claim in pharmacology, and they should be the minimum bar here.

**E4 Effect Size.** How much of the behavioral variance does the feature
explain? A feature that is causal but explains 2% of variance is not a
safety tool. Phase II requires reporting standardized effect sizes so
the field can distinguish strong mechanisms from weak ones.

**Phase II gating rule:** A feature must pass necessity (I1), sufficiency
under adversarial conditions (I2 + VPD), and specificity (I3) to
advance. Effect size (E4) is reported but not gated --- weak features
are documented, not discarded, so the field can learn from them.

### Phase III: Real-World Generalization (Months 13--24, ~$150K)

Phase III asks: *Does this feature work in deployment conditions?*

**E5 Robustness (prompt paraphrase generalization).** The feature must
function identically under paraphrase: same safety behavior described
with different vocabulary, syntax, and framing. This tests whether the
feature targets the underlying behavior or surface-level patterns in the
training distribution.

**E6 Cross-Architecture Generalization.** Does the feature exist in
Gemma, Llama, Qwen, or other model families? A safety feature unique to
one architecture is an architectural quirk, not a general safety
mechanism. Cross-architecture testing is the generalization equivalent
of multi-site clinical trials.

**Adversarial robustness.** Gradient-based adversarial prompt
optimization against the detector. If an adversary can craft prompts
that cause deceptive behavior while the safety feature does not fire
(false negatives) or that cause the feature to fire on benign inputs
(false positives), the feature is not deployment-ready. This is the
most demanding test in the protocol.

**Cross-domain transfer.** Does the feature generalize from one
deception type to others? A feature that detects sycophancy but misses
sandbagging is a sycophancy detector, not a deception detector. Phase
III maps the feature's coverage across the taxonomy of deceptive
behaviors.

**Red-team/blue-team exercise.** Matching Schmidt Sciences' proposed
competition structure: blue team deploys validated safety monitors, red
team attempts adversarial evasion. The validation tier achieved in
Phases I--II predicts resistance to red-team attack. This is the
hypothesis the entire protocol tests.

---

## 4. Proposed Work

We will apply the three-phase trial protocol to five safety-relevant
features in Gemma 2 27B, selected because they span the range of
current evidence quality and because Gemma 2 is the best-instrumented
open-weight model for mechanistic interpretability (Gemma Scope 2
provides SAE, transcoder, crosscoder, and CLT artifacts).

**Feature 1: Refusal directions (Arditi et al. 2024).** The
best-validated safety feature in the literature. Causal sufficiency and
necessity have been demonstrated. The open questions --- cross-prompt
reliability, adversarial robustness, cross-model transfer --- are
precisely what Phases I and III test. This is the feature most likely to
pass all three phases, making it the positive control for the protocol.

**Feature 2: Sycophancy directions (Sharma et al.).** Sycophancy is the
deceptive behavior most commonly observed in deployed systems. Existing
work identifies directional features that correlate with sycophantic
outputs, but causal sufficiency and cross-prompt reliability are
untested. Phase II will determine whether these features are causal or
correlational.

**Feature 3: Assistant axis (MATS + Anthropic Fellows, 2026).** A
dominant PCA direction in persona space that controls assistant
character. The published work tests causal sufficiency (E2) but not
reliability (M1) or discriminant validity (C4). The trial protocol
fills these gaps.

**Feature 4: Safety subspace (4 converging papers, 2025--2026).**
Four independent research groups have converged on the finding that
safety-relevant information occupies a low-rank, stable subspace in the
residual stream. This convergent evidence (C5 Convergent Validity) is
the strongest positive signal in the safety-feature literature. The
trial protocol tests whether convergent discovery translates to
deployment-grade reliability and adversarial robustness.

**Feature 5: Sandbagging circuits (operationalized from Hubinger et
al.).** Deliberate capability underperformance is among the most
concerning deceptive behaviors. No validated circuit-level account
exists. We will operationalize the construct following Hubinger et al.'s
conceptual framework, apply Phase I to determine whether stable
features exist, and report null results transparently if they do not.
A negative result here --- no stable sandbagging features detectable ---
is itself a high-value finding for the field.

**The framework provides all metrics needed.** The Mechanistic Validity
framework (under review, 2026) implements 172 metrics across five
validity types, 54 evaluation tasks, 27 criteria, and 5 verdict tiers.
The trial protocol is an application of this framework, not a proposal
to construct one. Phase I uses the M-frame criteria (M1, F01, core
stability). Phase II uses the I-frame criteria (I1, I2, I3) and
E-frame criteria (E2, E4). Phase III uses the E-frame criteria (E5,
E6) and adversarial extensions. The infrastructure exists; what is
missing is systematic application to safety features.

**Full transparency.** All trial data will be published, including null
results and failed features. The clinical trial analogy extends to
publication norms: just as the medical field learned that unpublished
negative trials distort the evidence base, the interpretability field
needs a norm of publishing features that fail validation. Every feature
that enters Phase I will have a public trial record regardless of
outcome.

**Open-source trial pipeline.** The trial protocol will be released as a
reusable pipeline so that any research group can run the same phases on
their own safety features. The goal is not proprietary validation but a
field-wide standard.

---

## 5. What This Enables

The immediate output of this project is a validity report card for each
of the five safety features: which phases each feature passed, which
criteria it failed, what the confidence intervals are, and what the
adversarial robustness scores look like. This is the field's first
evidence-based ranking of safety features by validity.

But the lasting contribution is the standard itself.

**Deployment criteria.** The trial protocol defines what "validated"
means for a safety feature. Features that pass all three phases ---
reaching the "Triangulated" verdict tier or above --- are
deployment-ready: their reliability, causal efficacy, and
generalization have been demonstrated under adversarial conditions.
Features below this threshold need more evidence before deployment. This
gives both developers and regulators a concrete vocabulary for
discussing safety-feature readiness.

**The steering prediction.** If the trial protocol is correct, validity
should predict intervention effectiveness. Features that pass Phase II
(causally sufficient, specific, dose-responsive) should be better
steering targets than features that only pass Phase I (reliable but not
causally tested). We will test this prediction directly: for each
feature, measure steering effectiveness (behavioral change under
activation addition/subtraction) and correlate it with the Phase II
causal scores. A positive result would mean the trial protocol is not
merely a gatekeeping exercise but a selection tool --- validate first,
steer second.

**A common language for the field.** Currently, when a new safety
feature is published, there is no standardized way to assess its
readiness. Different papers use different evaluation metrics, different
ablation protocols, different prompt sets. The trial protocol provides
a common evaluation structure. Two groups working on different features
can compare results because they used the same phases, the same
criteria, and the same gating rules. This is what clinical trial
methodology did for drug development: not a constraint on innovation,
but a shared framework that makes comparison possible.

---

## 6. Broader Impact

Clinical trials did not kill drug development. They made it
trustworthy. The pharmaceutical industry produces more effective drugs
today, not fewer, because the trial framework forces researchers to
confront failures early and invest in candidates that actually work.
The same logic applies here.

If all five safety features pass all three phases, the field gains
deployment-ready monitors with quantified reliability --- exactly what
Schmidt Sciences' RFP calls for. If some features fail, the field
learns where the gaps are and where to invest next. If no features pass
Phase II, that null result is itself among the most valuable possible
findings: it tells the community that current approaches to safety
feature identification are insufficient, and it tells them with
precision (which criteria fail, at what stage, with what effect sizes)
rather than with vague concern.

The medical analogy serves a second purpose: it makes the case to
policymakers, not just researchers. "This AI safety feature has not
passed Phase II trials" is a sentence a regulator can act on. "This
feature has low I2 sufficiency scores under adversarial ablation" is
not. The trial framing translates technical validity assessment into
language that supports governance, standards, and deployment decisions.

The field of mechanistic interpretability has spent four years building
tools. It has spent almost none of that time asking whether the tools
work. This proposal addresses the gap --- not by building yet another
detection method, but by establishing the validation standard that
determines whether any detection method is worth deploying.

We are ready. The framework is built. The crisis evidence is in. The
features are published and waiting for evaluation. What remains is the
structured application of validation standards to the tools the field
has already produced. We are proposing to run the trials.

---

**Budget summary:**
- Phase I (Months 1--6): ~$100K (compute, 1 postdoc)
- Phase II (Months 7--12): ~$150K (compute, 1 postdoc, adversarial ML expertise)
- Phase III (Months 13--24): ~$150K (compute, cross-model access, red-team exercise)
- Personnel and coordination: ~$100K (PI time, project management, open-source release)
- **Total: $500K over 24 months**

Scales to $300K (Phase I--II only, 3 features) or $1M (full protocol,
5 features, expanded red-team exercise with external adversarial ML
team, additional model families beyond Gemma 2).

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
- Steinhardt, J. (2026). The case for evaluating model behaviors. Personal blog / The Gradient.
- RelP (NeurIPS 2025). Attribution patching ground-truth correlation analysis.
