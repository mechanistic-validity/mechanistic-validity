# Schmidt Sciences AI Interpretability RFP — Four Draft Outlines

RFP: $300K–$1M, 1–3 years. Due May 26 2026.
Focus: detect/mitigate deceptive behaviors, move beyond academic benchmarks, leverage model internals.

---

## Outline A: "The Validation Layer" — Infrastructure, Not Another Detector

**Core pitch**: The field's bottleneck isn't detection methods — it's knowing whether detection methods work. We provide the missing validation infrastructure.

1. **The crisis** (1 page)
   - 99.7% of SAE features are unstable across training runs (DMSAE, NeurIPS 2025)
   - 93% of MI papers fail reproducibility checks (MechEvalAgent, Feb 2026)
   - SAEBench audit: 2 of 6 standard evaluations should not be used (May 2026)
   - Attribution patching — the field's default tool — correlates r=0.006 with ground truth (RelP, NeurIPS 2025)
   - Bottom line: any deception feature found with current methods has unknown validity

2. **What we built** (1 page)
   - 172 metrics across 5 validity types (construct, internal, external, measurement, interpretive)
   - 54 evaluation tasks, 27 criteria, 5 verdict tiers (Proposed → Validated)
   - Method-agnostic: evaluates SAE features, circuits, probing results, steering vectors alike
   - 13 worked case studies on published circuits (IOI, Greater-Than, Induction, etc.)

3. **Application to safety** (2 pages)
   - Deception detection as special case: every "this detects deception" claim is an MI claim subject to the same validation
   - Map: C1 falsifiability → what would disprove the feature? I2 sufficiency → does it causally produce deception? M1 reliability → consistent across prompts? C3 specificity → is it deception or just uncertainty?
   - Safety subspace finding: 4 independent papers show safety info lives in low-rank stable subspace — high validity by construction

4. **Proposed work** (2 pages)
   - Year 1: Full validity report cards for 5 known safety-relevant features (refusal directions, sycophancy directions, assistant axis, safety subspace, sandbagging circuits)
   - Year 2: Test whether validity predicts steering effectiveness (high-validity features → better intervention targets)
   - Year 3: Red-team/blue-team exercise — validated monitors vs unvalidated monitors under adversarial attack

5. **Why us, why now** (0.5 page)
   - Framework already built (not a proposal to build one)
   - ICML 2026 paper makes identical argument — convergent evidence the field is ready
   - Neel Nanda, Steinhardt, 30-author Open Problems paper all calling for this

**Tone**: Infrastructure/engineering. "We built the test suite. Let's use it."

---

## Outline B: "Clinical Trials for AI Safety" — The Medical Analogy

**Core pitch**: Drug approval requires Phase I–III trials. AI safety feature deployment requires nothing. We propose the equivalent validation pipeline.

1. **The analogy** (1 page)
   - Phase I (safety/toxicity) → M-frame: Is the feature reliably measurable? Is it stable? Does it have acceptable precision?
   - Phase II (efficacy) → I-frame: Does the feature causally produce/prevent the target behavior? Effect size?
   - Phase III (generalization) → E-frame: Does it work across prompts, models, scales, adversarial conditions?
   - Pre-registration → Claim specs: state predictions before running experiments
   - The current field is shipping drugs without any trials

2. **The failures that motivate trials** (1 page)
   - Same crisis data as Outline A
   - Add: thalidomide analogy — a safety feature that "works" on benchmarks but fails in deployment is worse than no feature (false confidence)
   - The refusal direction story: Arditi et al. found it, others used it for jailbreaking — validity determines whether a feature is a shield or a weapon

3. **The trial protocol** (2 pages)
   - Phase I (6 months): 8 reliability and measurement criteria, bootstrap stability, seed variance, prompt paraphrase invariance
   - Phase II (6 months): Causal necessity (ablation), sufficiency (knock-in), specificity (discriminant validity), dose-response (graded activation)
   - Phase III (12 months): Cross-prompt generalization, cross-model transfer, adversarial robustness, real-world deployment monitoring
   - Pre-registration: MechanisticClaimSpec with predictions, negative controls, rival explanations

4. **Proposed work** (1.5 pages)
   - Run Phase I–II trials on 5 safety-relevant features in Gemma 2 27B
   - Open-source the trial protocol as a reusable pipeline
   - Publish trial results with full transparency (including null results)

5. **Broader impact** (0.5 page)
   - If safety features pass → deployment-ready monitors with known reliability
   - If safety features fail → the field knows which features are invalid and why
   - Either way, the field gains a standard for what "validated" means

**Tone**: Regulatory/public health. "We wouldn't approve a drug this way. Why are we approving AI safety tools this way?"

---

## Outline C: "The Adversarial Robustness Gap" — Red Team Focus

**Core pitch**: Current safety features fail under adversarial conditions. Validated features survive. We measure the gap and close it.

1. **The adversarial problem** (1 page)
   - VPD finding: adversarial ablation reveals that most circuit claims identify causally wrong nodes (Sharkey et al., May 2026)
   - Standard ablation ≠ adversarial ablation: a feature that appears necessary under benign conditions may be unnecessary under worst-case
   - For safety: an adversary will exploit exactly these gaps
   - The red-team/blue-team structure Schmidt proposes requires knowing which detection methods survive adversarial pressure

2. **Validated vs unvalidated features** (1 page)
   - Hypothesis: features that pass causal sufficiency (I2) + adversarial ablation + cross-prompt reliability are harder to evade
   - Why: they target deep causal structure, not surface correlates. Adversaries can change surface features but not the model's causal mechanisms.
   - The DMSAE finding as cautionary tale: features that are training artifacts (99.7% of SAE features) are trivially evaded

3. **The measurement framework** (1.5 pages)
   - 172 metrics organized by what they test
   - Key metrics for adversarial robustness: VPD adversarial ablation, cross-prompt consistency, seed stability, superposition risk assessment
   - Verdict tiers as a robustness hierarchy: Proposed (evaded trivially) → Validated (requires model-level attack)

4. **Proposed experiments** (2 pages)
   - Phase 1: Classify 5 safety features by validity tier using the full framework
   - Phase 2: For each feature, measure adversarial evasion rate (gradient-based adversarial prompt optimization)
   - Phase 3: Test prediction: validity tier correlates with adversarial robustness
   - Phase 4: Deploy validated monitors in a controlled red-team exercise (Schmidt's competition design)

5. **Deliverables** (0.5 page)
   - Adversarial robustness scores for each safety feature
   - Correlation between validity and robustness (the key scientific result)
   - Open-source adversarial validation pipeline

**Tone**: Security/adversarial ML. "Your detectors don't survive red-teaming. Ours do, because we validated them first."

---

## Outline D: "Deception Is Not Special" — Generality Argument

**Core pitch**: Deception detection is not a separate research agenda — it's an application of MI that inherits all of MI's unsolved validation problems. Solve the general problem, get deception detection for free.

1. **The construct validity problem** (1 page)
   - "Deception" is not a natural kind in transformer computation
   - It's a human label on a cluster: strategic lying, sycophancy, sandbagging, goal misgeneralization, faithfulness failures
   - Each sub-behavior may have different mechanistic substrate
   - Building a "deception detector" without construct validity is searching for a concept that may not exist as a unified feature
   - C1 falsifiability: what result would disprove your "deception feature"?

2. **The general MI validation crisis** (1 page)
   - Same crisis evidence
   - Point: these problems affect ALL MI claims, not just deception ones
   - A deception feature found with an unreliable method is unreliable whether you call it "deception" or not

3. **The framework** (1.5 pages)
   - 5 validity types × 27 criteria × 5 verdict tiers
   - Works for any MI claim: feature descriptions, circuit claims, steering vectors, safety features
   - 172 implemented metrics with thresholds

4. **Deception as worked example** (2 pages)
   - Apply the framework to deception detection as a case study
   - Step through: C1 (define deception falsifiably) → C3 (distinguish from uncertainty) → I2 (causal test) → M1 (reliability) → E5 (generalization) → E6 (cross-model)
   - Show where current deception features fall on the verdict tier scale
   - The Neel Nanda argument formalized: "MI provides mechanistic evidence, not certainty" → the framework quantifies how much evidence

5. **Proposed work** (1 page)
   - Run the full protocol on deception-adjacent constructs (sycophancy, refusal, sandbagging, goal-misgeneralization)
   - For each: validity report card, verdict tier, identified gaps
   - Output: the field's first principled answer to "can MI detect deception?" with quantified confidence

**Tone**: Philosophy of science / measurement theory. "You're trying to detect something you haven't defined. Let's define it first."

---

## Comparative Assessment

| Angle | Strength | Risk | Schmidt alignment |
|---|---|---|---|
| A: Validation Layer | Most direct match to "move beyond benchmarks" | Could seem too infrastructure-focused, not enough safety | HIGH |
| B: Clinical Trials | Most compelling analogy for general audience | Analogy might feel forced; regulatory tone might not land with ML audience | MEDIUM-HIGH |
| C: Adversarial Robustness | Most directly actionable; matches competition design | Narrower scope; may miss the generality argument | HIGH |
| D: Deception Not Special | Most intellectually novel; strongest philosophy | Could seem dismissive of the RFP's specific focus on deception | MEDIUM |

**Recommendation**: Lead with A's infrastructure pitch, use B's analogy as framing device, include C's adversarial experiments as the concrete deliverable. D's construct validity argument belongs in the motivation section, not as the lead.
