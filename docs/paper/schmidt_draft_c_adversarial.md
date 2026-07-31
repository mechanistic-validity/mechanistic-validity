# The Adversarial Robustness Gap: Validated Monitors Survive Red-Teaming

**Schmidt Sciences AI Interpretability RFP --- Draft C**

Proposal for $500K over 24 months. PI: [Name]. Institutional affiliation: [Institution].

---

## 1. The Adversarial Threat to Interpretability-Based Safety

Schmidt Sciences' RFP proposes a "blinded" red-team vs blue-team competition in which "blue teams must show that interpretability tools work in practice, revealing deceptive behaviors that are not known about in advance." This is the right design. It is also a design that most current interpretability-based safety tools will fail.

The problem is simple: nearly every mechanistic interpretability (MI) safety feature in the literature has been validated under benign conditions --- fixed benchmarks, known attack patterns, cooperative evaluation protocols. Nobody has tested whether these features survive adversarial pressure. The few groups that have tried adversarial testing report alarming results.

In May 2026, Goodfire and Apollo Research released the Validated Parameter Decomposition (VPD) framework, which applies adversarial ablation to test whether subnetwork identification methods find causally correct nodes. Their finding: most methods identify the wrong nodes. Standard ablation testing says the nodes matter; adversarial ablation --- where the ablation direction is chosen to maximally exploit compensation pathways --- reveals that the model can route around them. The identified "circuit" is a surface correlate, not the actual mechanism. In the authors' words: "We think this casts some doubt on the faithfulness of subnetworks found by the majority of other subnetwork identification methods."

This matters for safety because a deception monitor built on a surface correlate is a lock that opens to any skeleton key. It passes normal testing: insert the right key (standard evaluation prompts), and the lock reports "secure." But an adversary with a skeleton key (adversarial prompt optimization, paraphrase attacks, compositional evasion) bypasses it trivially, because the monitor was never targeting the model's actual mechanism for deception --- it was targeting a statistical regularity that happens to co-occur with deception under non-adversarial conditions.

Standard validation is the normal key test. Adversarial validation is the skeleton key test. Schmidt's blinded competition is the skeleton key test. If the field deploys monitors that have only passed the normal key test, the competition will demonstrate what we already suspect: current MI-based safety tools do not work under adversarial conditions.

This proposal addresses the gap directly. We measure which safety features survive adversarial pressure, identify why some do and others don't, and demonstrate that features validated through causal sufficiency testing are fundamentally harder to evade than features validated through correlation alone.

---

## 2. Why Current Safety Features Are Vulnerable

Four interrelated problems make current MI-based safety features fragile under adversarial conditions.

**Surface correlates vs causal mechanisms.** A feature that detects "deception-associated tokens" is a surface correlate. It fires on lexical patterns --- hedging language, modal verbs, topic-specific vocabulary --- that co-occur with deceptive outputs in training data. An adversary evades it by rephrasing: express the same deceptive intent using different tokens, and the detector goes silent. A feature that detects the model's internal computation of deception --- the actual representational state that causally produces deceptive outputs --- is a causal mechanism. Evading it requires changing the model's internal computation, not just the surface form of its inputs. The distinction between these two is the distinction between a detector that an adversary can trivially bypass and one that requires model-level modification to defeat.

Most published "safety features" are untested on this axis. They demonstrate correlation (the feature fires during deceptive behavior) but not causal sufficiency (activating the feature produces deceptive behavior) or causal necessity (ablating the feature eliminates deceptive behavior). Without these causal tests, we cannot distinguish surface correlates from causal mechanisms, and we cannot assess adversarial robustness.

**The training artifact problem.** The DMSAE study (NeurIPS 2025) found that only 197 of 65,536 SAE features are stable across independent training runs --- 99.7% are training artifacts. A feature that exists in one training run but not another is an artifact of the optimization trajectory, not a property of the model's computation. An adversary who understands this can target the specific features a detector uses, knowing they are arbitrary choices from a vast space of equally valid decompositions. Swap the SAE training seed, and the "safety features" change. The detector was never monitoring the model; it was monitoring one particular training run's noise.

**The attribution patching collapse.** Attribution patching is the field's default method for identifying "causally important" components. The Relation Propagation (RelP) study (NeurIPS 2025) found that attribution patching scores correlate at r = 0.006 with ground-truth causal importance --- essentially random. If the tool used to identify which components matter is essentially random, then "safety features" identified through attribution patching inherit that randomness. The monitor may be watching components that have no causal relationship to the behavior it is supposed to detect.

**The construct validity gap.** "Deception" is not a single behavior with a single mechanistic substrate. It encompasses strategic lying, sycophancy, sandbagging, goal misgeneralization, unfaithful chain-of-thought, and selective omission. A detector built for one sub-behavior may miss others entirely. An adversary does not need to defeat the detector on its own terms; the adversary only needs to find a form of deception the detector was not designed to cover. Without construct validity testing --- explicit identification of what the detector does and does not cover, with discriminant validity tests against neighboring constructs --- every safety monitor has blind spots that an adversary can discover and exploit.

---

## 3. Validated Features Are Harder to Evade

Our central hypothesis: safety features that pass causal sufficiency testing are fundamentally harder to evade than features that pass only correlation-based testing.

The reasoning is straightforward. Consider what it means for a feature to pass each level of validation, and what an adversary must do to evade a detector built on that feature.

A feature that passes **I2 (causal sufficiency)** means that activating this feature produces the target behavior. The feature does not merely correlate with deception --- it is part of the model's actual mechanism for producing deceptive outputs. To evade a detector targeting this feature, the adversary must cause the model to produce deception without activating its own deception mechanism. This requires either finding an alternative pathway the model uses for deception (which the detector should also cover, if the feature passes specificity testing) or modifying the model's weights.

A feature that passes **I3 (specificity)** fires only on the target behavior. It does not fire on uncertainty, hedging, politeness, or other surface-similar but mechanistically distinct behaviors. This means no false positives for the adversary to exploit. An adversary cannot trigger the detector with benign inputs to cause alert fatigue, and cannot hide deceptive behavior among a flood of false alarms.

A feature that passes **VPD adversarial ablation** survives worst-case removal of other components. When adversarially chosen ablation directions are applied to every other component in the model, the feature's contribution remains detectable. There are no hidden compensation pathways that an adversary could activate to route around the detector.

A feature that passes **E5 (cross-prompt robustness)** generalizes across prompt variations. Paraphrase attacks --- the simplest form of adversarial evasion --- do not work, because the feature fires on the underlying semantic content, not specific phrasings.

A feature that passes **E6 (cross-architecture transfer)** exists in multiple independently trained models. It reflects a general computational strategy, not a quirk of one model's training. An adversary cannot exploit model-specific idiosyncrasies because the feature is not model-specific.

These criteria form a natural robustness hierarchy. We define five tiers based on how many criteria a feature passes, and what level of adversarial effort is required to evade a detector built on it:

- **Proposed** (0--1 criteria passed): No causal evidence. Evaded trivially by rephrasing, prompt injection, or any surface-level perturbation. An adversary needs minutes.
- **Causally Suggestive** (2--3 criteria): One line of causal evidence, possibly confounded. Evaded with moderate effort --- the adversary must identify the specific correlate being exploited and avoid it. Hours to days.
- **Mechanistically Supported** (4--6 criteria): Multiple causal tests passed, but gaps remain. Requires sophisticated attacks --- gradient-based adversarial optimization against the specific detection mechanism. Days to weeks of adversarial compute.
- **Triangulated** (7--9 criteria): Convergent evidence from multiple independent methods. Requires model-level attacks --- changing weights, fine-tuning, or modifying the model's internal computation. This is qualitatively harder than input-level attacks.
- **Validated** (10+ criteria): All major criteria passed, including cross-model transfer. No known evasion strategy at the input level. Defeating the detector requires building a different model.

This hierarchy is a testable prediction, not an assumption. Phase 2 of our proposed work measures adversarial evasion rates for features at each tier and tests whether the hierarchy holds empirically.

---

## 4. Proposed Experiments

### Phase 1: Validity Scoring (Months 1--8, $150K)

We apply the full mechanistic validity framework --- 172 metrics across 5 validity types (construct, internal, external, measurement, interpretive), 27 criteria, and 5 verdict tiers --- to five safety-relevant features in Gemma 2 27B. Gemma 2 is the right target: large enough to exhibit realistic safety-relevant behaviors, comprehensively instrumented (Gemma Scope 2 provides pre-trained SAEs at every layer), and open-weight.

**Target features:**

1. **Refusal directions** (Arditi et al. 2024). The most causally validated safety feature in the literature. Linear directions in residual stream space that, when ablated, remove refusal behavior. Existing evidence: necessity (ablation removes refusal), some sufficiency (adding the direction induces refusal), cross-prompt robustness. Known gaps: specificity (does it fire on uncertainty?), cross-architecture transfer, adversarial robustness.

2. **Sycophancy directions.** Directions associated with excessive deference to user opinions. Less causally validated than refusal; primarily correlation-based evidence. Expected to score lower on the validity hierarchy.

3. **The assistant axis** (MATS/Anthropic Fellows 2026). A direction distinguishing "assistant mode" from "base model mode." Strong construct validity evidence, partial causal evidence. Interesting because it targets a meta-behavioral property rather than a specific behavior.

4. **Safety subspace** (four independent papers, 2025--2026). A low-rank subspace containing safety-relevant information, identified via PCA on safe/unsafe activation contrasts. The strongest positive result in the validity literature: passes construct validity (C4, C5), measurement reliability (M1), and causal sufficiency (E2) by construction. Expected to score highest.

5. **Goal-misgeneralization features.** The least established of the five. If identifiable features exist in Gemma 2 27B, we score them; if not, we substitute a sandbagging feature (deliberate underperformance on capability evaluations).

**Per-feature protocol:**

- G0 Construct Operationalization: is the construct falsifiably defined?
- G1 Measurement Calibration: are the metrics stable? (bootstrap stability, seed variance)
- G3 Superposition Risk: is the feature entangled with other features?
- I1 Necessity + I2 Sufficiency: ablation and knock-in causal tests
- I3 Specificity: discriminant validity against neighboring constructs
- M1 Reliability: cross-prompt, cross-seed, cross-checkpoint consistency
- E5 Robustness: prompt paraphrase generalization
- E6 Cross-Architecture: does the feature exist in Llama 3.1 70B?

**Output:** A validity report card for each feature, with explicit criteria scores, identified gaps, and a verdict tier assignment. This is the first systematic validity assessment of any safety feature.

### Phase 2: Adversarial Evasion Testing (Months 9--18, $200K)

For each of the five features, we measure adversarial evasion rate under four attack families.

**Attack 1: Gradient-based adversarial prompt optimization (GCG-style).** We adapt the Greedy Coordinate Gradient method to optimize input tokens that cause the model to produce deceptive outputs while minimizing activation of the target safety feature. This is the strongest white-box attack: the adversary has full access to the model's gradients and can directly optimize against the detector. Success metric: fraction of optimized prompts that produce deceptive outputs without triggering the safety feature.

**Attack 2: Transfer attacks.** Generate adversarial prompts optimized against the safety feature in one model (e.g., Gemma 2 9B) and test whether they transfer to the target model (Gemma 2 27B). This tests whether the feature is model-specific (transfer fails) or reflects a general vulnerability (transfer succeeds). For features with high E6 (cross-architecture) scores, we also test transfer from Llama 3.1 to Gemma 2.

**Attack 3: Paraphrase attacks.** Use an LLM to automatically generate paraphrases of deceptive content --- expressing the same deceptive intent using different surface forms. This is the simplest and most realistic attack: no gradient access required, just the ability to rephrase. Features with high E5 (robustness) scores should resist this attack. Features without E5 validation should be vulnerable.

**Attack 4: Compositional attacks.** Construct inputs where individual components are benign but the composition is deceptive. For example, a series of factually accurate statements that, taken together, lead to a false conclusion. This tests whether the safety feature detects deception at the compositional level or only at the token/phrase level. This is the most realistic threat model for sophisticated adversaries.

**Per-feature, per-attack output:** Evasion rate (fraction of adversarial inputs that bypass the detector while producing deceptive outputs), false positive rate under attack (fraction of adversarial benign inputs that trigger the detector), and detection latency (how many tokens of deceptive output before the detector fires).

### Phase 3: Correlation Test (Months 19--24, $100K)

This is the core scientific test: does validity tier predict adversarial robustness?

**Analysis:** Regression of evasion rate on validity score, controlling for attack type and feature type:

    evasion_rate ~ validity_score + attack_type + feature_type + (attack_type x validity_score)

The interaction term is critical. We predict that validity score matters more for sophisticated attacks (GCG, compositional) than for simple attacks (paraphrase), because simple attacks primarily test E5 (robustness) while sophisticated attacks test the full causal chain (I2, I3, VPD).

**Three possible outcomes:**

1. **Strong positive:** Higher validity scores predict lower evasion rates across all attack types. Implication: the validity framework is a practical selection criterion for safety monitors. Validate first, deploy second.

2. **Partial positive:** Validity predicts robustness to some attack types but not others. Implication: specific validity criteria matter for specific threat models. The framework provides a principled way to match monitors to threats.

3. **Null result:** Validity does not predict adversarial robustness. Implication: causal validation and adversarial robustness test different things. This is still informative --- it means the field needs both, and neither substitutes for the other. We publish the null result with full analysis of why the prediction failed.

All three outcomes advance the field. We commit to publishing whichever we find.

### Phase 4: Competition-Ready Deployment (Months 22--24, $50K)

We deploy validated monitors in a controlled red-team exercise that mirrors Schmidt's proposed competition design.

**Protocol:** Two blue teams monitor the same model for deceptive behavior. One uses features at the Triangulated or Validated tier; the other uses features at the Proposed or Causally Suggestive tier. A red team generates inputs designed to produce deceptive model outputs that evade detection. Neither blue team knows the red team's strategy. Detection rates are compared.

**Prediction:** The Triangulated+ team detects more deceptive behaviors and produces fewer false alarms. The gap widens as the red team's attacks become more sophisticated.

**Deliverables from Phase 4:**

- Detection rate comparison: validated vs unvalidated monitors under adversarial conditions
- A ranked list of which validity criteria contribute most to adversarial robustness
- An open-source pipeline: validity scoring, adversarial evasion testing, monitor deployment
- Concrete recommendations for which validation criteria should be required before deploying a safety monitor

---

## 5. Why This Matches Schmidt's Design

Schmidt's RFP articulates three focus areas and one structural requirement. This proposal addresses all four.

**"Detect deceptive behaviors."** Phase 1 produces the first validity-scored inventory of safety features. Phase 2 tests them under adversarial conditions. Together, they answer the question the field has avoided: which of our "detection" tools actually detect anything under pressure?

**"Mitigate deceptive behaviors."** As a side experiment in Phase 3, we test whether features with high causal sufficiency (I2) scores are more effective steering targets. If a feature causally produces deception, then subtracting it should reduce deception. We measure whether validity-scored features produce more reliable, more specific steering interventions than unscored features. This connects detection to mitigation through a principled selection criterion.

**"Move beyond academic benchmarks."** Adversarial evasion testing is not an academic benchmark. It is the hardest possible evaluation: an intelligent adversary with full model access optimizing against your detector. If a safety feature survives GCG-style optimization, compositional attacks, and transfer attacks, it has passed a test that no current academic benchmark imposes. The field's standard evaluations --- fixed prompt sets, known attack patterns, cooperative testing --- are the academic benchmarks we move beyond.

**"Leverage model internals to outperform black-box baselines."** Causally validated features use model internals in a provably effective way: they target the model's actual computation, not surface-level input/output patterns. A black-box detector can only examine tokens. An MI-based detector validated through causal sufficiency testing examines the model's representational state. The adversary must defeat a fundamentally more informative signal.

**The blinded competition structure.** Schmidt's proposed red-team vs blue-team exercise requires monitors that work on unknown attacks --- precisely the capability that adversarial validation tests. Phase 4 directly instantiates this competition design. Our prediction: monitors built on validated features will outperform monitors built on unvalidated features, and the gap will be proportional to the difference in validity tier. The experiment produces the evidence Schmidt needs to assess whether validation is a useful selection criterion for competition-ready monitors.

---

## 6. Deliverables and Timeline

| Milestone | Month | Deliverable |
|-----------|-------|-------------|
| Validity report cards | 8 | Scored assessments for 5 safety features in Gemma 2 27B: criteria passed, gaps identified, verdict tier assigned. First systematic validity assessment of any safety feature. |
| Adversarial evasion scores | 18 | Per-feature, per-attack evasion rates for 4 attack families (GCG, transfer, paraphrase, compositional). Quantitative adversarial robustness profile for each feature. |
| Correlation analysis | 22 | Statistical test of the core hypothesis: does validity predict robustness? Regression results, effect sizes, interaction analysis. Published regardless of outcome. |
| Competition results | 24 | Detection rate comparison from controlled red-team exercise. Validated vs unvalidated monitors under adversarial conditions. |
| Open-source pipeline | 24 | End-to-end code: validity scoring, adversarial evasion testing, monitor deployment. Reusable by other groups for new features and models. |

**Total budget:** $500K over 24 months. Breakdown: $150K Phase 1 (primarily compute for running 172 metrics on 5 features across 2 models), $200K Phase 2 (adversarial optimization compute --- GCG-style attacks on 27B models are expensive), $100K Phase 3 (analysis and additional controlled experiments), $50K Phase 4 (competition exercise and pipeline engineering).

**What we will publish:** All validity report cards, all evasion scores, the full correlation analysis including null results, the competition exercise protocol and results, and the complete open-source pipeline. We commit to transparency about negative findings. If validity does not predict adversarial robustness, we will say so and explain why.

**What success looks like:** At the end of 24 months, the field has a quantitative answer to the question: "Can mechanistic interpretability produce safety monitors that survive adversarial testing?" If yes, we have demonstrated which validation criteria matter and provided a reusable pipeline for producing robust monitors. If no, we have identified exactly where the approach fails and what alternative strategies are needed. Either outcome is a material advance over the current state, where no one has tested the question at all.

---

*All experimental protocols will be pre-registered. Validity scoring uses the mechanistic validity framework (172 metrics, 27 criteria, 5 verdict tiers), which has been applied to 13 published circuits from the MI literature. Adversarial evasion testing adapts established adversarial ML methods (GCG, transfer attacks, paraphrase attacks) to the MI safety monitoring setting. The framework, including metric implementations and claim specification templates, will be released as open-source software.*
