# Mechanistic Validity — Perplexity deep-research queries

Purpose: gather the evidence base for the claim-level audit (mini-lit-review per claim),
sharpen the contribution delta, and ground Section 6 (self-validation). Run in Perplexity
deep research; paste results back per section.

## A. Per-claim audit evidence (the audit is claim-level, not paper-level)

**Reusable template** — swap in each claim:
> Comprehensive review of all published follow-up work, replications, criticisms, and failed
> replications of [CLAIM] in mechanistic interpretability of transformer language models,
> 2022–2026. Prioritize papers testing specificity, cross-model/cross-scale generalization,
> method-conditional faithfulness, and alternative/non-unique circuits. Give primary citations.

1. **IOI circuit** (Wang et al. 2023, GPT-2 Small): every follow-up on faithfulness method-dependence, specificity/negative controls, circuit non-uniqueness, and cross-model replication (incl. Miller et al. 2024, Meloux et al. 2025).
2. **Induction heads** (Olsson/Elhage 2022): cross-architecture and cross-scale replications, the in-context-learning connection, and any critiques or boundary conditions.
3. **Linear probing as evidence of model computation**: critiques of "decodable ≠ used," causal probing / DAS, control tasks (Hewitt & Liang), selectivity.
4. **Gender-bias circuits / causal mediation** (Vig et al. 2020): work on whether "bias" is separable from grammatical-gender knowledge in model representations.

## B. Sharpen the contribution delta (Reviewer 2)

5. Construct validity, measurement theory, and psychometrics applied to ML benchmarks and LLM evaluation (Bean et al. 2025, Freiesleben et al. 2025, and adjacent) — who has done this, on what unit of analysis, and what remains unaddressed for *causal mechanistic claims*.
6. Frameworks for evaluating interpretability-method reliability and faithfulness (MIB, SAEBench, causal scrubbing, faithfulness-metric critiques) — newest 2025–2026 work and how they define "validity."

## C. Ground Section 6 — self-validation (Reviewers 1, 4)

7. Preregistration and registered reports in machine learning / computational research: precedents, adoption, tooling, and critiques.
8. Ground-truth and synthetic benchmarks for validating interpretability methods (Tracr, clock/pizza grokking, toy models with known circuits) — candidates for a ground-truth Validated-tier calibration case.

## D. Method-agnostic scope (Reviewer 9)

9. Validity, reliability, and causal testing of SAE features and steering vectors (2024–2026): are features model properties or dictionary artifacts; steering-vector generalization and failure modes.

## E. Keep the motivation current (Reviewers 5, 6)

10. Recent (2025–2026) critiques of mechanistic-interpretability evaluation: circuit non-uniqueness, underdetermination, method-conditional faithfulness, baseline/null failures.
11. Safety-monitor validity: evaluations of deception probes, refusal directions, and chain-of-thought faithfulness, and their robustness to distribution shift or adversarial optimization.
12. Landmark false-positive / non-replication cases across sciences (dead-salmon fMRI, candidate-gene psychiatry, ORBITA stents, cold fusion, COVID-ML shortcuts) — primary citations, to make the cross-science section airtight.

## F. The two new non-circuit case studies (full audits)

**A-SAE. SAE feature.** Pick one representative, well-documented feature claim (e.g., a
Templeton et al. 2024 scaling-monosemanticity feature — deception/sycophancy/safety — or a
Cunningham/Bricken feature). Then:
> For SAE feature [X] claimed to represent [concept C]: what causal evidence exists that the
> model *uses* this feature (feature clamping / activation steering on the feature), not merely
> that [C] is decodable from it? What evidence bears on discriminant validity — is the feature
> separable from neighboring concepts (negation, fiction, uncertainty)? How were its labels
> assigned (max-activating examples vs. causal test)? Any baseline-separation / robustness
> critiques of the specific feature or of SAE feature interpretation generally (2024–2026)?

**A-steer. Steering vector.** Pick one representative vector (refusal direction, Arditi et al.
2024; or a sycophancy/truthfulness vector). Then:
> For steering vector [v] claimed to control [behavior C]: what evidence tests necessity
> (does ablating/removing the direction remove C?), specificity (does it affect only C or also
> unrelated behaviors?), dose-response (does scaling v graduate the effect?), and cross-model
> transfer? Is the direction narrated as a *representation of C* or only as a *behavioral lever*?
> Any evidence the vector fails under distribution shift or adversarial optimization?
