# Peer Review: Three Schmidt Sciences RFP Drafts

**Reviewer context**: Schmidt Sciences AI Interpretability RFP, $300K–$1M, 1–3 years. Focus: detect/mitigate deceptive behaviors, move beyond academic benchmarks, leverage model internals. Red-team/blue-team competition structure. Deadline May 26 2026.

---

## Summary Assessment

Three drafts pitch the same core framework (172 validity metrics, 5 verdict tiers) from different angles. All share the same evidence base (DMSAE 99.7%, MechEvalAgent 93%, SAEBench 2/6, RelP r=0.006, VPD adversarial ablation), the same target features (refusal, sycophancy, assistant axis, safety subspace, sandbagging), the same model (Gemma 2 27B), and the same budget ($500K). The question is which framing best matches what Schmidt reviewers want to fund.

| | Draft A: Validation Layer | Draft B: Clinical Trials | Draft C: Adversarial |
|---|---|---|---|
| **Words** | 3,517 | 3,380 | 3,238 |
| **Lead with** | Infrastructure gap | Medical analogy | Adversarial threat |
| **Tone** | Methodological authority | Regulatory urgency | Security pragmatism |
| **Core argument** | Evaluation criteria are the bottleneck | Unvalidated features are thalidomide | Validated features survive red-teaming |
| **Strongest section** | §2 Framework description | §3 Trial protocol | §3 Robustness hierarchy |
| **Weakest section** | §5 Team (placeholder) | §1 Opening (analogy may feel strained) | §4 Phase 3 (regression may be underpowered) |
| **Schmidt fit** | HIGH | MEDIUM-HIGH | HIGH |

---

## Major Comments

### MC1. All three drafts bury the lede on deception [Drafts A, B, C]

Schmidt's RFP is explicitly about *deceptive behaviors*. All three drafts spend 40-60% of their word budget on the general validation crisis before getting to deception specifically. A Schmidt reviewer's first question is: "What does this do for deception detection?" All three drafts answer this question well, but they answer it on page 3, not page 1.

**Recommendation**: The final version should open with a deception-specific hook. Example: "Five groups have published features claimed to detect deceptive behavior in LLMs. None has been validated against adversarial evasion, cross-model transfer, or discriminant validity. We propose the first systematic validity trial."

### MC2. The "already built" claim needs evidence [All drafts]

All three drafts claim the framework is "already built" with 172 metrics, 13 case studies, etc. But no URL, preprint, or publication is cited for the framework itself. A skeptical reviewer will ask: "Where can I see this? Has it been peer-reviewed?" Draft B mentions "(under review, 2026)" in passing but doesn't give an arXiv ID.

**Recommendation**: Include a preprint link or supplement. Even a GitHub repo URL would help. The "already built" claim is the proposal's strongest differentiator, but it's currently unverifiable.

### MC3. The 5-feature selection is identical across drafts but needs justification hierarchy [All]

All three propose the same 5 features, but the rationale for *why these five* varies in quality. Sandbagging circuits ("if proposed by the time evaluation begins") is a weak candidate — it signals uncertainty about whether there's anything to evaluate. Goal-misgeneralization features (Draft C) have the same problem.

**Recommendation**: Lead with the 3 strongest candidates (refusal, safety subspace, sycophancy). Present assistant axis as a stretch candidate with specific interest (meta-behavioral property). Replace the 5th slot with an explicit statement: "The fifth slot is reserved for the most mature safety feature published before the evaluation period begins." This is honest and signals awareness that the field is moving fast.

### MC4. The $500K budget is underleveraged [All]

All three propose $500K — the lower end of $300K–$1M. For a proposal claiming to be the "missing validation infrastructure for AI safety," $500K feels unambitious. The budget breakdown across all three is thin: "$150K compute, 1-2 postdocs." No advisory board, no external red-team collaboration, no cross-institution partnerships.

**Recommendation**: Propose $750K–$800K. Add: external adversarial ML collaborator for Phase 3 ($50K), advisory board with 2-3 external researchers ($25K), cross-model compute for E6 testing on Llama/Qwen ($50K). This signals confidence and seriousness without being excessive.

### MC5. Draft B's clinical trial analogy is both the greatest strength and risk

The thalidomide opening is powerful and memorable. But two risks:
1. **Proportionality**: Comparing SAE feature instability to birth defects may strike some reviewers as overwrought. The analogy works for the *structure* (Phase I/II/III) but breaks at the *stakes* level.
2. **Misfit with ML audience**: Grant reviewers for this RFP will likely be ML researchers, not medical scientists. The analogy may feel imported rather than native.

**Recommendation**: Keep the Phase I/II/III structure (it's excellent), soften the thalidomide opening to one sentence rather than a full paragraph, and lead with the refusal-direction-as-jailbreak story (which is native to the ML safety domain and makes the same point about double-edged features).

### MC6. Draft C's adversarial robustness hierarchy is the most testable claim

Draft C's five-tier robustness hierarchy (Proposed → Validated, mapped to adversarial effort levels from "minutes" to "requires building a different model") is the single most concrete, testable, and fundable claim across all three drafts. It gives reviewers a clear mental model and a falsifiable prediction.

**Recommendation**: This hierarchy should appear in whatever final version is produced, regardless of which draft's framing is chosen. It's the clearest answer to "what will we learn?"

### MC7. None of the drafts addresses the team question adequately

All three leave team details as placeholders. For a $300K–$1M grant, reviewers will weight team credibility heavily. "PI: [Name]" signals the draft is not submission-ready.

**Recommendation**: User-specific. Fill in PI, institution, 2-3 collaborators with specific roles, and any track record with the framework itself.

---

## Minor Comments

### m1. Draft A §2 is the clearest framework description
The five validity types + verdict tiers explanation in Draft A is crisper than the equivalent in B or C. Use A's version as the basis regardless of final framing.

### m2. Draft C's attack taxonomy is strong
The four attack families (GCG, transfer, paraphrase, compositional) are specific, well-justified, and map cleanly to different validity criteria. This is missing from Drafts A and B's Year 3 descriptions, which mention "adversarial prompt optimization" generically.

### m3. All drafts need a "risks and mitigations" section
Grant reviewers look for honest risk assessment. Key risks: (1) target features may be too immature for full evaluation, (2) Gemma 2 27B may not exhibit realistic deceptive behavior, (3) the correlation between validity and robustness may be null. All drafts mention null results briefly but don't frame risks proactively.

### m4. Citation consistency
Draft B has the most complete reference list. Drafts A and C cite papers inline but don't include a reference section (A) or include a minimal one (C). The final version needs a proper reference list.

### m5. The ICML 2026 parallel is undersold
The "Interpretability Can Be Actionable" paper (Orgad, Barez et al., ICML 2026) is the strongest external validation of the proposal's thesis. Draft A mentions it in §2; Drafts B and C barely reference it. The final version should cite it prominently in the opening: "ICML 2026 accepted a 12-author position paper arguing that evaluation criteria — not new methods — are the missing ingredient. This proposal provides the evaluation criteria."

### m6. Draft A's "deception is not special" argument is philosophically strong but strategically risky
Telling Schmidt "deception detection is not a separate research agenda" may read as dismissive of their specific focus. Reframe: "deception detection *benefits from* general validity infrastructure" rather than "deception detection *is just* general validity."

### m7. The counter-prediction in Draft A §4 (Year 2) is excellent
"Perhaps any reliably detectable direction is a good steering target regardless of causal status" — this is the kind of honest counter-prediction that signals scientific maturity. Include it in the final version.

---

## Recommendation: Composite Draft Strategy

**Lead framing**: Draft C's adversarial focus. Schmidt's RFP emphasizes red-team/blue-team and "move beyond academic benchmarks." The adversarial angle is the most directly responsive.

**Opening**: Draft C's adversarial threat (§1), shortened. Add deception-specific hook from MC1.

**Problem statement**: Draft A's crisis evidence (§1). It's the most complete and impactful.

**Framework description**: Draft A's §2 (clearest), condensed by 30%.

**Protocol**: Draft B's Phase I/II/III structure. It's the most intuitive and memorable framing for the methodology. Drop "clinical trials" from the title; keep the gating structure.

**Experiments**: Draft C's four phases. The attack taxonomy and robustness hierarchy are the most concrete.

**Robustness hierarchy**: Draft C's §3. This is the testable core claim.

**Broader impact**: Draft A's §6 + Draft B's "either outcome is informative" framing.

**Counter-predictions**: Draft A's Year 2 counter-prediction.

**Budget**: Scale to $750K as in MC4.

This composite takes the strongest section from each draft, eliminates redundancy, and leads with what Schmidt reviewers want to hear: "Your red-team competition will fail unless monitors are validated. Here's how to validate them."

---

## Scoring (1–5 scale, 5 = best)

| Criterion | Draft A | Draft B | Draft C |
|---|---|---|---|
| Alignment with RFP | 4 | 3.5 | 4.5 |
| Scientific rigor | 4.5 | 4 | 4 |
| Novelty/differentiation | 4 | 4.5 | 4 |
| Feasibility | 4.5 | 4 | 4 |
| Clarity of writing | 4 | 4.5 | 4 |
| Testable predictions | 3.5 | 3.5 | 5 |
| Persuasiveness | 4 | 4.5 | 4.5 |
| **Overall** | **4.1** | **4.1** | **4.3** |

Draft C edges ahead on RFP alignment and testable predictions. Draft B has the most memorable framing. Draft A has the strongest scientific foundation. The composite strategy outlined above should score ~4.7.
