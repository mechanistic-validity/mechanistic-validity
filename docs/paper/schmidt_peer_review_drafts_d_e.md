# Peer Review: Schmidt Sciences RFP Drafts D and E

**Reviewer context**: Schmidt Sciences AI Interpretability RFP, $300K--$1M, 1--3 years. Focus: detect/mitigate deceptive behaviors, move beyond academic benchmarks, leverage model internals. Competition structure. Deadline May 26 2026. Out-of-scope: "Work on interpretability methods without a clear application to studying deceptive behaviors in LLMs."

---

## Summary Assessment

Both drafts represent a significant improvement over Drafts A--C. The core advance: they lead with the PI's own behavioral research on deceptive failure modes (six papers, five frontier models, four domains) and frame the validity framework as a tool for evaluating deception monitors --- not as a general MI infrastructure project. This directly satisfies the RFP's explicit requirement that proposals must "assess impact on monitoring, steering, or assessing deceptive behaviors."

| | Draft D: Behavioral Ground Truth | Draft E: Monitor Trust |
|---|---|---|
| **Words** | ~3,100 | ~2,900 |
| **Lead with** | "We already know how models deceive" | "Should you trust a deception detector?" |
| **Tone** | Research program narrative | Pointed question, then answer |
| **Core argument** | We have the behavioral ground truth; now find the mechanism | The trust question is unsolved; we have the framework to solve it |
| **Strongest section** | §1 paper taxonomy (compelling, unique) | §1 opening question (instantly grabs reviewer) |
| **Weakest section** | §3 framework description (feels tacked on) | §2 paper taxonomy (slightly less vivid than D's) |
| **Schmidt fit** | VERY HIGH | VERY HIGH |
| **Desk rejection risk** | LOW | LOW |

Both are dramatically better positioned for this RFP than Drafts A--C.

---

## Major Comments

### MC1. Draft E's opening is the best hook across all five drafts [Draft E]

"Suppose a mechanistic interpretability team hands you a 'deception detector.' Should you trust it?" This is the single best opening sentence across all five drafts. It puts the reviewer in the position of the person who will USE the research output. It makes the proposal's value proposition immediate. Draft D's opening ("We already know how models deceive") is strong but positions the PI as the protagonist. Draft E positions the REVIEWER as the protagonist. For a grant committee deciding what to fund, that's more effective.

**Recommendation**: Whatever final version is produced, use Draft E's opening question or a close variant.

### MC2. Draft D's Section 1 is the most compelling evidence of PI uniqueness [Draft D]

The six-paper taxonomy in Draft D's §1 is the proposal's single greatest asset. No other applicant will have this. Six papers, five frontier models, four domains, three mechanistic archetypes --- this is not a plan to study deception. This is a demonstrated research program with results. Draft E covers the same material in §2 but more compactly and less vividly (the SunkBench archetype table is preserved, which is good, but the individual paper descriptions are thinner).

**Recommendation**: The final version needs the full vividness of Draft D's paper descriptions (especially the "internal over-committer / silent follower / surface detector" narrative) regardless of which opening is used.

### MC3. Both drafts handle the RFP deception requirement correctly [Both]

The previous drafts (A--C) framed deception as a special case of general MI validation. These drafts frame deception monitoring as the PRIMARY research question, with the validity framework as the tool. This is the correct framing for this RFP. Specific evidence:
- Both open with deception
- Both frame every experiment in terms of detecting/mitigating deceptive behaviors
- Both connect to Schmidt's competition design through deception-specific scenarios
- Neither contains the phrase "deception is not special" or "general MI validation"

No changes needed on this axis.

### MC4. The "Independent Researcher" PI designation is a significant risk [Both]

Both drafts list PI: Elliot Tower (Independent Researcher). For a $750K grant, this is the biggest red flag a reviewer will see. It raises questions about:
1. Where will the money go? (no institutional overhead, but also no institutional infrastructure)
2. Who provides compute? (no university cluster)
3. What happens if the PI is unavailable? (no department to absorb the project)
4. Is this a real research group? (no lab, no students, no institution)

The co-PI (Gagan Bansal) and collaborator (Ivan, Edinburgh) mitigate this somewhat, but the proposal needs to address institutional risk directly.

**Recommendations**:
- Consider whether any institutional affiliation is possible (even a visiting position or research affiliate status at UMass through Jensen, or Edinburgh through Ivan). A sentence like "Hosted at [Institution]" changes the reviewer's risk calculus dramatically.
- If genuinely independent, add a brief "Institutional Plan" paragraph: where compute will be hosted (cloud?), how funds will be administered (fiscal sponsor?), what governance structure exists. Schmidt reviewers will ask.
- Emphasize that the behavioral benchmarks and validity framework are already built and open-source --- reducing the institutional risk because the infrastructure doesn't depend on any one person.

### MC5. The Gemma 2 27B replication step needs justification [Both]

Both drafts propose replicating GEB-Bench and SunkBench experiments in Gemma 2 27B as the first step. But the original papers used DeepSeek-R1, GPT-5.4, GPT-4.1, and Llama-3.3-70B --- models where the behavioral ground truth is already established. The proposal should explain:
1. Why replicate in Gemma 2 specifically? (open weights needed for mechanistic analysis --- can't do activation analysis on closed-API models)
2. What if Gemma 2 27B doesn't exhibit the same deceptive behaviors? (it's smaller than most models tested; the behaviors may not transfer)
3. Is there a fallback? (Llama 3.1 70B is open-weight and larger)

**Recommendation**: Add 2--3 sentences justifying the Gemma choice (open weights + Gemma Scope SAEs) and acknowledging the risk with a fallback plan. "If Gemma 2 27B does not exhibit sunk-cost lock-in, we will use Llama 3.1 70B, which is open-weight and closer in scale to the models where these behaviors were originally observed."

### MC6. Paper M and Paper V are underused [Both]

The clinical hallucination paper (M) and the code vulnerability paper (V) are mentioned but their implications for mechanistic monitoring are underdeveloped. These papers demonstrate something specific and important: **black-box monitoring fails in characterizable ways** (confirmation hysteresis, stance-induced suppression). The proposal's implicit argument is that whitebox (mechanistic) monitoring should bypass these failures by reading internal state directly. But neither draft tests this claim explicitly.

**Recommendation**: Add an experiment (or fold into Experiment 2): "Do mechanistic deception features bypass confirmation hysteresis? When binary-framed behavioral monitoring fails to detect a hallucination, does the mechanistic feature still fire?" This is a direct comparison of whitebox vs. blackbox monitoring, which is exactly what the RFP means by "leverage model internals to outperform black-box baselines."

### MC7. The robustness hierarchy regression is underpowered [Both]

Same issue as Draft C (from the previous review): with only 5 features × 4 attack types = 20 data points, a regression of evasion rate on validity score will have low statistical power. Both drafts mention this implicitly ("sub-feature granularity" in the composite, "permutation-based testing" nowhere in D or E).

**Recommendation**: Explicitly state how you'll handle power: (a) use per-criterion scores rather than aggregate tiers (increasing from 5 data points to 27 × 5 = 135), (b) use permutation testing rather than parametric regression, (c) frame as effect-size estimation rather than hypothesis testing.

### MC8. Draft E's "Three Possible Outcomes" section is excellent [Draft E]

This is the best outcomes section across all five drafts. It's honest, specific, and makes the case that the proposal is informative regardless of result. Outcome 3 ("mechanistic features cannot reliably distinguish deceptive episodes") is especially strong --- it positions a null result as field-redirecting rather than just "we tried and failed." Draft D covers this in the feasibility paragraph but less compellingly.

**Recommendation**: Include Draft E's §5 in the final version, verbatim or close to it.

---

## Minor Comments

### m1. The SunkBench archetype table should appear early [Both]

The three-archetype table (internal over-committer / silent follower / surface detector) is the most memorable visual in both drafts. In Draft D it appears in §1 (good). In Draft E it appears in §2 (still fine, but slightly delayed). In the final version, this table should appear on page 1. It's the single best illustration of what the proposal is about.

### m2. "Nathan [Last Name]" is a placeholder [Both]

Fill this in before submission. If Nathan hasn't confirmed yet, remove him rather than submit with a placeholder.

### m3. The preprint/DOI section needs concrete URLs [Both]

Both drafts end with "[site URL]" and "[Zenodo DOI]" placeholders. These need to be real before submission. Even if the papers are rough, having live links is dramatically better than "[TBD]". Zenodo DOIs can be minted in minutes.

### m4. Draft D's §3 feels like an interruption [Draft D]

In Draft D, the flow goes: §1 (our papers, vivid) → §2 (crisis evidence, urgent) → §3 (validity framework, suddenly abstract and condensed). The tonal shift is jarring. Draft E handles this better by weaving the framework into the "solution" narrative.

**Recommendation**: In the final version, either integrate the framework description into the experimental protocol (as Draft E does) or move it to an appendix/footnote with the preprint link. Don't break the narrative momentum with a condensed framework summary.

### m5. Both drafts could cite Steinhardt more explicitly [Both]

Jacob Steinhardt's behavioral evaluation argument (May 2026) --- that the field under-invests in measuring what models DO versus what they CAN do --- is a direct endorsement of the proposal's approach. The PI's six papers are exactly "measuring what models do." Neither draft mentions Steinhardt, which is a missed opportunity for external validation of the research direction.

### m6. "Deceptive failure modes" vs. "deceptive behaviors" framing [Both]

Both drafts use "deceptive failure modes" in places. For the RFP, "deceptive behaviors" is the correct term --- it matches the RFP's language exactly. "Failure modes" sounds like the model is broken; "behaviors" sounds like the model is doing something the RFP is designed to study. Small language choice, but alignment with RFP terminology matters.

### m7. Neither draft addresses the page/word limit [Both]

The RFP is via SurveyMonkey Apply, which typically has specific field lengths. Both drafts are ~3,000 words, which may need to be compressed to fit form fields. The user should check the actual submission format and identify which sections map to which fields. The current drafts are structured as essays; the submission may need to be restructured as form responses.

---

## Recommendation: Composite Strategy

**Opening**: Draft E's question ("Should you trust a deception detector?") + crisis evidence bullets (compact).

**The behavioral research program**: Draft D's §1 in full vividness. All six papers, the three archetypes, the "this IS the deception monitoring problem" pivot. This is the PI's differentiator --- don't compress it.

**The monitor trust problem stated concretely**: Draft E's Paper M / confirmation hysteresis narrative. "Even a monitor with the right evidence can fail." This motivates the validity framework naturally.

**The framework**: One tight paragraph + link. Don't break narrative momentum. Draft E's approach (framework as "the solution") rather than Draft D's approach (framework as separate section).

**Experiments**: Both drafts have essentially the same experimental plan; use Draft E's numbering (4.1--4.10) for clarity. ADD the whitebox-vs-blackbox experiment from MC6.

**Outcomes**: Draft E's §5 verbatim.

**Team**: Same, but address MC4 (institutional risk) explicitly.

**Schmidt alignment**: Draft E's format (bullet-matched to RFP language).

---

## Scoring (1--5 scale, 5 = best)

| Criterion | Draft D | Draft E |
|---|---|---|
| Alignment with RFP | 4.5 | 5 |
| Deception focus (not general MI) | 5 | 5 |
| Scientific rigor | 4 | 4 |
| Novelty / differentiation | 5 | 4.5 |
| Feasibility | 3.5 | 3.5 |
| Testable predictions | 4 | 4.5 |
| Persuasiveness | 4.5 | 5 |
| PI credibility (papers + track record) | 5 | 5 |
| Institutional risk | 2.5 | 2.5 |
| **Overall** | **4.2** | **4.4** |

Draft E edges ahead on RFP alignment and persuasiveness (the opening question is extremely effective). Draft D has slightly stronger novelty presentation (the paper taxonomy is more vivid). Both score 5 on deception focus and PI credibility --- a massive improvement over Drafts A--C. The institutional risk (2.5) is the single biggest weakness and affects both equally.

The composite strategy outlined above should score ~4.8, with institutional risk remaining the key variable.
