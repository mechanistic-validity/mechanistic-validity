---
title: "Medical Microbiology"
description: "The non-isolability lens: what to do when the component cannot be separated from the system, and how Koch's postulates were revised three times to handle exactly that."
---

# The Medical Microbiology Lens

This lens asks one question: **what counts as causal evidence when the proposed component cannot be isolated?**

The other lenses share an assumption: that the component under study can be separated from its system. Genetics knocks out a gene, neuroscience lesions a region, pharmacology blocks a receptor. Interpretability often cannot isolate — backup name movers assume the role of name movers once those are ablated, and component sets sharing few edges reach comparable performance on the same task. Medical microbiology revised its evidence standards three times in response to the same difficulty, and what it contributes is those revisions rather than a single technique.

## Koch's postulates and their failures

[Koch's postulates](https://en.wikipedia.org/wiki/Koch%27s_postulates) (1890) require that the organism be found in all cases of the disease, isolated in pure culture, and shown to induce the disease when introduced into a healthy host. For culturable bacteria causing acute disease in tractable animal models, these work. For everything else, they break.

Viruses cannot be grown in pure culture. Unculturable organisms — *Mycobacterium leprae* has resisted axenic media for 150 years ([Gotor-Rivera et al. 2025](https://doi.org/10.3389/fmicb.2025.1708557)) — cannot be separated from their host. Rivers noted in 1936 that "it is unfortunate that so many workers blindly followed the rules, because Koch himself quickly realized that in certain instances all the conditions could not be met" (quoted in [Fredricks & Relman 1996](https://doi.org/10.1128/cmr.9.1.18)).

The field's response was not to abandon the postulates but to replace isolation with different forms of evidence, three times:

1. **Rivers (1937)** — for viruses. Dropped the pure-culture requirement. Substituted filtration, serial passage, and histological evidence.
2. **Falkow (1988)** — [molecular Koch's postulates](https://doi.org/10.1093/cid/10.Supplement_2.S274) for genes rather than organisms. The trait belongs to pathogenic members of the taxon, inactivating the gene responsible produces a measurable loss of virulence, and restoring the gene restores pathogenicity.
3. **Fredricks & Relman (1996)** — [sequence-based postulates](https://doi.org/10.1128/cmr.9.1.18) for organisms known only by their nucleic acid. Presence is graded (measured by amplification rather than culture), the sequence is absent or reduced in hosts without the disease, and the relationship tracks treatment.

Each revision dropped a requirement that could not be satisfied and replaced it with a different kind of evidence that could be. What remained constant was the logic: the putative cause must co-occur with the effect, its removal should reduce the effect, and the relationship should hold across hosts or conditions.

## The MI analog

In MI, the component often cannot be cleanly isolated from the computation it participates in. Ablating an attention head removes everything it does — the targeted computation, the backup computations, the residual stream contributions to downstream layers. The "pure culture" analog would be running the component in isolation with no residual stream context, which is incoherent: the component's behavior depends on its inputs. Circuit isolation (complement ablation) is the closest available, but it breaks the distributional assumptions of downstream components.

What transfers from microbiology is not the postulates themselves but the discipline's response to their failure:

- **Graded presence replaces presence-absence.** Fredricks & Relman's guideline (iii) requires that the organism's measured abundance track the disease severity. In MI: the mechanism's measured strength (activation patching effect, SAE feature activation, weight-space alignment) should track the capability's strength across training checkpoints, prompt difficulty, or model scale. This grounds I11 (onset coupling).

- **Evidence must be gathered at the level the claim is pitched at.** Fredricks & Relman's guideline (vi) requires cellular-level evidence for cellular-level claims. In MI: a claim about a subspace requires subspace-level evidence (DAS-IIA, not head-level ablation). A claim about a head requires head-level evidence (activation patching, not weight inspection alone). This grounds V2 (level-evidence match).

- **Partial satisfaction is the norm, not the exception.** Koch's third postulate cannot be satisfied for viruses or for organisms that cannot be grown in culture, and the field kept the postulate rather than weakening it. A criterion a claim cannot satisfy in principle records a limit on the evidence available in that setting, which is information about the setting. This grounds the verdict system's treatment of untested criteria.

## The disanalogy

The analogy has limits. Microbial causation concerns an organism external to the host, whereas a circuit is constitutive of the model implementing it; the closest counterpart to inoculating a naive host is a counterfactual circuit transplant, which moves a component into a model that lacks it but cannot make that model naive to the capability. The postulates also assume one causal organism per disease, an assumption interpretability cannot make while circuits sharing few edges reach comparable performance on the same task.

Causal attribution from detection alone is underdetermined: presence is equally consistent with colonization, past infection, and contamination ([Zautner et al. 2017](https://doi.org/10.3389/fmicb.2017.01210)). The same holds in MI — a head with high activation during a task may be a participant, a bystander, or a residual from an earlier training phase. What transfers is the response to non-isolability rather than the postulates themselves.

## Sources

| Source | Year | Field | Principle |
|---|---|---|---|
| Koch, "Über bakteriologische Forschung" | 1890 | Medical microbiology | **Koch's postulates** — four conditions for establishing that an organism causes a disease: presence in all cases, isolation in pure culture, induction in a new host, re-isolation |
| [Falkow, "Molecular Koch's postulates applied to microbial pathogenicity"](https://doi.org/10.1093/cid/10.Supplement_2.S274) | 1988 | Molecular biology | **Molecular Koch's postulates** — the virulence gene must be present in pathogenic strains, inactivating it must reduce virulence, and restoring it must restore pathogenicity |
| [Fredricks & Relman, "Sequence-based identification of microbial pathogens: a reconsideration of Koch's postulates"](https://doi.org/10.1128/cmr.9.1.18) | 1996 | Clinical microbiology | **Sequence-based postulates** — graded presence, tracking with treatment, cellular-level evidence; replaced isolation with molecular detection |
| [Zautner et al., "More pathogenicity or just more pathogens?"](https://doi.org/10.3389/fmicb.2017.01210) | 2017 | Diagnostic microbiology | **Detection-attribution gap** — multiplex detection detects more organisms, but detection is equally consistent with colonization, past infection, and contamination |
| [Gotor-Rivera et al., "Silence on the plate: revisiting the enigma of *Mycobacterium leprae* cultivation"](https://doi.org/10.3389/fmicb.2025.1708557) | 2025 | Microbiology | **Non-cultivability as a permanent obstruction** — some pathogens have never been grown in pure culture; Koch's second postulate is structurally unsatisfiable for them |
| [Jamrozik et al., "Key criteria for the ethical acceptability of COVID-19 human challenge studies"](https://doi.org/10.1016/j.vaccine.2020.10.075) | 2021 | Bioethics | **Controlled human infection model** — ethical conditions for deliberate infection of healthy volunteers; challenge cohorts bounded to healthy adults aged 18–30 |

## Validity type: [Internal validity](/mechanistic-validity/framework/validity-types/internal/)

> **Non-isolability does not excuse untested criteria.** Koch's field kept the postulate it could not satisfy rather than silently dropping it. A criterion that returns Untested is a statement about the evidence, not a free pass. The verdict must name it.

Medical microbiology contributes to the framework not through specific criteria of its own — unlike genetics or pharmacology, it does not ground a numbered criterion — but through three principles that shape how the criteria system works:

1. **Graded presence.** The mechanism's measured strength should co-vary with the capability's strength. This is the operating principle behind I11 (onset coupling) and I12 (offset coupling): onset coupling adapted from guideline (iii) of the Fredricks & Relman sequence-based postulates.

2. **Level-matched evidence.** Evidence must be gathered at the level the claim is pitched at. This is the operating principle behind V2 (level-evidence match).

3. **Partial satisfaction as verdict, not failure.** A claim that cannot satisfy a criterion in principle receives Untested on that criterion, not a penalty. This shapes the verdict system's treatment of inapplicable criteria.

## Evidence patterns

| Evidence pattern | What it establishes | Recommended language |
|---|---|---|
| Mechanism strength tracks capability across checkpoints | Graded co-occurrence; onset coupling | "Mechanism and capability co-emerge during training" |
| Mechanism persists after capability is fine-tuned away | Offset coupling fails; mechanism may be architectural | "Mechanism persists after capability removal; not tightly coupled" |
| Evidence is head-level for a subspace-level claim | Level mismatch; V2 not satisfied | "Head-level ablation; subspace claim requires subspace-level evidence" |
| Criterion structurally unsatisfiable | Limit of the evidence setting | "I2 returns Untested: circuit transplant into a model naive to the capability is incoherent for this architecture" |

## Verdicts

Medical microbiology shapes the verdict system rather than gating specific transitions:

- **All tiers:** An unsatisfiable criterion is reported as Untested with the obstruction named. It does not count as a failure, and it does not count as a pass.
- **Causally suggestive → Mechanistically supported:** contributes nothing. The tier turns on I2, I4 and E1.
- **Mechanistically supported → Triangulated:** contributes V2 (level-evidence match), through the discipline of stating what a graded presence licenses. I11 and I12 sit above this tier, and a claim reaching Triangulated without either is the expected case.
- **Triangulated → Validated:** contributes I11 (onset coupling), adapted from the Fredricks & Relman graded-presence guideline, and I12 (offset coupling), its converse — removing the capability should reduce the mechanism's strength. Validated requires that the mechanism and capability track each other in both directions.
