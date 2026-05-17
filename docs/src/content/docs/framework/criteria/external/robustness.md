---
title: "Robustness"
validity_type: "External"
criterion_id: "E5"
---

# Criterion E5 — Robustness

| | |
|---|---|
| Validity type | External |
| Pass condition | The claim survives prompt paraphrase, cross-scale transfer, and held-out task generalization |
| Evidence family | Behavioral, Structural |
| Minimum reporting | ≥1 of: new prompt distribution test; cross-scale weight transfer; held-out task transfer |
| Common failure mode | Testing only on the same prompt templates used for discovery |

## What this criterion requires

Robustness is the generalization criterion within the discovery conditions. Three forms, in ascending strength:

**Prompt-distribution robustness:** Circuit achieves comparable faithfulness on a new prompt distribution not used during discovery. For IOI: if discovered on Wang et al.'s 15 templates, test on 15 new templates with different names, verbs, and sentence structures.

**Cross-scale robustness:** Weight classifier trained on GPT-2 Small achieves non-zero F1 on a different model size (GPT-2 Medium, Pythia-160M). Tests whether the structural signature is general across scales, not specific to one model's random initialization. Operationalized via `c13invariance.py`.

**Held-out task generalization:** IIA trained on one task template transfers to a held-out template distribution. Test-retest across prompt families: Pearson r ≥ 0.8.

## Robustness vs. cross-architecture generalization (E6)

Robustness (E5) = mechanism survives variation *within* discovery conditions (different prompts, sizes).
Cross-architecture (E6) = mechanism found in a completely different model *family*.

Robustness is the prerequisite for the stronger cross-architecture claim.

## Minimum reporting rule

- Which form(s) of robustness were tested.
- For prompt-distribution: new distribution, sample size, faithfulness on new vs. original.
- For cross-scale: model size, transfer F1, null expectation.
- If robustness was not tested: external validity is a partial pass at best.
