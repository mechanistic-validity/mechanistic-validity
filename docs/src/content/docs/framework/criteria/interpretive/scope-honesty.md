---
title: "Scope Honesty"
validity_type: "Interpretive"
criterion_id: "V5"
---

# Criterion V5 — Scope Honesty

| | |
|---|---|
| Validity type | Interpretive |
| Pass condition | The verdict does not silently generalize beyond the evidence scope |
| Evidence family | N/A (criterion is about claim scope, not experiments) |
| Minimum reporting | Explicit scope restrictions in the verdict statement; named dimensions on which the claim has not been tested |
| Common failure mode | A claim proven on one prompt distribution is stated without scope restriction; a finding in GPT-2 Small is described as a finding about "transformers" |

## What this criterion requires

Scope honesty requires that every claim includes explicit restrictions on its scope — specifying the model, the task, the prompt distribution, and any other dimensions on which the claim has been tested.

A claim without scope restrictions is an overclaim. The evidence supports a claim only for the conditions under which it was obtained. Generalizing beyond those conditions requires either (a) additional evidence (robustness E5, cross-architecture E6) or (b) explicit acknowledgment that the generalization is speculative.

## Common scope overclaims

| What was shown | Overclaimed as |
|---|---|
| Finding in GPT-2 Small | "Transformers implement X" |
| Finding on 15 IOI templates | "The IOI mechanism" (without prompt distribution restriction) |
| Finding at one intervention strength | "Causal manipulation of X" (without graded-response characterization) |
| Finding on the final checkpoint | "GPT-2 Small's mechanism" (without checkpoint range) |
| Finding in one random seed | "The model's circuit" (without seed stability) |

## The scope statement format

Every verdict must include a scope statement of the form:

> **Scope:** [Model]: [model name and size]. [Task]: [task name and prompt distribution]. [Evidence conditions]: [ablation method(s), n prompts, seeds, checkpoints]. [Not tested]: [dimensions not yet investigated].

Example:
> **Scope:** GPT-2 Small (124M, 12L). SVA task on Linzen et al. (2016) held-out split (n=200, 3 seeds). Zero ablation at hook point `blocks.8.mlp.hook_post`. Not tested: cross-scale (GPT-2 Medium, Pythia), cross-prompt-family, causal specificity (control-axis IIA not computed).

## Relation to other criteria

Scope honesty is the final check in the interpretive layer. It cannot be satisfied if any of the other interpretive criteria (V1–V4) are not met, because level declaration, level–evidence match, narrative coherence, and alternative exclusion together determine what scope the evidence actually licenses. Scope honesty requires reporting that scope accurately.
