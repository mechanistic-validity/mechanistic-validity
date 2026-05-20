---
title: "D08 — Prompt Paraphrase Robustness"
description: "Measures whether circuit behavior is consistent across semantically equivalent prompt templates."
---

# D08 — Prompt Paraphrase Robustness

This framework asks: **Does the circuit's behavior depend on surface-level prompt wording, or on the underlying semantic structure?**

A circuit explanation should be robust to paraphrase: if the task is "identify the indirect object," the circuit should work regardless of whether the prompt says "Then, Mary gave a drink to" or "After that, Mary handed a drink to." Sensitivity to surface form suggests the circuit is exploiting template-specific cues rather than implementing the claimed algorithm.

Counterfactual consistency measures this by evaluating the circuit across a battery of semantically equivalent prompt variants and quantifying how much behavior varies. Low variance indicates a robust mechanism; high variance reveals template overfitting.

## Theoretical grounding

| Source | Year | Key contribution |
|---|---|---|
| [Wang et al., "Interpretability in the Wild"](https://arxiv.org/abs/2211.00593) | 2022 | IOI with multiple templates (ABBA/BABA variants) |
| [Geiger et al., "Causal Abstraction for Faithful Model Interpretability"](https://arxiv.org/abs/2301.04709) | 2023 | Intervention consistency across input distributions |
| [Conmy et al., "Towards Automated Circuit Discovery"](https://arxiv.org/abs/2304.14997) | 2023 | ACDC sensitivity to prompt distribution |
| [Hanna et al., "How does GPT-2 compute greater-than?"](https://arxiv.org/abs/2305.00586) | 2023 | Template robustness for year comparison |

## Core concept

Given a set of paraphrases \( \{x_1, \ldots, x_K\} \) for the same underlying task instance, robustness is measured by the coefficient of variation of the circuit's task metric:

\[
\text{CV} = \frac{\sigma_M}{\mu_M} = \frac{\text{std}(M(C, x_1), \ldots, M(C, x_K))}{\text{mean}(M(C, x_1), \ldots, M(C, x_K))}
\]

where \( M(C, x_k) \) is the circuit's faithfulness score (logit diff, KL, etc.) on paraphrase \( k \). Low CV indicates that the circuit implements a template-invariant computation. Alternatively, the consistency score compares the worst-case paraphrase to the best-case:

\[
\text{Consistency} = \frac{\min_k M(C, x_k)}{\max_k M(C, x_k)}
\]

## Metrics under D08

### Counterfactual Consistency (`34_counterfactual_consistency.py`)

Evaluates circuit faithfulness across multiple prompt template variants for each task, reporting per-template scores and their variance.

**What it establishes:** Whether the circuit captures template-invariant structure.
**What it does not establish:** Which specific template features the circuit is sensitive to (requires further ablation).

**Usage:**
```
uv run python 34_counterfactual_consistency.py --tasks ioi sva
```

## Reading the scores

| Pattern | What it means |
|---|---|
| CV < 0.05 | Highly robust — circuit is template-invariant |
| CV 0.05–0.15 | Mostly robust with minor template sensitivity |
| CV > 0.3 | Template-dependent — circuit exploits surface cues |
| Consistency > 0.9 | Worst case is close to best case |
| One template fails dramatically | Circuit relies on a specific positional pattern |

