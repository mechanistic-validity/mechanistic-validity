---
title: "D07 — Cross-Scale Transfer"
description: "Measures whether circuit structure discovered in one model size replicates in larger or smaller models."
---

# D07 — Cross-Scale Transfer

This framework asks: **Are these circuits a property of the algorithm, or an artifact of this particular model's size?**

Cross-scale transfer tests whether circuits discovered in GPT-2 small (117M) replicate in GPT-2 medium (345M) and GPT-2 large (774M). If the same heads and layers implement the same mechanism across scales, the circuit reflects a fundamental algorithmic structure rather than a contingent training outcome. This is the strongest form of generalization: invariance to model capacity.

Scale transfer also validates the discovery method. If a method finds structurally analogous circuits across model sizes without re-tuning hyperparameters, it is likely capturing genuine computational structure rather than overfitting to architectural details.

## Theoretical grounding

| Source | Year | Key contribution |
|---|---|---|
| [Olsson et al., "In-context Learning and Induction Heads"](https://arxiv.org/abs/2209.11895) | 2022 | Induction heads appear at all scales with consistent structure |
| [Conmy et al., "Towards Automated Circuit Discovery"](https://arxiv.org/abs/2304.14997) | 2023 | ACDC applied to multiple GPT-2 variants |
| [Wang et al., "Interpretability in the Wild"](https://arxiv.org/abs/2211.00593) | 2022 | IOI circuit in GPT-2 small as reference for larger models |
| [Hanna et al., "How does GPT-2 compute greater-than?"](https://arxiv.org/abs/2305.00586) | 2023 | Greater-than circuit verified across model scales |

## Core concept

For models at scales \( s_1 < s_2 \), let \( C_{s_1} \) be the circuit discovered at scale \( s_1 \). Cross-scale transfer measures structural correspondence via a mapping \( \phi: C_{s_1} \to C_{s_2} \) that aligns circuit components by relative position (layer fraction, head role):

\[
\text{Overlap}(s_1, s_2) = \frac{|\phi(C_{s_1}) \cap C_{s_2}|}{|C_{s_2}|}
\]

Because larger models have more heads and layers, exact component matching is replaced by role-based alignment: "the name mover heads at relative depth 0.6–0.8" rather than "head 9.6 specifically." The functional correspondence is verified by checking that the aligned components achieve comparable faithfulness (D01) on the same task prompts.

## Metrics under D07

### Cross-Model Invariance (`38_cross_model_invariance.py`)

Discovers circuits independently in GPT-2 small, medium, and large, then measures structural overlap via role-based alignment and functional equivalence on shared task prompts.

**What it establishes:** Whether the circuit is a scale-invariant algorithmic structure.
**What it does not establish:** Whether the mechanism is identical at the weight level — only that analogous components exist.

**Usage:**
```
uv run python 38_cross_model_invariance.py --tasks ioi sva --models gpt2 gpt2-medium gpt2-large
```

## Reading the scores

| Pattern | What it means |
|---|---|
| Overlap > 70% | Circuit is scale-invariant — genuine algorithmic structure |
| Overlap 40–70% | Core mechanism transfers, periphery varies by scale |
| Overlap < 30% | Circuit is scale-specific — likely an artifact of model size |
| Monotone with scale | Mechanism becomes more distributed at larger scale |
| Faithfulness preserved across scales | Functional equivalence confirmed |

