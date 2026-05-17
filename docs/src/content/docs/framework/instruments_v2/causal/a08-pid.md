---
title: "A08 — Partial Information Decomposition"
description: "Decomposing mutual information between circuit component pairs and the output into redundant, unique, and synergistic contributions."
---

# A08 — Partial Information Decomposition

This framework asks: **do two circuit components carry redundant, unique, or synergistic information about the output — and what does this tell us about circuit organization?**

Partial Information Decomposition (PID) takes a pair of source variables \( S_1, S_2 \) (e.g., two attention heads) and a target variable \( T \) (e.g., the logit difference) and decomposes the total mutual information \( I(S_1, S_2; T) \) into four non-negative atoms: redundancy (both carry the same information), unique information from \( S_1 \), unique information from \( S_2 \), and synergy (information available only from the joint but not from either alone). This decomposition reveals whether circuit components are backups for each other (redundant), specialized for different sub-computations (unique), or cooperating in a way that transcends their individual contributions (synergistic).

In MI terms, PID answers the "hydra effect" question: when you ablate one head and another compensates, is that because they carry redundant information (PID predicts compensation) or because they carry synergistic information (PID predicts the joint is not decomposable into individual contributions)?

## Theoretical grounding

| Source | Year | Key contribution |
|---|---|---|
| [Williams & Beer, arXiv 1004.2515](https://arxiv.org/abs/1004.2515) | 2010 | PID framework: redundancy, unique, synergy decomposition |
| [McGrath et al., arXiv 2307.15771](https://arxiv.org/abs/2307.15771) | 2023 | Hydra effect: backup heads compensate when primary is ablated |
| [Elhage et al., "A Mathematical Framework for Transformer Circuits"](https://transformer-circuits.pub/2021/framework/index.html) | 2021 | Compositional structure enabling information-theoretic analysis |
| [Olsson et al., "In-context Learning and Induction Heads"](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) | 2022 | Multiple induction heads as potential redundant/synergistic ensemble |

## Core concept: the four information atoms

For sources \( S_1, S_2 \) and target \( T \):

\[
I(S_1, S_2; T) = \underbrace{R(S_1, S_2; T)}_{\text{redundancy}} + \underbrace{U_1(S_1; T \setminus S_2)}_{\text{unique to } S_1} + \underbrace{U_2(S_2; T \setminus S_1)}_{\text{unique to } S_2} + \underbrace{Syn(S_1, S_2; T)}_{\text{synergy}}
\]

- **Redundancy:** Both heads carry the same information about the output. Ablating one should have minimal effect because the other provides a backup.
- **Unique (S1):** Only this head carries certain information. Ablating it should produce a deficit that nothing else compensates.
- **Synergy:** Information emerges only from the joint. Neither head alone carries it; both must be present. Ablating either one destroys the synergistic contribution entirely.

High redundancy between heads within a circuit predicts robustness to single-component ablation (the hydra effect). High synergy predicts brittleness — the circuit breaks if any synergistic component is removed.

## Instruments under A08

### C8 — PID Analysis (`08_pid.py`)

Computes the four PID atoms for all pairs of circuit components with respect to the task output. Uses a discretized mutual information estimator (binning activations into quantiles) to compute:

\[
R(S_1, S_2; T), \quad U_1, \quad U_2, \quad Syn(S_1, S_2; T)
\]

Reports the redundancy/synergy ratio for each pair and identifies the dominant information structure of the circuit.

**What it establishes:** Whether pairs of components carry overlapping (redundant) or complementary (unique/synergistic) information about the output. Predicts ablation robustness and compensation patterns.

**What it does not establish:** The causal direction of information flow (PID is symmetric in the sources). Does not identify *what* information is shared, only *how much*.

**Usage:**
```
uv run python 08_pid.py --tasks ioi --n-prompts 40
```

## Reading the scores

| Pattern | What it means |
|---|---|
| High redundancy between heads A and B | Backup/compensation structure; ablating one has limited effect |
| High unique info for head A | A carries specialized information no other component has |
| High synergy for pair (A, B) | Joint computation; both needed, neither alone suffices |
| Redundancy dominant across all pairs | Robust, distributed circuit; hard to break with single ablations |
| Synergy dominant across pairs | Fragile circuit; single ablations may catastrophically degrade |

## Practical considerations

PID estimation requires discretization of continuous activations. The instrument bins each component's activation into quantiles (default: 8 bins) before computing mutual information. This introduces a bias-variance tradeoff: more bins capture finer structure but require more data for reliable estimation. For circuits with more than ~20 components, computing all pairwise PID atoms becomes expensive — the instrument supports a `--top-k` flag to restrict analysis to the highest-AP components from A01.

The synergy/redundancy ratio is sensitive to the choice of target variable. Using logit-difference as \( T \) measures task-specific information sharing; using full logit vector captures broader representational overlap.

## Connection to other frameworks

A08 provides the information-theoretic explanation for patterns observed in A01 (activation patching) and A03 (CATE). When A01 shows that ablating a component has no effect, A08 can determine whether this is because of redundancy (another component carries the same information) or because the component is genuinely irrelevant. A05 (MDC/Glennan) predicts which components should interact based on weight-space organization; A08 verifies whether this interaction is redundant or synergistic. The hydra effect (McGrath et al. 2023) is a direct prediction of high-redundancy PID structure.
