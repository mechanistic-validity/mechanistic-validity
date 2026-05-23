---
title: "Economics -- Metrics & Protocols"
description: "Metrics and protocols for the economics lens: mechanism design, attention auctions, Shapley interactions, pairwise synergy, arbitrage search, game theory, Kyle lambda, and signal discrimination."
---

# Economics -- Metrics & Protocols

The [economics lens](/framework/lenses/supporting/economics) adapts concepts from market microstructure, mechanism design, auction theory, and cooperative game theory to evaluate neural circuit claims. This page documents each metric and protocol: what it measures, how it works, what it establishes, and how to interpret the results.

The economics lens contributes primarily to construct validity --- testing whether the circuit concept carves computation at its joints --- and external validity --- characterizing the quantitative structure of intervention effects. It is a **supporting lens**, meaning its criteria strengthen evidence from the core causal and neuroscience lenses but do not gate tier transitions on their own.

---

## Metrics

### ECON1 --- Mechanism Design

**Metric ID:** `ECON1.mechanism_design`
**Category:** Economics (extended)
**Reference:** Hurwicz 1960, "Optimality and Informational Efficiency in Resource Allocation Processes," *Mathematical Methods in the Social Sciences*

#### What it measures

Tests whether credit allocation across circuit heads is *incentive-compatible*. In mechanism design, a system is incentive-compatible when each agent's declared contribution aligns with its true contribution --- no agent can free-ride by overclaiming. The metric translates this to circuits: each head's **declared contribution** (its activation norm at the final position) is compared against its **true contribution** (the change in logit difference when the head is zero-ablated).

#### Procedure

1. For a given task, identify all circuit heads and generate prompts.
2. Compute the baseline logit difference across prompts.
3. For each circuit head:
   - **Declared contribution:** Mean activation norm of `hook_z` at the last sequence position.
   - **True contribution:** Baseline logit difference minus the mean logit difference after zero-ablating the head.
4. Compute the Spearman rank correlation between declared and true contribution vectors across all circuit heads.

#### Interpretation

| Outcome | Interpretation |
|---|---|
| Spearman $\rho > 0.5$ (pass) | Heads that activate strongly also contribute strongly. Credit allocation is incentive-compatible --- heads cannot free-ride. |
| Spearman $\rho \leq 0.5$ (fail) | Activation magnitude is a poor proxy for causal importance. Some heads may be "overvalued" (high activation, low contribution) or "undervalued." |

**What it establishes:** That the magnitude of a head's activation is a reliable signal of its functional importance within the circuit. This connects to the economics concept of "fair pricing" --- a head's weight/activation should reflect its actual contribution to the computation.

**What it does not establish:** That the circuit is minimal, sufficient, or uniquely necessary. A circuit can be incentive-compatible yet contain redundant components, as long as redundant heads also have low activation norms.

**Usage:**
```bash
uv run python ECON1_mechanism_design.py --tasks ioi sva --n-prompts 40
```

---

### ECON2 --- Attention Auction

**Metric ID:** `ECON2.attention_auction`
**Category:** Economics (extended)
**Reference:** Vickrey 1961, "Counterspeculation, Auctions, and Competitive Sealed Tenders," *Journal of Finance* 16:8--37

#### What it measures

Tests whether attention allocation follows auction-theoretic efficiency. In a Vickrey auction, the highest-value bidder wins and pays the second-highest bid, ensuring that bids truthfully reveal value. The analogy: attention weights are "bids" that positions place for influence over the output. The metric asks whether heads attend most to positions whose information is most valuable (measured by logit difference contribution from each source position).

#### Procedure

1. For each circuit head and each prompt:
   - Extract the attention pattern from the last sequence position to all source positions.
   - For each source position, zero its contribution through the head (zero the head's `hook_z` at that position) and measure the change in logit difference. This is the **position value**.
   - Compute the Spearman correlation between the attention weight vector (the "bids") and the position value vector (the "true values").
2. Average the per-prompt correlations to get a mean efficiency score per head.
3. Average across all circuit heads for the task-level score.

#### Interpretation

| Outcome | Interpretation |
|---|---|
| Mean efficiency $> 0.3$ (pass) | Attention is *allocatively efficient*: heads attend to positions in proportion to how much information those positions carry. |
| Mean efficiency $\leq 0.3$ (fail) | Attention allocation is misaligned with information value. The head may attend to positions for reasons unrelated to their contribution (e.g., positional bias, key-query geometry artifacts). |

**What it establishes:** That the attention mechanism is functioning as an efficient information aggregation system, not merely an associative pattern. High allocative efficiency is evidence that the circuit routes information purposefully.

**What it does not establish:** That the circuit is necessary or sufficient. A head can allocate attention efficiently to valuable positions without being causally necessary for the overall computation (another head may also attend to the same information).

**Usage:**
```bash
uv run python ECON2_attention_auction.py --tasks ioi sva --n-prompts 40
```

---

### H1 --- Hedonic Synergy (PAS + OCA)

**Metric ID:** `H1.hedonic_synergy`
**Category:** Causal (economics)
**Reference:** Chowdhury et al. 2025, "Hedonic Neurons," UMass, arXiv 2509.23684

#### What it measures

Quantifies pairwise interaction structure between circuit heads using two complementary measures from hedonic game theory:

- **PAS (Pairwise Ablation Synergy):** Measures whether the joint ablation effect of two heads exceeds the sum of their individual effects. Defined as:

$$
\phi_{\text{PAS}}(i,j) = -\left( \ell_{-\{i,j\}}(x) - \ell_{-i}(x) - \ell_{-j}(x) + \ell(x) \right)
$$

where $\ell(x)$ is the clean logit difference, $\ell_{-i}(x)$ is the logit difference after mean-ablating head $i$, and $\ell_{-\{i,j\}}(x)$ after ablating both.

- **OCA (Orthogonal Co-Activation):** A weight-space predictor of synergy. Measures whether heads that have orthogonal OV circuits (low weight-space overlap) tend to co-activate (high activation correlation):

$$
\phi_{\text{OCA}}(i,j) = \left(1 - |\cos(W_i, W_j)|\right) \cdot \rho(a_i, a_j)
$$

where $W_i$ is the flattened OV weight matrix for head $i$ and $\rho$ is the Pearson correlation of activation norms across prompts.

#### Procedure

1. Calibrate mean activations across prompts for mean-ablation baseline.
2. For each pair of circuit heads:
   - **PAS:** Run four forward passes per prompt (clean, ablate $i$, ablate $j$, ablate both). Compute the second-order interaction term and average over prompts.
   - **OCA:** Extract each head's OV weight matrix ($W_O \times W_V$), compute cosine similarity. Collect activation norms across prompts and compute correlation.
3. Classify each pair as *synergistic* (PAS $> 0$), *redundant* (PAS $< 0$), or *independent* (PAS $\approx 0$).

#### Interpretation

| Outcome | Interpretation |
|---|---|
| At least one pair has PAS $> 0$ (pass) | The circuit contains synergistic interactions: some head pairs contribute more jointly than the sum of their parts. Linear (first-order) attribution methods like EAP miss this structure. |
| All pairs PAS $\leq 0$ (fail) | All interactions are redundant or independent. Linear attribution may fully describe the circuit. |
| High OCA for synergistic pairs | Weight-space geometry (orthogonal OV circuits) predicts activation-level synergy. The synergy is "baked into" the architecture, not an artifact of specific inputs. |

**What it establishes:** Whether the circuit has genuine second-order structure --- interactions that cannot be captured by summing individual head contributions. This is direct evidence for or against the adequacy of linear attribution methods.

**What it does not establish:** The direction or mechanism of the synergy. A pair can be synergistic because they compute complementary sub-tasks, because one head's output gates the other's attention pattern, or for other reasons. PAS quantifies the magnitude of interaction but not its nature.

**Usage:**
```bash
uv run python H1_shapley_interactions.py --tasks ioi --n-prompts 40
```

---

### C10 --- Pairwise Ablation Synergy

**Metric ID:** `C10.pairwise_ablation_synergy`
**Category:** Causal
**Reference:** Chowdhury et al. 2025, "Hedonic Neurons," UMass

#### What it measures

A focused variant of H1 that computes only the PAS component (no OCA), with a pass criterion based on the magnitude of second-order effects rather than their sign. Where H1 asks "is there synergy?", C10 asks "are second-order interactions non-trivial?" --- both synergy and redundancy count as evidence of interaction structure.

#### Procedure

Same as the PAS computation in H1, but with a different pass criterion:

1. For each pair of circuit heads, compute PAS via four-way ablation (clean, ablate $i$, ablate $j$, ablate both).
2. Compute mean $|\text{PAS}|$ across all pairs.
3. Classify each pair as synergistic (PAS $>$ 0.02), redundant (PAS $<$ -0.02), or independent.

#### Interpretation

| Outcome | Interpretation |
|---|---|
| Mean $|\text{PAS}| > 0.02$ (pass) | Non-trivial second-order interactions exist. The circuit's behavior cannot be fully decomposed into independent, additive head contributions. |
| Mean $|\text{PAS}| \leq 0.02$ (fail) | Interactions are negligible. Individual head attributions (e.g., from activation patching) are sufficient to characterize the circuit. |

The per-pair breakdown is often more informative than the aggregate. A circuit where most pairs are independent but one pair has a large synergy score suggests a specific functional coupling worth investigating.

**Usage:**
```bash
uv run python 94_pairwise_synergy.py --tasks ioi sva --n-prompts 40
```

---

## Protocols

Protocols orchestrate multiple metrics and calibrations into a structured evaluation. Each protocol runs its constituent metrics, applies calibration checks, and produces a summary verdict.

### C12 --- Arbitrage Search

**Protocol ID:** `C12`
**Validity type:** Construct
**Metrics used:** `activation_patching`, `effect_size`, `cka`
**Calibrations:** Structural

#### Question

Can an alternative set of heads (same size as the circuit) achieve comparable performance? If yes, the circuit is not uniquely necessary --- there exists "arbitrage" (an alternative computational path). If no substitute achieves $\geq 90\%$ of circuit performance, the circuit is *non-fungible*.

#### Theoretical grounding

In financial markets, an arbitrage opportunity exists when equivalent assets have different prices. The analog for circuits: an arbitrage opportunity exists when the same computation is achievable through a different set of components. A circuit that routes through head 9.1 when head 7.3 could do the same job is "mispriced" --- the circuit boundary is drawn incorrectly.

This is distinct from minimality (C4). Minimality asks whether any component can be *removed*. Arbitrage asks whether any component can be *replaced*. A circuit can be minimal (every component is necessary) while failing arbitrage freedom (substitutes exist for each component). This happens when the circuit is one of several equally good configurations.

#### Procedure

1. Identify circuit heads for the target task.
2. For each circuit head, test all same-type components in the model as potential substitutes.
3. Measure task performance when each substitute replaces the original (one-at-a-time swap, rest of circuit held fixed).
4. Report the best substitute for each component and its performance ratio relative to the original.

#### Pass criteria

- No alternative component subset of equal or smaller size achieves $\geq 90\%$ of the circuit's task performance when substituted for any single circuit component.
- Secondary: the top-5 substitute candidates each recover $< 50\%$ of the circuit's task performance.

| Source | Year | Key contribution |
|---|---|---|
| Wang et al. | 2022 | IOI circuit identification with complement ablation |
| Conmy et al. | 2023 | ACDC: greedy search over component subsets |
| Hanna et al. | 2023 | Circuit uniqueness via systematic search |

---

### GT --- Game Theory Analysis

**Protocol ID:** `GT`
**Validity type:** External
**Metrics used:** `nash_equilibrium`, `banzhaf_power`, `core_stability`, `voting_power`, `envy_freeness`, `coalition_discovery`, `coalition_tracking`, `replicator_dynamics`, `nucleolus`
**Calibrations:** Structural

#### Question

Do circuit components form strategic equilibria? Can heads be modeled as rational agents in a cooperative game?

The protocol applies a suite of game-theoretic analyses, each asking a different question about the strategic structure of the circuit:

| Sub-metric | What it asks |
|---|---|
| **Nash equilibrium** | Could any single head unilaterally improve the circuit by changing its behavior? If not, the circuit is at a Nash equilibrium. |
| **Banzhaf power** | How often is each head pivotal --- i.e., its inclusion or exclusion flips the circuit from functional to non-functional? |
| **Core stability** | Is the current head allocation "in the core" --- no coalition of heads would prefer a different arrangement? |
| **Voting power** (Shapley-Shubik) | What fraction of orderings make each head the pivotal voter? This is the Shapley value in the voting-game formulation. |
| **Envy-freeness** | Does any head "envy" another's allocation --- i.e., would it prefer to receive the credit/activation budget assigned to another head? |
| **Coalition discovery** | Which subsets of heads form stable coalitions (jointly contributing more than individually)? |
| **Coalition tracking** | How do coalitions change under progressive ablation? Stable coalitions persist; fragile ones dissolve. |
| **Replicator dynamics** | If head "populations" evolve under selection pressure proportional to their contribution, which heads survive and which go extinct? |
| **Nucleolus** | What is the "fairest" allocation of credit that minimizes the maximum complaint of any coalition? |

#### Procedure

1. Run each sub-metric independently for the target task.
2. Apply structural calibrations (random circuit baselines, shuffled circuits).
3. Aggregate into a protocol-level summary: is the circuit at equilibrium? Is credit allocation fair? Are coalitions stable?

| Source | Year | Key contribution |
|---|---|---|
| Nash | 1950 | Equilibrium points in $N$-person games |
| Shapley | 1953 | A value for $N$-person games |
| Banzhaf | 1965 | Weighted voting power index |
| Taylor & Jonker | 1978 | Evolutionary stable strategies and game dynamics |

---

### WC\_M11 --- Kyle Lambda (Market Microstructure Price Impact)

**Protocol ID:** `WC_M11`
**Validity type:** Causal / External
**Metrics used:** `activation_patching`, `effect_size`, `eap`
**Calibrations:** Structural

#### Question

What is the "price impact" of each circuit component --- how much does a unit increase in activation magnitude change the task-relevant logit difference?

This adapts Kyle's (1985) model of market microstructure. In Kyle's framework, the price impact coefficient $\Lambda$ measures how much one unit of informed trading moves the market price. The neural circuit analog: $\Lambda$ measures how much one unit of activation norm shifts the model's output.

#### Procedure

For each attention head and MLP layer:

1. Collect activation norms at the target position across a batch of prompts.
2. Compute a task-agnostic logit difference (top-1 logit minus top-2 logit) as the dependent variable.
3. Fit a Ridge regression: $\Delta\text{logit}_i = \lambda_k \cdot |\text{activation}_{k,i}| + \epsilon_i$.
4. Report per-component $\lambda$ (price impact), $R^2$ (explanatory power), circuit depth ($1/|\lambda|$), and direction (excitatory if $\lambda > 0$, inhibitory if $\lambda < 0$).

#### Interpretation

| Property | High $|\lambda|$ | Low $|\lambda|$ |
|---|---|---|
| **Behavioral role** | Precision instrument: fires rarely but strongly moves the output | Noisy follower: fires often but barely changes output |
| **Circuit depth** ($1/|\lambda|$) | Low depth: easy to steer, sensitive to perturbation | High depth: robust to perturbation |
| **Information density** | High information per activation unit | Low information per activation unit |
| **Direction** ($\lambda > 0$: excitatory, $\lambda < 0$: inhibitory) | Activation increases target logit | Activation decreases target logit |

A component is flagged as **significant** when $R^2 > 0.05$ and $|\lambda| > 0.1$. Components below these thresholds have negligible price impact --- their activations do not meaningfully predict logit changes.

| Source | Year | Key contribution |
|---|---|---|
| Kyle | 1985 | Continuous auctions and insider trading: the original $\Lambda$ model |

---

### I15 --- Signal Discrimination

**Protocol ID:** `I15`
**Validity type:** Internal
**Metrics used:** `activation_patching`, `effect_size`, `cka`
**Calibrations:** Causal

#### Question

Does the circuit respond more strongly to task-relevant perturbations than to random perturbations of the same magnitude? A circuit that responds equally to both is not specifically computing the task --- it is merely a generic bottleneck sensitive to any input change.

This implements Kyle's distinction between *informed trading* (signal) and *noise trading* (random perturbation). A well-functioning circuit should discriminate the two, just as a well-functioning market responds differently to information-bearing and noise trades.

#### Procedure

1. Generate prompt pairs: each pair consists of a "clean" prompt and a "corrupted" prompt (different answer) for the same task.
2. For each pair:
   - **Task-relevant perturbation:** Patch circuit head activations from the corrupted prompt into the clean run. Measure the change in logit difference.
   - **Random perturbation:** Patch circuit head activations with random vectors of the same L2 norm as the task-relevant patch. Measure the change in logit difference.
3. Compute the **discrimination ratio:** mean task-relevant effect / mean random-perturbation effect.
4. Test statistical significance (Welch's $t$-test or permutation test, $p < 0.01$).

#### Interpretation

| Outcome | Interpretation |
|---|---|
| Discrimination ratio $\geq 2.0$ (pass) | The circuit responds preferentially to task-relevant perturbations. It has internal structure specifically tuned to the task. |
| Discrimination ratio $< 2.0$ (fail) | The circuit does not discriminate signal from noise. Its response to interventions is generic, not task-specific. |

**What it establishes:** That the circuit has internal organization aligned with task-relevant features, beyond generic sensitivity to any perturbation. This complements specificity (I3), which tests whether the circuit affects the target task more than other tasks. Signal discrimination tests the converse: whether the circuit is *affected by* task-relevant inputs more than random inputs.

**What it does not establish:** That the circuit is necessary (I1) or sufficient (I2). A circuit can discriminate signal from noise without being the only mechanism that does so.

| Source | Year | Key contribution |
|---|---|---|
| Woodward | 2003 | Interventionist theory: causes must be specific |
| Conant & Ashby | 1970 | Good regulators must model their system |
| Vig et al. | 2020 | Causal mediation analysis contrasting task-relevant vs random interventions |

---

## Reading the scores together

The economics metrics and protocols test different facets of circuit structure. They are most informative when combined:

| Evidence pattern | What it establishes |
|---|---|
| ECON1 pass + ECON2 pass | Credit allocation is honest and attention is efficient: the circuit's internal accounting is well-calibrated. |
| H1 synergy detected + C10 pass | The circuit has genuine multi-head interactions. Linear attribution (EAP, activation patching) underestimates the role of head pairs. |
| High OCA for synergistic pairs (H1) | Synergy is predicted by weight-space geometry, not just activation patterns. The interaction structure is architectural, not input-dependent. |
| WC\_M11 identifies high-$\lambda$ circuit heads | Circuit heads are precision instruments with high information density per activation unit. Non-circuit heads should have lower $\lambda$. |
| I15 pass + E9 pass (from lens page) | The circuit is both task-specific (discriminates signal from noise) and well-behaved (linear price impact). Strong evidence for a genuine computational mechanism. |
| C12 pass (arbitrage-free) + C4 pass (minimal) | The circuit is both irreducible (no component can be removed) and irreplaceable (no component can be substituted). The strongest structural evidence from the economics lens. |
| ECON1 fail + ECON2 pass | Attention is efficient but activation norms are misleading. Heads attend to the right positions but their activation magnitudes do not reflect their actual importance. |
| C12 fail (arbitrage exists) | Substitute components exist outside the circuit boundary. The circuit is one of several equivalent configurations. The claim should be qualified accordingly. |

---

## Relationship to other lenses

The economics lens overlaps with and complements several other lenses:

- **Pharmacology:** Pharmacology asks "how much effect?" (dose-response). Economics asks "how does the effect behave as a function of what you do?" (price impact, linearity, signal discrimination). Pharmacology's EC50 is the *threshold*; economics' $\Lambda$ is the *slope*.
- **Neuroscience:** Neuroscience's specificity criterion (I3) asks whether ablating the circuit affects the target task more than other tasks. Economics' signal discrimination (I15) asks the converse: whether the circuit is affected *by* task-relevant perturbations more than random ones.
- **Information theory:** Shapley interactions (H1) can be seen as an information-theoretic decomposition of joint effects. The distinction is that H1 uses the logit difference as the value function, while information-theoretic metrics use mutual information or entropy.
- **Geometry:** OCA's cosine similarity component connects to geometric analyses of weight-space structure. A geometric lens asks "are these vectors aligned?"; economics asks "does alignment predict functional interaction?"
