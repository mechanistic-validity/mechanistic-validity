---
title: "Genetics"
description: "The molecular biology lens: does the circuit behave like a genetic pathway — with epistasis, rescue, and instrument-valid causal structure?"
---

# The Genetics Lens

This lens asks one question: **does the circuit have the causal structure of a real genetic pathway?**

Molecular genetics provides the most rigorous methodology biology has developed for establishing that a component *implements* a function rather than merely correlating with it. The core toolkit — knockout, rescue, epistasis mapping, Mendelian randomization, sensitivity analysis — was refined over a century of work on organisms from *Drosophila* to *C. elegans* to human GWAS cohorts. Each technique answers a question that no other technique can answer alone. Knockout establishes necessity. Rescue establishes reversibility and sufficiency under damage. Epistasis establishes that components interact non-additively — that the circuit is more than the sum of its parts. Mendelian randomization establishes that an upstream variable affects the outcome *through* the circuit and not through a backdoor path. Sensitivity analysis (E-values) quantifies how robust a causal claim is to confounders the experimenter did not measure.

These techniques map directly onto mechanistic interpretability interventions. Ablation is knockout. Activation patching in the restoration direction is rescue. Pairwise ablation studies that reveal synergy or antagonism are epistasis mapping. Using upstream activations as instrumental variables for circuit-mediated effects is Mendelian randomization. Computing E-values for ablation effects is sensitivity analysis. The intellectual infrastructure already exists in MI — what is often missing is the systematic combination of these techniques and the interpretive framework that makes the combination more than the sum of its parts.

There is a disanalogy worth naming. In genetics, a knockout is permanent: the organism develops without the gene from the start, and compensatory mechanisms may or may not emerge over developmental time. In neural networks, ablation is instantaneous: the component is present during all of training and is removed only at inference. This means that neural network "knockouts" are closer to acute pharmacological blockade than to developmental gene deletion. The network never had the chance to develop compensatory pathways around the missing component, so the observed effect may overstate the component's true irreplaceability. Rescue experiments partially address this — if restoring the component recovers behavior after corruption, the deficit was not merely distributional disruption but a genuine loss of computational capacity.

## Key Distinctions

### Knockout vs knockdown

In genetics, a full knockout removes the gene entirely — no transcript, no protein, no residual function. A knockdown (e.g., via RNAi or CRISPRi) reduces expression without eliminating it. These are different interventions that yield different information. A gene can be essential at full knockout but dispensable at partial knockdown if the cell needs only 20% of normal expression. Conversely, a gene can appear dispensable under complete knockout (because compensatory pathways activate) but show a phenotype under partial knockdown (which disrupts the balance without triggering compensation).

In MI: zero ablation (replacing the component's output with the zero vector) is the analog of full knockout. Mean ablation is a partial knockdown — the component retains its average contribution but loses all input-dependent information. Resample ablation is a different kind of knockdown: the component retains in-distribution activation magnitudes but carries information from the wrong input. Rank-reduced ablation (projecting the component onto its top-k singular vectors) is a graded knockdown where the severity is controlled by $k$. These are qualitatively different interventions, not just different doses of the same intervention, and they probe different aspects of the circuit's function.

### Epistasis vs additivity

Fisher (1918) defined epistasis as the non-additive interaction between genetic loci. If gene A contributes effect $a$ and gene B contributes effect $b$, additivity predicts a joint effect of $a + b$. Epistasis is the deviation from this prediction: $\epsilon = f(A, B) - f(A) - f(B) + f(\emptyset)$, where $f$ is the phenotype and the arguments indicate which genes are knocked out. Positive epistasis (synergy) means the joint knockout is worse than expected — the genes cooperate. Negative epistasis (antagonism or buffering) means the joint knockout is milder than expected — the genes partially substitute for each other.

In MI: if ablating head 9.1 reduces IOI logit difference by 1.2 and ablating head 9.6 reduces it by 0.8, additivity predicts a joint ablation effect of 2.0. If the observed joint effect is 3.4, the synergy score $\epsilon = 3.4 - 1.2 - 0.8 = 1.4$ reveals that these heads cooperate — disrupting either alone leaves the other partially functional, but disrupting both collapses the computation. If the observed joint effect is 1.5, the antagonism score $\epsilon = 1.5 - 1.2 - 0.8 = -0.5$ reveals redundancy — the second ablation adds little because the first already forced the computation through alternative pathways. The sign and magnitude of $\epsilon$ reveal functional coupling that no single-component ablation can detect.

### Forward vs reverse genetics

Classical genetics distinguishes two strategies. Forward genetics starts with a phenotype (a mutant organism with an interesting trait) and works backward to identify the responsible gene. Reverse genetics starts with a gene (selected because its sequence or expression pattern is interesting) and disrupts it to observe the phenotypic consequence.

In MI: circuit discovery methods are forward genetics — we observe a behavior (IOI, induction, greater-than) and search for the components responsible, via activation patching, EAP, or causal scrubbing. Ablation studies are reverse genetics — we select a component (a head, a feature, a subspace) and disrupt it to observe the behavioral consequence. The two strategies complement each other. Forward genetics finds circuits but may miss components with subtle or redundant contributions. Reverse genetics characterizes individual components but may miss the circuit-level organization. A complete genetic analysis uses both: forward to discover, reverse to validate and extend.

### Instruments and exclusion

Mendelian randomization (MR) exploits a natural experiment: genetic variants are assigned at conception (effectively randomized with respect to postnatal confounders) and affect the outcome only through the gene they regulate. The critical assumption is the *exclusion restriction*: the instrument (genetic variant) affects the outcome *only* through the mediator (gene expression, protein level), not through any backdoor path. When the exclusion restriction holds, the causal effect of the mediator on the outcome is identified even in the presence of unmeasured confounders.

In MI: an upstream activation (e.g., a residual stream direction at an early layer) can serve as an instrument for a circuit's causal effect on the output. The exclusion restriction requires that this upstream activation affects the model's output *only* through the proposed circuit, not through any alternative pathway. If the instrument has a direct effect on the output that bypasses the circuit, the IV estimate is biased. Testing the exclusion restriction requires measuring whether the instrument predicts the outcome after controlling for the circuit's activations — any residual prediction violates exclusion. Overidentification tests (using multiple instruments and checking whether they agree) provide a falsifiable check.

## Analytical Constructs

### The epistasis matrix

The signature artifact of genetic evaluation is the epistasis matrix: a $k$-components $\times$ $k$-components matrix of pairwise interaction effects. For circuit components $\{c_1, \ldots, c_k\}$ and behavioral metric $M$:

$$\epsilon_{ij} = \Delta M(\text{ablate } c_i \text{ and } c_j) - \Delta M(\text{ablate } c_i) - \Delta M(\text{ablate } c_j)$$

where $\Delta M(\text{ablate } S) = M_{\text{clean}} - M_{\text{ablated}(S)}$.

The matrix reveals structure that individual or even ordered ablation studies cannot:

- **Diagonal entries** ($\Delta M(\text{ablate } c_i)$) — individual necessity effects. These are what most papers report and what the neuroscience lens's dissociation matrix also captures.
- **Off-diagonal entries** ($\epsilon_{ij}$) — pairwise epistasis. Positive values indicate synergy (the components cooperate and their joint removal is worse than predicted by additivity). Negative values indicate buffering (the components partially substitute for each other). Near-zero values indicate independence.
- **Row/column sums** — a component with uniformly positive epistasis interacts cooperatively with many circuit members, suggesting it is a hub. A component with uniformly negative epistasis is buffered by many alternatives.
- **Spectral structure** — eigendecomposition of the epistasis matrix reveals functional modules: components with correlated interaction profiles cluster together. This is the analog of genetic interaction profiles used in yeast systematic genetics ([Costanzo et al. 2010](https://doi.org/10.1126/science.1180823)).

The epistasis matrix extends the neuroscience lens's dissociation matrix in a specific direction. The dissociation matrix is circuits $\times$ tasks (each cell is a single ablation's effect on a single task). The epistasis matrix is components $\times$ components within a single circuit (each cell is the non-additive interaction between a pair of components on a single task). The dissociation matrix asks "is the circuit specific to the task?" The epistasis matrix asks "do the circuit's components functionally interact?"

To construct the matrix: for each pair $(c_i, c_j)$ in the circuit, ablate $c_i$ alone, $c_j$ alone, and both jointly. Compute $\epsilon_{ij}$. Fill the $k \times k$ grid. Inspect for modules of correlated interaction. Report the fraction of pairs with $|\epsilon_{ij}| > 2 \times \max(\Delta M(c_i), \Delta M(c_j))$ as the epistasis rate.

## Sources

| Source | Year | Field | Principle |
|---|---|---|---|
| [Fisher, "The correlation between relatives on the supposition of Mendelian inheritance"](https://doi.org/10.1017/S0080456800012163) | 1918 | Genetics | **Epistasis** — non-additive interaction between genetic loci; the deviation from additivity reveals functional coupling |
| [Brenner, "The genetics of *Caenorhabditis elegans*"](https://doi.org/10.1093/genetics/77.1.71) | 1974 | Genetics | **Forward genetics in a model organism** — systematic mutagenesis and phenotypic screening to map gene-to-function relationships |
| [Hartwell, Hopfield, Leibler & Murray, "From molecular to modular cell biology"](https://doi.org/10.1038/35011540) | 1999 | Systems Biology | **Genetic pathways as modules** — biological function is organized into discrete, interacting modules; the pathway, not the individual gene, is the unit of function |
| [Davey Smith & Hemani, "Mendelian randomization: genetic anchors for causal inference"](https://doi.org/10.1146/annurev-genom-090413-025437) | 2014 | Epidemiology | **Mendelian randomization** — using genetic variants as instruments to establish causation in the presence of unmeasured confounders; requires the exclusion restriction |
| [VanderWeele & Ding, "Sensitivity analysis in observational research"](https://doi.org/10.7326/M16-2607) | 2017 | Epidemiology | **E-values** — the minimum strength of unmeasured confounding that would explain away an observed association; quantifies robustness to hidden bias |
| [Costanzo et al., "The genetic landscape of a cell"](https://doi.org/10.1126/science.1180823) | 2010 | Systems Biology | **Systematic genetic interactions** — genome-wide pairwise knockout screen in yeast; interaction profiles cluster genes into functional modules; epistasis reveals pathway structure |
| [Staiger & Stock, "Instrumental variables regression with weak instruments"](https://doi.org/10.2307/2171753) | 1997 | Econometrics | **Weak instrument diagnostics** — F-statistic $> 10$ rule for instrument relevance; weak instruments produce biased IV estimates |

## Validity type: [Internal validity](/v2/validity-types/internal)

> **Genetic pathway as mechanism ([Hartwell et al. 1999](https://doi.org/10.1038/35011540)):** A genetic pathway is a module — a set of genes whose products interact to perform a discrete biological function. The pathway is identified not by individual gene knockouts alone, but by the *pattern* of epistatic interactions among its members and the *reversibility* of its disruption. A circuit in MI that passes knockout (necessity), rescue (reversibility), and epistasis (non-additive interaction) has the causal structure of a genetic pathway.

A single gene knockout tells you the gene matters. A rescue experiment tells you the deficit was specifically caused by the gene's absence and not by collateral damage. Epistasis mapping tells you the gene interacts with other genes in the pathway. Mendelian randomization tells you the effect operates through the pathway and not through a backdoor. Sensitivity analysis tells you how robust the whole story is to things you did not measure. Each test eliminates a class of alternative explanations that the others cannot.

This lens primarily contributes to internal validity, extending the neuroscience lens's I1--I5 with five additional criteria (I6--I10) that address interaction structure, reversibility, ordering, instrument validity, and confounding sensitivity. It also contributes one criterion to external validity (E7, allelic dose-response), extending the pharmacology lens's E1--E6.

## Criteria

| Code | Criterion | What it asks | Validity type |
|---|---|---|---|
| I6 | Epistatic interaction | Do circuit components interact non-additively? | Internal |
| I7 | Rescue reversibility | Does restoring a corrupted component recover the behavior? | Internal |
| I8 | Knockout ordering | Does ordered ablation reveal a dependency structure? | Internal |
| I9 | Instrument validity | Does an upstream variable affect the outcome only through the circuit? | Internal |
| I10 | Confounding sensitivity | How strong would an unmeasured confounder need to be to explain away the result? | Internal |
| E7 | Allelic dose-response | Does graded intervention across qualitatively different ablation types produce monotonic degradation? | External |

Epistatic interaction (I6) is the most distinctive contribution of this lens -- it tests whether the circuit is a functional unit with internal coupling, not merely a list of independently necessary components. Rescue reversibility (I7) provides the strongest evidence that an ablation deficit reflects genuine loss of computation rather than distributional disruption. Instrument validity (I9) is the most technically demanding and the least commonly tested in MI.

### Epistatic interaction (I6)

Pairwise or higher-order ablation of circuit components should reveal non-additive interactions.

A circuit whose components contribute additively is not a circuit in the genetic sense -- it is a collection of independent effects that happen to serve the same task. Epistasis is the signature of functional coupling: components that cooperate, compete, or buffer each other. The epistasis matrix (defined above) quantifies these interactions for every pair in the circuit.

The Shapley interaction index provides a formal measure. For components $i$ and $j$, the Shapley interaction index $I_{ij}$ is the expected change in $i$'s marginal contribution when $j$ is added to the coalition, averaged over all possible coalitions. A significantly non-zero $I_{ij}$ means the components' contributions are not separable -- knowing one's state changes the other's marginal importance.

**What it establishes.** The circuit's components interact functionally -- they are not independent contributors to the behavior. The pattern of interactions (synergy vs. buffering) reveals the circuit's internal organization.

**What it does not establish.** That the interactions are task-specific (the same components might interact on every task), or that the epistasis is mediated by direct information flow between the components (it could reflect shared dependence on a common upstream signal).

**Threshold.** At least one component pair with $|\epsilon_{ij}| > 2 \times \max(\Delta M(c_i), \Delta M(c_j))$, where $\epsilon_{ij}$ is the pairwise epistasis score. Shapley interaction index significantly non-zero ($p < 0.01$, permutation test with $\geq 1000$ shuffles).

**Minimum reporting.** The full $k \times k$ epistasis matrix for all circuit components. The fraction of pairs exceeding the $2\times$ threshold. Shapley interaction indices for at least the top-5 interacting pairs with $p$-values.

### Rescue reversibility (I7)

Corrupting a circuit component and then restoring it should recover the behavior.

Rescue is conceptually distinct from sufficiency (I2). Sufficiency asks: does the circuit, in isolation, reproduce the behavior? It patches clean activations into a corrupted run or ablates everything outside the circuit. Rescue asks a different question: if we first *damage* the circuit (corrupt, ablate, or noise a specific component) and then *restore* it (patch back the clean activation at that specific site), does the behavior recover? Rescue tests whether the deficit caused by corruption is *specifically reversible* by restoring the corrupted component, rather than being an irreversible cascade of downstream failures.

The distinction matters because a corrupted component can cause downstream disruption that persists even after the component itself is repaired. If mean-ablating head 9.1 pushes downstream residual stream norms off-manifold, restoring head 9.1 may not fix the downstream damage. A rescue that succeeds demonstrates that the deficit was caused by the specific loss of that component's contribution, not by collateral distributional damage. A rescue that fails despite restoring the correct activation is evidence that the observed necessity effect was partly an artifact of cascading disruption.

**What it establishes.** The behavioral deficit caused by corrupting the component is specifically attributable to the loss of that component's contribution and is reversible by restoring it. The corruption did not cause irreversible downstream damage.

**What it does not establish.** That the component is the only route for the computation (other components might also rescue the behavior if activated appropriately), or that the rescue would succeed under a different corruption method.

**Threshold.** Restoration recovers $\geq 80\%$ of clean performance, measured as the fraction of the clean-corrupt gap restored:

$$R_{\text{rescue}} = \frac{M_{\text{rescued}} - M_{\text{corrupt}}}{M_{\text{clean}} - M_{\text{corrupt}}} \geq 0.80$$

**Minimum reporting.** The corruption method, the restoration method, $M_{\text{clean}}$, $M_{\text{corrupt}}$, $M_{\text{rescued}}$, and $R_{\text{rescue}}$. If $R_{\text{rescue}} < 0.80$, report a distributional integrity check (e.g., residual stream norm at downstream positions before and after rescue) to diagnose whether the failure reflects irreversible cascading disruption.

### Knockout ordering (I8)

Ordered sequential ablation of circuit components should reveal a dependency structure.

In genetics, the order in which genes are knocked out in a pathway reveals the pathway's topology. If knocking out gene A (upstream) makes the subsequent knockout of gene B (downstream) redundant, A is upstream of B in the pathway. The ablation ordering test translates this directly: if ablating component $c_1$ first produces a large degradation, and ablating $c_2$ after $c_1$ produces little additional degradation, then $c_2$'s contribution was mediated through or redundant with $c_1$.

To construct the ordering: rank circuit components by their predicted importance (e.g., by individual ablation effect, by layer position, or by a causal ordering derived from path patching). Ablate components one at a time in this order, measuring cumulative degradation at each step. The resulting curve should be monotonically increasing if the predicted ordering reflects the true dependency structure.

**What it establishes.** The circuit has a dependency structure -- components are not interchangeable, and their contributions depend on which other components are intact. The ordering reveals which components are upstream (their removal makes downstream components redundant) and which are downstream (their removal has little effect once upstream components are gone).

**What it does not establish.** That the predicted ordering is the only correct ordering. Multiple valid orderings may exist if the circuit has parallel pathways.

**Threshold.** Monotonically increasing cumulative degradation curve. Spearman $\rho \geq 0.7$ between the predicted ordering (by individual ablation effect size or causal graph position) and the observed ordering (by marginal contribution when ablated in sequence).

**Minimum reporting.** The predicted ordering and its basis. The cumulative degradation curve. Spearman $\rho$ between predicted and observed orderings. If monotonicity is violated at any step, report which component violated it and by how much.

### Instrument validity (I9)

An upstream variable should affect the outcome only through the circuit, satisfying the exclusion restriction of Mendelian randomization.

Instrument validity adapts the logic of Mendelian randomization to MI. An *instrument* is an upstream activation (e.g., a residual stream direction at layer 0, or a specific embedding feature) that is correlated with the circuit's activations (relevance) and affects the model's output only through the circuit (exclusion). If an instrument satisfies both conditions, we can estimate the circuit's causal effect on the output even in the presence of unmeasured confounders (e.g., shared information in the residual stream that biases both the circuit and the output).

The two conditions are testable. *Relevance*: regress the circuit's activations on the instrument and check that the F-statistic exceeds 10 (the [Staiger & Stock (1997)](https://doi.org/10.2307/2171753) rule for avoiding weak-instrument bias). *Exclusion*: regress the output on the instrument while controlling for the circuit's activations. Any residual association between the instrument and the output, after controlling for the circuit, violates exclusion -- the instrument has a direct effect that bypasses the circuit. The Sargan/Hansen overidentification test formalizes this when multiple instruments are available: if all instruments yield the same IV estimate, exclusion is consistent; if they disagree, at least one instrument is invalid.

**What it establishes.** The circuit mediates the causal effect of the upstream variable on the outcome. The effect operates *through* the circuit, not around it. This is stronger than necessity (I1), which establishes only that removing the circuit changes the outcome -- not that the circuit is the mediating pathway.

**What it does not establish.** That the circuit is the *only* mediating pathway (other circuits may also mediate), or that the instrument is the only valid instrument (multiple instruments strengthen the claim if they agree).

**Threshold.** Instrument relevance: first-stage F-statistic $> 10$. Exclusion restriction: Sargan/Hansen overidentification test $p > 0.05$ (when $\geq 2$ instruments are available). If only one instrument is available, report the residual association between instrument and output after conditioning on circuit activations, with a permutation-based $p$-value.

**Minimum reporting.** The instrument(s) used, with justification for why they satisfy relevance and exclusion a priori. First-stage F-statistic. Overidentification test statistic and $p$-value (if $\geq 2$ instruments). Residual association test (if one instrument). The IV estimate of the circuit's causal effect alongside the naive (OLS) estimate -- divergence between the two indicates confounding that IV corrects for.

### Confounding sensitivity (I10)

The E-value or a similar sensitivity bound should quantify how strong an unmeasured confounder would need to be to explain away the observed causal effect.

This criterion is distinct from confound control (I5). I5 asks: "did you control for confounds?" -- it requires multi-method comparison and distributional integrity checks to rule out known confounds (off-manifold artifacts, cascading disruption, single-method bias). I10 asks the complementary question: "how robust is your claim to confounds you *did not* control for?" Every ablation study has potential unmeasured confounders -- information in the residual stream that correlates with both the ablated component and the output, backup pathways that partially compensate, or distributional effects that inflate the apparent necessity.

The E-value ([VanderWeele & Ding 2017](https://doi.org/10.7326/M16-2607)) quantifies this robustness. For an observed effect estimate $\text{RR}$ (risk ratio or analogous effect measure), the E-value is:

$$E = \text{RR} + \sqrt{\text{RR} \times (\text{RR} - 1)}$$

The E-value is the minimum strength of association (on the risk-ratio scale) that an unmeasured confounder would need to have with both the treatment and the outcome to fully explain away the observed effect. An E-value of 2.0 means the confounder would need to double both the exposure and the outcome risk to nullify the finding. An E-value of 5.0 means a fivefold confounder would be required -- plausible in some settings, implausible in others.

**What it establishes.** A quantitative bound on the vulnerability of the causal claim to unmeasured confounding. A high E-value does not prove the absence of confounders; it establishes that only a very strong confounder could explain the result away.

**What it does not establish.** That no such confounder exists. The E-value is a bound, not a proof. It also does not address confounders that were measured but improperly controlled for -- that is I5.

**Threshold.** E-value $\geq 2.0$. This means an unmeasured confounder would need to at least double both the treatment-outcome and confounder-outcome associations to explain away the observed effect.

**Minimum reporting.** The observed effect estimate and its scale (risk ratio, odds ratio, or standardized mean difference). The E-value for the point estimate. The E-value for the lower bound of the 95% confidence interval (this is the more conservative and more informative number -- if the CI lower-bound E-value is still $\geq 2.0$, the finding is robust even accounting for estimation uncertainty).

### Allelic dose-response (E7)

Graded intervention across qualitatively different ablation types should produce monotonic degradation.

This criterion extends the pharmacology lens's graded response (E2) in a specific direction. E2 varies *how much* of a single intervention type is applied (interpolating from 0 to 1 for one ablation method). E7 varies *what kind* of intervention is applied, across an ordered series of qualitatively different ablation types. The analogy is to allelic series in genetics: a set of mutations in the same gene with increasing severity (null, hypomorph, dominant negative), each producing a graded phenotype. If the phenotypic severity tracks the allelic severity, the gene-phenotype relationship is robust and the mechanism is well-defined.

In MI, the "allelic series" is a set of ablation types ordered by expected severity:

1. **Rank reduction** (retain top-$k$ singular vectors) -- mildest; removes only the least important subspace.
2. **Mean ablation** -- moderate; removes input-dependent information while retaining the average contribution.
3. **Resample ablation** -- moderate-to-strong; replaces with in-distribution but task-irrelevant signal.
4. **Zero ablation** -- strongest; removes the component's contribution entirely.

If the behavioral degradation is monotonically increasing across this series, the circuit's contribution is robust to the specific choice of ablation method -- the effect is a property of the circuit, not an artifact of a particular counterfactual distribution.

**What it establishes.** The circuit's behavioral contribution is consistent across qualitatively different intervention types. The finding is not an artifact of a single ablation method's distributional assumptions.

**What it does not establish.** That the ordering is the only valid ordering (some circuits may show different sensitivity profiles), or that the effect generalizes beyond the tested prompt distribution.

**Threshold.** Monotonic degradation across $\geq 3$ qualitatively different intervention types from the ordered series above. Rank correlation (Spearman $\rho$ or Kendall $\tau$) $\geq 0.8$ between the expected severity ordering and the observed degradation ordering.

**Minimum reporting.** The ablation types used, their expected severity ordering, and the observed degradation at each. The rank correlation. If monotonicity is violated, report which ablation type violated it and propose an explanation (e.g., mean ablation may produce stronger effects than resample ablation for components with large mean activations).

## Evidence Patterns

| Evidence pattern | What it establishes | Recommended language |
|---|---|---|
| Epistasis detected, no rescue | Non-additive coupling; reversibility unknown | "Components interact non-additively; rescue not tested" |
| Rescue succeeds, no epistasis | Reversible deficit; independence unknown | "Deficit is reversible; interaction structure not tested" |
| Epistasis + rescue + ordering | Structured, reversible, ordered pathway | "Circuit has genetic-pathway structure" |
| IV estimate matches naive estimate | No evidence of confounding through tested instruments | "IV and OLS estimates agree; unmeasured confounding unlikely on this pathway" |
| IV estimate diverges from naive | Confounding present; IV corrects for it | "Naive estimate biased by confounding; IV estimate of causal effect is [X]" |
| High E-value ($\geq 3.0$) | Robust to strong confounders | "E-value = [X]; a confounder of strength [X] required to explain away" |
| Low E-value ($< 2.0$) | Vulnerable to moderate confounders | "E-value = [X]; finding is sensitive to unmeasured confounding" |
| Monotonic allelic dose-response | Method-robust contribution | "Effect is monotonic across [k] ablation types" |

## Verdicts

The genetics lens strengthens verdict transitions primarily through internal validity:

- **Proposed → Causally suggestive:** The genetics lens does not gate this transition. I1 (necessity) from the neuroscience lens remains the entry requirement.
- **Causally suggestive → Mechanistically supported:** I7 (rescue reversibility) strengthens this transition by establishing that the ablation deficit is specifically reversible, not an artifact of cascading disruption. This complements I2 (sufficiency) from the neuroscience lens.
- **Mechanistically supported → Triangulated:** I6 (epistatic interaction) and I8 (knockout ordering) provide evidence from a different analytical framework (interaction structure rather than single-component intervention), strengthening the triangulation case. E7 (allelic dose-response) contributes cross-method robustness evidence.
- **Triangulated → Validated:** I9 (instrument validity) and I10 (confounding sensitivity) address threats to validity that no other lens tests -- backdoor confounding and hidden bias. A circuit that passes all genetics criteria alongside the criteria from other lenses has survived the most comprehensive set of causal tests available.

## Protocol

For a proposed circuit $C$ with components $\{c_1, \ldots, c_k\}$ and behavior $B$:

1. **Epistatic interaction.** For each pair $(c_i, c_j)$, ablate individually and jointly. Compute the epistasis matrix $\epsilon_{ij}$. Report the fraction of pairs with $|\epsilon_{ij}| > 2\times$ the larger individual effect. Compute Shapley interaction indices for the top interacting pairs.

2. **Rescue reversibility.** For each major component, corrupt it (mean or resample ablation), then restore the clean activation at that site. Report $R_{\text{rescue}}$ for each component. If any rescue fails ($R_{\text{rescue}} < 0.80$), diagnose whether the failure reflects cascading disruption (check downstream residual stream norms).

3. **Knockout ordering.** Predict an ablation ordering from individual effect sizes or causal graph structure. Ablate components sequentially in this order, recording cumulative degradation. Report the Spearman $\rho$ between predicted and observed orderings. Verify monotonicity.

4. **Instrument validity.** Identify at least one upstream variable as a candidate instrument. Test relevance (first-stage F $> 10$) and exclusion (residual association after conditioning on circuit activations). If $\geq 2$ instruments are available, run the overidentification test.

5. **Confounding sensitivity.** Compute the E-value for the principal necessity effect (from I1) and for the rescue effect (from I7). Report E-values for both the point estimate and the lower CI bound.

6. **Allelic dose-response.** Apply $\geq 3$ qualitatively different ablation types (rank reduction, mean, resample, zero) to the circuit. Report degradation at each type and the rank correlation with expected severity.

A skipped step must be named in the verdict.

## Case Studies

- [IOI Circuit](/framework/lenses_v6/examples/examples-ioi) -- epistasis between name-mover and S-inhibition heads; rescue via activation patching; partial knockout ordering from the Wang et al. circuit graph
