---
title: "Measurement Theory"
description: "The measurement validity lens: is the instrument that produced the number trustworthy?"
---

# The Measurement Theory Lens

This lens asks one question: **is the instrument that produced the number trustworthy?**

Every circuit finding begins with a number. An IIA score of 0.48. A faithfulness recovery of 87%. A logit difference of 3.10. The other lenses evaluate the claim that number supports — whether the causal logic holds, whether the effect generalizes, whether the interpretation is licensed. This lens evaluates something more basic: whether the number itself means what it appears to mean.

Measurement validity is the step MI most consistently skips. We run the instrument, get a number, and proceed directly to interpretation. What we skip is the question a measurement theorist would ask first: is this instrument reliable enough that the number is telling us about the model rather than about our choice of prompts? Is the score calibrated to anything we can interpret? Does the instrument measure the construct it claims to measure, or is it measuring its own capacity?

The distinction is the same one pharmacology makes between assay validation and drug efficacy. You validate the assay before interpreting what it measures. A failed assay produces numbers regardless — they just don't mean what you think.

## Key Distinctions

### Reliability vs validity

A measurement can be perfectly reliable (same result every time) and completely invalid (measuring the wrong thing). A probe that consistently returns 0.85 accuracy on a representation does not mean the representation encodes the claimed variable — it means the probe consistently extracts *something*. Reliability is necessary for validity but does not establish it.

In MI: bootstrap stability (F01) tells us our IIA score is reproducible. It does not tell us the score reflects the circuit's representation rather than the instrument's capacity to fit noise. A reliable instrument pointed at the wrong target produces confident wrong answers. This is why baseline separation (M3) exists as a separate criterion — it tests whether the instrument would produce similar scores on a model with no learned structure.

### Sensitivity vs specificity

Signal detection theory (Green & Swets 1966) separates two properties of any detection instrument: sensitivity (can it detect a real signal when one exists?) and specificity (does it correctly reject non-signals?). Hit rate alone is meaningless without the false alarm rate. A smoke detector that rings for everything has perfect sensitivity and zero specificity.

In MI: an instrument that identifies every head as "part of the circuit" has perfect sensitivity and zero specificity — it never misses a real component but also never rejects an irrelevant one. Conversely, a very conservative threshold might miss real components (low sensitivity) but never falsely includes irrelevant ones (high specificity). The $d'$ metric combines both into a single discriminability score. Current MI practice rarely reports false alarm rates — we report which heads are *in* the circuit but not how many non-circuit heads the method incorrectly flags.

### True score vs observed score

Classical test theory decomposes every measurement into true score plus error: $X = T + E$. The observed faithfulness score of 87% is not the circuit's true faithfulness — it is the true faithfulness plus whatever noise the prompt sample, random seed, and measurement procedure introduced. The proportion of variance attributable to the true score is the reliability coefficient.

In MI: when we report IIA = 0.48, we are reporting an observed score. The true score might be 0.52 (prompt sample was slightly unfavorable) or 0.44 (prompt sample was favorable). Without a confidence interval, we cannot know. Two circuits with observed scores of 0.48 and 0.52 may have overlapping true-score distributions — the apparent difference may be entirely measurement error. Reporting point estimates without confidence intervals invites over-interpretation of noise.

### Convergent vs discriminant validity

Campbell and Fiske (1959) argued that validity requires two things simultaneously: instruments measuring the same construct should agree (convergent validity), AND instruments measuring different constructs should disagree (discriminant validity). Agreement alone is not enough — if all your instruments agree about everything, they may share a bias rather than measuring a real signal.

In MI: if activation patching and weight-space analysis identify the same heads as the IOI circuit (convergent validity), that is strong evidence. But if they also identify the same heads for every other task (poor discriminant validity), the agreement reflects shared methodological bias rather than a real task-specific structure. The MTMM matrix formalizes this: cross-method agreement on the same circuit should exceed same-method agreement across different circuits.

## Analytical Constructs

### The multitrait-multimethod matrix

The signature artifact of measurement-theoretic evaluation is the multitrait-multimethod (MTMM) matrix (Campbell & Fiske 1959): a structured correlation table crossing k traits (circuits or mechanisms) with m methods (instruments or discovery procedures).

For k circuits measured by m methods, the MTMM matrix is a $km \times km$ correlation matrix with a specific block structure:

- **Monotrait-heteromethod correlations** (convergent validity) — do different methods agree about the same circuit? These should be high. If activation patching and weight-space analysis identify the same heads for the IOI circuit, that is convergent validity.
- **Heterotrait-monomethod correlations** (method effects) — do same-method measurements of different circuits correlate? These should be low. If activation patching gives similar scores to the IOI circuit and the Greater-Than circuit, that may reflect method bias rather than real similarity.
- **Heterotrait-heteromethod correlations** (discriminant validity) — do different methods measuring different circuits disagree? These should be lowest. This is the noise floor.

The validity condition: convergent > method effect > discriminant. Formally:

$$r(\text{trait}_i, \text{method}_a; \text{trait}_i, \text{method}_b) > r(\text{trait}_i, \text{method}_a; \text{trait}_j, \text{method}_a) > r(\text{trait}_i, \text{method}_a; \text{trait}_j, \text{method}_b)$$

In MI terms: the correlation between EAP-identified IOI circuit and weight-identified IOI circuit should exceed the correlation between EAP-identified IOI circuit and EAP-identified Greater-Than circuit, which should exceed the correlation between EAP-identified IOI circuit and weight-identified Greater-Than circuit.

To construct the matrix: identify k circuits and m discovery/evaluation methods. Run each method on each circuit. Compute pairwise Jaccard similarities (or correlation of attribution scores) between all km measurements. Arrange into the MTMM block structure. Check the validity ordering.

When the ordering is violated — when same-method correlations across circuits exceed cross-method correlations within circuits — the instruments share more variance with each other than with the construct they claim to measure. This is method bias, and it means the "circuit" may partly be an artifact of the discovery procedure.

## Sources

| Source | Year | Field | Principle |
|---|---|---|---|
| [Cronbach & Meehl, "Construct validity in psychological tests"](https://doi.org/10.1037/h0040957) | 1955 | Measurement Theory | **Reliability as prerequisite** — no construct validity claim is stronger than the measurement validity of the instrument supporting it |
| [Campbell & Fiske, "Convergent and discriminant validation by the multitrait-multimethod matrix"](https://doi.org/10.1037/h0046016) | 1959 | Measurement Theory | **MTMM and invariance** — an instrument is valid across contexts only if it produces comparable results under systematic variation of those contexts |
| [Green & Swets, *Signal Detection Theory and Psychophysics*](https://doi.org/10.1037/11187-000) | 1966 | Signal Detection | **$d'$ and AUROC/AUPRC** — separate discriminative ability from response bias; hit rate without false alarm rate is not sensitivity |
| [Lord & Novick, *Statistical Theories of Mental Test Scores*](https://archive.org/details/statisticaltheor00lord) | 1968 | Measurement Theory | **Classical test theory** — observed score = true score + error; reliability as the ratio of true-score variance to observed variance |
| [Cronbach, Gleser, Nanda & Rajaratnam, *The Dependability of Behavioral Measurements*](https://doi.org/10.1002/9781118619995) | 1972 | Measurement Theory | **Generalizability theory** — decompose error into identifiable sources (prompt sampling, seed variance, checkpoint) to know where measurement effort should go |
| [Hewitt & Liang, "A structural probe for finding syntax in word representations"](https://aclanthology.org/D19-1275/) | 2019 | Natural Language Processing | **Selectivity = linguistic accuracy $-$ control accuracy** — probe accuracy without a baseline measures instrument capacity, not representation structure |
| [Sutter et al., "How to evaluate satisfiability of interpretability claims"](https://arxiv.org/abs/2507.08802) | 2025 | Mechanistic Interpretability | **Baseline separation** — unconstrained nonlinear IIA achieves near-perfect scores on random-init models; the baseline is not optional |

## Validity type: [Measurement validity](/framework/validity-types/measurement)

> **Classical test theory (Lord & Novick 1968):** An observed score $X = T + E$, where $T$ is the true score and $E$ is measurement error. Reliability $\rho_{XX'} = \sigma^2_T / (\sigma^2_T + \sigma^2_E)$ is the proportion of observed variance attributable to the true score. An instrument with $\rho_{XX'} = 0.5$ carries as much noise as signal.

The difference between measurement theory and the other lenses is scope. The neuroscience lens asks whether a component implements a computation. The pharmacology lens asks whether the effect scales and generalizes. This lens asks whether the instrument that produced the numbers to evaluate those questions is itself reliable, calibrated, and measuring what it claims to measure. Instrument validity is prior to claim validity. A perfectly designed experiment with an unreliable instrument produces nothing.

Generalizability theory, developed by Cronbach and colleagues in 1972, extends classical test theory by decomposing the error term $E$ into identifiable sources: in our context, prompt sampling variance, random seed variance, and checkpoint variance. This decomposition matters for practice. If most of the variance is from prompt sampling, the fix is a larger prompt set. If most is from seed variance, the model itself is unstable and no prompt set will help. If most is from checkpoint variance, the mechanism is still being learned at the evaluated checkpoint. Knowing which source dominates tells us where effort should go.

## The criteria

### Reliability

An instrument whose output changes substantially under irrelevant perturbations cannot support any validity claim. If we resample prompts from the same distribution and the IIA score swings from 0.41 to 0.58, the score is a property of the specific prompt set, not of the circuit.

The Spearman-Brown formula connects current reliability to the prompt count needed to reach a target:

$$\rho_{nn'} = \frac{n \cdot \rho_{XX'}}{1 + (n-1) \cdot \rho_{XX'}}$$

where $n$ is the factor by which we multiply the number of prompts. If our current reliability is $\rho_{XX'} = 0.6$ on 50 prompts, doubling to 100 prompts gives $\rho_{nn'} = 2 \times 0.6 / (1 + 0.6) = 0.75$. This predicts whether a larger prompt set solves the problem or whether the variance is structural and a larger set won't help.

Conventional reliability thresholds from measurement theory (Nunnally 1978): below 0.5, the instrument is too noisy for any validity inference; 0.7 is acceptable; 0.9 is sufficient for interpretable small differences. These thresholds are not universal laws, but they provide orientation in the absence of domain-specific norms.

The most common reliability failure in current MI practice is discovery-evaluation overlap: the same prompts used to select the circuit are also used to evaluate it. The circuit was optimized to perform well on those prompts, so the apparent reliability is inflated. The fix is straightforward: hold out a prompt partition before running discovery and evaluate on it afterward.

**What to report.** Bootstrap the principal score across at least 100 prompt subsamples and report the 95% confidence interval. Compute split-half reliability: partition the prompt set, run the instrument on each half, report the Pearson correlation. Report internal consistency among circuit components if the circuit is large enough for it to be meaningful.

<details class="worked-example">
<summary>Worked example: bootstrap confidence intervals on IOI circuit faithfulness</summary>

[Wang et al. (2022)](https://arxiv.org/abs/2211.00593) report 87% faithfulness for the IOI circuit. This is the point estimate on the full evaluation set. To establish reliability, we can resample the evaluation prompts with replacement and recompute faithfulness on each bootstrap sample.

Suppose we draw 200 bootstrap samples of size 100 from the evaluation set and compute faithfulness on each. If the resulting distribution has mean 0.87 and standard deviation 0.06, the 95% confidence interval is approximately [0.75, 0.99]. That interval is wide. An instrument with $\sigma = 0.06$ on a score bounded between 0 and 1 has substantial prompt-sampling variance. The Spearman-Brown formula predicts that increasing from 100 to 400 prompts would reduce $\sigma$ to approximately 0.03, bringing the CI to [0.81, 0.93] — more interpretable.

A reliability check also reveals whether different prompt templates agree. If IOI faithfulness is 0.87 on the original template ("When Mary and John went to the store, John gave a drink to") but 0.61 on a paraphrased template, the score is template-specific and the reliability across templates is low. This is separate from the bootstrap CI, which only captures within-template prompt-sampling variance.
</details>

### Invariance

An instrument should give comparable results across model sizes and families. If IIA is 0.78 on GPT-2 Small and 0.31 on Pythia-160M, the difference could mean two things: the mechanism is weaker in Pythia, or the instrument is measuring something different in the two models. Invariance testing distinguishes these cases.

The measurement theory framework for invariance comes from confirmatory factor analysis. We test three levels sequentially. **Configural invariance**: the same constructs are present in both models (the same instrument structure is appropriate). **Metric invariance**: the loadings are equal across models (a unit change in the latent construct produces the same change in the measured score in both models). **Scalar invariance**: the intercepts are equal (a circuit with zero true effect produces the same baseline score in both models). Comparisons across models are only valid if at least metric invariance holds.

In practice, full measurement invariance testing is a substantial undertaking for MI instruments. A practical substitute is to include the untrained-model baseline for each model separately: if the baseline is 0.44 in GPT-2 Small and 0.29 in Pythia, the gap of 0.04 (trained minus random, GPT-2) vs. 0.02 (Pythia) is an apples-to-apples comparison even if the absolute scores differ.

**What to report.** At least two model sizes or families. The untrained-model baseline for each. Any observed differences characterized as potentially reflecting different mechanism strengths, different baseline levels, or potential instrument non-invariance.

### Baseline separation

Delta over a random-vector baseline and an untrained-model baseline should be substantially above zero.

This is the criterion whose absence most often produces false findings in current MI practice.

[Sutter et al. (NeurIPS 2025)](https://arxiv.org/abs/2501.07615) formally proved that unconstrained nonlinear IIA achieves near-perfect scores on random-initialization models. The alignment map has enough degrees of freedom to find a transformation that maps the source activations onto the target variable, regardless of whether the model's representation encodes that variable. The IIA score is a real measurement — it is a correct description of the alignment map's behavior. But without a baseline, it is not a measurement of the circuit's representation.

The minimum report for any IIA-based finding is three numbers: the score itself ($S_{\text{circuit}}$), the random-vector baseline ($S_{\text{random}}$), and the untrained-model baseline ($S_{\text{untrained}}$). The interpretable findings are:

$$\Delta_{\text{random}} = S_{\text{circuit}} - S_{\text{random}}$$
$$\Delta_{\text{arch}} = S_{\text{circuit}} - S_{\text{untrained}}$$

$\Delta_{\text{random}}$ tells us how much the model's actual representations contribute, over random directions. $\Delta_{\text{arch}}$ tells us how much the trained weights contribute, over the architectural prior (initialization structure, weight geometry). A large $S_{\text{circuit}}$ with a small $\Delta_{\text{random}}$ is a large number with a small finding. A modest $S_{\text{circuit}}$ with a large $\Delta_{\text{random}}$ and a large $\Delta_{\text{arch}}$ is a modest number with a genuine finding.

<details class="worked-example">
<summary>Worked example: interpreting IIA = 0.48 at L8.MLP for GPT-2 Small SVA</summary>

We measure IIA at layer 8's MLP and obtain 0.48. The published transcoder range for GPT-2 Small SVA is approximately 0.4–0.6. At first glance, 0.48 looks competitive with the literature.

Now add the baselines. Suppose we run the same alignment procedure on random unit vectors drawn from the same $d_{\text{model}}$-dimensional space, obtaining $S_{\text{random}} = 0.38$. We also run it on the same model before training (randomly initialized weights), obtaining $S_{\text{untrained}} = 0.33$.

The deltas are $\Delta_{\text{random}} = 0.48 - 0.38 = 0.10$ and $\Delta_{\text{arch}} = 0.48 - 0.33 = 0.15$. These are the actual findings. They say: the trained model's L8.MLP representations carry about 10 percentage points more causal information about SVA than random directions, and about 15 points more than the untrained architecture.

This is a real but modest signal. Whether it is a publishable finding depends on (a) whether the delta is stable across bootstrap resamples — if the CI on $\Delta_{\text{random}}$ is $[0.02, 0.18]$, the signal is real but noisy — and (b) whether the method has fewer parameters than DAS (which achieves 0.86–0.95), which would make a 0.10 delta at lower parameter cost an interesting result. Without the baselines, none of this analysis is possible.
</details>

### Sensitivity

A circuit with 12 components in a model with thousands of heads and neurons is a low-prevalence signal. In low-prevalence settings, AUROC can be misleadingly high while precision is poor — the instrument ranks circuit members above most non-members, but when it calls something a member, it is wrong most of the time.

Signal detection theory measures this with $d'$:

$$d' = z(\text{hit rate}) - z(\text{false alarm rate})$$

where $z$ is the inverse normal CDF. A $d' = 0$ means the instrument cannot distinguish circuit members from non-members at all. A $d' > 1$ indicates moderate discriminability. A $d' > 2$ is strong.

For circuit detection specifically, AUPRC (area under the precision-recall curve) is more informative than AUROC when the base rate is low. A circuit of 12 heads in a model with 144 total heads has a base rate of $12/144 \approx 0.08$. At this base rate, AUROC can reach 0.9 while precision is below 0.1 — the instrument correctly ranks circuit members above non-members most of the time, but when it calls something a member, it is almost always wrong.

**What to report.** AUPRC alongside AUROC for any circuit with fewer than 25 components. The base rate. Whether the reference circuit used to compute these metrics was discovered by the same instrument family, in which case agreement is partly mechanical.

![Signal Detection Framework — two-panel d-prime comparison showing standard vs high random baseline](/figures/signal_detection_minimal.svg)

### Calibration

A score is calibrated when we can locate it on a known scale. Without calibration, a number is a relative ranking within one experiment, not a measurement. Two papers reporting "87% faithfulness" may be measuring different quantities; calibration requires enough specificity to determine whether they are comparable.

The following table provides calibration reference points for common tasks and models:

| Task | Model | Metric | Full-model baseline | Circuit baseline | Recovery | Source |
|---|---|---|---|---|---|---|
| IOI | GPT-2 Small | Logit difference | 3.56 | 3.10 | 87% | [Wang et al. 2022](https://arxiv.org/abs/2211.00593) |
| Greater-Than | GPT-2 Small | Prob. difference | 81.7% | 72.7% | 89.5% | [Hanna et al. 2023](https://arxiv.org/abs/2305.00586) |
| SVA | GPT-2 Small | Logit diff / acc. | 0.70 | 0.65 | 93% | Lazo et al. 2025 |
| SVA (DAS) | GPT-2 Small | IIA | — | 0.86–0.95 | — | [Mueller et al. 2025](https://arxiv.org/abs/2504.13151) |
| SVA (transcoders) | GPT-2 Small | IIA | — | 0.4–0.6 | — | Published range |
| SVA (SAE features) | GPT-2 Small | IIA | — | Below raw neurons | — | [Mueller et al. 2025](https://arxiv.org/abs/2504.13151) |

All faithfulness numbers should be read as: "recovery under [ablation method] on [prompt distribution]." The IOI circuit's 87% is under mean ablation with the Wang et al. prompt set; [Miller et al. (2024)](https://arxiv.org/abs/2407.08734) show that different choices produce substantially different numbers for the same circuit. A new IIA score of 0.52 on GPT-2 Small SVA sits in the transcoder range and well below the DAS range — whether that is good or bad depends on the method's parameter count and the claim being made.

### Construct coverage

An instrument should measure what it claims to measure rather than a correlated proxy.

[Hewitt and Liang (EMNLP 2019)](https://arxiv.org/abs/1909.03368) showed this failure mode concretely for probes: a probe achieving 90% syntactic accuracy may achieve 85% on a control task where labels are shuffled into word-type statistics. The probe is measuring its own capacity, not the representation's structure. The selectivity — the 5 percentage point gap — is the valid measurement.

Sutter et al. (NeurIPS 2025) showed the same pattern for IIA: unconstrained nonlinear alignment maps achieve near-perfect IIA on random-initialization models. What gets measured is the map's flexibility, not the model's representational geometry. Linear IIA (DAS with a linear map) makes a specific, falsifiable claim about representational geometry: that the causal variable is linearly encoded. Nonlinear IIA makes a much weaker claim about which the alignment map architecture provides essentially no information.

The practical test is to vary the alignment map's capacity. If IIA remains high when the map dimension is reduced from $d_{\text{model}}$ to $d_{\text{model}} / 4$, the finding is robust to map complexity. If IIA collapses, it was measuring map flexibility. A control task — same probe architecture, labels that require no representational information — provides a direct test in the Hewitt and Liang sense.

**What to report.** The alignment map architecture stated explicitly. IIA measured across at least two map capacities. A control task at matched capacity if the construct coverage claim is central.

## Evidence patterns

| Evidence pattern | What it establishes | Recommended language |
|---|---|---|
| Score, no baselines | Instrument capacity | "Uncalibrated score; baselines pending" |
| Score + random baseline only | Signal over chance | "$\Delta_{\text{random}} = X$" |
| Score + both baselines | Signal over chance and arch prior | "$S = X$ ($\Delta_{\text{random}} = X_1$, $\Delta_{\text{arch}} = X_2$)" |
| AUROC, no AUPRC, low base rate | Ranking, not detection | "Ranks circuit members above non-members" |
| High IIA, collapsed with linear map | Map flexibility, not linear geometry | "IIA achievable; not linearly encoded" |

## Verdicts

Measurement validity gates the interpretation of every other evidence type:

- **Any verdict above Proposed** requires at least a bootstrap CI (reliability) and a random-vector baseline (baseline separation). Without these, a score is a data point, not a finding.
- **Causally suggestive → Mechanistically supported:** Requires calibration against at least one published reference point.
- **Mechanistically supported → Triangulated:** Requires invariance across at least two models and construct coverage confirmation.

## Protocol

For any reported score from a circuit evaluation instrument:

1. **Reliability.** Bootstrap across 100+ prompt subsamples; report 95% CI. Compute split-half correlation. If $\rho_{XX'} < 0.7$, apply Spearman-Brown to determine whether a feasible prompt increase would bring reliability above threshold.
2. **Invariance.** Test on at least two model sizes or families with separate untrained-model baselines.
3. **Baseline separation.** Report $S_{\text{random}}$, $S_{\text{untrained}}$, $\Delta_{\text{random}}$, and $\Delta_{\text{arch}}$. These are the primary reported quantities, not $S_{\text{circuit}}$ alone.
4. **Sensitivity.** AUPRC alongside AUROC for circuits with fewer than 25 components; state the base rate.
5. **Calibration.** Locate the score against at least one published baseline on the same task and model; state the ablation method and prompt distribution precisely.
6. **Construct coverage.** State the alignment map architecture; vary its capacity; run a control task at matched capacity if the representational geometry claim is central.

A skipped step must be named in the verdict.

## Case studies

For full worked examples applying all five lenses (including measurement validity) to published claims:

- [IOI Circuit](/framework/lenses_v6/examples/examples-ioi) — reliability untested; single prompt template
- [Induction Heads](/framework/lenses_v6/examples/examples-induction-heads) — multiple independent measurements converge
- [SAE Features](/framework/lenses_v6/examples/examples-sae-features) — baseline separation is the central question
- [Probing Classifiers](/framework/lenses_v6/examples/examples-probing) — measurement without construct coverage (Hewitt & Liang)
- [Othello World Model](/framework/lenses_v6/examples/examples-othello) — calibration question: linear decodability vs. world model
- [Grokking](/framework/lenses_v6/examples/examples-grokking) — full measurement validity (toy model, exact weights known)
