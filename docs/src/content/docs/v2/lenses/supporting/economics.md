---
title: "Economics"
description: "The market microstructure lens: are the circuit's weights fairly priced, its responses proportional, and its structure free of arbitrage?"
---

# The Economics Lens

Financial economics studies how prices aggregate information. A price is "fair" when it reflects all available information -- no one can profit by trading against it. The analogy to neural network weights is surprisingly precise: a weight is "fair" when training has converged -- no further gradient step can improve the loss. Kyle's (1985) model of market microstructure adds depth to this picture. In Kyle's framework, a market has three participants: an informed trader who knows the asset's true value, noise traders who trade randomly, and a market maker who sets prices by observing the aggregate order flow. The market maker cannot distinguish individual trades as informed or noisy, but the aggregate response reveals the information content. The key quantity is Lambda -- the price impact coefficient -- which measures how much one unit of trading moves the price.

Apply this to circuits. The way a circuit responds to interventions reveals whether it is robust (absorbs perturbations smoothly), whether it processes signal differently from noise (discriminates informed from random perturbations), and whether its structure is efficient (no cheaper substitute achieves the same computation). A circuit that responds identically to task-relevant and random perturbations is not performing computation -- it is merely reactive, the way an illiquid market overreacts to every trade regardless of information content. A circuit whose response scales linearly with intervention magnitude has well-behaved dynamics, the way a liquid market absorbs trades proportionally.

The pharmacology lens asks "how much effect?" The economics lens asks "how does the effect behave as a function of what you do?" Pharmacology's dose-response curve (E2) asks whether the effect is monotonic. Economics asks whether it is proportional. Pharmacology's selectivity (E3) asks whether the effect is larger on-target than off-target. Economics asks whether the circuit's response discriminates signal from noise in its inputs -- a finer-grained question that connects to how the circuit processes information, not just whether it is selective to intervention. And where pharmacology's therapeutic window characterizes the safe zone for a specific ablation, economics adds a question no other lens asks: is the circuit's structure efficient, or does an unexploited arbitrage opportunity reveal redundancy that other criteria miss?

## Key Distinctions

### Affinity is not price (connecting to pharmacology)

Pharmacology's distinction between affinity and efficacy has an economics analog. A component can be "expensive" -- high weight magnitude, large gradient, strong activation -- without being "fairly priced" in the economic sense. The price of a weight is what training set it to. The value of a weight is its actual contribution to the computation. Mispricing occurs when these diverge: a weight that is too large for its functional role (the optimizer overshot, or the component was important early in training and is now vestigial) or too small (the component carries signal that the optimizer has not yet fully capitalized on).

In MI: weight magnitude is often used as a proxy for importance. But a large weight on a component that contributes little to the task-relevant computation is mispriced -- like an overvalued stock whose price exceeds its fundamental value. The economics lens asks not "how large is the weight?" (affinity) or "how much does ablation hurt?" (efficacy) but "is the weight proportional to the component's actual contribution?" This is a calibration question that neither pharmacology nor neuroscience directly addresses.

### Informed vs noise trading

Kyle's key insight is that the market maker cannot distinguish informed from noise trades individually, but the aggregate response differs. Informed trading moves prices permanently -- the information gets incorporated into the consensus price and stays there. Noise trading moves prices temporarily -- the market reverts once the noise dissipates. The permanent price impact of informed trading is what allows information to be incorporated into prices at all. Without it, markets would not aggregate information.

In MI: a task-relevant perturbation is an informed trade. It carries signal about the computation the circuit performs. A random perturbation of equal magnitude is a noise trade. A well-functioning circuit -- one that genuinely computes something specific -- should respond more strongly to informed perturbations than to noise perturbations. The task-relevant perturbation should produce a larger, more systematic effect on the output because it engages the circuit's computational structure. The random perturbation should produce a smaller, noisier effect because it does not align with the circuit's functional organization. If the circuit responds equally to both, it is not discriminating signal from noise -- it is just a generic bottleneck that reacts to any perturbation, the way a panicking market maker moves prices on every trade regardless of content.

### Liquidity and resilience

A liquid market absorbs trades with small price impact. An illiquid market has large price impact -- small trades move prices dramatically. Liquidity is not a binary property; it is characterized by the shape of the price impact curve. Kyle's model predicts a linear relationship between order flow and price change for informed trading: twice the trade size produces twice the price move. Real markets deviate from this -- large trades have disproportionate impact (convex price impact), and some markets have threshold effects where small trades have no impact until a critical size is reached.

In MI: a "liquid" circuit absorbs perturbations smoothly. The behavioral effect scales linearly with perturbation magnitude -- half the ablation produces half the effect, twice the ablation produces twice the effect. An "illiquid" circuit has nonlinear responses: small perturbations do nothing (the circuit is buffered), then a slightly larger perturbation causes a disproportionate crash (the buffer is exhausted). The shape of the price impact curve -- linear, convex, or threshold -- characterizes the circuit's resilience. Linearity is a stronger claim than the monotonicity that pharmacology's graded response (E2) requires. A monotonically increasing but sharply convex curve passes E2 but reveals an illiquid circuit -- one that is fragile near its operating point.

### Arbitrage and efficiency

The efficient market hypothesis states that you cannot consistently earn excess returns because prices already reflect all available information. No arbitrage means no free lunch: there is no riskless profit available from rearranging your portfolio. An arbitrage opportunity exists when the same asset (or equivalent assets) have different prices in different markets -- you buy low, sell high, and pocket the difference.

In MI: the analog of an arbitrage opportunity is a substitution that achieves the same computation at lower cost. C4 (minimality) asks whether you can delete components from the circuit without losing performance. The arbitrage question is different: can you replace components in the circuit with non-circuit components that achieve the same performance? A circuit with an arbitrage opportunity has redundant structure in the substitution sense -- not because it has extra components, but because equivalent components exist outside it that could do the same job. If a non-circuit head can substitute for a circuit head with comparable task performance, the circuit boundary is not well-defined. The "price" the circuit pays for routing through its specific components is higher than necessary, because a cheaper alternative exists. The absence of arbitrage -- no substitution of comparable or smaller size achieves the circuit's performance -- is evidence that the circuit's structure is efficient: training has found the cheapest route, and there is no free lunch available from rerouting.

## Analytical Constructs

### The price impact curve

The signature artifact of economic evaluation is the price impact curve: a plot of intervention magnitude (x-axis) against behavioral effect (y-axis), with separate curves for task-relevant and random perturbations. In Kyle's model, the slope of this curve is Lambda -- the price impact coefficient. For informed trading, Kyle predicts a linear relationship.

The curve reveals structure that pharmacology's dose-response curve does not:

- **Lambda (slope)** -- the price impact coefficient. A steep slope means high sensitivity (illiquid). A shallow slope means the circuit absorbs perturbations easily (liquid).
- **Linearity** -- how well a straight line fits the curve. Kyle's model predicts linearity for informed trading. Deviation from linearity (threshold effects, convexity) characterizes the circuit's resilience profile.
- **Signal-to-noise ratio** -- the ratio $\Lambda_{\text{signal}} / \Lambda_{\text{noise}}$, where $\Lambda_{\text{signal}}$ is the slope for task-relevant perturbations and $\Lambda_{\text{noise}}$ is the slope for random perturbations. A circuit where both slopes are equal is processing all inputs the same way -- no task-specific computation is occurring. A circuit where $\Lambda_{\text{signal}} \gg \Lambda_{\text{noise}}$ is genuinely discriminating signal from noise.
- **Intercept** -- whether there is a threshold below which perturbations have no effect. A nonzero intercept on the task-relevant curve suggests buffering or redundancy. A zero intercept suggests the circuit is operating at capacity with no reserve.

To construct the curve: sweep perturbation magnitude across at least five values. At each magnitude, apply both a task-relevant perturbation (e.g., patching the indirect object's representation) and a random perturbation of equal norm drawn from the same activation subspace. Measure the behavioral effect for each. Plot both on shared axes. Fit a linear model to each curve and report $R^2$, slopes, and their ratio.

## Sources

| Source | Year | Field | Principle |
|---|---|---|---|
| [Fama, "Efficient capital markets: a review of theory and empirical work"](https://doi.org/10.2307/2325486) | 1970 | Financial Economics | **Efficient market hypothesis** -- prices reflect all available information; no arbitrage means no riskless profit from rearrangement |
| [Black & Scholes, "The pricing of options and corporate liabilities"](https://doi.org/10.1086/260062) | 1973 | Financial Economics | **No-arbitrage pricing** -- the absence of arbitrage constrains the relationship between assets; the analog is that the absence of substitution constrains the circuit's component set |
| [Kyle, "Continuous auctions and insider trading"](https://doi.org/10.2307/1913210) | 1985 | Market Microstructure | **Price impact (Lambda)** -- one unit of informed trading moves the price by Lambda; the slope of the intervention-effect curve is the circuit's price impact |
| [Glosten & Milgrom, "Bid, ask, and transaction prices in a specialist market with heterogeneously informed traders"](https://doi.org/10.1016/0304-405X(85)90044-3) | 1985 | Market Microstructure | **Adverse selection and the bid-ask spread** -- the spread compensates for the risk of trading against informed traders; the circuit's differential response to signal vs noise is the analog |
| [Hasbrouck, *Empirical Market Microstructure*](https://doi.org/10.1093/oso/9780195301649.001.0001) | 2007 | Market Microstructure | **Permanent vs transient price impact** -- informed trades have permanent impact; noise trades revert; the circuit's response to task-relevant vs random perturbations follows the same pattern |
| [Shleifer, *Inefficient Markets: An Introduction to Behavioral Finance*](https://doi.org/10.1093/0198292279.001.0001) | 2000 | Behavioral Finance | **Limits to arbitrage** -- even when mispricing exists, structural constraints can prevent correction; in MI, even when substitutes exist, the training process may not have found them |

## Validity type: [External validity](/v2/validity-types/external)

> **Kyle Lambda:** In Kyle's (1985) model, the equilibrium price impact is $\Lambda = \sigma_v / (2\sigma_u)$, where $\sigma_v$ is the standard deviation of the asset's fundamental value and $\sigma_u$ is the standard deviation of noise trading. The ratio captures how much one unit of order flow moves the price. In MI: $\sigma_v$ corresponds to the variance of task-relevant perturbation effects, $\sigma_u$ to the variance of random perturbation effects, and $\Lambda$ to the slope of the intervention-effect curve.

The economics lens contributes primarily to external validity by characterizing the quantitative structure of intervention effects -- not just how large they are (E4, effect magnitude) or whether they scale monotonically (E2, graded response), but whether the response is proportional and whether the circuit discriminates between signal and noise. The linearity question (E9) is a structural property of the dose-response curve that pharmacology does not test. The signal discrimination question (I15) extends internal validity by asking whether the circuit is genuinely computing something task-specific, tested from the information economics direction rather than the ablation direction. The arbitrage question (C12) extends construct validity by asking whether the circuit's component set is efficient in the substitution sense, which is orthogonal to minimality (C4).

## Criteria

| Code | Criterion | What it asks | Validity type |
|---|---|---|---|
| E9 | Price impact linearity | Does the intervention effect scale linearly with intervention magnitude? | External |
| I15 | Signal discrimination | Does the circuit respond more to task-relevant perturbations than random perturbations of equal magnitude? | Internal |
| C12 | Arbitrage freedom | Is there no cheaper way to achieve the same computation by rerouting through different components? | Construct |

### E9 -- Price impact linearity

The intervention effect should scale linearly with intervention magnitude. This is a structural claim about the shape of the dose-response curve that goes beyond monotonicity.

Pharmacology's graded response criterion (E2) asks whether the effect increases monotonically with intervention strength. This is necessary but weak -- a monotonically increasing curve can have any shape: threshold effects where nothing happens until a critical strength, convex curves where the effect accelerates, or sigmoidal curves with sharp transitions. Kyle's model predicts that in a well-functioning market, price impact is linear: twice the order flow produces twice the price move. The analog in MI is that twice the intervention magnitude should produce twice the behavioral effect, at least within the operating range.

Linearity matters because it characterizes resilience. A linear price impact curve means the circuit absorbs perturbations proportionally -- there are no fragile thresholds where a small increase in perturbation magnitude causes a disproportionate collapse. A convex curve means the circuit is robust to small perturbations but fragile to large ones. A threshold curve means the circuit has a buffer that masks small perturbations entirely, making dose-response data at low intervention strengths uninformative about the circuit's actual sensitivity.

**What it establishes.** The circuit's response to interventions is well-behaved and proportional within the tested range. The intervention-effect relationship can be characterized by a single parameter (the slope) rather than requiring a full nonlinear model. This makes effect predictions at untested intervention strengths possible and makes comparisons between circuits on different tasks commensurable.

**What it does not establish.** That the circuit is causally necessary (that is I1), that the effect is selective (that is E3/I3), or that the linear range extends beyond the tested magnitudes. Linearity within a tested range does not rule out threshold effects outside it.

**Threshold.** $R^2 \geq 0.8$ for a linear fit of behavioral effect vs intervention magnitude across at least 5 intervention strengths spanning the range from below the pharmacological threshold $\alpha_{\text{thresh}}$ (E2) to above the plateau $\alpha_{\text{plat}}$. Maximum residual as a fraction of the predicted value should be $< 0.2$ (no single point deviates by more than 20% from the linear prediction).

**Minimum reporting.**
- The price impact curve: behavioral effect (y-axis) vs intervention magnitude (x-axis) with at least 5 points
- Linear fit with $R^2$ and 95% confidence interval on the slope
- Maximum residual / predicted value, identifying which intervention strength produces the largest deviation
- Comparison to the pharmacological dose-response curve from E2, if available, to show whether monotonicity and linearity agree or diverge
- The intervention type (ablation fraction, patching interpolation, steering multiplier) named as part of the result

### I15 -- Signal discrimination

The circuit should respond more to task-relevant perturbations than to random perturbations of equal magnitude.

This criterion tests whether the circuit is genuinely performing a task-specific computation, approached from the information economics direction. The neuroscience lens tests specificity (I3) by asking whether ablating the circuit affects the target task more than other tasks -- a question about the circuit's role in the model's overall computation. Signal discrimination asks a complementary question: does the circuit respond differently to perturbations that carry task-relevant information versus perturbations that carry no information? This is Kyle's distinction between informed and noise trading applied to neural circuits.

The logic is as follows. A circuit that computes something specific must have internal structure aligned with the task-relevant features. Task-relevant perturbations engage this structure; random perturbations do not. If the circuit responds equally to both, its "computation" is not specific to the task -- it is a generic sensitivity to any perturbation, the way a bottleneck component degrades under any intervention. The signal-to-noise ratio of the circuit's response is a direct measure of its computational specificity, independent of the ablation-based specificity test (I3).

Concretely: at each of several perturbation magnitudes, apply a task-relevant perturbation (e.g., patching the key token's representation with a counterfactual) and a random perturbation of identical norm drawn from the same activation subspace. Measure the behavioral effect of each. The ratio of effects is the circuit's signal discrimination.

**What it establishes.** The circuit's response is structured: it responds preferentially to perturbations aligned with the task-relevant computation. This is evidence that the circuit has internal organization specifically tuned to the task, beyond generic sensitivity to input changes.

**What it does not establish.** That the circuit is necessary (I1) or sufficient (I2) for the computation. A circuit can discriminate signal from noise without being the only mechanism that does so. Signal discrimination also does not establish what the circuit computes -- only that it computes something specific to the task.

**Threshold.** Effect size ratio (mean task-relevant effect / mean random perturbation effect) $\geq 2.0$ at the intervention magnitude corresponding to the pharmacological $\alpha_{\text{thresh}}$ or, if that is not available, at the midpoint of the tested range. The difference between the two effect distributions must be statistically significant ($p < 0.01$, two-sided permutation test or Welch's $t$-test, with at least 50 perturbation samples per condition).

**Minimum reporting.**
- Effect distributions for task-relevant and random perturbations at each tested magnitude, reported as mean and standard deviation
- Effect size ratio with 95% confidence interval
- Statistical test, test statistic, and $p$-value
- Description of how the random perturbation was constructed (same subspace, same norm, independently sampled) to confirm that it is a valid noise baseline
- Comparison to I3 (specificity) if available, characterizing whether the two criteria agree or diverge

### C12 -- Arbitrage freedom

There should be no cheaper way to achieve the same computation by substituting non-circuit components for circuit components.

C4 (minimality) asks whether any component can be removed from the circuit without losing performance. This is the deletion question: is every component necessary? C12 asks the substitution question: for each component in the circuit, does any non-circuit component of equal or smaller size achieve comparable task performance when substituted in? If such a substitute exists, the circuit has an arbitrage opportunity -- its structure is not efficient because an equivalent computation is available through a different route.

The analogy to financial arbitrage is precise. An arbitrage opportunity exists when the same asset has different prices in different markets. In MI, an arbitrage opportunity exists when the same computation is achievable through different components at different "costs" (component count, activation norm, weight magnitude). A circuit that routes through head 9.1 when head 7.3 could do the same job at comparable performance is mispriced -- the circuit boundary is drawn incorrectly, or the circuit description is not capturing the real computational structure.

This criterion is genuinely new relative to the existing framework. C4 asks "can you make the circuit smaller by removing components?" C12 asks "can you make the circuit different by swapping components?" A circuit can be minimal in the deletion sense (every component is necessary) while failing arbitrage freedom (a non-circuit component can substitute for a circuit component). This happens when the circuit is one of several equally good configurations -- the training process landed on this one, but alternatives exist. Meloux et al. (2025) showed that multiple equally faithful circuits exist for the same task; C12 quantifies this at the component level.

**What it establishes.** The circuit's component set is efficient in the substitution sense: each component is not just necessary (removing it hurts) but irreplaceable (no substitute of comparable size achieves comparable performance). This is a stronger structural claim than minimality. It means the circuit boundary is well-defined not just by what is inside it but by the absence of viable alternatives outside it.

**What it does not establish.** That the circuit is the unique solution to the computational problem (multiple circuits can all be arbitrage-free). It also does not establish that the circuit's internal wiring is correct -- only that its component set cannot be cheaply substituted. An arbitrage-free circuit with incorrect wiring claims is still wrong about the mechanism.

**Threshold.** No alternative component subset of equal or smaller size achieves $\geq 90\%$ of the circuit's task performance when substituted for any single circuit component. As a secondary check: the top-5 substitute candidates (ranked by task performance when substituted in) each recover $< 50\%$ of the circuit's task performance. The substitute search should cover at least all same-type components (heads for heads, neurons for neurons) in the model.

**Minimum reporting.**
- For each circuit component: the best substitute found, its task performance when substituted in, and the performance ratio relative to the original circuit component
- The search space (which components were tested as potential substitutes) and the substitution procedure (one-at-a-time replacement, fixing the rest of the circuit)
- The threshold used and whether any component fails it
- Comparison to C4 (minimality): does the circuit pass minimality but fail arbitrage freedom, or vice versa?

## Evidence Patterns

| Evidence pattern | What it establishes | Recommended language |
|---|---|---|
| Linear price impact + signal discrimination (E9 + I15) | Proportional, task-specific response | "The circuit responds proportionally to perturbation magnitude ($R^2 = X$) and discriminates task-relevant from random perturbations (ratio $= Y$)" |
| Linear price impact without signal discrimination (E9 only) | Proportional but generic response | "The circuit responds proportionally; whether the response is task-specific is untested" |
| Signal discrimination without linearity (I15 only) | Task-specific but potentially fragile | "The circuit discriminates signal from noise (ratio $= Y$); the shape of the response curve is not characterized" |
| Arbitrage-free + minimal (C12 + C4) | Efficient and irreducible circuit | "The circuit is minimal (no component can be removed) and arbitrage-free (no component can be substituted)" |
| Minimal but not arbitrage-free (C4, C12 fails) | Necessary but replaceable components | "Every component is necessary, but substitutes exist for [components]; the circuit is one of several equivalent configurations" |
| Neither linear nor discriminating (E9 fails, I15 fails) | Generic bottleneck, not a task-specific circuit | "The component set responds generically to all perturbations; evidence for task-specific computation is absent" |

## Verdicts

- **Proposed to Causally suggestive:** The economics lens does not gate this transition. Causal evidence from the neuroscience lens (I1) is required. However, early signal discrimination evidence (I15) strengthens the case that the proposed circuit is task-specific rather than a bottleneck.
- **Causally suggestive to Mechanistically supported:** E9 (price impact linearity) strengthens the case for external validity alongside E2 (graded response) and E4 (effect magnitude). A circuit with a linear price impact curve and large effect magnitude has well-characterized quantitative behavior. I15 (signal discrimination) provides complementary evidence to I3 (specificity) for internal validity.
- **Mechanistically supported to Triangulated:** C12 (arbitrage freedom) provides construct validity evidence from a different direction than C4 (minimality) or C5 (convergent validity). A circuit that is minimal, arbitrage-free, and convergently identified by multiple methods has strong structural evidence from three independent tests.
- **Triangulated to Validated:** All three economics criteria (E9, I15, C12) contribute to the full evidence picture but are not individually required for validation. Their primary contribution is strengthening existing validity dimensions with evidence from a different theoretical tradition.

## Protocol

For a proposed circuit $C$ and behavior $B$:

1. **Price impact curve.** Sweep intervention magnitude across at least 5 values. At each magnitude, measure the behavioral effect. Fit a linear model and report $R^2$ and slope. Compare to the pharmacological dose-response curve (E2) if available.
2. **Signal discrimination.** At each intervention magnitude from step 1, apply both a task-relevant perturbation and a random perturbation of equal norm. Measure and compare the behavioral effects. Report the effect size ratio and statistical significance.
3. **Lambda comparison.** Fit separate linear models to the task-relevant and random perturbation curves. Report $\Lambda_{\text{signal}}$ and $\Lambda_{\text{noise}}$ (the slopes) and their ratio. A circuit with $\Lambda_{\text{signal}} / \Lambda_{\text{noise}} \gg 1$ is discriminating signal from noise at the structural level.
4. **Arbitrage search.** For each component in the circuit, test all same-type components in the model as potential substitutes. Measure task performance when each substitute replaces the original. Report the best substitute and its performance ratio.
5. **Efficiency summary.** Combine results from steps 1-4 with C4 (minimality) if available. Characterize the circuit as: (a) efficient and well-behaved (E9 + I15 + C12 pass), (b) well-behaved but replaceable (E9 + I15 pass, C12 fails), (c) task-specific but fragile (I15 passes, E9 fails), or (d) generic bottleneck (I15 fails).
6. A skipped step must be named in the verdict.

## Case Studies

For worked examples applying all lenses (including economics) to published claims:

- [IOI Circuit](/framework/lenses_v6/examples/examples-ioi) -- the economics lens can be applied via WC_M11 (Kyle Lambda); the existence of backup name-movers is a natural arbitrage test
- [Induction Heads](/framework/lenses_v6/examples/examples-induction-heads) -- signal discrimination is naturally high (induction heads respond specifically to repeated sequences, not random tokens)
- [Greater-Than](/framework/lenses_v6/examples/examples-greater-than) -- the successor heads' response to year-magnitude perturbations is a natural price impact curve
