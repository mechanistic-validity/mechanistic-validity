---
title: "Framework Overview"
description: "The Mechanistic Validity taxonomy: six layers, five validity types, seven description modes, verdict tiers."
---

# Framework Overview

When five metrics return five numbers on the same circuit, what do you conclude? Not the average. The answer depends on what each metric measures, whether they can fail for the same reasons, and whether baselines rule out artifacts. Going from a set of measurements to a verdict is itself an inference problem — one that hasn't been fully addressed by the field.

Analysis methods have properties: what they detect, what they're blind to, how they fail. When you model those properties, you can say what a set of metrics jointly establishes rather than just aggregating individual scores. This framework does that modeling. It names the link between each concrete measurement and the conclusion it supports, so the gap between "ran a metric" and "validated a claim" becomes explicit.

Every circuit claim is a chain from measurement to conclusion. Reading bottom-up (layer 1 → 6) is how you build a claim. Reading top-down (6 → 1) is how you audit one. Reading sideways across layer 2 is how you check whether independent methods agree.

<img src="/mechanistic-validity/figures/framework/pipeline-vert-example.png" alt="Mechanistic Validity Pipeline" width="100%"/>

---

## Layer 1 — Description Mode

The level of description the claim operates at. This has to be set first because it determines what evidence is relevant. A claim about *which components* are involved needs different metrics than a claim about *what computation* they perform.

| If the evidence establishes... | Tag |
|---|---|
| What function the system computes and why | `[computational]` |
| What procedure the system executes | `[algorithmic]` |
| What information is encoded and in what geometry | `[representational]` |
| Which components are involved | `[implementational–topographic]` |
| How components are wired | `[implementational–connectomic]` |
| Distributional properties of activations | `[implementational–statistical]` |
| What input-output transformation a component performs | `[implementational–functional]` |

**The tag is a property of the conclusion, not the metric.** Activation patching doesn't automatically license an algorithmic tag. DAS-IIA doesn't automatically license a representational one. The tag is determined by what the researcher claims, and it has to be licensed by the evidence.

**IOI example:** Wang et al. (2022) identify which attention heads are causally involved in indirect object identification — that's `[implementational–topographic]`. The follow-on claim that name mover heads "copy the indirect object to the output position" is `[implementational–functional]`, which requires a different evidence standard. These are two separate claims and the paper sometimes blurs the line.

---

## Layer 2 — Evidence Family

Classifies what *kind* of signal a metric produces. Two metrics from different families that agree are stronger evidence than two from the same family, because they can fail for structurally different reasons.

| Family | What it measures |
|---|---|
| Causal | Interventions and counterfactuals — what breaks when you change something |
| Structural | Weight-space analysis — what the model looks like without running it |
| Representational | Latent geometry and decodability — what information is present where |
| Behavioral | Held-out task performance and generalization — does the circuit do what it claims to do on new inputs |
| Information-theoretic | Coding properties and information flow |

**IOI example:** Hanna et al. (2023) replicate the Wang et al. circuit using different component definitions and find partial agreement. The disagreement isn't a failure — it's the actual finding. The two methods are sensitive to different properties of the same underlying phenomenon. Reporting overlap (Jaccard similarity over heads, for instance) as a primary result rather than declaring one method correct is the right move.

---

## Layer 3 — Metrics and Calibrations

**Metrics** are the concrete runnable tests. Activation patching, resample ablation, DAS-IIA, K-composition, copying score, spectral SVD of weight matrices. A metric produces a number. That number is not a finding by itself.

**Calibrations** determine whether a metric's numbers are trustworthy before you use them as evidence.

The three calibrations that matter most:

- **Random-vector baseline (F09):** Run the same metric with a random vector in place of the identified direction. If you get similar IIA, the result is not meaningful. Sutter et al. (NeurIPS 2025) showed that non-linear alignment maps achieve 100% IIA on untrained models — which means IIA without this baseline is uninterpretable.
- **Untrained model baseline (F10):** Run the same metric on a randomly initialized model. If it passes, the result reflects architecture, not learned behavior.
- **Bootstrap stability (F01):** Run the metric across seeds, dataset splits, or prompt variants. A result that collapses under resampling is noise.

**Example of what goes wrong without these:** An IIA of 0.48 at L8.MLP on the SVA task looks meaningful. Whether it is depends on whether the random-vector baseline is 0.12 (meaningful gap) or 0.44 (no gap). The number alone tells you nothing. The baseline is the result.

### Protocols

A **protocol** bundles multiple metrics and their calibrations into a single structured evaluation, organized around a specific analytical lens. Instead of running activation patching in isolation and interpreting the number, a protocol runs it alongside complementary metrics (ablation variants, sufficiency tests, baseline calibrations), checks their internal consistency, and produces a result that maps directly onto the criteria in Layer 4.

Each protocol is associated with a lens — an analytical tradition that determines which metrics are relevant, what baselines are required, and how results should be interpreted. The Neuroscience lens contributes protocols built around necessity/sufficiency interventions. The Pharmacology lens contributes protocols built around dose-response curves and cross-model generalization. The Measurement Theory lens contributes protocols that test whether the metrics themselves are reliable before their outputs are used as evidence.

**IOI example:** Protocol A01 (Structural Causal Models) runs four metrics on the IOI circuit — baseline logit diff, per-role ablation, activation patching, and causal scrubbing — then runs three calibrations (bootstrap stability, seed variance, ablation method invariance). The protocol's output isn't four separate numbers — it's a structured assessment of whether the circuit satisfies Pearl's causal hierarchy at the association, intervention, and counterfactual rungs.

### Synthesis

A **synthesis protocol** combines results across multiple protocols to produce findings that no individual metric or protocol can establish alone. Where a protocol asks "does this circuit pass this set of tests?", a synthesis protocol asks "what do the results across tests tell us about the circuit's structure?"

Examples of what synthesis protocols produce:

- **Functional parcellation** — which components serve which functional roles, derived from the pattern of which protocols each component passes or fails
- **Circuit flow graphs** — directed causal structure among components, inferred from the combination of pairwise patching results and Granger-style temporal ordering
- **Cross-method agreement maps** — where different protocols agree and disagree about circuit boundaries, treated as a finding about method sensitivity rather than noise to suppress

Synthesis is where the framework's meta-level positioning matters most. The individual metrics and protocols produce evidence; synthesis asks what that evidence jointly establishes — and where it conflicts, what the conflict reveals.

---

## Layer 4 — Criteria

27 falsifiable conditions across five validity types. Each has a clear pass/fail threshold.

| Validity type | # criteria | What it covers |
|---|---|---|
| Construct | 5 | Is the thing being measured well-defined? Falsifiability, structural plausibility, task specificity, minimality, convergent validity |
| Internal | 5 | Is the causal claim justified? Necessity, sufficiency, specificity, consistency, confound control |
| External | 6 | Does the result generalize? Intervention reach, graded response, selectivity, effect magnitude, robustness, cross-architecture replication |
| Measurement | 6 | Are the metrics trustworthy? Reliability, invariance, baseline separation, sensitivity, calibration, construct coverage |
| Interpretive | 5 | Is the verdict properly scoped? Level declaration, level–evidence match, narrative coherence, alternative exclusion, scope honesty |

**The dependency order matters.** You cannot establish internal validity with unreliable metrics. You cannot establish external validity before you have a local causal result. Skipping steps is the most common structural error in circuit papers — not fabrication, just skipping.

**IOI example of a construct validity failure:** The original Wang et al. paper defines the circuit by running path patching and calling the result the circuit. The circuit name then comes from what the metrics found, not from an independent definition. That's circular — it's the construct validity criterion "falsifiability" failing. The construct should be defined before the metrics are run, so there's some chance the metrics could fail to find it.

**IOI example of a measurement validity failure:** Several papers report IOI replication with a single random seed and call the result robust. Bootstrap stability (F01) requires running across seeds. Results that were reported as stable often show variance of ±15-20 percentage points across seeds when this is actually done.

---

## Layer 5 — Validity Type

The five abstract dimensions a circuit claim has to address. A type is satisfied when all its criteria are met. Partial satisfaction is reported as partial validity — not suppressed.

A claim can be internally valid (the causal result is real in one setting) without being externally valid (it doesn't generalize). A claim can pass all four empirical types while failing interpretive validity (the verdict is stated at a level the evidence doesn't support).

**Cross-architecture example:** Multiple papers demonstrate necessity and sufficiency for induction heads within GPT-2 Small, then generalize to "induction circuits implement in-context learning across transformer models." That move from one model to a general claim is an external validity claim — specifically cross-architecture generalization. It needs to be tested, not inferred. When Olsson et al. (2022) actually test across model sizes and families, the result holds reasonably well. When other papers make the same leap without testing it, the interpretive validity criterion "scope honesty" fails.

---

## Layer 6 — Verdict

The claim itself, stated with explicit scope. A verdict needs to name: the component(s), the attributed computation or behavior, the model and task, the verdict-strength tier, and the description-mode tag.

The "unaddressed" clause is not optional. A verdict can be *Mechanistically supported* with external validity explicitly unaddressed. That's a complete verdict. Leaving "unaddressed" implicit means silent upgrading becomes possible when the paper gets cited.

---

## Verdict Tiers

| Tier | What it requires |
|---|---|
| **Proposed** | No intervention evidence — structural or representational only |
| **Causally suggestive** | Necessity shown via ablation; sufficiency not established |
| **Mechanistically supported** | Necessity + sufficiency (ablation + patching, at least 2 ablation variants) |
| **Triangulated** | All internal criteria met, plus at least 1 external and 1 construct criterion |
| **Validated** | All five validity types addressed with explicit baselines and convergent evidence |
| **Underdetermined** | Evidence present but consistent with multiple mechanisms |
| **Disconfirmed** | Fails decisively on a key criterion |

Most circuit discovery papers land at *Causally suggestive* or *Mechanistically supported*. That is not a failure. It's an honest description of what activation patching and ablation actually establish.

**Example (SVA):** "L8.MLP causally mediates subject-verb number agreement in GPT-2 Small on the Linzen dataset" — *Mechanistically supported* `[implementational–functional]`; external validity and interpretive validity unaddressed.

The two "unaddressed" flags are part of the verdict, not a disclaimer appended afterward.

---

## Dependency Order

Validity types gate each other. The order is:

1. **Construct** — define what you're looking for before measuring it
2. **Measurement** — calibrate metrics before interpreting their outputs as evidence
3. **Internal** — establish a local causal result using trustworthy measurements
4. **External** — generalize only after the local result is solid
5. **Interpretive** — audit the assembled verdict against the description-mode tag

Common violations in practice:

| Pattern | What it looks like in a paper |
|---|---|
| Circular construct | The circuit is named for what the metrics found |
| Uncalibrated IIA | IIA 0.48 reported without random-vector or untrained-model baseline |
| Single-seed generalization | "Robust across tasks" from one random seed |
| Cross-arch before local | Cross-model results reported before within-model necessity is established |
| Algorithmic tag from ablation | "Implements SVA" concluded from ablation alone, which only licenses functional or topographic tags |

---

## On Multi-Metric Disagreement

When two metrics identify different sets of components as "the circuit," reporting the overlap as a first-class result is more informative than picking a winner. A Jaccard similarity of 0.3 between path patching and activation patching on IOI isn't a problem to resolve — it's telling you something about the construct, the metrics, or both. Suppressing the disagreement loses that information.

Hanna et al. (2023) and the original Wang et al. IOI paper find different head sets depending on ablation method and patching variant. The disagreement is real and has not been resolved. The appropriate verdict is *Underdetermined* on circuit boundaries, with the specific source of disagreement flagged.

---

## Measurement vs. Evidence

An IIA score of 0.48 at L8.MLP is a measurement. Whether it constitutes evidence that L8.MLP implements a mechanism is a validity question — one that requires the random-vector baseline, the untrained-model baseline, the cross-task control, and the replication rate across seeds. Evidence tells you what you measured. Validity tells you what you can conclude from it.
