# Preregistration: Ground-Truth Calibration of Mechanistic Validity

**Status:** FROZEN. No model had been trained when this was sealed.
**Timestamp:** 2026-08-01T02:12:56Z
**SHA-256 of this document at freeze** (computed with this line and the
two above absent, so it is reproducible by stripping them): `f43505a34b02012fd1af5fd0596c6066ee2d07c0ab2f54ac450acc60f3a2d1a1`
**Freeze commit:** recorded in the commit that sets this status.

**Terminology note.** This is a genuine preregistration rather than a held-out
analysis pre-specification. No model has been trained, no claim has been audited,
and no criterion has been scored. What is frozen here is the model suite, the
ground-truth labelling procedure, the claims to be audited, the criteria to be
run, the decision rules, and the analysis. The risk this hash protects against is
the one the framework itself names as a failure mode: adjusting tier thresholds
or claim wording after seeing which way the verdicts fall.

---

## 0. Disclosure: what is known when this document is written

Mechanistic Validity assigns claims to tiers running from Proposed through
Causally Suggestive, Mechanistically Supported, and Triangulated, to Validated.
In the audit of fifteen published claims reported in the paper, no claim reaches
Validated. The top tier is therefore currently empty, and a reader is entitled to
ask whether it is reachable at all or whether the thresholds are simply
unfalsifiable. That gap is the reason this study exists.

It is also known that one-layer transformers trained to generalization on modular
addition implement the task through a Fourier construction, and that such models
learn one of two mechanisms, conventionally called Clock and Pizza. Both use
circular embeddings, so the Fourier structure they share does not separate them;
they are distinguished by gradient symmetricity and distance irrelevance
(\S2).
The two mechanisms are behaviourally equivalent: both compute modular addition
correctly, and no measurement of task performance separates them.

Nothing has been computed for this study. No models have been trained, and the
distribution of Clock versus Pizza solutions under the training conditions below
is not known to the author.

---

## 1. Why this testbed

The calibration requires a mechanism known independently of the framework doing
the auditing. Modular addition supplies one, and the Clock/Pizza dichotomy
supplies something stronger: two known mechanisms rather than one. That converts
the study from a test of reachability into a test of discrimination. A framework
that certifies the true mechanism at Validated has shown only that its top tier
is attainable. A framework that certifies the true mechanism and denies the tier
to a behaviourally identical wrong mechanism has shown that its criteria carry
information task performance does not.

The indirect object identification circuit is excluded. Its mechanism-level
ground truth is precisely what the literature disputes, so it cannot serve as a
reference against which to calibrate.

---

## 2. Models and ground-truth labelling

Between twelve and sixteen one-layer transformers are trained on addition modulo
113, varying the training condition reported to select between Clock and Pizza
solutions. Each trained model is assigned a ground-truth label before any claim
about it is audited.

**The selection lever, specified.** An earlier draft of this section left the
"training conditions" unnamed and stated that labelling would use "the
Fourier-domain diagnostic." Both are corrected here against
\citet{zhong2024clock}, read directly.

The lever is the **attention rate** $\alpha$. The post-softmax attention matrix
$M$ is replaced by

$$M' = M\alpha + J(1-\alpha)$$

with $J$ the all-ones matrix, so $\alpha = 1$ retains attention and $\alpha = 0$
gives a constant attention matrix. The paper reports a phase transition in
$\alpha$: *"The Clock algorithm dominates when the attention rate is higher than
the phase change point, and the Pizza algorithm dominates when the attention
rate is lower than the point."* The transition point rises with model width, so
width is held fixed at 128 and $\alpha$ is swept.

**The labelling diagnostic, corrected.** Fourier-domain structure does **not**
separate Clock from Pizza — both are circular-embedding algorithms, and
circularity is the precondition for entering the study rather than the
discriminator. Labelling instead uses the paper's two metrics:

- **Gradient symmetricity** $s_g \in [-1,1]$: the mean cosine similarity between
  $\partial Q_{abc}/\partial E_a$ and $\partial Q_{abc}/\partial E_b$ over
  input-output triples. Pizza has symmetric gradients, Clock asymmetric.
  Reference values from the paper's own Model A (no attention, Pizza) and
  Model B (attention, Clock): **99.37% and 33.36%**.
- **Distance irrelevance** $q \in [0,1]$: the ratio of the mean within-diagonal
  standard deviation of the correct-logit matrix $L_{ij} = Q_{ij,i+j}$ to its
  overall standard deviation. Paper's reported ranges: **Pizza 0–0.4, Clock
  0.4–1**; its Model A and Model B give **0.17 and 0.85**.

A model is labelled **Pizza** if $s_g$ is high and $q < 0.4$, **Clock** if $s_g$
is low and $q > 0.4$, and **ambiguous** if the two metrics disagree or $q$ falls
near the 0.4 boundary. Ambiguous models are excluded and counted, per the rule
below. Circularity of the embeddings is checked first as an entry condition; a
model without circular embeddings implements neither algorithm and is excluded
separately.

A model enters the study if and only if the two diagnostics above assign it
unambiguously to Clock or to Pizza. Models labelled ambiguous are excluded, and
the number excluded is reported. The study requires at least four
confirmed models of each type; if the training sweep does not produce this,
additional models are trained under the same procedure until it does, and the
total number trained is reported.

Matched random-weight and untrained models are retained as controls for baseline
separation (M2).

Training these models is inexpensive. The labelling step, not the training, is
the operation this study depends on, and it is performed once and frozen before
auditing begins.

---

## 3. Claims audited

Four claims are written for each model, in the form the paper uses throughout:
a causal variable is implemented by a carrier that performs a role in computing
an output. All four are written before any criterion is scored, and none is
revised afterwards.

**True.** The correct mechanism for that model, correctly described.

**Wrong-mechanism.** The other algorithm, asserted of this model. Because the two
mechanisms produce identical behaviour, this is the hard negative on which the
study turns.

**Neither.** A plausible but incorrect account, either memorisation through a
lookup table or a Fourier claim at the wrong frequencies.

**Partial.** The right algorithm with the wrong specific components or
frequencies, included to test whether the middle tiers resolve degrees of
correctness rather than only correct against incorrect.

---

## 4. Criteria

The full criterion suites are scored. The following are pre-specified as those
expected to carry the discrimination, and the prediction that they do is itself
tested in H3.

Measurement validity contributes reliability under repetition (M1), separation
from random-weight and untrained controls (M2), and stability across seeds (M3).
Internal validity contributes necessity and sufficiency through ablation of the
claimed carriers (I1, I2), specificity (I3), and double dissociation (I4).
External validity contributes cross-model generalisation across models sharing a
ground-truth label (E4) and graded response (E5). Construct validity contributes
discriminant validity (C4), which is expected to do most of the work, since the
question of whether a Clock claim can be separated from a Pizza claim is exactly
a question about neighbouring constructs. Interpretive validity contributes
level declaration and level-evidence match (V1, V2).

Criterion statuses use the paper's five-level scale. Untested, Inapplicable, and
Disconfirmed are recorded as distinct outcomes and are never collapsed into a
single negative.

---

## 5. Predictions

All predictions are directional. Each is stated with the test that decides it and
the threshold that constitutes a pass.

### 5.1 Reachability (H1)

**H1 (primary, confirmatory).** The true-mechanism claim reaches Validated on at
least one model.

**Rationale.** The Validated tier requires all five validity types to be
addressed. On a model whose mechanism is known and whose scale permits the full
measurement and interpretive suites to be run, there is no principled obstacle to
meeting that bar. If the tier cannot be reached here, it cannot be reached
anywhere, and the thresholds are miscalibrated rather than demanding.

**Test.** Tier assignment for the true claim on each model.

**Pass threshold.** At least one model on which the true claim is scored
Validated with all five validity types addressed.

### 5.2 Discrimination (H2)

**H2 (primary, confirmatory).** The wrong-mechanism claim is denied Validated.

**Rationale.** A Clock claim asserted of a Pizza model is false, and the
framework should say so. Because the two mechanisms are behaviourally identical,
nothing in the model's task performance can supply the discrimination; it has to
come from the criteria.

**Test.** Tier assignment for wrong-mechanism claims compared against true claims,
within model.

**Pass threshold.** The wrong-mechanism claim scores a strictly lower tier than
the true claim on at least 80% of models, with a binomial confidence interval
reported.

**Known tension.** The wrong-mechanism claim is not a random claim. It describes
a real algorithm that the same architecture does implement under other training
conditions, so several criteria will legitimately return support for it. The
prediction is that the tier separates them, not that every criterion does. This
is noted in advance rather than discovered when the numbers arrive.

### 5.3 Discrimination for the stated reason (H3)

**H3 (secondary, confirmatory).** Where the framework discriminates, the criteria
responsible are discriminant validity (C4) or specificity (I3).

**Rationale.** The paper's account of why the framework works is that it asks
whether a construct can be separated from its neighbours and whether a component
does this task rather than everything. If discrimination instead arises from
unrelated criteria, the tier structure produces the right answer for reasons the
paper does not give, and the explanation needs revising even though the verdicts
are correct.

**Test.** For each model where H2's ordering holds, record which criteria differ
between the true and wrong claims, and report the distribution.

**Pass threshold.** C4 or I3 differs in at least 80% of discriminating cases,
after Benjamini-Hochberg correction (Section 6).

### 5.4 Value beyond existing metrics (H4)

**H4 (primary, confirmatory).** Standard faithfulness does not discriminate
Clock from Pizza claims where the framework does.

**Rationale.** This is the hypothesis on which the paper's contribution rests. If
a faithfulness or logit-recovery score already separates true from wrong
mechanisms, the framework's apparatus is redundant and should be reported as
such. The two mechanisms were selected precisely because behaviour cannot
separate them, so the expected result is that faithfulness is uninformative here.
Confirming that expectation is what licenses the claim that validity criteria
measure something faithfulness does not.

**Test.** Faithfulness and logit-recovery for true against wrong claims, with
confidence intervals, reported alongside tier assignments.

**Pass threshold.** Faithfulness scores for true and wrong claims are
statistically indistinguishable, with overlapping intervals, while tier
assignment differs at the rate specified in H2.

### 5.5 Rejection of the null claim (H5)

**H5 (secondary, confirmatory).** A claim matching neither ground truth scores
below Mechanistically Supported.

**Rationale.** A framework that discriminates between two sophisticated
alternatives but cannot reject an obviously wrong account is too permissive in
its middle range. This is the floor test that complements H2's ceiling test.

**Test.** Tier assignment for neither-claims.

**Pass threshold.** Neither-claims score at or below Causally Suggestive on at
least 90% of models, after Benjamini-Hochberg correction.

### 5.6 Inter-rater reliability (H6)

**H6 (robustness, pass/fail).** Criterion-level assignments are reproducible
across independent raters.

**Rationale.** The paper argues that the structured validity profile, rather than
the tier, is the primary output, on the grounds that disagreement then localises
to individual falsifiable criterion judgements. That argument requires the
criterion judgements to be reproducible, which has not been measured.

**Test.** Two raters independently score a shared subset of at least 25% of
claims. Agreement is computed at the criterion level, not the tier level.

**Pass threshold.** Cohen's κ ≥ 0.7 at the criterion level.

**Blinding.** The rater scoring criteria should not know the model's ground-truth
label wherever this is operationally possible. Where it is not possible, this is
recorded per criterion rather than glossed.

### 5.7 Partial claims (H7)

**H7 (exploratory, no pass/fail).** Partial claims score between neither-claims
and true claims.

**Status.** Exploratory, and excluded from the correction family. Reported
descriptively whatever the outcome, since the study is not powered to resolve
middle-tier ordering across four models per condition.

---

## 6. Multiple comparisons

H1, H2, and H4 are primary. H3 and H5 are secondary and confirmatory, and
Benjamini-Hochberg correction is applied across them at q = 0.10. H6 and H7 are
robustness and exploratory respectively, and are excluded from the correction
family.

If H2 fails, H3 is uninterpretable and is reported descriptively only, since
there are no discriminating cases whose criteria can be examined.

---

## 7. What would falsify the framework

Three outcomes count against Mechanistic Validity rather than against this study,
and each is reported without reframing.

If both the true and wrong claims reach Validated, the framework certifies a
false mechanism, and the tier structure is too permissive at its ceiling. If
neither reaches Validated on any model, the top tier is unreachable under
conditions maximally favourable to reaching it, and the thresholds require
revision. If faithfulness discriminates as well as the tier system, the framework
supplies vocabulary rather than measurement, and the paper should say so.

---

## 8. Relationship to the cross-view study

This study is deliberately the pilot for the cross-view predictive validity
experiment specified in `mechanistic-views/paper/PREREG_cross_view_predictive_validity.md`.
The two share a testbed, the logic of the claim bank, and the handling of
ground truth. Claim specification, criterion scoring, ground-truth labelling, and
grouped analysis are built once here at small scale and reused there across a
Tracr suite. Sequencing the smaller study first is a decision about
infrastructure as much as about publication order.

---

## 9. Limitations stated in advance

A one-layer model trained on modular arithmetic is not a frontier model, and this
study establishes only that the tier structure can discriminate a known mechanism
from a behaviourally equivalent wrong one at toy scale. Whether it does so at
scale is untested here.

The ground-truth label is supplied by the Fourier diagnostic rather than by
construction, so it is method-mediated. This is far weaker mediation than in the
IOI case, where the mechanism itself is contested, but it is not the definitional
ground truth a compiled program would provide.

The audit is run by the framework's author. The preregistered decision rules and
the inter-rater component reduce this exposure without removing it.

---

## 10. Execution order

1. Freeze this document and compute the SHA-256 hash.
2. Train the model suite under the specified conditions.
3. Assign ground-truth labels by Fourier diagnostic; record assignments and
   exclusions in `results/clockpizza_ground_truth.json`.
4. Write all four claims per model. Freeze them.
5. Score the criterion suites, blind to ground truth where possible.
6. Run H1 through H5, then H6 on the shared rater subset, then H7.
7. Apply Benjamini-Hochberg correction to H3 and H5.
8. Report everything to `results/clockpizza_calibration_results.json`.

Step 1 completes before step 2. Steps 3 and 4 complete before step 5, and no
claim may be revised after step 4.

---

## 11. Compute

One model per Modal job, at most ten concurrent, detached with a 24-hour timeout.
All dependencies pinned exactly, matplotlib installed regardless of use. Progress
bars and timestamps in every loop. Run names are human-readable, of the form
`mechval-clockpizza-model03-criteria-suite`. Every result is written to a JSON
file in the repository and never read back from logs. A single-model smoke test
runs locally before the sweep launches.

---

## 12. Artifacts to hash

| File | Contents |
|---|---|
| `PREREG_clockpizza_calibration_v2.md` | This document |
| `results/clockpizza_ground_truth.json` | Model labels and exclusions, once computed |

Hashes stored in `PREREG_clockpizza_calibration_v2_sha256.txt`.

---

## 13. Deviation log

Any departure from this document after the freeze hash is recorded here with date
and reason.

| Date | Deviation | Reason |
|---|---|---|
| | | |
