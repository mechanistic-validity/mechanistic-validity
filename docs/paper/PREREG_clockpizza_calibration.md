# Preregistration — Ground-Truth Calibration of Mechanistic Validity

**Paper:** Mechanistic Validity, §9 "Validating the Framework Itself"
**Status:** DRAFT — freeze before running. Commit, record SHA here and in the paper, then run.
**Freeze SHA:** `________` (fill on commit)

---

## 0. What this study is for

§9 claims the framework can be turned on itself. This is the experiment that makes that true. The Validated tier is currently **empty** — no published claim reaches it — so a skeptic can say the top tier is untested and the thresholds are unfalsifiable. This study reaches it against a known answer, and more importantly tests whether the framework can **deny** the tier to a wrong claim.

Reachability alone is weak. Discrimination is the real test.

---

## 1. Why modular arithmetic, and why Clock vs Pizza

A one-layer transformer trained to generalization on modular addition implements the computation through a Fourier construction whose structure is characterizable in closed form. Critically, models learn **one of two distinct mechanisms** — "Clock" or "Pizza" — that are **behaviorally equivalent** (both compute mod-p addition correctly) but mechanistically different, and are separable by a Fourier-domain signature (pre-unembed activations show a large mode at \(2k \bmod p\) for Pizza, at \(k\) only for Clock).

This is the ideal calibration testbed precisely because **behavior cannot distinguish the two**. Any framework that discriminates them must be using something other than task performance — which is the entire claim of Mechanistic Validity.

IOI is excluded: its mechanism-level ground truth is exactly what is disputed.

---

## 2. Preregistered hypotheses and decision rules

Let each trained model have ground-truth label \(g \in \{\text{Clock}, \text{Pizza}\}\) assigned by the Fourier diagnostic **before** any claim is audited.

| ID | Hypothesis | Decision rule | If it fails |
|---|---|---|---|
| **H1 (reachability)** | The true-mechanism claim reaches **Validated**. | ≥1 model where the correct claim is scored Validated with all five validity types addressed. | The top tier is unreachable in principle; thresholds are miscalibrated and must be revised. |
| **H2 (discrimination)** | The **wrong-mechanism** claim (asserting Clock on a Pizza model, or vice versa) is denied Validated. | Wrong-mechanism claim scores a strictly lower tier than the true claim in **≥80%** of models. | The framework certifies a false mechanism — a direct falsification. Report it. |
| **H3 (right reason)** | Discrimination is driven by **C4 (discriminant) and/or I3 (specificity)**, not by behavioral criteria. | In ≥80% of discriminating cases, the criterion that differs is C4 or I3. | The tiers discriminate by accident; the criterion structure is not doing the work it claims. |
| **H4 (added value)** | **Standard faithfulness fails to discriminate** Clock from Pizza claims where the framework succeeds. | Faithfulness/logit-recovery scores for true vs wrong claims are statistically indistinguishable (CI overlaps), while tier assignment differs. | The framework adds nothing over existing metrics. The most damaging possible result — publish it. |
| **H5 (null claim rejected)** | A mechanism claim matching **neither** ground truth (e.g. memorization/lookup, or wrong-frequency Fourier) scores below Mechanistically Supported. | Neither-claims score ≤ Causally Suggestive in ≥90% of models. | The framework is too permissive at the middle tiers. |
| **H6 (inter-rater)** | Criterion-level assignments are reproducible across independent raters. | Criterion-level agreement ≥ 0.7 (Cohen's κ or equivalent) on a shared subset. | Verdicts are rater-dependent; the structured profile is not operational. |

**H4 is the hypothesis that justifies the paper's existence.** It is reported regardless of outcome.

---

## 3. Models

- Train **12–16** one-layer transformers on mod-\(p\) addition (\(p = 113\)), varying the conditions known to select between Clock and Pizza solutions (attention-rate / architecture / seed).
- Assign ground truth by the Fourier diagnostic. **Discard** models whose signature is ambiguous, and record how many were discarded.
- Target: at least 4 confirmed Clock and 4 confirmed Pizza models.
- Also train **matched random-weight and untrained controls** for M2 baseline separation.

Training is cheap (minutes per model); the ground-truth labelling is the load-bearing step, not the training.

---

## 4. Claims audited per model

Four claims per model, all stated in the paper's standard form (*causal variable Z implemented by carrier U performing role R*):

1. **TRUE** — the correct mechanism, correctly described.
2. **WRONG-MECHANISM** — the *other* algorithm, asserted of this model. Behaviorally equivalent; the hard negative.
3. **NEITHER** — a plausible but incorrect account (memorization/lookup table, or a Fourier claim at the wrong frequencies).
4. **PARTIAL** — right algorithm family, wrong specific components/frequencies. Tests middle-tier resolution.

Claims are written **before** auditing and are not revised after seeing criterion results.

---

## 5. Criteria run

The full suites are run, but the following are pre-specified as the ones expected to discriminate:

- **M1** reliability (test–retest), **M2** baseline separation (vs random/untrained controls), **M3** stability (across seeds)
- **I1** necessity (ablation of claimed carriers), **I2** sufficiency, **I3** specificity, **I4** double dissociation
- **E4** cross-model generalization (does the claimed mechanism appear across models sharing a ground-truth label?), **E5** graded response
- **C4** discriminant validity — *the key criterion*: can the claimed mechanism be separated from its neighbor (the other algorithm)?
- **V1–V2** description-mode declaration and level–evidence match

Criterion statuses use the paper's five-level scale, and **Untested / Inapplicable / Disconfirmed are recorded distinctly** and never collapsed.

---

## 6. Analysis

- Primary: tier assignment per claim type, per model. Report the full 4 (claim types) × N (models) matrix.
- H2/H5: proportion of models where ordering holds, with binomial CIs.
- H3: for each discriminating pair, record *which* criterion differed; report the distribution.
- H4: faithfulness/logit-recovery for true vs wrong claims, with CIs, alongside tier assignments.
- H6: two independent raters on a shared subset (≥25% of claims), criterion-level agreement.
- Blinding: the rater scoring criteria should not know the model's ground-truth label where operationally possible; record where it was not possible.

---

## 7. Stated limitations

- **Toy scope.** A one-layer model on modular arithmetic is not a frontier model. The study establishes that the tier structure *can* discriminate a known mechanism from a behaviorally equivalent wrong one; it does not establish that it does so at scale.
- **Ground truth is method-mediated.** The Clock/Pizza label comes from the Fourier diagnostic, not from construction. Far less contested than IOI, but not definitional — unlike a compiled Tracr program.
- **Author-run.** Mitigated by the preregistered decision rules and the inter-rater component, not eliminated.

---

## 8. Relationship to the MechViews study

This is deliberately the **pilot** for `mechanistic-views/paper/PREREG_cross_view_predictive_validity.md`. It shares the testbed (modular arithmetic with known mechanisms), the claim-bank logic (true / wrong / neither / partial), and the ground-truth handling. Infrastructure built here — claim specification, criterion scoring, ground-truth labelling, grouped analysis — is reused there at larger scale on Tracr. Build it once.

---

## 9. Compute plan (Modal, ≤10 parallel)

- One model per job; ≤10 concurrent. `--detach`, `timeout=86400`.
- Pin all dependencies with `==`; install matplotlib regardless.
- `tqdm` + timestamps in every loop.
- Human-readable run names: `mechval-clockpizza-model03-criteria-suite`.
- **All results to JSON files** in the repo; never parsed from stdout.
- Local smoke test on one model before launching the sweep.

---

## 10. Deviation log

| Date | Deviation | Reason |
|---|---|---|
| | | |
