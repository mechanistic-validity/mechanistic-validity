---
title: "Run a Convergence Check"
---

# F — How to Run a Convergence Check: Multi-Instrument Agreement Protocol

Convergent validity (C5) requires that ≥2 instruments from different evidence families agree on which components are circuit members. This guide defines the protocol for computing and reporting instrument agreement.

---

## Step 1: Collect nominations from each instrument

For each instrument run, produce a **ranked list of component nominations**: the set of heads/MLPs the instrument identifies as circuit-relevant, in order of signal strength.

| Instrument | Evidence family | Nomination output |
|---|---|---|
| Weight classifier | Structural | Top-k heads by F1 score; threshold: F1 ≥ 0.70 |
| EAP attribution | Causal | Top-k heads by attribution score; threshold: top quintile |
| DAS-IIA | Representational | Top-k positions by IIA score; threshold: IIA ≥ 0.40 |
| Activation patching | Causal | Top-k heads by patching effect; threshold: Δ ≥ 0.10 × full-model metric |
| Zero ablation | Causal | Heads whose ablation degrades metric by ≥ 10% of full-model value |

---

## Step 2: Compute pairwise Jaccard similarity

For any two nomination sets A and B (both thresholded to binary circuit/non-circuit):

```
Jaccard(A, B) = |A ∩ B| / |A ∪ B|
```

Interpretation:
| Jaccard | Interpretation |
|---|---|
| ≥ 0.7 | Strong convergence — instruments agree substantially |
| 0.4–0.7 | Moderate convergence — core components shared, periphery differs |
| 0.1–0.4 | Weak convergence — some overlap but substantial disagreement |
| < 0.1 | Near-zero convergence — instruments nominate different components → *Underdetermined* |

---

## Step 3: Compute the convergence matrix

Report all pairwise Jaccard values in a matrix:

```
              Weight  EAP    DAS-IIA  Patching  Zero-abl
Weight         1.00   [J1]   [J2]     [J3]      [J4]
EAP            [J1]   1.00   [J5]     [J6]      [J7]
DAS-IIA        [J2]   [J5]   1.00     [J8]      [J9]
Patching       [J3]   [J6]   [J8]     1.00      [J10]
Zero-abl       [J4]   [J7]   [J9]     [J10]     1.00
```

If any off-diagonal Jaccard < 0.1 for instruments from different evidence families, the claim is *Underdetermined* pending a discriminating experiment (see [G_handle-disagreement.md](G_handle-disagreement.md)).

---

## Step 4: Compute the consensus set

The **consensus circuit** is the intersection of nominations from ≥2 instruments from different evidence families:

```python
consensus = set(weight_nominations) & set(eap_nominations)
# or for stricter: voted in by ≥3 instruments
consensus_3way = {c for c in all_components if sum(c in s for s in all_nominations) >= 3}
```

Report:
- Consensus circuit (components nominated by ≥2 instruments from different families)
- Majority circuit (components nominated by ≥3 instruments of any family)
- Jaccard between consensus circuit and each individual instrument's set

---

## Step 5: Report the convergence result

```
## Convergence Check: [Task] circuit in [Model]

Instruments run: [list]
Nomination threshold: [threshold per instrument]

Nominations:
  Weight classifier: {L8H6, L9H7, L10H2, ...} (k=[n])
  EAP attribution:   {L7H3, L9H1, L11H5, ...} (k=[n])
  DAS-IIA:           {L8.MLP, L9H6, ...}       (k=[n])

Pairwise Jaccard:
  Weight ∩ EAP:      [J1]
  Weight ∩ DAS-IIA:  [J2]
  EAP ∩ DAS-IIA:     [J3]

Consensus set (≥2 instruments, different families): {[components]}
Convergent validity (C5): [✓ Jaccard ≥ 0.5 across ≥1 pair] / [✗ Jaccard < 0.1 → Underdetermined]

Verdict impact: [upgrade to Triangulated] / [remain at Mechanistically supported] / [Underdetermined]
```

---

## The current project status

The weight-circuit and EAP-circuit have been compared (Jaccard ≈ 0). This places the SVA and IOI circuit claims at *Underdetermined*. The convergence check has been run; the result is that convergence has failed. See [G_handle-disagreement.md](G_handle-disagreement.md) for the resolution protocol.
