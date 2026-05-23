---
title: "Tier 4: Triangulated"
description: "Multiple independent lines of evidence converge on the same mechanistic account — no single method's failure would collapse the claim."
---

# Verdict Tier 4: Triangulated

| | |
|---|---|
| Tier | 4 of 5 (progressive) |
| What it means | Multiple methods with non-overlapping assumptions converge on the same mechanism |
| Minimum evidence | C5 (multi-method convergence) + E5 (external robustness) + V2 (cross-procedure agreement) + nomological network density |
| Upgrade to Validated | Completeness — every component's function characterized, quantitative predictions confirmed, scope boundary tested |
| Downgrade to Mechanistically supported | If convergence fails (methods disagree on core components) or external robustness is refuted |

## What this tier establishes

A Triangulated claim has been confirmed by methods whose failure modes do not overlap. The key property is *robustness to methodological critique*: if activation patching is shown to have distributional artifacts, the weight-space analysis still stands. If the behavioral test has confounds, the attention pattern evidence is independent.

This is a qualitative transition, not merely "more evidence." A single methodology, no matter how well-executed, produces findings conditional on that methodology's assumptions. Triangulation means the finding survives the failure of any single method's assumptions. The claim's epistemic status is fundamentally different from a well-replicated single-method result.

Convergence is formalized via the robust core: the intersection of circuits identified by each independent method. Claims about the robust core are more strongly supported than claims about the union.

## Example verdict statement

> **Verdict:** Triangulated — `[implementational-topographic]`
> **Claim:** Induction heads (L5H5, L5H1 in GPT-2 Small) implement in-context copying via QK composition with previous-token heads.
> **Met:** C5 (attention pattern analysis + QK weight decomposition + training dynamics + behavioral ablation all converge), E5 (mechanism found in GPT-2 Small, Medium, and Large), V2 (manual circuit identification and ACDC agree on core heads, Jaccard = 0.72)
> **Open:** $I_{\text{fun}}$ (complete component-level function for all supporting heads), quantitative prediction (novel prediction not yet tested)
> **Scope:** GPT-2 family, in-context copying of arbitrary tokens, sequences with repeated subsequences

## Minimum reporting for this tier

- At least two methods named, with their assumptions explicitly stated
- Jaccard similarity (or equivalent overlap measure) between circuits identified by each method
- The robust core (intersection) identified and distinguished from method-specific periphery
- External robustness evidence: distributions or model sizes tested beyond discovery context
- At least three independently testable predictions, with at least two confirmed by different methods

## Upgrade and downgrade

| Direction | What's required |
|---|---|
| → Validated | Every component's input-output function characterized ($I_{\text{fun}}$). At least one novel quantitative prediction confirmed post-hoc. Scope boundary explicitly tested (mechanism fails just outside scope). Coverage $\kappa > 0.9$. |
| → Mechanistically supported (downgrade) | Methods are shown to share a hidden assumption (their "independence" was illusory). Or external robustness fails: mechanism does not transfer to claimed distributions/sizes. |

## What convergence is NOT

Running the same method twice (e.g., activation patching with different hyperparameters) is *replication*, not triangulation. The methods must have non-overlapping failure modes — if one fails due to a distributional assumption, the other must not share that assumption. Two variants of patching (mean vs. resample) are closer to replication than triangulation because both assume the same causal model of intervention.

## Characteristic occupants

- **Induction heads** ([Olsson et al., 2022](https://arxiv.org/abs/2209.11895)) — the strongest candidate in the literature, confirmed by attention pattern analysis, QK composition analysis, training dynamics (phase transition), cross-model search, and behavioral ablation
- **IOI circuit robust core** — the subset of heads where [Wang et al. (2022)](https://arxiv.org/abs/2211.00593) manual analysis, [Conmy et al. (2023)](https://arxiv.org/abs/2304.14997) ACDC, and EAP-IG all agree

## Key references

- Olsson et al. (2022). *In-context Learning and Induction Heads.* [arXiv:2209.11895](https://arxiv.org/abs/2209.11895)
- Conmy et al. (2023). *Towards Automated Circuit Discovery.* [arXiv:2304.14997](https://arxiv.org/abs/2304.14997)
- Wang et al. (2022). *Interpretability in the Wild.* [arXiv:2211.00593](https://arxiv.org/abs/2211.00593)
- Lakatos, I. (1978). *The Methodology of Scientific Research Programmes.* [doi:10.1017/CBO9780511621123](https://doi.org/10.1017/CBO9780511621123)
- Miller et al. (2024). *Faithfulness Metrics for Circuit Discovery.* [arXiv:2407.08734](https://arxiv.org/abs/2407.08734)
