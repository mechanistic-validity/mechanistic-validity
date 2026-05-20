---
title: "Pick a Mode Tag"
---

# C — How to Pick a Mode Tag: Decision Tree

Use this decision tree before writing any verdict.

---

## The decision tree

```
Start: What is the claim about?
│
├─► "The model predicts / performs / scores on task X."
│     └─► [functional]
│
├─► "Component X encodes / represents information about variable Y."
│     └─► M3 (baseline-separated IIA) established?
│           ├─► Yes → [representational]
│           └─► No  → [functional]
│
├─► "Component X causally implements / is necessary for computation Y."
│     └─► I1 (necessity) established?
│           ├─► No  → use [representational] or [functional]
│           └─► Yes → I2 (sufficiency) established?
│                       ├─► No  → [causal-mechanistic] Causally suggestive
│                       └─► Yes → [causal-mechanistic] Mechanistically supported or above
│
├─► "The weights of component X implement computation Y."
│     └─► C2 (structural plausibility) established?
│           ├─► No  → use [representational] or [causal-mechanistic]
│           └─► Yes → I1 causal support?
│                       ├─► No  → [structural-mechanistic] Proposed
│                       └─► Yes → [structural-mechanistic] Causally suggestive or above
│
└─► "The mechanism generalizes across [model / task / prompts]."
      └─► E5 (robustness) established?
            ├─► No  → use non-transportable tag with scope restriction
            └─► Yes → [transportable]
```

---

## Common wrong choices

| What researcher writes | Evidence supports | Correct tag |
|---|---|---|
| "L8.MLP implements SVA" (no circuit-only pass) | I1 only | `[causal-mechanistic]` *Causally suggestive* |
| "Circuit generalizes to other models" (no cross-arch test) | Neither E5 nor E6 | `[causal-mechanistic]` *Triangulated* with scope restriction |
| "Factor represents subject-verb number" (IIA=0.48, no baseline) | M3 partial | `[representational]` *Causally suggestive* (pending baselines) |

---

## The tag does not determine the tier

Mode tag (description level) and verdict tier (strength of evidence) are **independent**.

- `[representational]` *Validated* — thoroughly validated representational claim
- `[causal-mechanistic]` *Causally suggestive* — causal claim with only necessity established
- `[structural-mechanistic]` *Triangulated* — structural claim corroborated by multiple metrics

Always state both.
