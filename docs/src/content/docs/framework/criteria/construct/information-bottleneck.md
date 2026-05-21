---
title: "Information Bottleneck"
validity_type: "Construct"
criterion_id: "C9"
---

# Criterion C9 — Information Bottleneck

| | |
|---|---|
| Validity type | Construct |
| Pass condition | I(circuit; output) / I(input; output) >= 0.7 with compression ratio dim(circuit)/dim(input) <= 0.3 |
| Evidence family | Information-theoretic |
| Minimum reporting | I(circuit; output), I(input; output), sufficiency ratio, compression ratio, MI estimation method and confidence bounds |
| Common failure mode | Reporting mutual information without compression ratio (high MI from passing everything through is trivial) |
| Lens | Information Theory |

## What this criterion requires

The information bottleneck criterion asks whether the circuit compresses the input to a low-dimensional sufficient statistic for the output. A circuit that retains all input information is not performing a computation — it is a pass-through. A circuit that retains almost no information about the output is not performing the claimed computation. The criterion requires both: high sufficiency (most output-relevant information is retained) and high compression (most input information is discarded).

The criterion is satisfied when:

1. **The sufficiency ratio I(circuit; output) / I(input; output) >= 0.7.** The circuit's internal representation captures at least 70% of the mutual information between input and output. This ensures the circuit is not discarding task-critical information.
2. **The compression ratio dim(circuit) / dim(input) <= 0.3.** The circuit's representation has at most 30% the dimensionality of the full input. This ensures the circuit is actually compressing, not just copying.
3. **Both conditions hold simultaneously.** High sufficiency without compression is trivial (pass everything through). High compression without sufficiency means the circuit is discarding relevant information.

This criterion does not establish the direction of information flow within the circuit, nor does it establish causal necessity. A circuit may be a sufficient statistic for the output without being the only such statistic — there may be redundant pathways. The criterion also depends on the MI estimation method; different estimators can give substantially different values on the same data.

## Minimum reporting rule

- I(circuit; output) with confidence bounds.
- I(input; output) with confidence bounds.
- Sufficiency ratio and whether it meets the >= 0.7 threshold.
- Compression ratio dim(circuit) / dim(input).
- MI estimation method used (e.g., MINE, binning, KSG) and any hyperparameters.
- If sufficiency is high but compression is low, note that this is a trivial pass and does not support a bottleneck interpretation.
