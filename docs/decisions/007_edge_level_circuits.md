# ADR-007: Edge-Level Circuit Flexibility

**Date:** 2025-05-19
**Status:** Implemented (schema only — edge-level metrics not yet built)

## Decision

Extend `ComputationalEdge` with optional fields for edge-level circuits:
- `weight: float | None` — edge weight (importance score from EAP/ACDC)
- `source_component: tuple[int, int] | None` — specific source component (head or neuron)
- `target_component: tuple[int, int] | None` — specific target component

This allows the framework to represent both:
1. **Node-level circuits** (most existing work): "These heads implement duplicate token detection"
2. **Edge-level circuits** (EAP, ACDC): "The connection from head 0.1 to head 9.9 with weight 0.73 carries the IOI signal"

## Problem

Different circuit discovery methods produce different granularity:
- Manual analysis (Wang et al. IOI): roles with head lists (node-level)
- EAP (Syed et al.): weighted edges between specific components
- ACDC: edges selected by thresholding
- MIB Track 1: edge sets

The framework needs to represent all of these without forcing one representation onto all methods.

## Alternatives Considered

1. **Node-only** (status quo) — Can't represent EAP output, incompatible with MIB Track 1
2. **Edge-only** — Would require converting node-level circuits to edge sets, lossy
3. **Separate EdgeCircuit class** — Parallel hierarchy, more code to maintain

## Justification

- Optional fields: existing node-level specs don't set them, no breaking changes
- When weight/source_component/target_component are set, the edge carries richer information
- Future: edge-level role_ablation could ablate specific connections (path patching) rather than entire roles
- Compatible with MIB Track 1 edge sets

## Impact

- `spec.py`: 3 optional fields on ComputationalEdge
- No metric changes yet — edge-level intervention requires path patching, which is a separate implementation
