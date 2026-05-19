# ADR-006: Mixed Component Support (Heads + MLPs + Neurons)

**Date:** 2025-05-19
**Status:** Implemented

## Decision

Extend `ComputationalStep` to map to multiple component types simultaneously:
- `maps_to_heads: list[tuple[int, int]]` — attention heads (layer, head)
- `maps_to_mlps: list[int]` — full MLP layers
- `maps_to_neurons: list[tuple[int, int]]` — individual MLP neurons (layer, neuron_idx)
- `maps_to_features: list[tuple[int, str, int]]` — SAE/factor features (layer, artifact_type, feature_idx) — future use

The `role_ablation` metric handles mixed ablation: attention heads via `hook_z` zeroing, MLPs via `hook_mlp_out`, neurons via `mlp.hook_pre`.

## Problem

Greater-than circuit (Hanna et al. 2023) explicitly requires MLP layers 8-11 for the year comparison computation. Attention-only ablation gives 33% confirmation rate. The framework couldn't represent or ablate MLP components.

More broadly, many circuits involve MLPs (IOI backup name movers use MLP10, induction heads interact with MLP layers). Restricting to attention heads misses a large fraction of the circuit.

## Alternatives Considered

1. **Attention heads only** (status quo) — Can't represent greater-than, SVA, or any MLP-heavy circuit
2. **Separate MLP step type** — Adds a new model class, more complexity for the same result
3. **Component union type** — `maps_to: list[Component]` where Component is a tagged union. Over-engineered for current needs.

## Justification

- Simple extension: add 3 list fields to ComputationalStep, each defaults to empty
- `component_types` property returns which types are active: `["attention", "mlp", "neuron"]`
- Intervention side: `_get_step_components()` returns `{"heads": set, "mlps": list, "neurons": list}`
- Ablation hooks: each component type has its own hook factory (`make_ablation_hook`, `_make_mlp_ablation_hooks`, `_make_neuron_ablation_hooks`)
- Measurement side: `_measure_components()` sums activation norms across all component types

## Hook details

| Component | Hook point | Ablation | Measurement |
|-----------|-----------|----------|-------------|
| Attention head (L, H) | `blocks.{L}.attn.hook_z` | Zero head H's output | L2 norm at last pos |
| MLP layer L | `blocks.{L}.hook_mlp_out` | Zero entire output | L2 norm at last pos |
| Neuron (L, N) | `blocks.{L}.mlp.hook_pre` | Zero neuron N's pre-activation | Abs value at last pos |

## Impact

- `spec.py`: 3 new fields on ComputationalStep + `component_types` property
- `role_ablation.py`: `_make_mlp_ablation_hooks()`, `_make_neuron_ablation_hooks()`, `_measure_components()`, updated `_run_targeted()`
- `greater_than/claim_spec.py`: `mlp_year_processing` step with `maps_to_mlps=[8, 9, 10, 11]`
