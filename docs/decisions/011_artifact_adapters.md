# ADR-011: Artifact Adapter Pattern for SAEs/Factors

**Date:** 2025-05-19
**Status:** Implemented (schema + SAE/FactorBank adapters)

## Decision

Define an `ArtifactAdapter` base class with `load()`, `directions()`, `activations()`, `ablate()`, and `metadata()` methods. Concrete adapters wrap specific artifact types:

- `SAEAdapter` wraps `sae_lens.SAE`
- `FactorBankAdapter` wraps factorized-circuits `FactorBankSAE`

Future adapters: transcoders, crosscoders, linear probes.

`ArtifactManifest` (Pydantic) captures metadata: artifact_type, target_model, hook_point, d_in, d_sae, construction params.

## Justification

- Different artifact types share the same operations: encode input, get directions, ablate features
- Metrics like `role_ablation` need to ablate SAE features, factor bank features, transcoder features — same interface, different internals
- `maps_to_features: list[tuple[int, str, int]]` on ComputationalStep stores (layer, artifact_type, feature_idx), connecting the spec to the adapter layer

## Alternatives Considered

1. **Direct sae_lens dependency** — Can't support factor banks, transcoders, or future artifacts
2. **No abstraction** — Each metric would need artifact-type-specific code paths
3. **Plugin system** — Over-engineered; we know the artifact types in advance

## Impact

- `lib/artifacts/adapter.py`: base class + manifest
- `lib/artifacts/sae.py`: SAEAdapter
- `lib/artifacts/factor_bank.py`: FactorBankAdapter
- `lib/artifacts/__init__.py`: re-exports
