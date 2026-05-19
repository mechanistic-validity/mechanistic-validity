"""CircuitSpec — universal circuit schema for mechanistic interpretability.

Supports head-based, MLP, SAE/factor, and hybrid circuits.
Backward-compatible with the existing dict-based interface via to_dict().
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CircuitSpec:
    roles: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    bands: dict[str, tuple] = field(default_factory=dict)
    pathways: list[tuple[str, str]] = field(default_factory=list)
    mlp_nodes: dict[str, list[int]] = field(default_factory=dict)
    features: dict[str, list[tuple[int, str, int]]] = field(default_factory=dict)
    source: str = "published"
    model_family: str = "gpt2"
    paper_ref: str | None = None

    def get_all_heads(self) -> set[tuple[int, int]]:
        return {h for heads in self.roles.values() for h in heads}

    def get_all_edges(self) -> set[tuple[int, int, int, int]]:
        edges: set[tuple[int, int, int, int]] = set()
        for src_role, dst_role in self.pathways:
            for s in self.roles.get(src_role, []):
                for d in self.roles.get(dst_role, []):
                    if s[0] < d[0]:
                        edges.add((s[0], s[1], d[0], d[1]))
        return edges

    def to_dict(self) -> dict:
        return {"roles": self.roles, "bands": self.bands, "pathways": self.pathways}

    def to_mib_edges(self) -> dict[tuple[int, int, int, int], float]:
        return {e: 1.0 for e in self.get_all_edges()}

    def to_interpbench_edges(self) -> list[tuple[int, int, int, int]]:
        return list(self.get_all_edges())
