"""Shared types for task prompt generators."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


class TokenizerLike(Protocol):
    """Minimal tokenizer interface — HF, TransformerLens, and MIB all satisfy this."""
    def encode(self, text: str, **kwargs: Any) -> list[int]: ...


@dataclass
class TaskPrompt:
    text: str
    target_correct: str
    target_incorrect: str
    metadata: dict[str, Any] = field(default_factory=dict)


TaskBuilder = Callable[..., list[TaskPrompt]]


def __getattr__(name: str):
    if name == "CircuitSpec":
        from mechanistic_validity.lib.tasks.spec import CircuitSpec
        return CircuitSpec
    if name in ("CircuitTask", "HeadCircuitTask"):
        from mechanistic_validity.lib.tasks import task as _task_mod
        return getattr(_task_mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "TokenizerLike", "TaskPrompt", "TaskBuilder",
    "CircuitSpec", "CircuitTask", "HeadCircuitTask",
]
