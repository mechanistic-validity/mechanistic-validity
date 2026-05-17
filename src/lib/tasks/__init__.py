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
