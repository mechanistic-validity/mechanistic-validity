"""CircuitTask — base class for all circuit evaluation tasks.

Concrete base (not ABC). Override get_circuit() and get_prompts().
HeadCircuitTask is a convenience subclass for tasks defined by
ROLES/BANDS/PATHWAYS modules.
"""
from __future__ import annotations

from mechanistic_validity.lib.tasks import TaskPrompt, TokenizerLike
from mechanistic_validity.lib.tasks.spec import CircuitSpec


class CircuitTask:
    task_id: str = ""
    model_family: str = "gpt2"
    source: str = "published"
    paper_ref: str | None = None

    def get_circuit(self) -> CircuitSpec:
        raise NotImplementedError(f"{type(self).__name__} must implement get_circuit()")

    def get_prompts(self, tokenizer: TokenizerLike, n_prompts: int = 40, seed: int = 42) -> list[TaskPrompt]:
        raise NotImplementedError(f"{type(self).__name__} must implement get_prompts()")

    def get_metric(self) -> str:
        return "logit_diff"

    def get_baselines(self) -> dict:
        return {}


class HeadCircuitTask(CircuitTask):
    _circuit_module = None
    _prompt_fn = None

    def get_circuit(self) -> CircuitSpec:
        m = self._circuit_module
        return CircuitSpec(
            roles=m.ROLES,
            bands=m.BANDS,
            pathways=m.PATHWAYS,
            source=self.source,
            model_family=self.model_family,
            paper_ref=self.paper_ref,
        )

    def get_prompts(self, tokenizer: TokenizerLike, n_prompts: int = 40, seed: int = 42) -> list[TaskPrompt]:
        return self._prompt_fn(tokenizer, n_prompts=n_prompts, seed=seed)
