from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class VLMAdapter(ABC):
    """Common adapter contract. Reported metrics must call :meth:`generate`."""

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def prepare_inputs(self, image_path: str | Path, question: str, answer: str | None = None) -> dict[str, Any]: ...

    @abstractmethod
    def generate(self, image_path: str | Path, question: str, generation_config: dict[str, Any]) -> str: ...

    @abstractmethod
    def teacher_forced_loss(self, image_path: str | Path, question: str, target: str): ...

    @abstractmethod
    def get_named_modules(self) -> dict[str, Any]: ...

    @abstractmethod
    def get_editable_mlp_modules(self) -> dict[str, Any]: ...

    @abstractmethod
    def clone_or_reload_base(self) -> None: ...
