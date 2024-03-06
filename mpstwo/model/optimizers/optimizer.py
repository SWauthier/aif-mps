from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mpstwo.model import MPSTwo


class Optimizer(ABC):
    def __init__(self, model: MPSTwo, lr: float) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.two_site: bool

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def state_dict(self) -> dict:
        return {"lr": self.lr}

    def load_state_dict(self, state_dict: dict) -> None:
        for k, v in state_dict.items():
            setattr(self, k, v)
