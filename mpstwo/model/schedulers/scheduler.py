from abc import ABC, abstractmethod
from typing import Any

from mpstwo.model.optimizers import Optimizer


class Scheduler(ABC):
    def __init__(self, optimizer: Optimizer) -> None:
        super().__init__()
        self.optimizer = optimizer

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.step(*args, **kwargs)

    @abstractmethod
    def step(self, *args, **kwargs) -> Any:
        ...

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state_dict: dict) -> None:
        for k, v in state_dict.items():
            setattr(self, k, v)
