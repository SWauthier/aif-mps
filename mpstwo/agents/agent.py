from typing import Protocol, Sequence

import torch

from mpstwo.data.datastructs.tensor_dict import TensorDict


class GymAgent(Protocol):
    def act(self, observation: torch.Tensor | TensorDict) -> tuple[torch.Tensor, dict]:
        ...

    def reset(self) -> None:
        ...

    def action_size(self) -> Sequence[int] | None:
        ...


class MPSAgent(Protocol):
    def act(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ...

    def reset(self) -> None:
        ...
