from typing import Optional

import gymnasium as gym
import torch

from mpstwo.data.datastructs.tensor_dict import TensorDict


class RandomAgent:
    def __init__(
        self,
        action_space: gym.Space,
        repeat_p: float = -1.0,
        options: Optional[torch.Tensor] = None,
    ) -> None:
        self.action_space = action_space
        self.repeat_p = repeat_p
        self.options = options
        self.num_actions = action_space.shape
        self.last_action = None

    def action_size(self) -> tuple[int, ...] | None:
        return self.num_actions

    def act(self, observation: torch.Tensor | TensorDict) -> tuple[torch.Tensor, dict]:
        batch_size = observation.shape[0]

        if torch.rand(1) <= self.repeat_p:
            if self.last_action is not None:
                new_action = self.last_action
            else:
                new_action = self.random_action(batch_size)
        else:
            new_action = self.random_action(batch_size)
        self.last_action = new_action
        return new_action, {}

    def random_action(self, batch_size: int) -> torch.Tensor:
        if self.options is not None:
            indices = torch.randint(0, self.options.shape[0], (batch_size,))
            return self.options[indices]
        else:
            actions = [
                torch.tensor(self.action_space.sample()) for _ in range(batch_size)
            ]
            return torch.stack(actions, dim=0)

    def reset(self) -> None:
        self.last_action = None
