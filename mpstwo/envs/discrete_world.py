from typing import Any, Optional, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class DiscreteWorld(gym.Env):
    def __init__(
        self,
        length: int = 11,
        init_pos: int = 0,
        init_vel: Optional[int] = None,
        cyclical: bool = False,
    ):
        self.length = length
        self.init_pos = init_pos
        self.init_vel = init_vel
        self.cyclical = cyclical

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Discrete(self.length)

        self.reset()

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        if self.init_vel is not None:
            self.vel += action.item()
        else:
            self.vel = action.item()
        self.pos += self.vel
        if self.cyclical:
            self.pos %= self.length
        elif self.pos >= self.length - 1:
            self.pos = self.length - 1
            self.vel = 0
        elif self.pos < 1:
            self.pos = 0
            self.vel = 0
        return np.array([self.pos]), 0, False, False, {}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.pos = self.init_pos % self.length if self.cyclical else self.init_pos
        self.vel = self.init_vel if self.init_vel is not None else 0
        return np.array([self.pos]), {}
