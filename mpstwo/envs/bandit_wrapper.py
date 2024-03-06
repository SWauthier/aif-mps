from typing import Any, SupportsFloat

import numpy as np
from gym_bandits.bandit import BanditEnv
from gymnasium.utils import seeding


class Bandit(BanditEnv):
    _np_random: np.random.Generator | None = None

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, rew, done, _ = super().step(action)
        return np.array(obs), rew, done, False, {}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)
        obs = np.array(super().reset())
        return obs, {}

    @property
    def np_random(self) -> np.random.Generator:
        """Returns the environment's internal :attr:`_np_random`

        If not set, will initialise with a random seed.

        Returns:
            Instances of `np.random.Generator`
        """
        if self._np_random is None:
            self._np_random, seed = seeding.np_random()
        return self._np_random

    @np_random.setter
    def np_random(self, value: np.random.Generator):
        self._np_random = value
