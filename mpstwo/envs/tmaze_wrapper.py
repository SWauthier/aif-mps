from typing import Any, Optional, Sequence, SupportsFloat

import numpy as np
from gym import spaces
from pymdp.envs import TMazeEnv


class TMaze(TMazeEnv):
    def __init__(self, reward_probs: Optional[Sequence] = None):
        super().__init__(reward_probs)

        self.action_space = spaces.MultiDiscrete([4, 1])
        self.observation_space = spaces.MultiDiscrete(self.num_obs)

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        actions = action.copy()
        obs = self._get_observation()
        if obs[0] in [1, 2]:
            actions[0] = obs[0]
        obs = super().step(actions)
        return np.array(obs), 0, False, False, {}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        # seeding not currently supported in pymdp
        state = None
        if options is not None:
            state = options.get("state")
        obs = super().reset(state)
        return np.array(obs), {}
