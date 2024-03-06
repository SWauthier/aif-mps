"""
    Modified version of the Mountain Car environment
    with only a noisy position observation
    
    Used in https://www.frontiersin.org/articles/10.3389/fncom.2020.574372/full
"""
from typing import Any, SupportsFloat

import numpy as np
from gymnasium import spaces
from gymnasium.envs.classic_control.continuous_mountain_car import (
    Continuous_MountainCarEnv,
)

_TYPES = {"position": slice(0, 1), "velocity": slice(1, 2), "both": slice(0, 2)}


class MountainCar(Continuous_MountainCarEnv):
    """Our own adaptation of the OpenAI Mountain Car Continuous environment with
    - only position in observation (no velocity information)
    - configurable gaussian noise on the observation
    """

    def __init__(
        self,
        noise: float = 0.0,
        init_velocity: float = 0.0,
        discrete=False,
        obs_type: str = "position",
    ):
        """Initialize the environment

        Arguments:
            noise           -- the amount of extra additive white noise on the
                              observations
            init_velocity   -- the initial velocity of the car

        Returns:
            self
        """
        self.noise = noise
        self.init_velocity = init_velocity
        self.type_slice = _TYPES.get(obs_type, slice(0, 1))
        super().__init__()

        if discrete:
            self.discrete = True
            self.action_space = spaces.Discrete(3)
            self.actions = (-1, 0, 1)
        else:
            self.discrete = False
            self.action_space = spaces.Box(
                low=self.min_action, high=self.max_action, shape=(1,)
            )
        self.observation_space = spaces.Box(
            low=self.low_state[self.type_slice], high=self.high_state[self.type_slice]
        )

        self.step_count = 0

        self.reset()

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        """Do one step in the environment

        Arguments:
            action -- Tensor of shape (batch_size, action_length)

        Returns:
            Dict -- With keys 'position', 'ground_truth',
                    'action', 'reward'.
        """
        if self.discrete:
            selected_action = self.actions[action.item()]
        else:
            selected_action = action.item()

        state, _, terminated, _, _ = super().step(np.array(selected_action))
        position = state[0]

        reward = 1 if terminated else 0

        over_position = position >= self.max_position
        under_position = position <= self.min_position
        self.step_count += 1
        too_many_steps = self.step_count >= 200
        truncated = over_position or under_position or too_many_steps

        return self._get_observation(), reward, terminated, truncated, {}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment to an optional intial position,
        if no position is specified the car will spawn at a random
        position.

        Keyword Arguments:
            init_position -- optional initial_position (default: None)

        Returns:
            Dict -- With keys 'position' and 'ground_truth'.
        """
        super(Continuous_MountainCarEnv, self).reset(seed=seed)
        init_position = None
        if options is not None:
            init_position = options.get("init_position", None)

        if init_position is None:
            state, _ = super().reset(options=options)
            init_position = state[0]
        elif not isinstance(init_position, float):
            init_position = init_position.item()

        if self.init_velocity is None:
            direction = self.np_random.choice([-1, 1])
            vel = self.np_random.uniform(low=0, high=self.max_speed)
            init_velocity = direction * vel
        else:
            init_velocity = self.init_velocity

        self.state = np.array([init_position, init_velocity])

        self.step_count = 0
        return self._get_observation(), {}

    def _get_observation(self) -> np.ndarray:
        obs = self.state[self.type_slice] + self.np_random.normal(0, self.noise)
        return obs

    def get_ground_truth(self) -> np.ndarray:
        return self.state[self.type_slice]
