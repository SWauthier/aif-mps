from typing import Any, Optional, SupportsFloat

import numpy as np
from gymnasium.envs.toy_text.frozen_lake import DOWN, LEFT, RIGHT, UP, FrozenLakeEnv
from gymnasium.spaces import flatdim


class FrozenLake(FrozenLakeEnv):
    def __init__(
        self,
        render_mode: Optional[str] = None,
        desc=None,
        map_name="4x4",
        is_slippery=True,
        how_slippery=None,
    ):
        super().__init__(render_mode, desc, map_name, is_slippery)
        if is_slippery and how_slippery is not None:
            nA = flatdim(self.action_space)
            nS = flatdim(self.observation_space)
            if isinstance(how_slippery, list):
                rates = how_slippery
            else:
                rates = [(1 - how_slippery) / 2, how_slippery, (1 - how_slippery) / 2]

            def to_s(row, col):
                return row * self.ncol + col

            def inc(row, col, a):
                if a == LEFT:
                    col = max(col - 1, 0)
                elif a == DOWN:
                    row = min(row + 1, self.nrow - 1)
                elif a == RIGHT:
                    col = min(col + 1, self.ncol - 1)
                elif a == UP:
                    row = max(row - 1, 0)
                return (row, col)

            self.P = {s: {a: [] for a in range(nA)} for s in range(nS)}

            for row in range(self.nrow):
                for col in range(self.ncol):
                    s = to_s(row, col)
                    for a in range(4):
                        li = self.P[s][a]
                        letter = self.desc[row, col]
                        if letter in b"GH":
                            li.append((1.0, s, 0, True))
                        else:
                            for b, rate in zip([(a - 1) % 4, a, (a + 1) % 4], rates):
                                newrow, newcol = inc(row, col, b)
                                newstate = to_s(newrow, newcol)
                                newletter = self.desc[newrow, newcol]
                                done = bytes(newletter) in b"GH"
                                rew = float(newletter == b"G")
                                li.append((rate, newstate, rew, done))

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, rew, term, trunc, info = super().step(action.item())
        return np.array(obs), rew, term, trunc, info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        return np.array(obs), info
