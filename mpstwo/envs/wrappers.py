from abc import ABC, abstractmethod
from typing import Any, Optional

import gymnasium as gym
import numpy as np
import torch

from mpstwo.data.datastructs import Dict, TensorDict
from mpstwo.envs.env import Env


class EnvWrapper(ABC):
    def __init__(self, wrappee: Env) -> None:
        self.__dict__["wrappee"] = wrappee

    def __getattr__(self, name: str) -> Any:
        return getattr(self.wrappee, name)

    def __setattr__(self, __name: str, __value: Any) -> None:
        if __name not in self.__dict__ and hasattr(self.wrappee, __name):
            setattr(self.wrappee, __name, __value)
        else:
            self.__dict__[__name] = __value

    @abstractmethod
    def step(self, action: torch.Tensor) -> TensorDict:
        pass

    @abstractmethod
    def reset(self, *args: Any, **kwargs: Any) -> TensorDict:
        pass

    def get_action_space(self) -> gym.Space:
        return self.wrappee.action_space

    def get_observation_space(self) -> gym.Space:
        return self.wrappee.observation_space


class DictPacker(EnvWrapper):
    def __init__(self, wrappee: Env, obs_keys: Optional[list[str]] = None) -> None:
        EnvWrapper.__init__(self, wrappee)
        if obs_keys is None:
            obs_keys = ["observation"]
        self.obs_keys = obs_keys

    def step(self, action: torch.Tensor) -> TensorDict:
        """Action should be a tensor"""
        act = action.squeeze(0).detach().cpu().numpy()
        obs, reward, terminated, truncated, _ = self.wrappee.step(act)
        obs = DictPacker.__parse_obs(obs)
        reward = torch.tensor([reward], dtype=torch.float32).unsqueeze(0)
        terminated = torch.tensor([terminated], dtype=torch.float32).unsqueeze(0)
        truncated = torch.tensor([truncated], dtype=torch.float32).unsqueeze(0)
        t = TensorDict(
            {
                self.obs_keys[0]: obs,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "action": action,
            }
        )
        try:
            ground_truth = self.wrappee.get_ground_truth()
            t["ground_truth"] = DictPacker.__parse_obs(ground_truth)
        except AttributeError:
            pass
        return t

    def get_observation_space(self) -> dict[str, gym.Space]:
        return Dict({self.obs_keys[0]: self.wrappee.observation_space})

    @staticmethod
    def __parse_obs(obs) -> torch.Tensor:
        if obs.dtype == np.uint8 and hasattr(obs, "shape"):
            # is an array and is uint8 -> probably an image
            obs = obs.copy().swapaxes(2, 0)
            obs = torch.from_numpy(obs).float().unsqueeze(0)
            obs /= 255.0
        elif hasattr(obs, "shape"):
            # is just an array
            obs = torch.from_numpy(obs).float().unsqueeze(0)
        else:
            obs = torch.tensor([obs], dtype=torch.float32).unsqueeze(0)
        return obs

    def reset(self, *args: Any, **kwargs: Any) -> TensorDict:
        obs, _ = self.wrappee.reset(*args, **kwargs)
        obs = DictPacker.__parse_obs(obs)
        t = TensorDict({self.obs_keys[0]: obs})
        try:
            ground_truth = self.wrappee.get_ground_truth()
            t["ground_truth"] = DictPacker.__parse_obs(ground_truth)
        except AttributeError:
            pass
        return t
