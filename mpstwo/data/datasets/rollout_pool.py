import logging
import time
from typing import Callable, Optional

import numpy as np
import torch

from mpstwo.agents import GymAgent
from mpstwo.data.datasets import Dataset
from mpstwo.data.datastructs import TensorDict, stack
from mpstwo.envs import EnvWrapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RolloutPool(Dataset):
    def __init__(
        self,
        env: EnvWrapper,
        agent: GymAgent,
        sequence_length: int = 100,
        epoch_size: int = 1000,
        show: bool = False,
        num_workers: int = 0,
        transform: Optional[Callable] = None,
    ) -> None:
        """RolloutPool does rollouts on-the-fly to fill mini_batches

        Arguments:
            env -- environment
            agent -- agent

        Keyword Arguments:
            sequence_length -- length of sequences (default: {100})
            epoch_size -- artificial length of data set (default: {1000})
            show -- whether to render sequences (default: {False})
            num_workers -- number of workers (default: {0})
            transform -- transformation on data set (default: {None})
        """
        self.env = env
        self.agent = agent
        self.sequence_length = sequence_length
        self.epoch_size = epoch_size
        self.show = show
        Dataset.__init__(self, transform, num_workers)

    def __len__(self) -> int:
        return self.epoch_size

    def _get_item(self, idx: int) -> TensorDict:
        # generate a new seed for each sequence
        seed = self._seed()
        torch.manual_seed(seed)
        np.random.seed(seed)
        # do a rollout
        sequence = do_rollout(
            self.env,
            self.agent,
            max_sequence_length=self.sequence_length,
            show=self.show,
        )
        for key, value in sequence.items():
            value.detach()
            sequence[key] = value.squeeze(0)

        return TensorDict(sequence)

    def _seed(self) -> int:
        t = time.time()
        diff = t - int(t)
        return int(diff * 100000)


def do_rollout(
    env: EnvWrapper,
    agent: GymAgent,
    max_sequence_length: int = 100,
    rollouts: int = 1,
    start_states: Optional[torch.Tensor] = None,
    break_on_done: bool = False,
    render: bool = False,
    show: bool = False,
) -> TensorDict:
    """
    Gets a single experience sequence of length `max_sequence_length`
    by doing a role-out with the agent and environment, the
    agent and environment should be compatible
    Arguments:
    `env`                 - the environment, step function should return a Dict
    `agent`               - an agent compatible with the environment
    `max_sequence_length` - the max length of a sequence, default 100
    `rollouts`            - the number of rollouts to do
    `start_states`        - a tensor with first dim = rollouts if the env supports
                            supplying an initial state with the reset method
    `render`              - whether or not to add rgb rendering to the state
    `show`                - show the rollouts
    """

    if start_states is not None:
        # check if rollouts is not set together with start_states ...
        if rollouts != 1 and start_states.shape[0] != rollouts:
            print(
                "WARNING, first dimension of start states does not match rollouts arg!"
            )

        rollouts = start_states.shape[0]

    sequences = []
    for i in range(rollouts):
        sequence = []

        agent.reset()
        # first step only contains observation from env.reset()
        if start_states is None:
            step = env.reset()
        else:
            step = env.reset(start_states[i, ...])

        # initialize with zero action and reward
        step.action = torch.zeros(agent.action_size()).unsqueeze(0)  # type: ignore
        step.reward = torch.tensor([0.0]).unsqueeze(0)
        action, state = agent.act(step)
        step.update(state)
        if render:
            # try to store the pixel buffer if applicable
            pix_buff = env.render(mode="rgb_array")
            if pix_buff is not None:
                pix_buff = (
                    pix_buff.copy().swapaxes(2, 0).swapaxes(1, 2)
                )  # HWC -> CHW and remove neg.strides
                pix_buff = pix_buff / 255  # normalize
                pix_buff = torch.as_tensor(pix_buff).float().unsqueeze(0)
                step.update({"pixels": pix_buff})
        if show:
            env.render(mode="human")
        sequence.append(TensorDict(step))

        for _ in range(max_sequence_length - 1):
            step = env.step(action)
            action, state = agent.act(step)
            step.update(state)
            if render:
                pix_buff = env.render(mode="rgb_array")
                if pix_buff is not None:
                    pix_buff = (
                        pix_buff.copy().swapaxes(2, 0).swapaxes(1, 2)
                    )  # HWC -> CHW and remove neg.strides
                    pix_buff = pix_buff / 255  # normalize
                    pix_buff = torch.as_tensor(pix_buff).float().unsqueeze(0)
                    step.update({"pixels": pix_buff})
            if show:
                env.render(mode="human")
            sequence.append(TensorDict(step))

            if break_on_done and env.done:
                break

        rollout = stack(*sequence).unsqueeze(0)

        # and merge those into a sequences batch
        if rollouts == 1:
            # early stop when we only need 1 rollout
            return rollout

        sequences.append(rollout)

    return stack(*sequences)
