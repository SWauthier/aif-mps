from collections import deque
from typing import Callable, Optional

import torch

from mpstwo.data.datastructs import TensorDict, cat
from mpstwo.model.mpstwo import MPSTwo
from mpstwo.utils.distributions import Q_obs, entropy, kl_divergence, likelihood
from mpstwo.utils.mapping import Map


class DiscreteFreeEnergyAgent:
    def __init__(
        self,
        model: MPSTwo,
        *,
        act_map: Map,
        obs_map: Map,
        pref_dist: Optional[torch.Tensor] = None,
        horizon: int,
        imagine_future: Optional[str] = None,
        recursion: bool = True,
    ) -> None:
        model.right_canonical()
        self._model = model
        self._act_dim = self._model.feature_dim_act
        self._obs_dim = self._model.feature_dim_obs
        self._device = self._model.device
        self._act_map = act_map
        self._obs_map = obs_map
        self._pref_dist = pref_dist
        self._horizon = horizon
        self._imagine_future = imagine_future if imagine_future is not None else ""
        self.recursion = recursion
        self.reset()

    def reset(self) -> None:
        self._actions = torch.zeros((1, 1, 1), device=self._device)
        self._observations = torch.tensor((), device=self._device)
        self._g = torch.tensor((), device=self._device)
        self._i = 0

    def act(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Take an action based on an observation

        Arguments:
        observation -- tensor of the form (batches=1, time steps=1, variable=1)
        """
        self._observations = torch.cat((self._observations, observation), dim=1)
        sequence = TensorDict(
            {
                "action": self._actions,
                "observation": self._observations,
            }
        )
        horizon = self._horizon
        if hasattr(self._model, "physical_legs"):
            horizon = min(horizon, self._model.physical_legs - self._i - 1)

        if self.recursion:
            g = self.expected_free_energy_rec(sequence, horizon)
        else:
            g = self.expected_free_energy_iter(sequence, horizon)
        self._g = torch.cat((self._g, g.unsqueeze(1)), dim=1)
        new_action = torch.argmin(g).view(1, -1, 1)
        self._actions = torch.cat((self._actions, new_action), dim=1)
        self._i += 1
        return self._actions[:, -1, :], self._g[:, -1, :]

    def expected_free_energy_next_action(self, sequence: TensorDict) -> torch.Tensor:
        """Expected free energy of next action

        E_Q(o, s | pi)[log Q(o | pi) - log P(o) - log Q(o | s, pi)]

        E_Q(o, s | pi)[log Q(o | pi) - log P(o)
            = sum_o,s Q(o, s | pi) (log Q(o | pi) - log P(o))
            = sum_o Q(o | pi) (log Q(o | pi) - log P(o))
            = D_KL(Q(o | pi) || P(o))

        -E_Q(o, s | pi)[log Q(o | s, pi)]
            = -sum _o,s Q(o, s | pi) log Q(o | s, pi)
            = -sum _s Q(s | pi) (sum _o Q(o | s, pi) log Q(o | s, pi))
            = -sum _s Q(s | pi) H(Q(o | s, pi))

        Q(s | pi) = 1 for t + 1, since there is only 1 state
        """
        q_obs = eval_func(
            Q_obs,
            self._model,
            sequence,
            act_map=self._act_map,
            obs_map=self._obs_map,
        ).abs()
        lik = eval_func(
            likelihood,
            self._model,
            sequence,
            act_map=self._act_map,
            obs_map=self._obs_map,
        ).abs()
        kl = kl_divergence(q_obs, self._pref_dist) if self._pref_dist is not None else 0
        h = entropy(lik)
        return kl + h

    ##########################################################################
    # RECURSIVE SOLUTION
    ##########################################################################

    def expected_free_energy_rec(
        self, sequence: TensorDict, horizon: int
    ) -> torch.Tensor:
        """Expected free energy in sophisticated inference

        Returns a vector G_a
        """
        g_u1 = self.expected_free_energy_next_action(sequence)
        if horizon > 1:
            exp_g_u2 = self.expected_free_energy_subsequent_actions(sequence, horizon)
            g_u1 += exp_g_u2
        return g_u1

    def expected_free_energy_subsequent_actions(
        self, sequence: TensorDict, horizon: int
    ) -> torch.Tensor:
        """Expected free energy of subsequent actions

        E_Q(o | pi) Q(pi_+1 | o)[G(o, pi_+1)]
        """
        q_o1_u1 = eval_func(
            Q_obs,
            self._model,
            sequence,
            act_map=self._act_map,
            obs_map=self._obs_map,
        ).abs()

        if self._imagine_future == "classic":
            # classic (for-loop)
            exp_g_u2 = torch.zeros(self._act_dim, device=self._device)
            for u1 in range(self._act_dim):
                for o1 in range(self._obs_dim):
                    act = torch.tensor([[[u1]]], device=self._device)
                    obs = torch.tensor([[[o1]]], device=self._device)
                    imagined_sequence = TensorDict(
                        {
                            "action": act,
                            "observation": obs,
                        }
                    )
                    imagined_sequence = cat(sequence, imagined_sequence)
                    g_u2 = self.expected_free_energy_rec(imagined_sequence, horizon - 1)
                    q_u2_o1 = torch.softmax(-g_u2, dim=-1)
                    exp_g_u2[u1] += q_o1_u1[u1, o1] * torch.dot(q_u2_o1, g_u2)

        elif self._imagine_future == "full":
            # large matrix
            step = self._horizon - horizon
            if hasattr(self._model, "physical_legs"):
                step = min(step, self._model.physical_legs - self._i - 1 - horizon)
            base_seq = sequence[:, : sequence.shape[1] - step]

            actions = torch.cartesian_prod(
                *[torch.arange(self._act_dim, device=self._device)] * (step + 1)
            ).view(-1, step + 1, 1)
            observations = torch.cartesian_prod(
                *[torch.arange(self._obs_dim, device=self._device)] * (step + 1)
            ).view(-1, step + 1, 1)
            act = torch.cat(
                (
                    base_seq["action"].repeat_interleave(
                        self._act_dim ** (step + 1), dim=0
                    ),
                    actions.tile(base_seq["action"].shape[0], 1, 1),
                ),
                dim=1,
            )
            obs = torch.cat(
                (
                    base_seq["observation"].repeat_interleave(
                        self._obs_dim ** (step + 1), dim=0
                    ),
                    observations.repeat(base_seq["observation"].shape[0], 1, 1),
                ),
                dim=1,
            )
            imagined_sequence = TensorDict(
                {
                    "action": act.repeat_interleave(obs.shape[0], dim=0),
                    "observation": obs.repeat(act.shape[0], 1, 1),
                }
            )
            g_u2 = self.expected_free_energy_rec(imagined_sequence, horizon - 1)
            q_u2_o1 = torch.softmax(-g_u2, dim=-1)
            exp_g_u2 = torch.sum(
                q_o1_u1
                * torch.sum(q_u2_o1 * g_u2, dim=-1).view(
                    -1, self._act_dim, self._obs_dim
                ),
                dim=-1,
            )

        else:
            # large matrix with pruning
            larger = q_o1_u1 >= 0.01
            next_step = larger.nonzero()
            b, act, obs = next_step.split(1, dim=1)
            repeats = torch.bincount(b.flatten())
            imagined_sequence = sequence.repeat_interleave(repeats, dim=0)
            next_sequence = TensorDict(
                {"action": act.view(-1, 1, 1), "observation": obs.view(-1, 1, 1)}
            )
            imagined_sequence = cat(imagined_sequence, next_sequence)

            g_u2_int = self.expected_free_energy_rec(imagined_sequence, horizon - 1)
            g_u2 = torch.full(
                (larger.numel(), g_u2_int.shape[-1]),
                g_u2_int.max().item(),
                device=self._device,
                dtype=g_u2_int.dtype,
            )
            g_u2[larger.view(-1)] = g_u2_int
            q_u2_o1 = torch.softmax(-g_u2, dim=-1)
            exp_g_u2 = torch.sum(
                q_o1_u1
                * torch.sum(q_u2_o1 * g_u2, dim=-1).view(
                    -1, self._act_dim, self._obs_dim
                ),
                dim=-1,
            )

        return exp_g_u2

    ##########################################################################
    # ITERATIVE SOLUTION
    # Only pruning method has been implemented
    ##########################################################################

    def expected_free_energy_iter(
        self, sequence: TensorDict, horizon: int
    ) -> torch.Tensor:
        start = self._create_new_frame(horizon, sequence, None)
        stack = deque([start])
        while len(stack) > 0:
            frame = dict_to(stack.pop(), self._device)
            if self._has_next_child(frame):
                child = self._get_next_child(frame)
                stack.append(dict_to(frame, "cpu"))
                stack.append(dict_to(child, "cpu"))
            else:
                return_value = self._get_return_value(frame)
                if frame["parent"] is not None:
                    self._pass_to_parent(frame, return_value)
        return self._get_return_value(start)

    @staticmethod
    def _create_new_frame(
        horizon: int, sequence: TensorDict, parent: Optional[dict]
    ) -> dict:
        # create an empty frame
        frame = {
            "horizon": None,
            "parent": None,  # the parent frame
            "return_value": None,  # the return value
            "local": {
                "sequence": None,
                "q_o1_u1": None,
                "larger": None,
            },  # the local variables
        }

        # fill in the fields
        frame["horizon"] = horizon
        frame["local"]["sequence"] = sequence
        frame["parent"] = parent
        return frame

    @staticmethod
    def _has_next_child(frame: dict) -> bool:
        return frame["horizon"] > 1 and frame["return_value"] is None

    def _get_next_child(self, frame: dict) -> dict:
        imagined_sequence, local = self.imagined_sequence(frame["local"]["sequence"])
        frame["local"].update(local)
        child = self._create_new_frame(frame["horizon"] - 1, imagined_sequence, frame)
        return child

    def _get_return_value(self, frame: dict) -> torch.Tensor:
        efe = self.expected_free_energy_next_action(frame["local"]["sequence"])
        if frame["horizon"] > 1:
            g_u2 = torch.full(
                (frame["local"]["larger"].numel(), frame["return_value"].shape[-1]),
                frame["return_value"].max().item(),
                device=self._device,
                dtype=frame["return_value"].dtype,
            )
            g_u2[frame["local"]["larger"].view(-1)] = frame["return_value"]
            q_u2_o1 = torch.softmax(-g_u2, dim=-1)
            exp_g_u2 = torch.sum(
                frame["local"]["q_o1_u1"]
                * torch.sum(q_u2_o1 * g_u2, dim=-1).view(
                    -1, self._act_dim, self._obs_dim
                ),
                dim=-1,
            )
            return efe + exp_g_u2
        return efe

    @staticmethod
    def _pass_to_parent(frame: dict, return_value: torch.Tensor) -> None:
        frame["parent"]["return_value"] = return_value

    def imagined_sequence(self, sequence: TensorDict) -> tuple[TensorDict, dict]:
        q_o1_u1 = eval_func(
            Q_obs,
            self._model,
            sequence,
            act_map=self._act_map,
            obs_map=self._obs_map,
        ).abs()
        larger = q_o1_u1 >= 0.01
        next_step = larger.nonzero()
        b, act, obs = next_step.split(1, dim=1)
        repeats = torch.bincount(b.flatten())
        imagined_sequence = sequence.repeat_interleave(repeats, dim=0)
        next_sequence = TensorDict(
            {"action": act.view(-1, 1, 1), "observation": obs.view(-1, 1, 1)}
        )
        imagined_sequence = cat(imagined_sequence, next_sequence)
        local = {"q_o1_u1": q_o1_u1, "larger": larger}
        return imagined_sequence, local


def eval_func(
    func: Callable,
    model: MPSTwo,
    sequence: TensorDict,
    act_map: Map,
    obs_map: Map,
    **kwargs,
) -> torch.Tensor:
    obs = obs_map(sequence["observation"]).to(model.dtype).to(model.device)
    act = act_map(sequence["action"]).to(model.dtype).to(model.device)

    if kwargs is None:
        return func(model, obs, act, canonical="right")
    else:
        return func(model, obs, act, canonical="right", **kwargs)


def dict_to(d: dict, other: str | torch.device | torch.dtype | torch.Tensor):
    for k, v in d.items():
        if k == "local":
            for m, w in v.items():
                if isinstance(w, (torch.Tensor, TensorDict)):
                    d["local"][m] = w.to(other)
        else:
            if isinstance(v, (torch.Tensor, TensorDict)):
                d[k] = v.to(other)
    return d
