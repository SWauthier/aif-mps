from typing import Optional

import torch

from mpstwo.agents.discrete_fe_agent import DiscreteFreeEnergyAgent, eval_func
from mpstwo.data.datastructs.tensor_dict import TensorDict, cat
from mpstwo.model.mpstwo import MPSTwo
from mpstwo.utils.distributions import Q_obs, entropy, kl_divergence, likelihood
from mpstwo.utils.mapping import MultiOneHotMap


class MultiDiscreteFreeEnergyAgent(DiscreteFreeEnergyAgent):
    def __init__(
        self,
        model: MPSTwo,
        *,
        act_map: MultiOneHotMap,
        obs_map: MultiOneHotMap,
        pref_dist: Optional[list[torch.Tensor]],
        horizon: int,
        imagine_future: str = "full"
    ) -> None:
        super().__init__(
            model,
            act_map=act_map,
            obs_map=obs_map,
            pref_dist=None,
            horizon=horizon,
        )
        if pref_dist is not None:
            assert len(obs_map.nums) == len(pref_dist)
        self._nums = obs_map.nums
        self._pref_dist = pref_dist
        self._imagine_future = imagine_future

    def reset(self) -> None:
        self._actions = torch.zeros(
            (1, 1, len(self._act_map.nums)), device=self._device  # type: ignore
        )
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
        if hasattr(self._model, "physical_legs"):
            horizon = min(self._horizon, self._model.physical_legs - self._i - 1)
        else:
            horizon = self._horizon

        g = self.expected_free_energy_rec(sequence, horizon)
        self._g = torch.cat((self._g, g.unsqueeze(1)), dim=1)
        new_action = self._act_map.invert_argmax(torch.argmin(g, dim=-1).unsqueeze(-1))  # type: ignore
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
        f = []
        for i, _ in enumerate(self._nums):
            sum_dims = tuple(
                q_obs.dim() - 1 + j for j, _ in enumerate(self._nums) if j != i
            )
            q_ = q_obs.reshape(q_obs.shape[:-1] + tuple(self._nums)).sum(sum_dims)
            kl = (
                kl_divergence(q_, self._pref_dist[i])
                if self._pref_dist is not None
                else torch.zeros_like(q_).sum(-1)
            )
            f.append(kl)
            lik_ = lik.reshape(lik.shape[:-1] + tuple(self._nums)).sum(sum_dims)
            h = entropy(lik_)
            f.append(h)
        return torch.stack(f, dim=0).sum(dim=0)

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
        q__ = []
        for i, _ in enumerate(self._nums):
            sum_dims = tuple(
                q_o1_u1.dim() - 1 + j for j, _ in enumerate(self._nums) if j != i
            )
            q__.append(
                q_o1_u1.reshape(q_o1_u1.shape[:-1] + tuple(self._nums)).sum(sum_dims)
            )

        if self._imagine_future == "full":
            if hasattr(self._model, "physical_legs"):
                step = min(
                    self._horizon - horizon,
                    self._model.physical_legs - self._i - 1 - horizon,
                )
            else:
                step = self._horizon - horizon
            actions = torch.cartesian_prod(
                *[torch.arange(self._act_dim, device=self._device)] * (step + 1)
            ).view(-1, step + 1)
            actions = self._act_map.invert_argmax(actions)  # type: ignore
            observations = torch.cartesian_prod(
                *[torch.arange(self._obs_dim, device=self._device)] * (step + 1)
            ).view(-1, step + 1)
            observations = self._obs_map.invert_argmax(observations)  # type: ignore
            idx = sequence.shape[1] - step
            base = sequence[0, :idx].unsqueeze(0)
            act = torch.cat(
                (
                    base["action"].repeat_interleave(
                        self._act_dim ** (step + 1), dim=0
                    ),
                    actions.tile(base["action"].shape[0], 1, 1),
                ),
                dim=1,
            )
            obs = torch.cat(
                (
                    base["observation"].repeat_interleave(
                        self._obs_dim ** (step + 1), dim=0
                    ),
                    observations.repeat(base["observation"].shape[0], 1, 1),
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
            sublists = [[..., 0, i + 1] for i, _ in enumerate(self._nums)]
            sublist = [...] + list(range(len(self._nums) + 1))
            exp_g_u2 = torch.einsum(
                *[item for t in zip(q__, sublists) for item in t],
                torch.sum(q_u2_o1 * g_u2, dim=-1).view(-1, self._act_dim, *self._nums),
                sublist,
                [..., 0]
            )

        else:
            # classic
            exp_g_u2 = torch.zeros(self._act_dim, device=self._device)
            for u1 in range(self._act_dim):
                for o1 in range(self._obs_dim):
                    act = self._act_map.invert_argmax(  # type: ignore
                        torch.tensor([[u1]], device=self._device)
                    )
                    obs = self._obs_map.invert_argmax(  # type: ignore
                        torch.tensor([[o1]], device=self._device)
                    )
                    imagined_sequence = TensorDict(
                        {
                            "action": act,
                            "observation": obs,
                        }
                    )
                    imagined_sequence = cat(sequence, imagined_sequence)
                    g_u2 = self.expected_free_energy_rec(imagined_sequence, horizon - 1)
                    q_u2_o1 = torch.softmax(-g_u2, dim=-1)
                    q_o_u = torch.prod(
                        torch.cat(
                            [
                                q[..., u1, int(obs[..., i].item())]
                                for i, q in enumerate(q__)
                            ]
                        )
                    )
                    exp_g_u2[u1] += q_o_u * torch.sum(q_u2_o1 * g_u2)
        return exp_g_u2
