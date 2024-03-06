import torch

from mpstwo.agents.discrete_surprisal_agent import DiscreteSurprisalAgentNoTree, eval_func
from mpstwo.data.datastructs.tensor_dict import TensorDict, cat
from mpstwo.model.mpstwo import MPSTwo
from mpstwo.utils.distributions import Q_obs, cross_entropy
from mpstwo.utils.mapping import MultiOneHotMap


class MultiDiscreteSurprisalAgent(DiscreteSurprisalAgentNoTree):
    def __init__(
        self,
        model: MPSTwo,
        *,
        act_map: MultiOneHotMap,
        obs_map: MultiOneHotMap,
        pref_dist: list[torch.Tensor],
        horizon: int,
        imagine_future: str = "full"
    ) -> None:
        if pref_dist is not None:
            assert len(obs_map.nums) == len(pref_dist)
        super().__init__(
            model,
            act_map=act_map,
            obs_map=obs_map,
            pref_dist=pref_dist,  # type: ignore
            horizon=horizon,
        )
        self._nums = obs_map.nums
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

        g = self.expected_surprise(sequence, horizon)
        self._g = torch.cat((self._g, g.unsqueeze(1)), dim=1)
        new_action = self._act_map.invert_argmax(torch.argmin(g, dim=-1).unsqueeze(-1))  # type: ignore
        self._actions = torch.cat((self._actions, new_action), dim=1)
        self._i += 1
        return self._actions[:, -1, :], self._g[:, -1, :]

    def surprise_next_action(self, sequence: TensorDict) -> torch.Tensor:
        """Expected surprise of next action

        E_Q(o | pi)[- log P(o)]
        """
        q_obs = eval_func(
            Q_obs,
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
            s = cross_entropy(q_, self._pref_dist[i])
            f.append(s)
        return torch.stack(f, dim=0).sum(dim=0)

    def surprise_subsequent_actions(
        self, sequence: TensorDict, horizon: int
    ) -> torch.Tensor:
        """Expected free energy of subsequent actions

        E_Q(o | pi) Q(pi_+1 | o)[G(o, pi_+1)]
        """
        q_obs = eval_func(
            Q_obs,
            self._model,
            sequence,
            act_map=self._act_map,
            obs_map=self._obs_map,
        ).abs()
        q__ = []
        for i, _ in enumerate(self._nums):
            sum_dims = tuple(
                q_obs.dim() - 1 + j for j, _ in enumerate(self._nums) if j != i
            )
            q__.append(
                q_obs.reshape(q_obs.shape[:-1] + tuple(self._nums)).sum(sum_dims)
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
            g_u2 = self.expected_surprise(imagined_sequence, horizon - 1)
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
                    g_u2 = self.expected_surprise(imagined_sequence, horizon - 1)
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
