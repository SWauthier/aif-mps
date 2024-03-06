from typing import Callable

import torch

from mpstwo.data.datastructs import TensorDict, cat
from mpstwo.model.mpstwo import MPSTwo
from mpstwo.utils.distributions import Q_obs, cross_entropy
from mpstwo.utils.mapping import Map


class DiscreteSurprisalAgent:
    def __init__(
        self,
        model: MPSTwo,
        *,
        act_map: Map,
        obs_map: Map,
        pref_dist: torch.Tensor,
        horizon: int,
        prune_threshold: float = 0.01,
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
        self._prune_threshold = prune_threshold
        self.reset()

    def reset(self) -> None:
        self._actions = torch.zeros((1, 1, 1), device=self._device)
        self._observations = torch.tensor((), device=self._device)
        self._q_obs = []
        self._search_tree = []
        self._imagined_sequence = None
        self._g = torch.tensor((), device=self._device)
        self._i = 0

    def act(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Take an action based on an observation

        Arguments:
        observation -- tensor of the form (batches=1, time steps=1, variable=1)
        """
        self._observations = torch.cat(
            (self._observations, observation.view(1, 1, 1)), dim=1
        )
        sequence = TensorDict(
            {
                "action": self._actions,
                "observation": self._observations,
            }
        )
        horizon = self._horizon
        if hasattr(self._model, "physical_legs"):
            horizon = min(horizon, self._model.physical_legs - self._i - 1)

        if self._imagined_sequence is None:
            self._build_search_tree(sequence, horizon)
        else:
            self._prune_search_tree(sequence)
            if horizon == self._horizon:
                self._add_layer(self._imagined_sequence, 1)

        g = self.expected_surprisal()
        self._g = torch.cat((self._g, g.unsqueeze(1)), dim=1)
        new_action = torch.argmin(g).view(1, -1, 1)
        self._actions = torch.cat((self._actions, new_action), dim=1)
        self._i += 1
        return self._actions[:, -1, :], self._g[:, -1, :]

    def expected_surprisal(self) -> torch.Tensor:
        """Expected surprise in sophisticated formalism

        Returns a vector G_a
        """
        s_u_int = self._search_tree[-1]
        for i in range(len(self._search_tree) - 2, -1, -1):
            larger = self._q_obs[i] >= self._prune_threshold
            s_u = torch.full(
                (larger.numel(), s_u_int.shape[-1]),
                s_u_int.max().item(),
                device=self._device,
                dtype=s_u_int.dtype,
            )
            s_u[larger.view(-1)] = s_u_int
            q_u2_o1 = torch.softmax(-s_u, dim=-1)
            s = torch.sum(
                self._q_obs[i]
                * torch.sum(q_u2_o1 * s_u, dim=-1).view(-1, self._act_dim, self._obs_dim),
                dim=-1,
            )
            s_u_int = self._search_tree[i] + s
        return s_u_int

    def _build_search_tree(self, sequence, horizon) -> None:
        self._surprisal(sequence)
        if horizon > 1:
            self._add_layer(sequence, horizon)

    def _surprisal(self, sequence):
        q_obs = eval_func(
            Q_obs,
            self._model,
            sequence,
            act_map=self._act_map,
            obs_map=self._obs_map,
        ).abs()
        self._q_obs.append(q_obs)

        surprisal = cross_entropy(q_obs, self._pref_dist)
        self._search_tree.append(surprisal)
        return surprisal

    def _add_layer(self, sequence, horizon) -> None:
        larger = self._q_obs[-1] >= self._prune_threshold
        next_step = larger.nonzero()
        b, act, obs = next_step.split(1, dim=1)
        repeats = torch.bincount(b.flatten())
        imagined_sequence = sequence.repeat_interleave(repeats, dim=0)
        next_sequence = TensorDict(
            {"action": act.view(-1, 1, 1), "observation": obs.view(-1, 1, 1)}
        )
        imagined_sequence = cat(imagined_sequence, next_sequence)
        self._imagined_sequence = imagined_sequence

        self._build_search_tree(imagined_sequence, horizon - 1)

    def _prune_search_tree(self, sequence) -> None:
        new_action = sequence["action"][:, -1, :].item()
        new_observation = sequence["observation"][:, -1, :].item()
        masks = []

        larger = self._q_obs[0] >= self._prune_threshold
        next_step = larger.nonzero()
        _, act, obs = next_step.split(1, dim=1)
        mask = torch.logical_and(act.flatten() == new_action, obs.flatten() == new_observation).nonzero(as_tuple=True)
        masks.append(mask)
        
        for i in range(1, len(self._q_obs) - 1):
            larger = self._q_obs[i] >= self._prune_threshold
            next_step = larger.nonzero()
            b, _, _ = next_step.split(1, dim=1)
            mask = torch.isin(b.flatten(), mask[0]).nonzero(as_tuple=True)
            masks.append(mask)

        del self._q_obs[0]
        del self._search_tree[0]

        for i, m in enumerate(masks):
            self._q_obs[i] = self._q_obs[i][m]
            self._search_tree[i] = self._search_tree[i][m]

        if self._imagined_sequence is not None:
            self._imagined_sequence = self._imagined_sequence[mask[-1]]

        if self._search_tree[0].shape[0] == 0:
            horizon = len(self._search_tree)
            self._search_tree.clear()
            self._build_search_tree(sequence, horizon)


class DiscreteSurprisalAgentNoTree:
    def __init__(
        self,
        model: MPSTwo,
        *,
        act_map: Map,
        obs_map: Map,
        pref_dist: torch.Tensor,
        horizon: int,
        prune_threshold: float = 0.01,
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
        self._prune_threshold = prune_threshold
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
        self._observations = torch.cat(
            (self._observations, observation.view(1, 1, 1)), dim=1
        )
        sequence = TensorDict(
            {
                "action": self._actions,
                "observation": self._observations,
            }
        )
        horizon = self._horizon
        if hasattr(self._model, "physical_legs"):
            horizon = min(horizon, self._model.physical_legs - self._i - 1)

        g = self.expected_surprise(sequence, horizon)
        self._g = torch.cat((self._g, g.unsqueeze(1)), dim=1)
        new_action = torch.argmin(g).view(1, -1, 1)
        self._actions = torch.cat((self._actions, new_action), dim=1)
        self._i += 1
        return self._actions[:, -1, :], self._g[:, -1, :]

    def expected_surprise(self, sequence: TensorDict, horizon: int) -> torch.Tensor:
        """Expected surprise in sophisticated formalism

        Returns a vector G_a
        """
        g_u1 = self.surprise_next_action(sequence)
        if horizon > 1:
            g_u1 += self.surprise_subsequent_actions(sequence, horizon)
        return g_u1

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
        return cross_entropy(q_obs, self._pref_dist)

    def surprise_subsequent_actions(
        self, sequence: TensorDict, horizon: int
    ) -> torch.Tensor:
        """Expected surprise of subsequent actions

        E_Q(o | pi) [G(pi_+1)]
        """
        q_obs = eval_func(
            Q_obs,
            self._model,
            sequence,
            act_map=self._act_map,
            obs_map=self._obs_map,
        ).abs()

        # large matrix with pruning
        larger = q_obs >= self._prune_threshold
        next_step = larger.nonzero()
        b, act, obs = next_step.split(1, dim=1)
        repeats = torch.bincount(b.flatten())
        imagined_sequence = sequence.repeat_interleave(repeats, dim=0)
        next_sequence = TensorDict(
            {"action": act.view(-1, 1, 1), "observation": obs.view(-1, 1, 1)}
        )
        imagined_sequence = cat(imagined_sequence, next_sequence)

        s_u_int = self.expected_surprise(imagined_sequence, horizon - 1)
        s_u = torch.full(
            (larger.numel(), s_u_int.shape[-1]),
            s_u_int.max().item(),
            device=self._device,
            dtype=s_u_int.dtype,
        )
        s_u[larger.view(-1)] = s_u_int
        q_u2_o1 = torch.softmax(-s_u, dim=-1)
        s = torch.sum(
            q_obs
            * torch.sum(q_u2_o1 * s_u, dim=-1).view(-1, self._act_dim, self._obs_dim),
            dim=-1,
        )
        return s


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
