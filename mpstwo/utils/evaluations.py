import itertools
from typing import Any, Optional

import torch

from mpstwo.agents import MPSAgent
from mpstwo.data.datasets import Dataset
from mpstwo.data.datastructs import TensorDict, cat, stack
from mpstwo.envs import EnvWrapper
from mpstwo.model import MPSTwo
from mpstwo.utils.mapping import Map


def check_probability(dataset: Dataset, sequence: TensorDict) -> torch.Tensor:
    """Check the probability of a sequence in the dataset

    Arguments:
        dataset -- the dataset to be checked
        sequence -- the sequence to be checked

    Returns:
        the percentage of occurence in the dataset
    """
    counter = torch.tensor(0)
    for i in range(len(dataset)):
        sample = dataset[i]
        if sample == sequence:
            counter += torch.tensor(1)
    return counter / len(dataset)


def check_distribution(dataset: Dataset) -> tuple[torch.Tensor, torch.Tensor]:
    """Check the distribution of a dataset with one-hot vectors

    Arguments:
        dataset -- the dataset to be checked

    Returns:
        two tensors with counts that when normalized yield:
        obs -- P(o_t)
        act -- P(a_t)
    """
    obs = torch.zeros_like(dataset[0]["observation"], dtype=torch.float32)
    act = torch.zeros_like(dataset[0]["action"], dtype=torch.float32)
    for i in range(len(dataset)):
        sample = dataset[i]
        o, a = sample["observation"], sample["action"]
        obs += o.real
        act += a.real
    return obs, act


def check_conditional_distribution(
    dataset: Dataset, condition: TensorDict
) -> torch.Tensor:
    """Check a specific conditional distribution of a dataset with one-hot vectors

    Depending on whether len(a) == len(o) or len(a) == len(o) + 1,
    the method will return P(a_t | a_<t, o_<t) or P(o_t | a_<=t, o_<t), resp.,
    with a = condition["action"] and o = condition["observation"].

    Arguments:
        dataset -- the dataset to be checked
        condition -- a TensorDict of the given history

    Returns:
        a tensors with counts that when normalized yields:
        P(o_t | a_<=t, o_<t) if len(a) == len(o) + 1
        P(a_t | a_<t, o_<t) if len(a) == len(o)
    """
    obslen, actlen = dataset[0]["observation"].shape[-1], dataset[0]["action"].shape[-1]
    o_c = torch.nn.functional.one_hot(
        condition["observation"].to(torch.int64), obslen
    ).to(dataset[0]["observation"])
    a_c = torch.nn.functional.one_hot(condition["action"].to(torch.int64), actlen).to(
        dataset[0]["action"]
    )
    if len(o_c) == len(a_c):
        plen = actlen
    elif len(o_c) + 1 == len(a_c):
        plen = obslen
    else:
        raise ValueError("len(o) must be == len(a) or == len(a) - 1")
    p = torch.zeros(plen)
    for i in range(len(dataset)):
        sample = dataset[i]
        o, a = sample["observation"], sample["action"]
        if torch.equal(a[: len(a_c)], a_c) and torch.equal(o[: len(o_c)], o_c):
            p += (a[len(a_c)] if len(o_c) == len(a_c) else o[len(o_c)]).real
    return p


def check_conditional_distributions(
    dataset: Dataset, timelen: Optional[int] = None
) -> tuple[list[list[torch.Tensor]], list[list[torch.Tensor]]]:
    """Check the conditional distributions of a dataset with one-hot vectors

    Arguments:
        dataset -- the dataset to be checked

    Returns:
        two tensors with counts that when normalized yield:
        obs -- P(o_t | a_<=t, o_<t)
        act -- P(a_t | a_<t, o_<t)
    """
    if timelen is None:
        timelen = len(dataset[0]["observation"])
    obslen, actlen = dataset[0]["observation"].shape[-1], dataset[0]["action"].shape[-1]
    obs, act = [[] for _ in range(timelen)], [[] for _ in range(timelen)]
    for t in range(timelen * 2):
        for comb in itertools.product(
            *([range(actlen), range(obslen)] * (t // 2) + [range(actlen)] * (t % 2))
        ):
            o_c = torch.tensor(comb[1::2])
            a_c = torch.tensor(comb[0::2])
            if comb[1::2]:
                o_c = torch.nn.functional.one_hot(o_c, obslen)
            else:
                o_c = o_c.reshape((-1, obslen))
            if comb[0::2]:
                a_c = torch.nn.functional.one_hot(a_c, actlen)
            else:
                a_c = a_c.reshape((-1, actlen))
            if t % 2:
                obs[t // 2].append(torch.zeros(obslen))
                for i in range(len(dataset)):
                    sample = dataset[i]
                    o, a = sample["observation"], sample["action"]
                    if torch.equal(a[: t // 2 + 1], a_c) and torch.equal(
                        o[: t // 2], o_c
                    ):
                        obs[t // 2][-1] += o[t // 2].real
            else:
                act[t // 2].append(torch.zeros(actlen))
                for i in range(len(dataset)):
                    sample = dataset[i]
                    o, a = sample["observation"], sample["action"]
                    if torch.equal(a[: t // 2], a_c) and torch.equal(o[: t // 2], o_c):
                        act[t // 2][-1] += a[t // 2].real

    return obs, act


def normalization(model: MPSTwo):
    """Check the normalization of an MPS

    Arguments:
        model -- an MPSTwo model

    Returns:
        the norm of the model
    """
    return model.norm()


def density_matrix(model: MPSTwo):
    """Check the trace of the density matrices of an MPS.

    Arguments:
        model -- an MPSTwo model

    Returns:
        a tuple of:
        densities -- a density matrix for each physical leg
        traces -- the trace of each density matrix
    """
    densities = []
    traces = []
    for idx in range(model.physical_legs):
        left = torch.ones((1, 1), dtype=model.dtype, device=model.device)
        for i, m in enumerate(model.matrices):
            if i == idx:
                left = torch.einsum("LM,LaoR,MapS->opRS", left, m, m.conj())
            else:
                left = torch.einsum("...LM,LaoR,MaoS->...RS", left, m, m.conj())
        right = torch.ones((1, 1), dtype=model.dtype, device=model.device)
        dens = torch.einsum("...RS,RS->...", left, right.conj())
        densities.append(dens)
        traces.append(torch.trace(dens))
    return densities, traces


def expected_observation(
    model: MPSTwo, obs: torch.Tensor, act: torch.Tensor, observation_map: Map
):
    """Compute the observation with the largest probability at the end
    of a sequence.

    Arguments:
        model -- an MPSTwo model
        obs -- a sequence of observations
        act -- a sequence of actions
        observation_map -- a mapping from observation to Hilbert space

    Returns:
        a tuple of:
        o_exp -- the expected observation
        o_max -- the observation with the largest probability
        o_prob -- an array of the probabilities of each observation
    """
    left = torch.tensor([1])
    for i in range(model.physical_legs - 1):
        left = torch.einsum(
            "l,a,o,laor->r", left, act[0, i], obs[0, i], model.matrices[i]
        )

    right = torch.tensor([1])
    o_contr = torch.einsum(
        "l,a,laor,r->o",
        left,
        act[0, model.physical_legs - 1],
        model.matrices[model.physical_legs - 1],
        right,
    )
    o_prob = torch.einsum("o,o->o", o_contr.conj(), o_contr).abs()
    o_prob = o_prob / torch.sum(o_prob)

    o_exp = torch.arange(0, model.feature_dim_obs).to(o_prob) @ o_prob
    o_max = observation_map.invert(o_prob)
    return o_exp, o_max, o_prob


def expected_action(
    model: MPSTwo,
    obs: torch.Tensor,
    act: torch.Tensor,
    action_map: Map,
    missing_idx: int = -1,
):
    """Compute the action with the largest probability at a position in
    a sequence.

    Arguments:
        model -- an MPSTwo model
        obs -- a sequence of observations
        act -- a sequence of actions
        action_map -- a mapping from action to Hilbert space

    Keyword Arguments:
        missing_idx -- the index of the action to be computed
                       (default: {-1})

    Returns:
        a tuple of:
        a_exp -- the expected action
        a_max -- the action with the largest probability
        a_prob -- an array of the probabilities of each action
    """
    if missing_idx == -1:
        missing_idx = act.shape[1]
    left = torch.tensor([1])
    for i in range(missing_idx):
        left = torch.einsum(
            "l,a,o,laor->r", left, act[0, i], obs[0, i], model.matrices[i]
        )

    right = torch.tensor([1])
    for i in range(model.physical_legs - 1, missing_idx, -1):
        right = torch.einsum(
            "laor,a,o,r->l", model.matrices[i], act[0, i], obs[0, i], right
        )

    a_contr = torch.einsum(
        "l,o,laor,r->a",
        left,
        obs[0, missing_idx],
        model.matrices[missing_idx],
        right,
    )
    a_prob = torch.einsum("a,a->a", a_contr.conj(), a_contr).abs()
    a_prob = a_prob / torch.sum(a_prob)

    a_exp = torch.arange(0, model.feature_dim_act).to(a_prob) @ a_prob
    a_max = action_map.invert(a_prob)
    return a_exp, a_max, a_prob


def masked_observation(
    model: MPSTwo,
    obs: torch.Tensor,
    act: torch.Tensor,
    observation_map: Map,
    mask: torch.Tensor,
):
    """Compute the observation with the largest probability at selected
    positions in a sequence.

    Arguments:
        model -- an MPSTwo model
        obs -- a sequence of observations
        act -- a sequence of actions
        observation_map -- a mapping from observation to Hilbert space
        mask -- a mask in the form of a list

    Returns:
        a dictionary with keys corresponding to index in the sequence
        and values containing the observation with the largest
        probability
    """
    pred_o = {}

    c_matrices = []
    for i, m in enumerate(model.matrices):
        if mask[i]:
            c_matrices.append(torch.einsum("a,o,laor->lr", act[0, i], obs[0, i], m))
        else:
            c_matrices.append(torch.einsum("a,laor->lor", act[0, i], m))

    missing = torch.arange(len(mask))[mask.logical_not()]
    for j in missing:
        left = torch.tensor([[1]])
        for i in range(j):
            left = torch.einsum(
                "lm,lr,ms->rs", left, c_matrices[i], c_matrices[i].conj()
            )

        right = torch.tensor([[1]])
        for i in range(model.physical_legs - 1, j, -1):
            if c_matrices[i].dim() < 3:
                right = torch.einsum(
                    "lr,ms,rs->lm", c_matrices[i], c_matrices[i].conj(), right
                )
            else:
                right = torch.einsum(
                    "lor,mos,rs->lm",
                    c_matrices[i],
                    c_matrices[i].conj(),
                    right,
                )

        o_prob = torch.einsum(
            "lm,lor,mos,rs->o",
            left,
            c_matrices[j],
            c_matrices[j].conj(),
            right,
        ).abs()
        o_prob = o_prob / torch.sum(o_prob)
        o_s = torch.argmax(o_prob)
        o_state = torch.zeros(model.feature_dim_obs)
        o_state[o_s] = 1
        o_sample = observation_map.invert(o_state)

        c_matrices[j] = torch.einsum(
            "o,lor->lr", o_state.to(c_matrices[j]), c_matrices[j]
        )
        pred_o[j.item()] = o_sample

    return pred_o


def masked_action(
    model: MPSTwo,
    obs: torch.Tensor,
    act: torch.Tensor,
    action_map: Map,
    mask: torch.Tensor,
):
    """Compute the action with the largest probability at selected
    positions in a sequence.

    Arguments:
        model -- an MPSTwo model
        obs -- a sequence of observations
        act -- a sequence of actions
        action_map -- a mapping from action to Hilbert space
        mask -- a mask in the form of a list

    Returns:
        a dictionary with keys corresponding to index in the sequence
        and values containing the action with the largest probability
    """
    pred_a = {}

    c_matrices = []
    for i, m in enumerate(model.matrices):
        if mask[i]:
            c_matrices.append(torch.einsum("a,o,laor->lr", act[0, i], obs[0, i], m))
        else:
            c_matrices.append(torch.einsum("o,laor->lar", obs[0, i], m))

    missing = torch.arange(len(mask))[mask.logical_not()]
    for j in missing:
        left = torch.tensor([[1]])
        for i in range(j):
            left = torch.einsum(
                "lm,lr,ms->rs", left, c_matrices[i], c_matrices[i].conj()
            )

        right = torch.tensor([[1]])
        for i in range(model.physical_legs - 1, j, -1):
            if c_matrices[i].dim() < 3:
                right = torch.einsum(
                    "lr,ms,rs->lm", c_matrices[i], c_matrices[i].conj(), right
                )
            else:
                right = torch.einsum(
                    "lar,mas,rs->lm",
                    c_matrices[i],
                    c_matrices[i].conj(),
                    right,
                )

        a_prob = torch.einsum(
            "lm,lar,mas,rs->a",
            left,
            c_matrices[j],
            c_matrices[j].conj(),
            right,
        ).abs()
        a_prob = a_prob / torch.sum(a_prob)
        a_s = torch.argmax(a_prob)
        a_state = torch.zeros(model.feature_dim_act)
        a_state[a_s] = 1
        a_sample = action_map.invert(a_state)

        c_matrices[j] = torch.einsum(
            "a,lar->lr", a_state.to(c_matrices[j]), c_matrices[j]
        )
        pred_a[j.item()] = a_sample

    return pred_a


def masked_sequence(
    model: MPSTwo,
    obs: torch.Tensor,
    act: torch.Tensor,
    observation_map: Map,
    action_map: Map,
    mask: torch.Tensor,
):
    """Compute the action with the largest probability at selected
    positions in a sequence.

    Arguments:
        model -- an MPSTwo model
        obs -- a sequence of observations
        act -- a sequence of actions
        observation_map -- a mapping from observation to Hilbert space
        action_map -- a mapping from action to Hilbert space
        mask -- a mask in the form of a list

    Returns:
        a dictionary with keys corresponding to index in the sequence
        and values containing tuples of the action with the largest
        probability and corresponding observation with the largest
        probabilty
    """
    pred = {}

    c_matrices = []
    for i, m in enumerate(model.matrices):
        if mask[i]:
            c_matrices.append(torch.einsum("a,o,laor->lr", act[0, i], obs[0, i], m))
        else:
            c_matrices.append(m)

    missing = torch.arange(len(mask))[mask.logical_not()]
    for j in missing:
        left = torch.tensor([[1]])
        for i in range(j):
            left = torch.einsum(
                "lm,lr,ms->rs", left, c_matrices[i], c_matrices[i].conj()
            )

        right = torch.tensor([[1]])
        for i in range(model.physical_legs - 1, j, -1):
            if c_matrices[i].dim() < 3:
                right = torch.einsum(
                    "lr,ms,rs->lm", c_matrices[i], c_matrices[i].conj(), right
                )
            else:
                right = torch.einsum(
                    "laor,maos,rs->lm",
                    c_matrices[i],
                    c_matrices[i].conj(),
                    right,
                )

        a_prob = torch.einsum(
            "lm,laor,maos,rs->a",
            left,
            c_matrices[j],
            c_matrices[j].conj(),
            right,
        ).abs()
        a_prob = a_prob / torch.sum(a_prob)
        a_s = torch.argmax(a_prob)
        a_state = torch.zeros(model.feature_dim_act)
        a_state[a_s] = 1
        a_sample = action_map.invert(a_state)

        o_prob = torch.einsum(
            "lm,laor,a,maos,rs->o",
            left,
            c_matrices[j],
            a_state.to(c_matrices[j]),
            c_matrices[j].conj(),
            right,
        ).abs()
        o_prob = o_prob / torch.sum(o_prob)
        o_s = torch.argmax(o_prob)
        o_state = torch.zeros(model.feature_dim_obs)
        o_state[o_s] = 1
        o_sample = observation_map.invert(o_state)

        c_matrices[j] = torch.einsum(
            "a,o,laor->lr",
            a_state.to(c_matrices[j]),
            o_state.to(c_matrices[j]),
            c_matrices[j],
        )
        pred[j.item()] = (a_sample, o_sample)

    return pred


def agent_rollout(
    *,
    env: EnvWrapper,
    agent: MPSAgent,
    sequence_length: int,
    rollouts: int = 1,
    reset_seed: Optional[list[int] | int] = None,
    reset_options: Optional[list[dict[str, Any]] | dict[str, Any]] = None,
    break_on_done: bool = False,
    show: bool = True,
    render: bool = False,
) -> TensorDict:
    if isinstance(reset_seed, list) and isinstance(reset_options, list):
        assert len(reset_seed) == len(reset_options), "list lengths must match"
        if len(reset_seed) != rollouts:
            print("WARNING: rollouts adjusted to equal number of seeds and options")
            rollouts = len(reset_seed)

    sequences = []
    for i in range(rollouts):
        sequence = []

        agent.reset()
        seed = (
            reset_seed[i % len(reset_seed)]
            if isinstance(reset_seed, list)
            else reset_seed
        )
        options = (
            reset_options[i % len(reset_options)]
            if isinstance(reset_options, list)
            else reset_options
        )
        step = env.reset(seed=seed, options=options).to(agent._device)  # type: ignore
        act_dim = env.get_action_space().shape or (1,)
        step.action = torch.zeros((1,) + act_dim, device=agent._device)  # type: ignore
        step.reward = torch.zeros((1, 1), device=agent._device)  # type: ignore
        step.terminated = torch.zeros((1, 1), device=agent._device)  # type: ignore
        step.truncated = torch.zeros((1, 1), device=agent._device)  # type: ignore
        step = step.unsqueeze(1)
        sequence.append(step)

        if show:
            print("\ninitial observation:", step.observation.cpu().numpy())

        if render:
            env.render()

        for j in range(1, sequence_length):
            action, efe = agent.act(step.observation)
            step = env.step(action.type(torch.int)).to(agent._device)  # type: ignore
            step = step.unsqueeze(1)
            sequence.append(step)

            if render:
                env.render()

            if show:
                print("\nTimestep:", j)
                print("EFE:", efe.cpu().numpy())
                print("action:", action.cpu().numpy())
                print("observation:", step.observation.cpu().numpy())

            if break_on_done and (step.terminated.item() or step.truncated.item()):
                break

        rollout = cat(*sequence)
        sequences.append(rollout)

    return stack(*sequences)
