import math
import pathlib
import sys
import traceback
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import torch
from torch.distributions import Categorical
from tqdm import tqdm

from mpstwo.agents import MultiDiscreteSurprisalAgent, RandomAgent
from mpstwo.data.datasets import MemoryPool, RolloutPool
from mpstwo.data.datastructs import Dict, TensorDict
from mpstwo.envs import DictPacker, TMaze
from mpstwo.model import MPSTrainer, MPSTwo
from mpstwo.model.cutoff_schedulers import cutoff_scheduler_factory
from mpstwo.model.optimizers import SGD
from mpstwo.model.schedulers import scheduler_factory
from mpstwo.utils import get_model_path, save_config
from mpstwo.utils.distributions import entropy
from mpstwo.utils.evaluations import (
    agent_rollout,
    expected_action,
    expected_observation,
    masked_action,
    masked_observation,
    masked_sequence,
)
from mpstwo.utils.mapping import Map, MultiOneHotMap

config = Dict(
    {
        "log_dir": "tmaze_%Y%m%d_%H%M%S",
        "environment": Dict(
            {
                "reward_probs": [1, 0],
            }
        ),
        "pool": Dict(
            {
                "sequence_length": 3,
                "rollouts": 5000,
                "random": False,
            }
        ),
        "optimizer": Dict({"lr": 1e-3}),
        "trainer": Dict(
            {
                "batch_size": 100,
                "epochs": 500,
                "save_epoch": 50,
            }
        ),
        "model": Dict(
            {
                "init_mode": "positive",
                "max_bond": 24,
                "cutoff": 0.03,
                "dtype": "torch.complex128",
            }
        ),
        "device": "cuda",
    }
)

prev_model: MPSTwo


def make_env(config):
    return DictPacker(TMaze(**config.environment))


def make_dataset(config: Dict):
    dtype = eval(config.model.get("dtype", "torch.float32"))

    # init the agent
    agent = RandomAgent(_env.get_action_space())

    # init xp pool
    pool = RolloutPool(
        _env,
        agent,
        sequence_length=config.pool.sequence_length,
        epoch_size=config.pool.rollouts,
    )
    train = MemoryPool()

    if config.pool.random:
        for i in range(config.pool.rollouts):
            sample = pool[i]
            for k, v in sample.items():
                if k == "observation":
                    sample[k] = observation_map(v).type(dtype)
                elif k == "action":
                    sample[k] = action_map(v).type(dtype)
            train.push_no_update(sample)
    else:
        for c0 in range(2):
            for a1 in range(4):
                p1 = a1
                rmin1 = 1 if a1 in [1, 2] else 0
                rmax1 = 3 if a1 in [1, 2] else 1
                for r1 in range(rmin1, rmax1):
                    for c1 in range(2):
                        for a2 in range(4):
                            p2 = a2
                            if a1 in [1, 2]:
                                p2 = a1
                            if p1 in [1, 2] and p2 in [1, 2]:
                                rmin2 = r1
                                rmax2 = r1 + 1
                            elif p1 == 3 and p2 in [1, 2]:
                                rmin2 = 2 - (c1 + p2) % 2
                                rmax2 = rmin2 + 1
                            elif p2 in [1, 2]:
                                rmin2 = 1
                                rmax2 = 3
                            else:
                                rmin2 = 0
                                rmax2 = 1
                            for r2 in range(rmin2, rmax2):
                                cmin2 = c1 if a1 == 3 and a2 == 3 else 0
                                cmax2 = c1 + 1 if a1 == 3 and a2 == 3 else 2
                                for c2 in range(cmin2, cmax2):
                                    sequence = TensorDict(
                                        {
                                            "action": torch.tensor(
                                                [[0, 0], [a1, 0], [a2, 0]]
                                            ),
                                            "observation": torch.tensor(
                                                [[0, 0, c0], [p1, r1, c1], [p2, r2, c2]]
                                            ),
                                        }
                                    )
                                    for k, v in sequence.items():
                                        if k == "observation":
                                            sequence[k] = observation_map(v).type(dtype)
                                        elif k == "action":
                                            sequence[k] = action_map(v).type(dtype)
                                    train.push_no_update(sequence)
    train._update_table()

    validate = MemoryPool(
        sequence_length=config.pool.sequence_length,
        sequence_stride=config.pool.sequence_length + 1,
    )
    for i in range(config.trainer.batch_size):
        sample = pool[i]
        for k, v in sample.items():
            if k == "observation":
                sample[k] = observation_map(v).type(dtype)
            elif k == "action":
                sample[k] = action_map(v).type(dtype)
        validate.push(sample)

    return train, validate


def run(config: Dict):
    train, validate = make_dataset(config)

    my_mps = MPSTwo(
        config.pool.sequence_length,
        feature_dim_obs=math.prod(_env.num_obs), # type: ignore
        feature_dim_act=math.prod(_env.num_controls), # type: ignore
        **config.model,
        device=config.device,
    )
    global prev_model
    prev_model = my_mps.clone()

    optimizer = SGD(my_mps, **config.optimizer)
    scheduler = None
    if hasattr(config, "scheduler"):
        scheduler = scheduler_factory(optimizer, config.scheduler)
    cutoff_scheduler = None
    if hasattr(config, "cutoff_scheduler"):
        cutoff_scheduler = cutoff_scheduler_factory(my_mps, config.cutoff_scheduler)

    trainer = MPSTrainer(
        my_mps,
        train,
        val_dataset=validate,
        feature_map_obs=observation_map,
        feature_map_act=action_map,
        optimizer=optimizer,
        scheduler=scheduler,
        cutoff_scheduler=cutoff_scheduler,
        **config.trainer,
        log_dir_suffix=config.log_dir,
        device=config.device,
        custom_callback=convergence_callback,
        experiment_config=config,
    )

    torch.save(train, trainer._log_dir / "train_set.pt")
    save_config(config, trainer._log_dir / "config.log")

    try:
        trainer.train(config.trainer.epochs)
    except Exception:
        print(traceback.format_exc())
        raise
    finally:
        trainer.close()


def run_param_search(config: Dict):
    iterations = config.get("iterations", 10)
    initializations = ["positive", "random_eye"]
    cutoffs = [0.2, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01, 0.005]
    set_sizes = [100, 204, 1000, 10000, -1]
    max_size = max(set_sizes)
    epochs = config.trainer.epochs
    if config.get("use_ssu", False):
        config.scheduler = Dict(
            {"TwoSite": Dict({"frequency": 10, "first_epoch_update": False})}
        )
        config.model.bond_dim = math.prod(_env.num_obs)

    for init in initializations:
        config.model.init_mode = init
        istr = "pos" if init == "positive" else "eye"
        for cutoff in cutoffs:
            config.model.cutoff = cutoff
            for size in set_sizes:
                config.pool.random = size > 0
                config.pool.rollouts = size if size > 0 else 204
                sstr = size if size > 0 else "fixed"
                config.trainer.epochs = (
                    epochs
                    * (max_size // config.trainer.batch_size)
                    // ((size if size > 0 else 204) // config.trainer.batch_size)
                )
                for i in range(iterations):
                    config.log_dir = f"tmaze_{istr}_cutoff{cutoff}_set{sstr}_{i}"
                    log_dir = pathlib.Path("./MPS_" + config.log_dir)
                    if not log_dir.exists():
                        run(config)
                    else:
                        print(f"Skipping {log_dir}")


def convergence_callback(
    model: MPSTwo, obs: torch.Tensor, act: torch.Tensor
) -> list[tuple[str, Any, str]]:
    result = []
    global prev_model
    global first

    result.append(
        (
            "Metrics/neg_log_change",
            (-(model @ prev_model).abs().log() / model.physical_legs).item(),
            "scalar",
        )
    )
    prev_model = model.clone()

    result.append(("Metrics/norm-1", model.norm().abs().item() - 1, "scalar"))

    return result


def generate_sample(model):
    observations = torch.empty((model.physical_legs, len(_env.num_obs)))
    actions = torch.empty((model.physical_legs, len(_env.num_controls)))

    model.left_canonical()

    right = torch.tensor([1])
    for p in range(model.physical_legs - 1, -1, -1):
        contr = torch.einsum("laor,r->lao", model.matrices[p], right)

        # action
        a_prob = torch.einsum("lao,lao->a", contr.conj(), contr)
        a_prob = a_prob / torch.sum(a_prob)
        a_s = Categorical(a_prob).sample()
        a_state = torch.zeros(model.feature_dim_act)
        a_state[a_s] = 1

        # observation
        o_prob = torch.einsum(
            "lao,lbo,a,b->o",
            contr.conj(),
            contr,
            a_state.reshape(-1),
            a_state.reshape(-1),
        )
        o_prob = o_prob / torch.sum(o_prob)
        o_s = Categorical(o_prob).sample()
        o_state = torch.zeros(model.feature_dim_obs)
        o_state[o_s] = 1

        right = torch.einsum(
            "laor,a,o,r->l",
            model.matrices[p],
            a_state.reshape(-1),
            o_state.reshape(-1),
            right,
        )

        observations[p] = observation_map.invert(o_state)
        actions[p] = action_map.invert(a_state)
    return TensorDict({"observation": observations, "action": actions}).unsqueeze(0)


def eval_func(
    func: Callable,
    model: MPSTwo,
    sequence: TensorDict,
    obs_mapping: Optional[dict] = None,
    **kwargs,
):
    if obs_mapping is None:
        obs_mapping = {"observation": "observation"}

    model.right_canonical()
    if sequence["action"].dim() == 2:
        sequence = sequence.unsqueeze(0)
    obs = observation_map(sequence[obs_mapping["observation"]]).to(model.dtype)
    act = action_map(sequence["action"]).to(model.dtype)

    if kwargs is None:
        return func(model, obs, act)
    else:
        return func(model, obs, act, **kwargs)


def evaluate_accuracy(config: Dict):
    model_path = get_model_path(config)
    model = MPSTwo.load_full(model_path).to("cpu")
    print("\n== Generated sample ==")
    sample = generate_sample(model)
    print("act:", sample.action)
    print("obs:", sample.observation)

    agent = RandomAgent(_env.get_action_space(), repeat_p=0.5)
    pool = RolloutPool(_env, agent, sequence_length=config.pool.sequence_length)

    print("\n== Expected observation ==")
    sequence = pool[0]
    print("act:", sequence.action)
    print("obs:", sequence.observation)
    o_exp = eval_func(
        expected_observation, model, sequence, observation_map=observation_map
    )
    print("gt:", sequence.observation[-1])
    print("pred obs:")
    print("\tEV:", o_exp[0])
    print("\targmax:", o_exp[1])
    print("\tprobs:")
    print(o_exp[2].reshape(4, 3, 2))
    acc = 0
    tests = 1000
    for i in range(tests):
        sequence = pool[i]
        o_exp = eval_func(
            expected_observation, model, sequence, observation_map=observation_map
        )
        # print(sequence.observation, o_exp[1])
        acc += torch.all(sequence.observation[-1][:-1] == o_exp[1][:-1]) / tests
    print("acc:", acc)

    print("\n== Expected action ==")
    sequence = pool[0]
    print("act:", sequence.action)
    print("obs:", sequence.observation)
    pred = torch.zeros((config.pool.sequence_length, len(_env.num_controls)))
    for i in range(config.pool.sequence_length):
        a_exp = eval_func(
            expected_action, model, sequence, missing_idx=i, action_map=action_map
        )
        pred[i] = a_exp[1]
        # print("Index", i,":", a_exp)
    print("predicted actions (index i indicates action i left out):")
    print(pred)

    print("\n== Masked observation ==")
    sequence = pool[0]
    print("act:", sequence.action)
    print("obs:", sequence.observation)
    mask = [1, 0, 1]
    print("mask:", mask)
    print("gt obs:", sequence.observation[~torch.tensor(mask, dtype=torch.bool)])
    pred = eval_func(
        masked_observation, model, sequence, observation_map=observation_map, mask=mask
    )
    print("pred obs (index: observation):", pred)

    print("\n== Masked action ==")
    sequence = pool[0]
    print("act:", sequence.action)
    print("obs:", sequence.observation)
    mask = [1, 0, 1]
    print("mask:", mask)
    print("gt act:", sequence.action[~torch.tensor(mask, dtype=torch.bool)])
    pred = eval_func(masked_action, model, sequence, action_map=action_map, mask=mask)
    print("pred act (index: action):", pred)

    print("\n== Masked sequence ==")
    sequence = pool[0]
    print("act:", sequence.action)
    print("obs:", sequence.observation)
    mask = [1, 0, 1]
    print("mask:", mask)
    print("gt act:", sequence.action[~torch.tensor(mask, dtype=torch.bool)])
    print("gt obs:", sequence.observation[~torch.tensor(mask, dtype=torch.bool)])
    pred = eval_func(
        masked_sequence,
        model,
        sequence,
        observation_map=observation_map,
        action_map=action_map,
        mask=mask,
    )
    print("pred sequence (index: (action, observation)):", pred)
    new_act = sequence.action
    new_obs = sequence.observation
    for i in pred.keys():
        new_act[i] = pred[i][0]
        new_obs[i] = pred[i][1]
    print("pred act:", new_act)
    print("pred obs:", new_obs)

    print("\n== Probability of possible paths ==")
    print("## immediately right ##")
    sequence = TensorDict(
        {
            "action": torch.tensor([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]]),
            "observation": torch.tensor(
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
            ),
        }
    )
    print("act:", sequence.action)
    print("obs:", sequence.observation[:-1])
    pred = eval_func(
        expected_observation, model, sequence, observation_map=observation_map
    )
    print("argmax:")
    print(pred[1])
    print("probs:")
    print(pred[2].reshape(4, 3, 2))
    print("H:", entropy(pred[2]).item())
    print("## immediately left ##")
    sequence = TensorDict(
        {
            "action": torch.tensor([[0.0, 0.0], [0.0, 0.0], [2.0, 0.0]]),
            "observation": torch.tensor(
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
            ),
        }
    )
    print("act:", sequence.action)
    print("obs:", sequence.observation[:-1])
    pred = eval_func(
        expected_observation, model, sequence, observation_map=observation_map
    )
    print("argmax:")
    print(pred[1])
    print("probs:")
    print(pred[2].reshape(4, 3, 2))
    print("H:", entropy(pred[2]).item())
    print("## cue, right | reward right ##")
    sequence = TensorDict(
        {
            "action": torch.tensor([[0.0, 0.0], [3.0, 0.0], [1.0, 0.0]]),
            "observation": torch.tensor(
                [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
            ),
        }
    )
    print("act:", sequence.action)
    print("obs:", sequence.observation[:-1])
    pred = eval_func(
        expected_observation, model, sequence, observation_map=observation_map
    )
    print("argmax:")
    print(pred[1])
    print("probs:")
    print(pred[2].reshape(4, 3, 2))
    print("H:", entropy(pred[2]).item())
    print("## cue, left | reward right ##")
    sequence = TensorDict(
        {
            "action": torch.tensor([[0.0, 0.0], [3.0, 0.0], [2.0, 0.0]]),
            "observation": torch.tensor(
                [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
            ),
        }
    )
    print("act:", sequence.action)
    print("obs:", sequence.observation[:-1])
    pred = eval_func(
        expected_observation, model, sequence, observation_map=observation_map
    )
    print("argmax:")
    print(pred[1])
    print("probs:")
    print(pred[2].reshape(4, 3, 2))
    print("H:", entropy(pred[2]).item())
    print("## cue, right | reward left ##")
    sequence = TensorDict(
        {
            "action": torch.tensor([[0.0, 0.0], [3.0, 0.0], [1.0, 0.0]]),
            "observation": torch.tensor(
                [[0.0, 0.0, 0.0], [3.0, 0.0, 1.0], [0.0, 0.0, 0.0]]
            ),
        }
    )
    print("act:", sequence.action)
    print("obs:", sequence.observation[:-1])
    pred = eval_func(
        expected_observation, model, sequence, observation_map=observation_map
    )
    print("argmax:")
    print(pred[1])
    print("probs:")
    print(pred[2].reshape(4, 3, 2))
    print("H:", entropy(pred[2]).item())
    print("## cue, left | reward left ##")
    sequence = TensorDict(
        {
            "action": torch.tensor([[0.0, 0.0], [3.0, 0.0], [2.0, 0.0]]),
            "observation": torch.tensor(
                [[0.0, 0.0, 0.0], [3.0, 0.0, 1.0], [0.0, 0.0, 0.0]]
            ),
        }
    )
    print("act:", sequence.action)
    print("obs:", sequence.observation[:-1])
    pred = eval_func(
        expected_observation, model, sequence, observation_map=observation_map
    )
    print("argmax:")
    print(pred[1])
    print("probs:")
    print(pred[2].reshape(4, 3, 2))
    print("H:", entropy(pred[2]).item())


def evaluate_agent(config):
    model_path = get_model_path(config)
    model = MPSTwo.load_full(model_path)
    prob_pos = torch.softmax(torch.tensor([0.0, 0.0, 0.0, 0.0]), dim=-1)
    prob_rew = torch.softmax(torch.tensor([0.0, 3.0, -3.0]), dim=-1)
    prob_ctx = torch.softmax(torch.tensor([0.0, 0.0]), dim=-1)
    prob = [prob_pos, prob_rew, prob_ctx]
    agent = MultiDiscreteSurprisalAgent(
        model,
        act_map=action_map,
        obs_map=observation_map,
        pref_dist=[p.to(model.device) for p in prob],
        horizon=2,
        imagine_future=config.get("future", "classic"),
    )
    state0 = np.empty(2, dtype=object)
    state0[0] = torch.nn.functional.one_hot(torch.tensor(0), 4).numpy()
    state0[1] = torch.nn.functional.one_hot(torch.tensor(0), 2).numpy()
    state1 = np.empty(2, dtype=object)
    state1[0] = torch.nn.functional.one_hot(torch.tensor(0), 4).numpy()
    state1[1] = torch.nn.functional.one_hot(torch.tensor(1), 2).numpy()
    reset_options = [{"state": state0}, {"state": state1}]
    rollouts = agent_rollout(
        env=_env,
        agent=agent,
        sequence_length=config.pool.sequence_length,
        rollouts=2,
        reset_seed=None,
        reset_options=reset_options,
        break_on_done=False,
        show=True,
        render=False,
    )
    print("\nfound:", (rollouts.observation[:, -1, 1] == 1).cpu().numpy())
    for k, v in rollouts.items():
        print(k)
        print(v.cpu().numpy())


def model_accuracy(
    model: MPSTwo, obs: torch.Tensor, act: torch.Tensor, observation_map: Map
) -> Any:
    left = torch.ones(1, dtype=model.dtype, device=model.device).unsqueeze(0)
    for i, m in enumerate(model.matrices[:-1]):
        left = torch.einsum("bl,ba,bo,laor->br", left, act[:, i], obs[:, i], m)
    right = torch.ones(1, dtype=model.dtype, device=model.device).unsqueeze(0)
    o_contr = torch.einsum(
        "bl,ba,laor,br->bo",
        left,
        act[:, model.physical_legs - 1],
        model.matrices[model.physical_legs - 1],
        right,
    )
    o_prob = torch.einsum("bo,bo->bo", o_contr.conj(), o_contr).abs()
    o_prob = o_prob / torch.sum(o_prob, dim=-1, keepdim=True)

    o_pred = observation_map.invert(o_prob)
    o_true = observation_map.invert(obs[:, -1].abs())
    obs_inv = observation_map.invert(obs.abs())

    ctx_matters = torch.logical_and(obs_inv[:, 1, 0] == 3, obs_inv[:, 2, 0] == 3)
    only_pos_matters = obs_inv[:, 1, 0] == 0
    correct = torch.where(
        ctx_matters,
        torch.eq(o_pred, o_true).all(dim=-1),
        torch.where(
            only_pos_matters,
            torch.eq(o_pred[:, 0], o_true[:, 0]),
            torch.eq(o_pred[:, :-1], o_true[:, :-1]).all(dim=-1),
        ),
    )
    acc = correct.float().mean()
    return acc.item()


def batch_create_figures(config: Dict):
    inits = ["pos", "eye"]
    cutoffs = [0.2, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01, 0.005]
    set_sizes = [100, 204, 1000, 10000, "(204)"]
    seeds = config.get("seeds", 10)
    config.show = False
    config.save = True
    accuracy = torch.zeros((len(inits), len(cutoffs), len(set_sizes), seeds))
    rolls = 100
    reward = torch.zeros((len(inits), len(cutoffs), len(set_sizes), seeds, rolls))
    negloglik = torch.zeros_like(accuracy)
    parameters = torch.zeros_like(accuracy)
    svs = []

    config.pool.random = False
    test_set, _ = make_dataset(config)

    prob_pos = torch.softmax(torch.tensor([0.0, 0.0, 0.0, 0.0]), dim=-1)
    prob_rew = torch.softmax(torch.tensor([0.0, 3.0, -3.0]), dim=-1)
    prob_ctx = torch.softmax(torch.tensor([0.0, 0.0]), dim=-1)
    prob = [prob_pos, prob_rew, prob_ctx]
    state0 = np.empty(2, dtype=object)
    state0[0] = torch.nn.functional.one_hot(torch.tensor(0), 4).numpy()
    state0[1] = torch.nn.functional.one_hot(torch.tensor(0), 2).numpy()
    state1 = np.empty(2, dtype=object)
    state1[0] = torch.nn.functional.one_hot(torch.tensor(0), 4).numpy()
    state1[1] = torch.nn.functional.one_hot(torch.tensor(1), 2).numpy()
    reset_options = [{"state": state0}, {"state": state1}]

    folder_path = pathlib.Path(config.get("folder_path", "tmaze"))
    fig_path = folder_path / "figures"
    fig_path.mkdir(parents=True, exist_ok=True)
    for path in tqdm(folder_path.glob("MPS_*")):
        config.model_path = path
        model_path = get_model_path(config)

        init = [s for s in path.name.split("_") if s in inits][0]
        cutoff = [s for s in path.name.split("_") if "cutoff" in s][0].replace("cutoff", "")
        set_size = [s for s in path.name.split("_") if "set" in s][0].replace("set", "")
        seed = int([s for s in path.name.split("_") if s.isnumeric()][0])
        init_i = inits.index(init)
        cutoff_i = cutoffs.index(float(cutoff))
        set_size_i = set_sizes.index(int(set_size) if set_size.isnumeric() else "(204)")

        model = MPSTwo.load_full(model_path).to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        correct = 0
        nlls = []
        for sequence in test_set:
            sequence = sequence.unsqueeze(0).to(model.device)
            correct += model_accuracy(
                model,
                sequence["observation"],
                sequence["action"],
                observation_map=observation_map,
            )

            probs = model(sequence["observation"], sequence["action"])
            nlls.append(-torch.log(probs))

        accuracy[init_i, cutoff_i, set_size_i, seed] = correct / len(test_set)

        negloglik[init_i, cutoff_i, set_size_i, seed] = sum(nlls) / len(nlls)

        parameters[init_i, cutoff_i, set_size_i, seed] = model.get_parameters()

        agent = MultiDiscreteSurprisalAgent(
            model,
            act_map=action_map,
            obs_map=observation_map,
            pref_dist=[p.to(model.device) for p in prob],
            horizon=2,
        )
        rollouts = agent_rollout(
            env=_env,
            agent=agent,
            sequence_length=config.pool.sequence_length,
            rollouts=rolls,
            reset_options=reset_options,
            show=False,
        )
        reward[init_i, cutoff_i, set_size_i, seed, :] = rollouts.observation[:, -1, 1] == 1

        if config.get("verbose", False):
            print(accuracy[init_i, cutoff_i, set_size_i, seed])
            print(negloglik[init_i, cutoff_i, set_size_i, seed])
            print(parameters[init_i, cutoff_i, set_size_i, seed])
            print(reward[init_i, cutoff_i, set_size_i, seed, :])

        if (
            (not set_size.isnumeric() or int(set_size) == set_sizes[0])
            and init == "pos"
            and seed == 1
        ):
            for b, _ in enumerate(model.matrices[:-1]):
                svs_i = model.get_sv(b)[:24]
                svs_i = (svs_i / svs_i[0]).tolist()
                ind_i = [i + 1 for i, _ in enumerate(svs_i)]
                df_i = pd.DataFrame(
                    {
                        "cutoff": cutoff,
                        "bond": b,
                        "value": svs_i,
                        "singular value": ind_i,
                        "overfit": set_size.isnumeric(),
                    }
                )
                svs.append(df_i)

    svs = pd.concat(svs)

    torch.save(accuracy, fig_path / "accuracy.pt")
    torch.save(reward, fig_path / "reward.pt")
    torch.save(negloglik, fig_path / "negloglik.pt")
    torch.save(parameters, fig_path / "parameters.pt")
    torch.save(svs, fig_path / "svs.pt")


if __name__ == "__main__":
    if len(sys.argv) > 2:
        for kv in sys.argv[2:]:
            kvl = kv.split("=")
            try:
                exec(f"config.{kvl[0]} = {kvl[1]}")
            except NameError:
                exec(f"config.{kvl[0]} = '{kvl[1]}'")
            except SyntaxError:
                exec(f"config.{kvl[0]} = '{kvl[1]}'")

_env = make_env(config)
observation_map = MultiOneHotMap(_env.num_obs)
action_map = MultiOneHotMap(_env.num_controls)

if __name__ == "__main__":
    func = getattr(sys.modules[__name__], sys.argv[1])
    with torch.inference_mode():
        func(config)
