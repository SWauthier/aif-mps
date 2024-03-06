import pathlib
import shutil
import sys
import traceback
from typing import Any, Callable, Optional

import numpy as np
import torch
from gymnasium.spaces import flatdim
from PIL import Image
from tqdm import tqdm

from mpstwo.agents import DiscreteSurprisalAgent, RandomAgent
from mpstwo.data.datasets import MemoryPool, RolloutPool
from mpstwo.data.datastructs import Dict, TensorDict
from mpstwo.envs import DictPacker, FrozenLake
from mpstwo.model import MPSTrainer, MPSTwo
from mpstwo.model.cutoff_schedulers import cutoff_scheduler_factory
from mpstwo.model.optimizers import SGD
from mpstwo.model.schedulers import scheduler_factory
from mpstwo.utils import get_model_path, save_config
from mpstwo.utils.distributions import (
    Q_obs,
    Q_state,
    entropy,
    kl_divergence,
    likelihood,
    # likelihood_,
)
from mpstwo.utils.evaluations import (
    agent_rollout,
    check_conditional_distribution,
    check_distribution,
    # density_matrix,
    expected_action,
    expected_observation,
    masked_action,
    masked_observation,
    masked_sequence,
    # normalization,
)
from mpstwo.utils.mapping import OneHotMap
from mpstwo.utils.visualizations import (
    plot_energy,
    to_bar,
    to_bar_rgb,
    visualize_sequence,
)

DATA_SET_SIZE = 100000
BATCH_SIZE = 256
config = Dict(
    {
        "log_dir": "frozenlake_%Y%m%d_%H%M%S",
        "environment": Dict({"desc": ["SFF", "FHF", "FFG"], "is_slippery": False}),
        "pool": Dict(
            {
                "sequence_length": 5,
                "rollouts": (DATA_SET_SIZE // BATCH_SIZE + 1) * BATCH_SIZE,
                "val_no_overlap": False,
            }
        ),
        "optimizer": Dict({"lr": 1e-3}),
        "scheduler": Dict(
            {
                "TwoSite": Dict({"frequency": 5, "first_epoch_update": False}),
                #         "Han": Dict({"safe_loss_threshold": 0.75, "lr_shrink_rate": 0.1}),
                #         "Annealing": Dict({"change_factor": 2}),
            }
        ),
        # "cutoff_scheduler": Dict(
        #     {
        #         "Step": Dict({"epoch": 10, "new_value": 0.01}),
        #     }
        # ),
        "trainer": Dict(
            {
                "log_to": "wandb",
                "batch_size": BATCH_SIZE,
                "epochs": 14,
                "save_epoch": 1,
            }
        ),
        "model": Dict(
            {
                "init_mode": "positive",
                "max_bond": 128,
                "cutoff": 0.03,
                "dtype": "torch.complex128",
            }
        ),
        "device": "cuda",
    }
)

train_set: MemoryPool
prev_model: MPSTwo
first = True


def make_env(config: Dict):
    return DictPacker(FrozenLake(**config.environment, render_mode="ansi"))


def make_dataset(config: Dict):
    dtype = eval(config.model.get("dtype", "torch.float32"))

    _env.reset()
    # init the agent
    agent = RandomAgent(_env.get_action_space())

    # init xp pool
    pool = RolloutPool(
        _env,
        agent,
        sequence_length=config.pool.sequence_length,
        epoch_size=config.pool.rollouts,
    )

    print("Generating training set...")
    train = MemoryPool()
    for i in tqdm(range(config.pool.rollouts)):
        sample = pool[i]
        for k, v in sample.items():
            if k == "observation":
                sample[k] = observation_map(v.unsqueeze(-1)).type(dtype)
            elif k == "action":
                sample[k] = action_map(v.unsqueeze(-1)).type(dtype)
        # if sample not in train:
        train.push_no_update(sample)
    train._update_table()
    # print(f"Number of samples: {len(train)}")

    print("Generating validation set...")
    validate = MemoryPool()
    if config.pool.val_no_overlap:
        with tqdm(total=config.trainer.batch_size) as pbar:
            while len(validate) < config.trainer.batch_size:
                sample = pool[0]
                for k, v in sample.items():
                    if k == "observation":
                        sample[k] = observation_map(v.unsqueeze(-1)).type(dtype)
                    elif k == "action":
                        sample[k] = action_map(v.unsqueeze(-1)).type(dtype)
                if sample not in train:
                    validate.push(sample)
                    pbar.update(1)
    else:
        for i in tqdm(range(config.trainer.batch_size)):
            sample = pool[i]
            for k, v in sample.items():
                if k == "observation":
                    sample[k] = observation_map(v.unsqueeze(-1)).type(dtype)
                elif k == "action":
                    sample[k] = action_map(v.unsqueeze(-1)).type(dtype)
            validate.push(sample)

    return train, validate


def run(config: Dict):
    global first
    first = True

    train, validate = make_dataset(config)
    global train_set
    train_set = train

    my_mps = MPSTwo(
        config.pool.sequence_length,
        feature_dim_obs=flatdim(_env.get_observation_space()["observation"]),
        feature_dim_act=flatdim(_env.get_action_space()),
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
    form = "_4x4" if config.environment.get("desc", None) is None else ""
    slip = "slip" if config.environment.is_slippery else "nons"
    iterations = config.get("iterations", 5)
    initializations = config.get("initializations", ["positive", "random_eye"])
    cutoffs = config.get("cutoffs", [0.1, 0.05, 0.03, 0.01])
    seq_start = config.get("seq_start", 5)
    seq_end = config.get("seq_end", 8)
    seq_stride = config.get("seq_stride", 1)
    seq_lengths = list(range(seq_start, seq_end, seq_stride))

    for init in initializations:
        config.model.init_mode = init
        istr = "pos" if init == "positive" else "eye"
        for cutoff in cutoffs:
            config.model.cutoff = cutoff
            for size in seq_lengths:
                config.pool.sequence_length = size
                for i in range(iterations):
                    config.log_dir = (
                        f"frozenlake{form}_{slip}_{istr}_cutoff{cutoff}_seq{size}_{i}"
                    )
                    log_dir = pathlib.Path("./MPS_" + config.log_dir)
                    if not log_dir.exists():
                        try:
                            run(config)
                        except Exception:
                            shutil.rmtree(log_dir)
                    else:
                        print(f"Skipping {log_dir}")


def convergence_callback(
    model: MPSTwo, obs: torch.Tensor, act: torch.Tensor
) -> list[tuple[str, Any, str]]:
    result = []
    global prev_model
    global first
    log_to_tensorboard = config.trainer.get("log_to", None) == "tensorboard"
    ltransform = to_bar_rgb if log_to_tensorboard else to_bar
    ltype = "image" if log_to_tensorboard else "figure"
    fourbyfour = model.feature_dim_obs == 16 and model.physical_legs > 6

    if first:
        global train_set
        o, _ = check_distribution(train_set)
        for i in range(model.physical_legs):
            log = ltransform(o[i].real.tolist())
            result.append((f"Probabilities/marginal_{i}_data", log, ltype))

        i = model.physical_legs - 1
        for j in range(action_map.dims):
            seq = TensorDict(
                {
                    "observation": observation_map.invert(obs[0, :i].real),
                    "action": action_map.invert(act[0, :i].real),
                }
            )
            result.append(("Probabilities/sequence", str(seq), "text"))
            seq["action"] = torch.cat(
                (seq["action"], torch.tensor([j]).to(seq["action"]))
            )
            obs1_ = check_conditional_distribution(train_set, seq)
            log = ltransform(obs1_.real.tolist())
            result.append((f"Probabilities/likelihood{i}_{j}_data", log, ltype))

        if fourbyfour:
            sequence = TensorDict(
                {
                    "action": torch.tensor(
                        [
                            [0.0, 2.0, 2.0, 1.0, 1.0, 1.0],
                            [0.0, 1.0, 1.0, 2.0, 2.0, 1.0],
                            [0.0, 1.0, 1.0, 2.0, 1.0, 2.0],
                        ]
                    ),
                    "observation": torch.tensor(
                        [
                            [0.0, 1.0, 2.0, 6.0, 10.0, 14.0],
                            [0.0, 4.0, 8.0, 9.0, 10.0, 14.0],
                            [0.0, 4.0, 8.0, 9.0, 13.0, 14.0],
                        ]
                    ),
                }
            )
        else:
            sequence = TensorDict(
                {
                    "action": torch.tensor(
                        [[0.0, 2.0, 2.0, 1.0], [0.0, 1.0, 1.0, 2.0]]
                    ),
                    "observation": torch.tensor(
                        [[0.0, 1.0, 2.0, 5.0], [0.0, 3.0, 6.0, 7.0]]
                    ),
                }
            )
        for i in range(sequence.shape[0]):
            for j in range(action_map.dims):
                seq = sequence[i]
                seq["action"] = torch.cat(
                    (seq["action"], torch.tensor([j]).to(seq["action"]))
                )
                obs1_ = check_conditional_distribution(train_set, seq)
                log = ltransform(obs1_.real.tolist())
                result.append(
                    (f"Probabilities/likelihood2goal{i}_{j}_data", log, ltype)
                )
        first = False

    result.append(
        (
            "Metrics/neg_log_change",
            (-(model @ prev_model).abs().log() / model.physical_legs).item(),
            "scalar",
        )
    )
    prev_model = model.clone()

    result.append(("Metrics/norm-1", model.norm().abs().item() - 1, "scalar"))

    # for i, s in enumerate(model.get_sv(1)[:20]):
    #     result.append((f"Singular Values Bond 1/{i}", s.item(), "scalar"))
    # model.right_canonical()

    for i in range(model.physical_legs):
        q_o = model.Q_oi(i)
        log = ltransform(q_o.real.tolist())
        result.append((f"Probabilities/marginal{i}", log, ltype))
        for k, e in enumerate(q_o.real.tolist()):
            result.append((f"Probabilities/marginal{i}_{k}", e, "scalar"))

    i = model.physical_legs - 1
    for j in range(action_map.dims):
        t_prob = model.likelihood(obs[:1, :i], act[:1, :i])
        a = action_map(torch.tensor([j])).type(model.dtype).to(model.device)
        lik = torch.einsum("a,Bao->Bo", a, t_prob).squeeze()
        log = ltransform(lik.real.tolist())
        result.append((f"Probabilities/likelihood{i}_{j}", log, ltype))
        for k, e in enumerate(lik.real.tolist()):
            result.append((f"Probabilities/likelihood{i}_{j}_{k}", e, "scalar"))

    if fourbyfour:
        sequence = TensorDict(
            {
                "action": torch.tensor(
                    [
                        [[0.0], [2.0], [2.0], [1.0], [1.0], [1.0]],
                        [[0.0], [1.0], [1.0], [2.0], [2.0], [1.0]],
                        [[0.0], [1.0], [1.0], [2.0], [1.0], [2.0]],
                    ]
                ),
                "observation": torch.tensor(
                    [
                        [[0.0], [1.0], [2.0], [6.0], [10.0], [14.0]],
                        [[0.0], [4.0], [8.0], [9.0], [10.0], [14.0]],
                        [[0.0], [4.0], [8.0], [9.0], [13.0], [14.0]],
                    ]
                ),
            }
        )
    else:
        sequence = TensorDict(
            {
                "action": torch.tensor(
                    [
                        [[0.0], [2.0], [2.0], [1.0]],
                        [[0.0], [1.0], [1.0], [2.0]],
                    ]
                ),
                "observation": torch.tensor(
                    [
                        [[0.0], [1.0], [2.0], [5.0]],
                        [[0.0], [3.0], [6.0], [7.0]],
                    ]
                ),
            }
        )
    for i in range(sequence.shape[0]):
        for j in range(action_map.dims):
            t_prob = model.likelihood(
                observation_map(sequence["observation"][i : i + 1])
                .to(model.dtype)
                .to(model.device),
                action_map(sequence["action"][i : i + 1])
                .to(model.dtype)
                .to(model.device),
            )
            a = action_map(torch.tensor([j])).type(model.dtype).to(model.device)
            lik = torch.einsum("a,Bao->Bo", a, t_prob).squeeze()
            log = ltransform(lik.real.tolist())
            result.append((f"Probabilities/likelihood2goal{i}_{j}", log, ltype))
            for k, e in enumerate(lik.real.tolist()):
                result.append(
                    (f"Probabilities/likelihood2goal{i}_{j}_{k}", e, "scalar")
                )

    return result


def eval_func(
    func: Callable,
    model: MPSTwo,
    sequence: TensorDict,
    obs_mapping: Optional[dict] = None,
    **kwargs,
):
    if obs_mapping is None:
        obs_mapping = {"observation": "observation"}

    if sequence["action"].dim() == 1:
        sequence = sequence.unsqueeze(0).unsqueeze(-1)
    elif sequence["action"].dim() == 2:
        sequence = sequence.unsqueeze(-1)
    obs = observation_map(sequence[obs_mapping["observation"]]).to(model.dtype)
    act = action_map(sequence["action"]).to(model.dtype)

    if kwargs is None:
        return func(model, obs, act)
    else:
        return func(model, obs, act, **kwargs)


def evaluate_accuracy(config: Dict):
    model_path = get_model_path(config)
    model = MPSTwo.load_full(model_path)
    model.right_canonical()
    agent = RandomAgent(_env.get_action_space(), repeat_p=0.5)
    pool = RolloutPool(_env, agent, sequence_length=config.pool.sequence_length)

    # print("\n== Normalization ==")
    # print(normalization(model))

    # print("\n== Normalization of densities ==")
    # dens, traces = density_matrix(model)
    # for i, d in enumerate(dens):
    #     print(traces[i], d)

    # sequence = pool[0].unsqueeze(0).unsqueeze(-1)

    # print("\n== Likelihood through density ==")
    # dens = eval_func(likelihood_, model, sequence[:, :-1])
    # act = action_map(sequence.action)
    # dens = torch.einsum("Ba,Baop->Bop", act[:, -1], dens)
    # print("density matrix:")
    # print(dens)
    # print("trace: ", torch.trace(dens.squeeze(0)))
    # print("probabilities:")
    # print(torch.einsum("Boo->Bo", dens).view(_env.nrow, _env.ncol))

    # print("\n== Likelihood ==")
    # lik = eval_func(likelihood, model, sequence[:, :-1])
    # act = action_map(sequence.action).to(model.dtype)
    # lik = torch.einsum("Ba,Bao->Bo", act[:, -1], lik)
    # print("probabilities:")
    # print(lik.abs().view(_env.nrow, _env.ncol).numpy())
    # print("sum:", lik.abs().sum().item())

    print("\n== Expected observation ==")
    sequence = pool[0].to(model.dtype)
    print("act:", sequence["action"].abs().int().numpy())
    print("obs:", sequence["observation"].abs().int().numpy())
    o_exp = eval_func(
        expected_observation, model, sequence, observation_map=observation_map
    )
    print("gt:", sequence["observation"][-1].abs().int().numpy())
    print("pred obs:")
    print("\tEV:", o_exp[0].abs().item())
    print("\targmax:", o_exp[1].abs().item())
    print("\tprobs:")
    print(o_exp[2].abs().view(_env.nrow, _env.ncol).numpy())

    print("\n== Expected action ==")
    sequence = pool[0].to(model.dtype)
    print("act:", sequence["action"].abs().int().numpy())
    print("obs:", sequence["observation"].abs().int().numpy())
    pred = torch.zeros((config.pool.sequence_length))
    for i in range(config.pool.sequence_length):
        a_exp = eval_func(
            expected_action, model, sequence, missing_idx=i, action_map=action_map
        )
        pred[i] = a_exp[1]
        # print("Index", i,":", a_exp)
    print("predicted actions (index i indicates action i left out):")
    print(pred.abs().numpy())

    print("\n== Masked observation ==")
    sequence = pool[0].to(model.dtype)
    print("act:", sequence["action"].abs().int().numpy())
    print("obs:", sequence["observation"].abs().int().numpy())
    mask = [1, 0, 0, 1, 1, 0, 0, 1, 0, 1][: config.pool.sequence_length]
    print("mask:", mask)
    mask = torch.tensor(mask, dtype=torch.bool)
    print("gt obs:", sequence.observation[mask.logical_not()].abs().int().numpy())
    pred = eval_func(
        masked_observation, model, sequence, observation_map=observation_map, mask=mask
    )
    print("pred obs (index: observation):", pred)

    print("\n== Masked action ==")
    sequence = pool[0].to(model.dtype)
    print("act:", sequence["action"].abs().int().numpy())
    print("obs:", sequence["observation"].abs().int().numpy())
    mask = [1, 0, 0, 1, 1, 0, 0, 1, 0, 1][: config.pool.sequence_length]
    print("mask:", mask)
    mask = torch.tensor(mask, dtype=torch.bool)
    print("gt act:", sequence.action[mask.logical_not()].abs().int().numpy())
    pred = eval_func(masked_action, model, sequence, action_map=action_map, mask=mask)
    print("pred act (index: action):", pred)

    print("\n== Masked sequence ==")
    sequence = pool[0].to(model.dtype)
    print("act:", sequence["action"].abs().int().numpy())
    print("obs:", sequence["observation"].abs().int().numpy())
    mask = [1, 0, 0, 1, 1, 0, 0, 1, 0, 1][: config.pool.sequence_length]
    print("mask:", mask)
    mask = torch.tensor(mask, dtype=torch.bool)
    print("gt act:", sequence.action[mask.logical_not()].abs().int().numpy())
    print("gt obs:", sequence.observation[mask.logical_not()].abs().int().numpy())
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
    print("pred act:", new_act.abs().numpy())
    print("pred obs:", new_obs.abs().numpy())
    visualize_sequence(
        _env,
        TensorDict(
            {
                "observation": new_obs.type(torch.int8),
                "action": new_act.type(torch.int8),
            }
        ),
    )

    print("== Transitions ==")
    print("From starting position...")
    sequence = TensorDict(
        {
            "action": torch.tensor([0]),
            "observation": torch.tensor([0]),
        }
    )
    for i, d in zip(range(action_map.dims), ["LEFT", "DOWN", "RIGHT", "UP"]):
        print("Action:", d)
        t_prob = eval_func(likelihood, model, sequence)
        act = action_map(torch.tensor([i])).to(model.dtype)
        lik = torch.einsum("a,Bao->Bo", act, t_prob).squeeze()
        print(lik.abs().view(_env.nrow, _env.ncol).numpy())
        print("sum:", lik.abs().sum().numpy())

    print("\nAfter going right...")
    sequence = TensorDict(
        {
            "action": torch.tensor([0, 2]),
            "observation": torch.tensor([0, 1]),
        }
    )
    for i, d in zip(range(action_map.dims), ["LEFT", "DOWN", "RIGHT", "UP"]):
        print("Action:", d)
        t_prob = eval_func(likelihood, model, sequence)
        act = action_map(torch.tensor([i])).to(model.dtype)
        lik = torch.einsum("a,Bao->Bo", act, t_prob).squeeze()
        print(lik.abs().view(_env.nrow, _env.ncol).numpy())
        print("sum:", lik.abs().sum().numpy())


def evaluate_action_selection(config: Dict):
    model_path = get_model_path(config)
    model = MPSTwo.load_full(model_path).to("cpu")
    model.right_canonical()

    print("\n== Action selection ==")
    start = torch.where(torch.tensor(_env.desc.ravel() == b"S"))[0]
    holes = torch.where(torch.tensor(_env.desc.ravel() == b"H"))[0]
    goal = config.get("goal", torch.where(torch.tensor(_env.desc.ravel() == b"G"))[0])
    pref = torch.zeros(observation_map.dims).to(model.device)
    pref[holes] = -5.0
    pref[goal] = 20.0
    print("preferences")
    print(pref)
    prob = torch.softmax(pref, dim=-1)

    print("first action")
    sequence = TensorDict(
        {
            "action": torch.tensor([0]),
            "observation": torch.tensor([0]),
        }
    )
    q_obs = eval_func(Q_obs, model, sequence, idx=1)
    lik = eval_func(likelihood, model, sequence, idx=1)
    q_s = eval_func(Q_state, model, sequence, idx=1)
    kl11 = kl_divergence(q_obs, prob)
    h11 = entropy(lik)
    print("KL", kl11)
    print("H", h11)
    print(kl11 + h11)

    q_obs = eval_func(Q_obs, model, sequence, idx=2)
    lik = eval_func(likelihood, model, sequence, idx=2)
    q_s = eval_func(Q_state, model, sequence, idx=2)
    kl21 = kl_divergence(q_obs, prob)
    h21 = torch.sum(q_s * entropy(lik), dim=2)
    print("KL")
    print("p", kl21)
    print("H:")
    print("p", h21)
    print("F")
    print(kl21 + h21)

    print("EFE")
    print((kl11 + h11).unsqueeze(-1) + (kl21 + h21))

    print("\nlast action")
    for a in range(action_map.dims):
        for p in range(observation_map.dims):
            sequence = TensorDict(
                {
                    "action": torch.tensor([0.0, a]),
                    "observation": torch.tensor([0.0, p]),
                }
            )
            q_obs = eval_func(Q_obs, model, sequence, idx=1)
            lik = eval_func(likelihood, model, sequence, idx=1)
            kl21 = kl_divergence(q_obs, prob)
            h21 = entropy(lik)
            print(f"({a}, {p})")
            print(" KL:", kl21)
            print("  H:", h21)
            print("EFE:", kl21 + h21)

    import time

    print("\n== Sophisticated ==")

    print("\nQ(o_t | o_{<t}, pi)")
    print("first action")
    print("first step")
    init_a = torch.tensor([0.0])
    init_o = torch.tensor([0.0])
    sequence = TensorDict({"action": init_a, "observation": init_o})
    q_obs = eval_func(Q_obs, model, sequence, idx=1)
    lik = eval_func(likelihood, model, sequence, idx=1)
    kl1 = kl_divergence(q_obs, prob)
    h1 = entropy(lik)
    print("KL")
    print(kl1)
    print("H")
    print(h1)
    print("G")
    f1 = kl1 + h1
    print(f1)

    print("\nsecond step")
    q_o_u = q_obs

    print("--> using loop")
    start = time.time()
    f2 = torch.zeros(4)
    for a in range(action_map.dims):
        for p in range(observation_map.dims):
            sequence = TensorDict(
                {
                    "action": torch.tensor([0.0, a]),
                    "observation": torch.tensor([0.0, p]),
                }
            )
            q_obs = eval_func(Q_obs, model, sequence, idx=1)
            lik = eval_func(likelihood, model, sequence, idx=1)
            kl2 = kl_divergence(q_obs, prob)
            h2 = entropy(lik)
            g = kl2 + h2
            q_u_o = torch.softmax(-g, dim=-1)

            f2[a] += q_o_u[:, a, p] @ torch.sum(q_u_o * g, dim=-1)

    end = time.time()
    print("Elapsed time:", end - start)

    print("G")
    print(f2)

    print("total G")
    print(f1 + f2)

    print("\n--> using matrix")
    start = time.time()
    pos_act = torch.arange(action_map.dims).type(init_a.dtype)
    pos_obs = torch.arange(observation_map.dims).type(init_o.dtype)
    init_a = torch.cartesian_prod(init_a, pos_act)
    init_o = torch.cartesian_prod(init_o, pos_obs)
    sequence = TensorDict(
        {
            "action": init_a.repeat_interleave(len(init_o), dim=0),
            "observation": init_o.repeat(len(init_a), 1),
        }
    )
    q_obs = eval_func(Q_obs, model, sequence, idx=1)
    lik = eval_func(likelihood, model, sequence, idx=1)
    kl2 = kl_divergence(q_obs, prob)
    h2 = entropy(lik)
    g = kl2 + h2
    q_u_o = torch.softmax(-g, dim=-1)
    f2 = torch.einsum(
        "bao,bao->ba",
        q_o_u,
        torch.sum(q_u_o * g, dim=-1).view(-1, action_map.dims, observation_map.dims),
    )

    end = time.time()
    print("Elapsed time:", end - start)

    print("G")
    print(f2)

    print("total G")
    print(f1 + f2)

    print("\nLEFT, DOWN, RIGHT, UP")


def evaluate_sophisticated(config: Dict):
    model_path = get_model_path(config)
    model = MPSTwo.load_full(model_path).to("cpu")
    model.right_canonical()

    # start = torch.where(torch.tensor(_env.desc.ravel() == b"S"))[0]
    holes = torch.where(torch.tensor(_env.desc.ravel() == b"H"))[0]
    goal = config.get("goal", torch.where(torch.tensor(_env.desc.ravel() == b"G"))[0])
    pref = torch.zeros(observation_map.dims).to(model.device)
    pref[holes] = -5.0
    pref[goal] = 20.0
    print("preferences")
    print(pref.view(_env.nrow, _env.ncol).numpy())
    prob = torch.softmax(pref, dim=-1)

    print("\nQ(o_t | o_{<t}, pi)")
    print("first action")
    print("first step")
    init_a = torch.tensor([0.0])
    init_o = torch.tensor([0.0])
    sequence = TensorDict({"action": init_a, "observation": init_o})
    q_obs = eval_func(Q_obs, model, sequence, idx=1).abs()
    lik = eval_func(likelihood, model, sequence, idx=1).abs()
    kl1 = kl_divergence(q_obs, prob)
    h1 = entropy(lik)
    print("KL")
    print(kl1.numpy())
    print("H")
    print(h1.numpy())
    print("G")
    f1 = kl1 + h1
    print(f1.numpy())

    print("\nsecond step")
    q_o_u = q_obs

    pos_act = torch.arange(action_map.dims).type(init_a.dtype)
    pos_obs = torch.arange(observation_map.dims).type(init_o.dtype)
    init_a = torch.cartesian_prod(init_a, pos_act)
    init_o = torch.cartesian_prod(init_o, pos_obs)
    sequence = TensorDict(
        {
            "action": init_a.repeat_interleave(len(init_o), dim=0),
            "observation": init_o.repeat(len(init_a), 1),
        }
    )
    q_obs = eval_func(Q_obs, model, sequence, idx=1).abs()
    lik = eval_func(likelihood, model, sequence, idx=1).abs()
    kl2 = kl_divergence(q_obs, prob)
    h2 = entropy(lik)
    g = kl2 + h2
    q_u_o = torch.softmax(-g, dim=-1)
    f2 = torch.einsum(
        "bao,bao->ba",
        q_o_u,
        torch.sum(q_u_o * g, dim=-1).view(-1, action_map.dims, observation_map.dims),
    )

    print("G")
    print(f2.numpy())

    print("total G")
    print((f1 + f2).numpy())

    print("\nactions: LEFT, DOWN, RIGHT, UP")


def evaluate_agent(config: Dict):
    model_path = get_model_path(config)
    model = MPSTwo.load_full(model_path).to("cpu")
    # if torch.cuda.is_available():
    #     model = model.to(config.device)
    start = torch.where(torch.tensor(_env.desc.ravel() == b"S"))[0]
    holes = torch.where(torch.tensor(_env.desc.ravel() == b"H"))[0]
    goal = config.get("goal", torch.where(torch.tensor(_env.desc.ravel() == b"G"))[0])
    pref = torch.zeros(observation_map.dims).to(model.device)
    pref[holes] = -5.0
    pref[goal] = 20.0
    pref_dist = torch.softmax(pref, dim=-1)
    horizon = config.get(
        "horizon",
        (
            torch.abs(goal % _env.ncol - start % _env.ncol)
            + torch.abs(
                torch.div(goal, _env.ncol, rounding_mode="floor")
                - torch.div(start, _env.ncol, rounding_mode="floor")
            )
        ).item(),
    )
    agent = DiscreteSurprisalAgent(
        model,
        act_map=action_map,
        obs_map=observation_map,
        pref_dist=pref_dist,
        horizon=horizon,
        # recursion=config.get("recursion", True),
    )
    print("Running agent...")
    print(f"Using horizon: {horizon}")
    print("Using preferred state:")
    print(pref.view(_env.nrow, _env.ncol).cpu().numpy())
    agent.reset()
    t = _env.reset()
    r = _env.render()
    if isinstance(r, np.ndarray):
        save_path = model_path.parent.parent
        img = Image.fromarray(r).convert("RGB")
        img.save(save_path / f"frozenlake{observation_map.dims}_0.eps")
    elif r is not None:
        print(r)

    for i in range(1, model.physical_legs):
        print("Timestep:", i)
        a, g = agent.act(t.observation.to(model.device))
        t = _env.step(a.type(torch.int))
        print("EFE:", g.tolist())
        r = _env.render()
        if isinstance(r, np.ndarray):
            poss = ["left", "down", "right", "up"]
            plot_energy(
                g.squeeze().cpu(),
                show=False,
                save=True,
                ylabel=None if isinstance(agent, DiscreteSurprisalAgent) else "G",
                xlabel="action",
                xlabs=poss,
                xrot=45,
                width=1 if isinstance(agent, DiscreteSurprisalAgent) else 1.5,
                height=1,
                save_loc=save_path,
                save_suffix=".eps",
            )

            img = Image.fromarray(r).convert("RGB")
            img.save(save_path / f"frozenlake{observation_map.dims}_{i}.eps")
        elif r is not None:
            print(r)

        if t.terminated.item() or t.truncated.item():
            break

    print("Done.")


def batch_create_figures(config: Dict):
    inits = ["pos", "eye"]
    cutoffs = [0.1, 0.05, 0.03, 0.01]
    seq_start = config.get("seq_start", 5)
    seq_end = config.get("seq_end", 8)
    seq_stride = config.get("seq_stride", 1)
    seq_lengths = list(range(seq_start, seq_end, seq_stride))
    seeds = config.get("seeds", 5)
    config.show = False
    config.save = True
    rolls = config.get("rolls", 1)
    reward = torch.zeros((len(inits), len(cutoffs), len(seq_lengths), seeds, rolls))
    parameters = torch.zeros((len(inits), len(cutoffs), len(seq_lengths), seeds))

    start = torch.where(torch.tensor(_env.desc.ravel() == b"S"))[0]
    holes = torch.where(torch.tensor(_env.desc.ravel() == b"H"))[0]
    goal = torch.where(torch.tensor(_env.desc.ravel() == b"G"))[0]
    pref = torch.zeros(observation_map.dims)
    pref[holes] = config.get("pref_hole", -5.0)
    pref[goal] = config.get("pref_goal", 20.0)
    pref_dist = torch.softmax(pref, dim=-1)
    horizon = config.get(
        "horizon",
        (
            torch.abs(goal % _env.ncol - start % _env.ncol)
            + torch.abs(
                torch.div(goal, _env.ncol, rounding_mode="floor")
                - torch.div(start, _env.ncol, rounding_mode="floor")
            )
        ).item(),
    )

    folder_path = pathlib.Path(config.get("folder_path", "frozenlake"))
    fig_path = folder_path / "figures"
    fig_path.mkdir(parents=True, exist_ok=True)
    for path in tqdm(folder_path.glob("MPS_*")):
        if config.get("verbose", False):
            print(path)
        model_path = path / "model"
        model_path = max(
            model_path.iterdir(), key=lambda x: int(x.name.split("-")[-1].split(".")[0])
        )

        name_split = path.name.split("_")
        init = [s for s in name_split if s in inits][0]
        cutoff = [s for s in name_split if "cutoff" in s][0].replace("cutoff", "")
        seq_length = [s for s in name_split if "seq" in s][0].replace("seq", "")
        seed = int([s for s in name_split if s.isnumeric()][0])
        init_i = inits.index(init)
        cutoff_i = cutoffs.index(float(cutoff))
        seq_length_i = seq_lengths.index(int(seq_length))

        model = MPSTwo.load_full(model_path).to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        parameters[init_i, cutoff_i, seq_length_i, seed] = model.get_parameters()

        agent = DiscreteSurprisalAgent(
            model,
            act_map=action_map,
            obs_map=observation_map,
            pref_dist=pref_dist.to(model.device),
            horizon=horizon,
            prune_threshold=1 / observation_map.dims,
        )
        rollouts = agent_rollout(
            env=_env,
            agent=agent,
            sequence_length=model.physical_legs,
            rollouts=rolls,
            show=False,
        )
        reward[init_i, cutoff_i, seq_length_i, seed, :] = (
            rollouts.observation[:, -1] == goal.item()
        )
        if config.get("verbose", False):
            print(reward[init_i, cutoff_i, seq_length_i, seed, :])

    torch.save(reward, fig_path / "reward.pt")
    torch.save(parameters, fig_path / "parameters.pt")


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
observation_map = OneHotMap(flatdim(_env.get_observation_space()["observation"]))
config.model.bond_dim = observation_map.dims
action_map = OneHotMap(flatdim(_env.get_action_space()))

if __name__ == "__main__":
    func = getattr(sys.modules[__name__], sys.argv[1])
    with torch.inference_mode():
        func(config)
