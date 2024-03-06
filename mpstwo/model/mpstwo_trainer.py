from __future__ import annotations

import logging
import pathlib
from copy import deepcopy
from time import strftime
from typing import TYPE_CHECKING, Any, Callable, Optional

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

try:
    import wandb
except ImportError:
    pass

from mpstwo.model.log import Log
from mpstwo.model.losses import NLLLoss

if TYPE_CHECKING:
    from mpstwo.data.datasets.dataset import Dataset
    from mpstwo.data.datastructs import Dict
    from mpstwo.model import MPSTwo
    from mpstwo.model.cutoff_schedulers import CutoffScheduler
    from mpstwo.model.optimizers import Optimizer
    from mpstwo.model.schedulers import Scheduler
    from mpstwo.utils.mapping import Map

logger = logging.getLogger(__name__)


class MPSTrainer:
    def __init__(
        self,
        model: MPSTwo,
        train_dataset: Dataset,
        optimizer: Optimizer,
        *,
        val_dataset: Optional[Dataset] = None,
        feature_map_obs: Optional[Map] = None,
        feature_map_act: Optional[Map] = None,
        key_mapping: Optional[dict[str, str]] = None,
        loss: Optional[Callable] = None,
        batch_size: int = 32,
        scheduler: Optional[Scheduler] = None,
        cutoff_scheduler: Optional[CutoffScheduler] = None,
        num_workers: int = 0,
        log_epoch: int = 1,
        save_epoch: int = 1,
        vis_batch_size: Optional[int] = None,
        custom_callback: Optional[
            Callable[[MPSTwo, torch.Tensor, torch.Tensor], list[tuple[str, Any, str]]]
        ] = None,
        log_dir_suffix: Optional[str] = None,
        device: str | torch.device = "cpu",
        log_to: str = "tensorboard",
        experiment_config: Optional[Dict] = None,
        **kwargs: Any,
    ) -> None:
        self.device = device

        self._model = model.to(self.device)

        # data parameters
        self._train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
        )
        self._val_loader = None
        if val_dataset is not None:
            self._val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                drop_last=True,
            )
        self.batch_size = batch_size
        self.feature_map_obs = feature_map_obs
        self.feature_map_act = feature_map_act

        if key_mapping is None:
            self.key_mapping = {"observation": "observation", "action": "action"}
        else:
            self.key_mapping = key_mapping

        # SGD parameters
        self.loss = NLLLoss()
        if loss:
            self.loss = loss
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._cutoff_scheduler = cutoff_scheduler

        # logging
        if log_dir_suffix is not None:
            log_dir = strftime(f"MPS_{log_dir_suffix}")
        else:
            log_dir = strftime("MPS_%Y%m%d_%H%M%S")
        self._log_dir = pathlib.Path("./" + log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Logging to {self._log_dir}")
        self._model_dir = self._log_dir / "model"
        self._model_dir.mkdir(parents=True, exist_ok=True)
        self._optim_dir = self._log_dir / "optimizer"
        self._optim_dir.mkdir(parents=True, exist_ok=True)

        self._log_epoch = log_epoch
        self._save_epoch = save_epoch
        self._vis_epoch = {
            "before_rate": self._log_epoch,
            "n": self._log_epoch * 25,
            "after_rate": self._log_epoch * 10,
        }

        if vis_batch_size is None:
            vis_batch_size = batch_size
        else:
            vis_batch_size = min(batch_size, vis_batch_size)
        self._vis_batch = {}
        self._vis_batch["train"] = next(iter(self._train_loader))[0:vis_batch_size, ...]
        if self._val_loader:
            self._vis_batch["val"] = next(iter(self._val_loader))[0:vis_batch_size, ...]

        log_to_tensorboard = log_to == "tensorboard"
        if not log_to_tensorboard:
            try:
                project = "tn"
                run = self._log_dir.name
                if log_dir_suffix is not None:
                    name_split = log_dir_suffix.split("_")
                    project = name_split[0]
                    run = "_".join(name_split[1:])

                if experiment_config is not None:
                    experiment_config = deepcopy(experiment_config)
                    experiment_config["model"].update(self._model.hyperparameters)

                wandb.init(
                    project=project,
                    name=run,
                    config=experiment_config,
                )

                self._log_callback = self._wandb_log_callback
            except Exception as err:
                # revert to tensorboard logging
                logger.warning(
                    f"Failed to initialize wandb: {err}, "
                    f"reverting to tensorboard logging."
                )
                log_to_tensorboard = True

        if log_to_tensorboard:
            train_log_dir = self._log_dir / "train"
            train_log_dir.mkdir(parents=True, exist_ok=True)
            self._train_writer = SummaryWriter(log_dir=train_log_dir)
            if self._val_loader is not None:
                val_log_dir = self._log_dir / "val"
                val_log_dir.mkdir(parents=True, exist_ok=True)
                self._val_writer = SummaryWriter(log_dir=val_log_dir)

            self._log_callback = self._tensorboard_log_callback

        self._custom_callback = custom_callback

    def train(self, epochs: int, start_epoch: int = 0) -> None:
        self._initial_log(start_epoch)

        for epoch in range(start_epoch + 1, start_epoch + epochs + 1):
            logger.info(f"{epoch}/{start_epoch + epochs}")
            train_logs = self._epoch()
            logger.info(
                f"Train loss: "
                f'{torch.tensor(train_logs.logs["Loss/loss"]["value"]).flatten().mean()}'
            )

            val_logs = None
            if self._val_loader:
                val_logs = self._validate()
                logger.info(
                    f"Validation loss: "
                    f'{torch.tensor(val_logs.logs["Loss/loss"]["value"]).flatten().mean()}'
                )

            self._schedule_callback(train_logs, val_logs, epoch)
            self._log_callback(train_logs, val_logs, epoch)
            self._save_model_callback(epoch)

    def _epoch(self) -> Log:
        """Train over training data"""
        logs = Log()
        for _, input_dict in enumerate(tqdm(self._train_loader)):
            obs = self.embed_input(input_dict[self.key_mapping["observation"]]).to(
                self.device
            )
            act = self.one_hot_input(input_dict[self.key_mapping["action"]]).to(
                self.device
            )
            self._model.update(obs, act, self._optimizer)
            probs = self._model(obs, act)
            self.loss(probs)
            logs += self.loss.logs
        return logs

    def _validate(self) -> Log:
        """Validate over val data"""
        logs = Log()
        for _, input_dict in enumerate(tqdm(self._val_loader)):
            obs = self.embed_input(input_dict[self.key_mapping["observation"]]).to(
                self.device
            )
            act = self.one_hot_input(input_dict[self.key_mapping["action"]]).to(
                self.device
            )
            probs = self._model(obs, act)
            self.loss(probs)
            logs += self.loss.logs
        return logs

    def _tensorboard_log_callback(
        self, train_logs: Log, val_logs: Optional[Log] = None, epoch: int = 0
    ) -> None:
        """Generic tensorboard logger

        Receives a dictionary with key: ( value, type )
        :param train_logs: logs should be written as above
        :param val_logs: logs should be written as above
        :param epoch: epoch number
        """
        vis_epoch = (
            self._vis_epoch["before_rate"]
            if epoch < self._vis_epoch["n"]
            else self._vis_epoch["after_rate"]
        )
        if epoch % vis_epoch == 0:
            train_logs, val_logs = self._log_visualization(train_logs, val_logs, epoch)

        if epoch % self._log_epoch == 0:
            train_logs.to_writer(self._train_writer, epoch)
            if val_logs is not None:
                val_logs.to_writer(self._val_writer, epoch)

            # Visualize learning rate if there is a scheduler
            if self._scheduler is not None:
                self._train_writer.add_scalar("Loss/lr", self._optimizer.lr, epoch)

    def _wandb_log_callback(
        self, train_logs: Log, val_logs: Optional[Log] = None, epoch: int = 0
    ) -> None:
        vis_epoch = (
            self._vis_epoch["before_rate"]
            if epoch < self._vis_epoch["n"]
            else self._vis_epoch["after_rate"]
        )
        if epoch % vis_epoch == 0:
            # Get visualization on eval-batch
            train_logs, val_logs = self._log_visualization(train_logs, val_logs, epoch)

        wandb_logs = {}
        if epoch % self._log_epoch == 0:
            wandb_logs.update(train_logs.to_wandb(prefix="train"))
            if val_logs is not None:
                wandb_logs.update(val_logs.to_wandb(prefix="val"))

            # Visualize learning rate if there is a scheduler
            if self._scheduler is not None:
                wandb_logs.update({"lr": self._optimizer.lr})

        wandb_logs["epoch"] = epoch
        wandb.log(wandb_logs)

    def _log_visualization(
        self,
        train_logs: Log,
        val_logs: Optional[Log] = None,
        epoch: int = 0,
    ) -> tuple[Log, Optional[Log]]:
        for i, d in enumerate(self._model.matrices[:-1]):
            train_logs.add("Bonds/bond_dim_" + str(i), d.shape[-1], "scalar")
        train_logs.add("Bonds/cutoff", self._model.cutoff, "scalar")

        if self._custom_callback is not None:
            _logs = "val" if val_logs is not None else "train"
            obs = self.embed_input(
                self._vis_batch[_logs][self.key_mapping["observation"]]
            ).to(self.device)
            act = self.one_hot_input(
                self._vis_batch[_logs][self.key_mapping["action"]]
            ).to(self.device)
            callback_info = self._custom_callback(self._model, obs, act)
            if val_logs is not None:
                for info in callback_info:
                    val_logs.add(*info)
            else:
                for info in callback_info:
                    train_logs.add(*info)

        return train_logs, val_logs

    def _save_model_callback(self, epoch: int) -> None:
        """Save all the information of the MPS into a folder

        Is called every epoch to store the model if epoch is a multiple of
        self._save_epoch
        :param epoch: current epoch during training
        """
        if epoch % self._save_epoch == 0:
            torch.save(
                self._model.clone().to("cpu"), self._model_dir / f"model-{epoch:04d}.pt"
            )
            torch.save(
                self._optimizer.state_dict(), self._optim_dir / f"optim-{epoch:04d}.pt"
            )

            if self._scheduler is not None:
                torch.save(
                    self._scheduler.state_dict(),
                    self._optim_dir / f"scheduler-{epoch:04d}.pt",
                )

            if self._cutoff_scheduler is not None:
                torch.save(
                    self._cutoff_scheduler.state_dict(),
                    self._optim_dir / f"cutoff_scheduler-{epoch:04d}.pt",
                )

    def _schedule_callback(
        self, train_logs: Log, val_logs: Optional[Log] = None, epoch: int = 0
    ) -> None:
        """
        Callback function for scheduling the optimizer learning rate
        :param train_logs: training logs can be used to determine new lr value
        :param val_logs: validation logs can be used to determine new lr value
        """
        if self._scheduler is not None:
            loss = torch.tensor(train_logs.logs["Loss/loss"]["value"]).flatten().mean()
            self._scheduler(loss.item())

        if self._cutoff_scheduler is not None:
            self._cutoff_scheduler(epoch)

    def _initial_log(self, start_epoch: int) -> None:
        # compute initial batch and log this
        # also visualize ground truth
        with torch.no_grad():
            train_logs = Log()
            obs = self.embed_input(
                self._vis_batch["train"][self.key_mapping["observation"]]
            ).to(self.device)
            act = self.one_hot_input(
                self._vis_batch["train"][self.key_mapping["action"]]
            ).to(self.device)
            probs = self._model(obs, act)
            self.loss(probs)
            train_logs += self.loss.logs
            if self._scheduler is not None:
                loss = (
                    torch.tensor(train_logs.logs["Loss/loss"]["value"]).flatten().mean()
                )
                self._scheduler(loss.item())
            if self._cutoff_scheduler is not None:
                self._cutoff_scheduler(start_epoch)

            val_logs = None
            if self._val_loader:
                val_logs = Log()
                obs = self.embed_input(
                    self._vis_batch["val"][self.key_mapping["observation"]]
                ).to(self.device)
                act = self.one_hot_input(
                    self._vis_batch["val"][self.key_mapping["action"]]
                ).to(self.device)
                probs = self._model(obs, act)
                self.loss(probs)
                val_logs += self.loss.logs

            self._log_callback(train_logs, val_logs, start_epoch)

    def embed_input(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Embed input_data into separate local feature spaces

        Args:
            input_data (Tensor): Input with shape [batch_size, model.physical_legs],
                                 or [batch_size, model.physical_legs, feature_dim_obs].
                                 In the latter case, the data is assumed to already
                                 be embedded, and is returned unchanged.

        Returns:
            embedded_data (Tensor): Input embedded into a tensor with shape
                                    [batch_size, model.physical_legs, feature_dim_obs]
        """
        assert input_data.dim() in [2, 3], input_data.shape
        assert input_data.size(1) == self._model.physical_legs

        # If input already has a feature dimension, return it as is
        if input_data.dim() == 3 and input_data.size(2) != 1:
            if input_data.size(2) != self._model.feature_dim_obs:
                raise ValueError(
                    f"input_data has wrong shape to be unembedded or pre-embedded data"
                    f" (input_data.shape = {list(input_data.shape)},"
                    f" feature_dim_obs = {self._model.feature_dim_obs})"
                )
            return input_data

        elif input_data.dim() == 3 and input_data.size(2) == 1:
            input_data = input_data.squeeze(2)

        # Apply a custom embedding map if it has been defined by the user
        if self.feature_map_obs is not None:
            embedded_data = self.feature_map_obs(input_data)

            # Make sure our embedded input has the desired size
            assert embedded_data.shape == torch.Size(
                [
                    input_data.size(0),
                    self._model.physical_legs,
                    self._model.feature_dim_obs,
                ]
            )

        # Otherwise, use a simple linear embedding map with feature_dim_obs = 2
        else:
            if self._model.feature_dim_obs != 2:
                raise RuntimeError(
                    f"self.feature_dim_obs = {self._model.feature_dim_obs}, "
                    "but default feature_map_obs requires self.feature_dim_obs = 2"
                )

            embedded_data = torch.stack([input_data, 1 - input_data], dim=2)

        return embedded_data

    def one_hot_input(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Convert a batch of numbers from the set {0, 1,..., num_value-1} into their
        one-hot encoded counterparts
        """
        assert input_data.dim() in [2, 3], input_data.shape
        assert input_data.size(1) == self._model.physical_legs

        # If input already has a one-hot dimension, return it as is
        if input_data.dim() == 3 and input_data.size(2) != 1:
            if input_data.size(2) != self._model.feature_dim_act:
                raise ValueError(
                    f"input_data has wrong shape to be unembedded or pre-embedded data"
                    f" (input_data.shape = {list(input_data.shape)},"
                    f" feature_dim_act = {self._model.feature_dim_act})"
                )
            return input_data

        # Apply a custom one-hot map if it has been defined by the user
        if self.feature_map_act is not None:
            one_hot_data = self.feature_map_act(input_data)

            # Make sure our embedded input has the desired size
            assert one_hot_data.shape == torch.Size(
                [
                    input_data.size(0),
                    self._model.physical_legs,
                    self._model.feature_dim_act,
                ]
            ), f"Input shape: {one_hot_data.shape}"

        # Otherwise, use a simple one-hot encoding
        else:
            one_hot_data = torch.zeros(
                [
                    input_data.size(0),
                    self._model.physical_legs,
                    self._model.feature_dim_act,
                ]
            )

            for i, sequence in enumerate(input_data):
                for j, action in enumerate(sequence):
                    one_hot_data[i, j, action] = 1.0

        return one_hot_data

    def close(self):
        """Close the trainer

        Used when training multiple times in the same process.
        """
        try:
            self._train_writer.close()
            self._val_writer.close()
        except AttributeError:
            pass

        try:
            wandb.finish()
        except ImportError:
            pass

    def load(self, checkpoint_epoch):
        """Load model and trainer

        Arguments:
            checkpoint_epoch -- epoch from which to restore
        """
        # load optimizer state
        optim_path = sorted(list(self._optim_dir.glob("optim*.pt")))[-1]
        optim_state_dict = torch.load(optim_path)
        self._optimizer.load_state_dict(optim_state_dict)

        # load scheduler state
        if self._scheduler:
            sched_path = sorted(list(self._optim_dir.glob("sched*.pt")))[-1]
            scheduler_state_dict = torch.load(sched_path)
            self._scheduler.load_state_dict(scheduler_state_dict)

        # load cutoff scheduler state
        if self._cutoff_scheduler:
            sched_path = sorted(list(self._optim_dir.glob("cutoff*.pt")))[-1]
            cutoff_state_dict = torch.load(sched_path)
            self._cutoff_scheduler.load_state_dict(cutoff_state_dict)
