import logging

from mpstwo.model.optimizers import Optimizer
from mpstwo.model.schedulers.scheduler import Scheduler

logger = logging.getLogger(__name__)


class Han(Scheduler):
    def __init__(
        self, optimizer: Optimizer, safe_loss_threshold: float, lr_shrink_rate: float
    ) -> None:
        super().__init__(optimizer)
        self._prev_loss = None

        self.safe_loss_threshold = safe_loss_threshold
        self.lr_shrink_rate = lr_shrink_rate

    def step(self, loss: float, *args, **kwargs) -> None:
        if (
            self._prev_loss is not None
            and loss - self._prev_loss > self.safe_loss_threshold
        ):
            logger.info(
                f"lr={self.optimizer.lr:1.3e} " f"is too large to continue safely"
            )
            self.optimizer.lr *= self.lr_shrink_rate

            if self.optimizer.lr < 1e-10:
                logger.info(
                    f"Learning rate has become negligible: " f"{self.optimizer.lr:1.3e}"
                )
                raise ValueError("Learning rate has become negligible")
        self._prev_loss = loss

    def state_dict(self) -> dict:
        d = super().state_dict()
        d["_prev_loss"] = self._prev_loss
        d["safe_loss_threshold"] = self.safe_loss_threshold
        d["lr_shrink_rate"] = self.lr_shrink_rate
        return d
