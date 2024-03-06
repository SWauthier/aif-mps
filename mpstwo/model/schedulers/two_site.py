import logging

from mpstwo.model.optimizers import Optimizer
from mpstwo.model.schedulers.scheduler import Scheduler

logger = logging.getLogger(__name__)


class TwoSite(Scheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        frequency: int = 10,
        first_epoch_update = True,
    ) -> None:
        """A scheduler for two-site updates

        Arguments:
            optimizer -- the optimizer

        Keyword Arguments:
            frequency -- frequency of two-site updates (default: {10})
        """
        super().__init__(optimizer)
        self.optimizer.two_site = False
        self.frequency = frequency
        self._counter = int(not first_epoch_update)

    def step(self, *args, **kwargs) -> None:
        if self._counter % self.frequency == 0:
            self.optimizer.two_site = True
            self._counter = 1
        else:
            self.optimizer.two_site = False
            self._counter += 1

    def state_dict(self) -> dict:
        d = super().state_dict()
        d["frequency"] = self.frequency
        d["_counter"] = self._counter
        return d
