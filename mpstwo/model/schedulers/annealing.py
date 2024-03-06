import logging
from collections import deque
from statistics import mean, stdev

from mpstwo.model.optimizers import Optimizer
from mpstwo.model.schedulers.scheduler import Scheduler

logger = logging.getLogger(__name__)


class Annealing(Scheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        change_factor: float = 2,
        histlen: int = 50,
        small_sd_threshold: float = 0.001,
        timer: int = 25,
    ) -> None:
        """A learning rate scheduler performing annealing

        Arguments:
            optimizer -- the optimizer

        Keyword Arguments:
            change_factor -- factor to be multiplied or divided (default: {2})
            maxlr -- maximum learning rate (default: {2e-4})
            histlen -- length of history to take into account (default: {50})
            small_sd_threshold -- low stdev threshold (default: {0.01})
            large_sd_threshold -- high stdev threshold (default: {0.5})
            timer -- how long to keep lr high when annealing (default: {10})
        """
        super().__init__(optimizer)
        self._loss_history = deque(maxlen=histlen)

        self.change_factor = change_factor
        self.small_sd_threshold = small_sd_threshold
        self.timer = timer - 1
        self.i_timer = -1

    def step(self, loss: float, *args, **kwargs) -> None:
        if len(self._loss_history) == self._loss_history.maxlen:
            hist_mean = mean(self._loss_history)

            if self.i_timer > 0:
                self.i_timer -= 1

            elif self.i_timer == 0:
                self.i_timer -= 1
                self.optimizer.lr /= self.change_factor
                self._loss_history.clear()

            elif self.small_sd() and self.within_99p(loss, hist_mean):
                self.optimizer.lr *= self.change_factor
                self.i_timer = self.timer
                self.small_sd_threshold /= self.change_factor

        self._loss_history.append(loss)

    def within_99p(self, x: float, mu: float) -> bool:
        sd = stdev(self._loss_history)
        z = (x - mu) / sd
        return abs(z) < 3

    def small_sd(self) -> bool:
        sd = stdev(self._loss_history)
        return sd < self.small_sd_threshold

    def state_dict(self) -> dict:
        d = super().state_dict()
        d["_loss_history"] = self._loss_history
        d["change_factor"] = self.change_factor
        d["small_sd_threshold"] = self.small_sd_threshold
        d["timer"] = self.timer
        d["i_timer"] = self.i_timer
        return d
