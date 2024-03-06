import logging

from mpstwo.model.cutoff_schedulers.cutoff_scheduler import CutoffScheduler
from mpstwo.model.mpstwo import MPSTwo

logger = logging.getLogger(__name__)


class Linear(CutoffScheduler):
    def __init__(self, model: MPSTwo, epoch: int, new_value: float, steps: int) -> None:
        super().__init__(model)
        self.epoch = epoch
        self.new_value = new_value
        self.steps = steps

    def step(self, epoch: int) -> None:
        if self.steps > epoch - self.epoch > 0:
            self.model.cutoff -= (self.model.cutoff - self.new_value) / (
                self.steps - (epoch - self.epoch)
            )

    def state_dict(self) -> dict:
        d = super().state_dict()
        d["epoch"] = self.epoch
        d["new_value"] = self.new_value
        d["steps"] = self.steps
        return d
