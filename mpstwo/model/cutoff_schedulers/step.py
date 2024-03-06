import logging

from mpstwo.model.cutoff_schedulers.cutoff_scheduler import CutoffScheduler
from mpstwo.model.mpstwo import MPSTwo

logger = logging.getLogger(__name__)


class Step(CutoffScheduler):
    def __init__(self, model: MPSTwo, epoch: int, new_value: float) -> None:
        super().__init__(model)
        self.epoch = epoch
        self.new_value = new_value

    def step(self, epoch: int) -> None:
        if epoch > self.epoch:
            self.model.cutoff = self.new_value

    def state_dict(self) -> dict:
        d = super().state_dict()
        d["epoch"] = self.epoch
        d["new_value"] = self.new_value
        return d
