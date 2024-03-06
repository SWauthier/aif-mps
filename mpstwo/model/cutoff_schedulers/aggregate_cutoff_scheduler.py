from typing import Any

from mpstwo.model.cutoff_schedulers.cutoff_scheduler import CutoffScheduler


class AggregateCutoffScheduler(CutoffScheduler):
    def __init__(self, *scheduler: CutoffScheduler) -> None:
        self.schedulers = scheduler

    def step(self, *args, **kwargs) -> Any:
        return [s(*args, **kwargs) for s in self.schedulers]

    def state_dict(self) -> dict:
        return {i: s.state_dict() for i, s in enumerate(self.schedulers)}

    def load_state_dict(self, state_dict: dict) -> None:
        for i, s in enumerate(self.schedulers):
            s.load_state_dict(state_dict[i])
