from mpstwo.model.cutoff_schedulers.aggregate_cutoff_scheduler import (
    AggregateCutoffScheduler,
)
from mpstwo.model.cutoff_schedulers.cutoff_scheduler import CutoffScheduler
from mpstwo.model.cutoff_schedulers.linear import Linear
from mpstwo.model.cutoff_schedulers.step import Step
from mpstwo.model.mpstwo import MPSTwo


def cutoff_scheduler_factory(model: MPSTwo, definition: dict):
    schedulers = []
    for c, kwargs in definition.items():
        cls = globals()[c]
        schedulers.append(cls(model, **kwargs))
    return AggregateCutoffScheduler(*schedulers)
