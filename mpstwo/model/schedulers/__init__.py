from mpstwo.model.optimizers.optimizer import Optimizer

from .aggregate_scheduler import AggregateScheduler
from .annealing import Annealing
from .han import Han
from .scheduler import Scheduler
from .two_site import TwoSite


def scheduler_factory(optimizer: Optimizer, definition: dict):
    schedulers = []
    for c, kwargs in definition.items():
        cls = globals()[c]
        schedulers.append(cls(optimizer, **kwargs))
    return AggregateScheduler(*schedulers)
