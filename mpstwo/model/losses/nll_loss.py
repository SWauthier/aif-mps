import torch

from mpstwo.model.log import Log


class NLLLoss:
    def __init__(self, aggregate=torch.mean) -> None:
        self._loss = torch.zeros(0)
        self._aggregate = aggregate

    def __call__(self, probs: torch.Tensor) -> torch.Tensor:
        """Calculate the NLL averaged on the training set"""
        self._loss = -self._aggregate(torch.log(probs))
        return self._loss

    @property
    def logs(self):
        """
        :return: the logs for all components in this loss function
        """
        log = Log()
        if self._aggregate is not None:
            log.add("Loss/loss", self._loss.detach().cpu(), "scalar")
        else:
            for v in self._loss.detach().cpu():
                log.add("Loss/loss", v, "scalar")
        return log
