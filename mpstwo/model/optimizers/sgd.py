from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from mpstwo.model.optimizers.optimizer import Optimizer
from mpstwo.utils.operations import contract, normalize

if TYPE_CHECKING:
    from mpstwo.model import MPSTwo


class SGD(Optimizer):
    def __init__(
        self,
        model: MPSTwo,
        lr: float,
        two_site: bool = True,
        cumulants: bool = True,
    ) -> None:
        super().__init__(model, lr)
        self.cumulants = cumulants
        self.two_site = two_site

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        if self.cumulants:
            return self._sgd_cumulants(*args, **kwds)
        else:
            return self._sgd(*args, **kwds)

    def _sgd(self) -> torch.Tensor:
        raise NotImplementedError

    def _sgd_cumulants(
        self,
        m1: torch.Tensor,
        m2: torch.Tensor,
        obs: torch.Tensor,
        act: torch.Tensor,
        left: torch.Tensor,
        right: torch.Tensor,
        going_right: bool,
    ) -> torch.Tensor:
        """Gradient descent with cumulants"""
        merged_matrix = contract(m1, m2)
        gradient = self._gradient(merged_matrix, obs, act, left, right)

        if self.two_site:
            updated_matrix = merged_matrix - self.lr * 2 * gradient
        else:
            if going_right:
                gradient = torch.einsum("laocpr,mcpr->laom", gradient, m2.conj())
            else:
                gradient = torch.einsum("laocpr,laos->scpr", gradient, m1.conj())
            updated_matrix = (m1 if going_right else m2) - self.lr * 2 * gradient
            updated_matrix = normalize(updated_matrix)

        return updated_matrix

    def _gradient(
        self,
        merged_matrix: torch.Tensor,
        obs: torch.Tensor,
        act: torch.Tensor,
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> torch.Tensor:
        psi = self.model.get_psi_cumulants(merged_matrix, obs, act, left, right)
        psi_prime = self.model.get_psi_prime_cumulants(obs, act, left, right)
        if (psi == 0).sum():
            print(
                f"Error: With batch_size={psi.shape[0]}, "
                f"{(psi == 0).sum()} psi's == 0. "
                f"At position(s):\n\t"
                f"{(psi == 0).nonzero().ravel().cpu().numpy()}"
            )
            raise ZeroDivisionError("Some psi's == 0")
        Z_prime = merged_matrix.clone()

        batch_dim = psi_prime.shape[0]
        return Z_prime.add_(
            -psi_prime.conj().reshape(batch_dim, -1)
            .div_(psi.conj().reshape(batch_dim, 1))
            .mean(0).reshape_as(Z_prime)
        )

    def state_dict(self) -> dict:
        d = super().state_dict()
        d["cumulants"] = self.cumulants
        d["two_site"] = self.two_site
        return d
