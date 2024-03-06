from typing import Optional

import torch


def contract(*t: torch.Tensor) -> torch.Tensor:
    if len(t) > 2:
        return torch.tensordot(t[0], contract(*t[1:]), dims=1)
    elif len(t) == 2:
        return torch.tensordot(t[0], t[1], dims=1)
    elif len(t) == 1:  # no effect
        return t[0]
    else:
        raise ValueError("contract requires inputs")


def normalize(m: torch.Tensor) -> torch.Tensor:
    return m / torch.linalg.norm(m)


def transfer_normalize(t: torch.Tensor, idx: Optional[int] = None, eps: float = 1e-20):
    dims = list(range(t.dim()))
    if idx is not None:
        dims.remove(idx)
    dims = tuple(dims)
    cond = (torch.abs(t) > eps).sum(dims, keepdim=True).type(torch.bool)
    norm = torch.linalg.vector_norm(t, dim=dims, keepdim=True)
    t /= torch.where(cond, norm, 1)
    return t


def outer(*m: torch.Tensor) -> torch.Tensor:
    """Outer product of a series of vectors with leading batch dimensions."""
    r = torch.ones(1, device=m[0].device)
    for a in m:
        r = torch.einsum("b...,bv->b...v", r, a)
    return r
