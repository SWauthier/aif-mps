from __future__ import annotations

from typing import Sequence, overload

import torch
from typing_extensions import Self  # move to typing in Python 3.11

from mpstwo.data.datastructs import Dict


class TensorDict(Dict):
    """Separate datastructure for mapping from str to torch.Tensor
    Allow to index both on key and slice
    """

    @overload
    def __getitem__(self, __key: str) -> torch.Tensor:
        ...

    @overload
    def __getitem__(self, __key: int | slice | Sequence[int | slice]) -> TensorDict:
        ...

    def __getitem__(
        self, __key: str | int | slice | Sequence[int | slice]
    ) -> torch.Tensor | TensorDict:
        if isinstance(__key, str):
            return dict.__getitem__(self, __key)
        else:
            item = {k: dict.__getitem__(self, k)[__key] for k in self.keys()}
            return TensorDict(item)

    def to(self, other: str | torch.device | torch.dtype | torch.Tensor) -> Self:
        for key, value in self.items():
            if value is not None:
                self[key] = value.to(other)
        return self

    def detach(self) -> Self:
        for key, value in self.items():
            if value is not None:
                self[key] = value.detach()
        return self

    def squeeze(self, dim: int) -> TensorDict:
        """Squeeze a dimension of the TensorDict"""
        d = {key: _check_type(value, value.squeeze(dim)) for key, value in self.items()}
        return TensorDict(d)

    def unsqueeze(self, dim: int) -> TensorDict:
        """Unsqueeze a dimension of the TensorDict"""
        d = {
            key: _check_type(value, value.unsqueeze(dim)) for key, value in self.items()
        }
        return TensorDict(d)

    def repeat_interleave(self, repeats: int | torch.Tensor, dim: int) -> TensorDict:
        d = {key: value.repeat_interleave(repeats, dim) for key, value in self.items()}
        return TensorDict(d)

    def clone(self) -> TensorDict:
        clone = {key: value.clone() for key, value in self.items()}
        return TensorDict(clone)

    @property
    def shape(self) -> tuple[int, ...]:
        """Get first two dimension shapes of the TensorDict values
        :return: first two dimensions, assuming these are batch and time
        """
        key = list(self.keys())[0]
        return tuple(self[key].shape[:2])

    @property
    def device(self) -> torch.device:
        key = list(self.keys())[0]
        return self[key].device

    def __eq__(self, __o: TensorDict) -> bool:
        if self.keys() != __o.keys():
            return False
        for key in self.keys():
            if not torch.equal(self[key], __o[key]):
                return False
        return True

    def __hash__(self):
        return hash(
            tuple((k, tuple(self[k].flatten().tolist())) for k in sorted(self.keys()))
        )


def cat(*dicts: TensorDict) -> TensorDict:
    """Merge TensorDicts in the time dimension"""
    if len(dicts) == 1:
        return dicts[0]
    merged = {}
    s1 = dicts[0]
    for i in range(1, len(dicts)):
        s2 = dicts[i]
        for key, value in s1.items():
            if key in s2:
                if value.shape[0] != s2[key].shape[0]:
                    # repeat in batch dimension if shapes are not the same
                    factor = int(s2[key].shape[0] / value.shape[0])
                    sizes = [factor]
                    for _ in range(value.dim() - 1):
                        sizes.append(1)
                    value = value.repeat(sizes)
                merged_value = torch.cat((value, s2[key]), dim=1)
                merged[key] = _check_type(value, merged_value)
            else:
                merged[key] = value
        s1 = merged

    return TensorDict(merged)


def stack(*dicts: TensorDict) -> TensorDict:
    """Stack TensorDicts in the batch dimension"""
    if len(dicts) == 1:
        return dicts[0]
    merged = {}
    s1 = dicts[0]
    for i in range(1, len(dicts)):
        s2 = dicts[i]
        for key, value in s1.items():
            if key in s2:
                merged_value = torch.cat((value, s2[key]), dim=0)
                merged[key] = _check_type(value, merged_value)
            else:
                merged[key] = value
        s1 = merged
    return TensorDict(merged)


def _check_type(value, result):
    """Helper function to make sure distributions are also
    converted correctly"""
    if value.__class__.__name__ != "Tensor":
        constructor = globals()[value.__class__.__name__]
        return constructor(result)
    return result
