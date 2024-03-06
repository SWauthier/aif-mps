from __future__ import annotations

from typing import Any

from typing_extensions import Self  # move to typing in Python 3.11


class Dict(dict[str, Any]):
    """Dommel dictionary implementation.

    Adds:
     - dot.notation access to dictionary attributes
     - update of nested dictionaries
    """

    def __getitem__(self, __key: str) -> Any:
        __value = super().__getitem__(__key)
        if isinstance(__value, dict):
            __value = Dict(__value)
            self[__key] = __value
        return __value

    def __getattr__(self, __name: str) -> Any:
        try:
            return self.__getitem__(__name)
        except KeyError:
            raise AttributeError(__name)

    def __setattr__(self, __name: str, __value: Any) -> None:
        return super().__setitem__(__name, __value)

    def __delattr__(self, __name: str) -> None:
        return super().__delitem__(__name)

    def __getstate__(self) -> Self:
        return self

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.update(state)

    def update(self, *__map: dict[str, Any], **kwargs: Any) -> None:
        """Update the self dict.

        :param dicts: optional list of dictionaries to include
        :param other: key value pairs to include
        The original dict is overwritten left to right, so the right most dict
        overwrites te left most, and key values overwrite that again.
        """
        others = {}
        for __m in __map:
            others.update(__m)
        others.update(kwargs)
        for k, v in others.items():
            if isinstance(v, dict):
                if k in self.keys() and isinstance(self[k], dict):
                    self[k].update(v)
                else:
                    self[k] = Dict(v)
            else:
                self[k] = v

    def dict(self) -> dict[str, Any]:
        """Convert to regular dict (e.g. to export to file)."""
        result = dict()
        for k, v in self.items():
            if type(v) is Dict:
                v = v.dict()
            result[k] = v
        return result
