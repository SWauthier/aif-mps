import logging
import os
from collections import deque
from os import PathLike
from typing import Any, Callable, Optional, Sequence

import h5py
import numpy as np
import torch
from typing_extensions import Self  # move to typing in Python 3.11

from mpstwo.data.datasets import SequenceDataset, retry
from mpstwo.data.datasets.dataset import Dataset
from mpstwo.data.datastructs import TensorDict

logger = logging.getLogger(__name__)


class MemoryPool(SequenceDataset):
    """MemoryPool stores sequences in a deque buffer in memory
    max_size specifies the maximum size of the buffer.
    The pool can be read / written from / to a directory of h5 files.
    """

    def __init__(
        self,
        max_size: Optional[int] = None,
        transform: Optional[Callable] = None,
        keys: Optional[str] = None,
        num_workers: int = 0,
        device: str | torch.device = "cpu",
        directory: Optional[str | PathLike] = None,
        interval: Optional[Sequence] = None,
        sequence_length: int = -1,
        sequence_stride: int = 1,
        shuffle: bool = False,
        cutoff: bool = True,
        **kwargs: Any,
    ) -> None:
        self._keys = keys
        self._buffer = deque(maxlen=max_size)
        self._device = torch.device(device)
        SequenceDataset.__init__(
            self,
            sequence_length,
            sequence_stride,
            shuffle,
            cutoff,
            transform,
            num_workers,
        )
        if directory:
            self.load(directory, interval)

    def _load_sequences(self):
        keys = list(range(len(self._buffer)))
        lengths = [self._buffer[i].shape[0] for i in range(len(self._buffer))]
        return keys, lengths

    def _load_sequence(self, key, indices):
        raw_sequence = self._buffer[key]
        if indices.dtype.kind == "u":
            indices = indices.astype(np.int64)
        indices = torch.as_tensor(indices, dtype=torch.long)
        return raw_sequence[indices]

    def push_no_update(self, sequence: TensorDict) -> None:
        s = TensorDict({})
        for key, value in sequence.items():
            if self._keys is None or key in self._keys:
                value = value.detach()
                value = value.to(self._device)
                # we squeeze the batch dimension here (if present)
                # so it can be added automatically when sampling
                if value.shape[0] == 1:
                    value = value.squeeze(0)
                s[key] = value

        self._buffer.append(s)
    
    def push(self, sequence: TensorDict) -> None:
        self.push_no_update(sequence)
        self._update_table()

    def dump(self, path: str | PathLike, compression: int = 9) -> None:
        if not os.path.exists(path):
            os.makedirs(path)

        for i in range(len(self._buffer)):
            file_name = str(i) + ".h5"
            out_path = os.path.join(path, file_name)
            sequence = self._buffer[i]
            with h5py.File(out_path, "w") as f:
                for key, value in sequence.items():
                    f.create_dataset(
                        key,
                        data=value,
                        compression="gzip",
                        compression_opts=compression,
                    )

    def load(self, path: str | PathLike, interval: Optional[Sequence] = None) -> Self:
        file_list = os.listdir(path)
        filtered_list = [item for item in file_list if ".h5" in item]

        # Train, validation split functionality
        if interval is None:
            interval = [0, 1.0]
        b = int(interval[0] * len(filtered_list))
        e = int(interval[1] * len(filtered_list))
        filtered_list = filtered_list[b:e]

        for name in filtered_list:
            file_path = os.path.join(path, name)
            d = self._load(file_path)
            self._buffer.append(TensorDict(d))

        self._update_table()
        return self

    @retry
    def _load(self, path: str | PathLike) -> dict:
        with h5py.File(path, "r") as f:
            # include all keys when no keys specified
            if not self._keys:
                keys = f.keys()
            else:
                keys = self._keys
            return {
                key: self._create_entry(value[:])
                for key, value in f.items()
                if key in keys
            }

    def _create_entry(self, value: np.ndarray) -> torch.Tensor:
        value = value[:]
        if value.dtype == "uint16":
            value = value.astype(np.int32)
        return torch.as_tensor(value).to(self._device)

    def wrap(self, pool: Dataset) -> Self:
        """Wraps a dataset into this MemoryPool.

        Arguments:
            pool -- an mpstwo.data.datasets.Dataset

        Returns:
            this MemoryPool
        """
        for i in range(len(pool)):
            sequence = pool[i].unsqueeze(0)
            if self._keys is not None:
                filtered = {}
                for k, v in sequence.items():
                    if k in self._keys:
                        filtered[k] = v.detach().clone()
                sequence = TensorDict(filtered)
            self.push(sequence)
        return self
