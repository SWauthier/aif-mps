from abc import ABC, abstractmethod
from functools import wraps
from typing import Any, Callable, Optional

import numpy as np
import torch.utils.data as data

from mpstwo.data.datastructs import TensorDict


class Dataset(data.Dataset, ABC):
    """An abstract class representing a dommel Dataset

    Dommel datasets differ with PyTorch datasets in that they return
    TensorDicts instead of tensor tuples.

    In addition to inheriting from torch's Dataset, it also adds methods to
    sample a random batch of data. Sampling is implemented by default using
    a DataLoader, but you can also use your own DataLoader.
    """

    def __init__(
        self, transform: Optional[Callable] = None, num_workers: int = 0, **kwargs: Any
    ) -> None:
        self._num_workers = num_workers
        self._batch_size = 0
        self._dataloader = None
        self._transform = transform

    def __getitem__(self, index: int) -> TensorDict:
        if index >= len(self):
            raise IndexError(f"Index out of bounds: {index}")

        item = self._get_item(index)

        if self._transform:
            item = self._transform(item)

        return item

    def sample(self, batch_size: int = 1) -> TensorDict:
        """returns a dictionary of tensors
        (shape = (batch_size, sequence_length))
        """
        if batch_size != self._batch_size:
            # reinstantiate the DataLoader if the batch size changes during
            # calls (should we better fix batch_size at construction?)
            self._batch_size = batch_size
            self._create_dataloader()
        try:
            x = next(self._itr)
        except StopIteration:
            self._create_dataloader()
            x = next(self._itr)
        if x[list(x.keys())[0]].shape[0] != self._batch_size:
            # recreate the dataloader and resample!
            self._create_dataloader()
            x = next(self._itr)
        return TensorDict(x)

    def _create_dataloader(self) -> None:
        self._dataloader = data.DataLoader(
            self,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=self._num_workers,
            drop_last=True,
        )
        self._itr = iter(self._dataloader)

    @abstractmethod
    def _get_item(self, idx: int | slice | tuple[int, ...]) -> TensorDict:
        """get item at index idx"""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """get the length of the dataset"""
        pass


class SequenceDataset(Dataset, ABC):
    """SequenceDataset is a dommel dataset that handles sequences of data

    Subsequences of sequence_length and stride can be sampled, and also
    sequence indices can be shuffled.

    If cutoff is False, the last subsequence with length < sequence_length
    is also kept.
    """

    def __init__(
        self,
        sequence_length: int = -1,
        sequence_stride: int = 1,
        shuffle: bool = False,
        cutoff: bool = True,
        transform: Optional[Callable] = None,
        num_workers: int = 0,
        **kwargs: Any,
    ):
        Dataset.__init__(self, transform, num_workers, **kwargs)
        self._sequence_length = sequence_length
        self._sequence_stride = sequence_stride
        self._cutoff = cutoff
        if self._sequence_stride == 0:
            raise ValueError("Invalid sequence_stride: 0")

        self._shuffle = shuffle
        self._update_table()

    @abstractmethod
    def _load_sequences(self) -> tuple[list, list]:
        """return a list with a key for each raw sequence
        and a list with their corresponding lengths"""
        pass

    @abstractmethod
    def _load_sequence(self, key: int, indices: np.ndarray) -> TensorDict:
        """return a tensordict with a key and indices"""
        pass

    def _update_table(self) -> None:
        self._sequences, sequence_lengths = self._load_sequences()
        if self._sequence_length > 0:
            extra = 0 if self._cutoff else 1
            total_indices = sum(
                max(length - self._sequence_length + extra + 1, 0)
                for length in sequence_lengths
            )
            # generate the table
            table = np.zeros(
                (total_indices, 1 + self._sequence_length), dtype=np.uint32
            )
            i = 0
            if not self._shuffle:
                for j, length in enumerate(sequence_lengths):
                    k = 0
                    while k + self._sequence_length <= length:
                        indices = np.arange(k, k + self._sequence_length)
                        table[i] = np.array([j, *indices], dtype=np.uint32)
                        i += 1
                        if self._sequence_stride > 0:
                            k += self._sequence_stride
                        else:
                            o = self._sequence_length + self._sequence_stride
                            k += o
                    if not self._cutoff:
                        indices = np.arange(k, length)
                        fill = np.empty((k + self._sequence_length - length), np.uint32)
                        fill[:] = np.iinfo(np.uint32).max
                        table[i] = np.array([j, *indices, *fill], dtype=np.uint32)
                        i += 1
                self._lookup_table = table[0:i]
            else:
                for j, length in enumerate(sequence_lengths):
                    r = length - self._sequence_length
                    for k in range(r):
                        indices = np.random.choice(
                            np.arange(0, length), self._sequence_length, replace=False
                        )
                        table[i] = np.array([j, *indices], dtype=np.uint32)
                        i += 1
                self._lookup_table = table
        else:
            table = {
                i: np.array([i] + list(range(0, length)))
                for i, length in enumerate(sequence_lengths)
            }
            self._lookup_table = table

    def __len__(self) -> int:
        return len(self._lookup_table)

    def _get_item(self, idx: int) -> TensorDict:
        lookup = self._lookup_table[idx]
        key = self._sequences[lookup[0]]
        indices = lookup[1:]
        indices = indices[np.where(indices != np.iinfo(np.uint32).max)]
        return self._load_sequence(key, indices)


class ConcatDataset(data.ConcatDataset, Dataset):
    """Wraps Pytorch ConcatDataset as a dommel dataset,
    providing the sample() method for convenience
    """

    def __init__(self, datasets: list[data.Dataset]) -> None:
        Dataset.__init__(self, None, 0)
        data.ConcatDataset.__init__(self, datasets)


def retry(func):
    """
    A Decorator to retry a function for a certain amount of attempts
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        attempts = 0
        max_attempts = 100
        last_error = ""
        while attempts < max_attempts:
            try:
                return func(*args, **kwargs)
            except (OSError, PermissionError) as e:
                attempts += 1
                last_error = e
        raise OSError(f"Retry failed: {last_error}")

    return wrapper
