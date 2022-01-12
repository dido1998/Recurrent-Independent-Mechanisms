import numpy as np
from torch.utils.data.sampler import BatchSampler, Sampler, SubsetRandomSampler
from typing import Iterator
#from ..prelude import Array

from typing import Any, Callable, Iterable, Sequence, Tuple, TypeVar, Union
from utils.device import Device


try:
    from typing import GenericMeta, NamedTupleMeta  # type: ignore

    class GenericNamedMeta(NamedTupleMeta, GenericMeta):
        pass
except ImportError:
    from typing import NamedTupleMeta  # type: ignore
    GenericNamedMeta = NamedTupleMeta  # type: ignore

T = TypeVar('T')
Self = Any 


class Array(Sequence[T]):
    @property
    def shape(self) -> tuple:
        ... 

    def squeeze(self) -> Self:
        ... 

    def transpose(self, *args) -> Self:
        ... 

    def __rsub__(self, value: Any) -> Self:
        ... 

    def __truediv__(self, rvalue: Any) -> Self:
        ... 



def sample_indices(n: int, k: int) -> np.ndarray:
    """Sample k numbers from [0, n)
       Based on https://github.com/chainer/chainerrl/blob/master/chainerrl/misc/random.py
    """
    if 3 * k >= n:
        return np.random.choice(n, k, replace=False)
    else:
        selected = np.repeat(False, n)
        rands = np.random.randint(0, n, size=k * 2)
        j = k
        for i in range(k):
            x = rands[i]
            while selected[x]:
                if j == 2 * k:
                    rands[k:] = np.random.randint(0, n, size=k)
                    j = k
                x = rands[i] = rands[j]
                j += 1
            selected[x] = True
        return rands[:k]


class FeedForwardBatchSampler(BatchSampler):
    def __init__(self, nsteps: int, nworkers: int, batch_size: int) -> None:
        super().__init__(
            SubsetRandomSampler(range(nsteps * nworkers)),
            batch_size=batch_size,
            drop_last=True
        )


class RecurrentBatchSampler(Sampler):
    def __init__(self, nsteps: int, nworkers: int, batch_size: int) -> None:
        if batch_size % nsteps > 0:
            raise ValueError('batch_size must be a multiple of nsteps')
        self.nsteps = nsteps
        self.nworkers = nworkers
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[Array[int]]:
        env_num = self.batch_size // self.nsteps
        total, step = self.nsteps * self.nworkers, self.nworkers
        perm = np.random.permutation(self.nworkers)
        for end in np.arange(env_num, self.nworkers + 1, env_num):
            workers = perm[end - env_num: end]
            batches = np.stack([np.arange(w, total, step) for w in workers], axis=1)
            yield batches.flatten()

    def __len__(self) -> int:
        return (self.nsteps * self.nworkers) // self.batch_size

