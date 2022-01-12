from functools import reduce
import operator
from torch import Tensor
from typing import Any, Iterable


def iter_prod(it: Iterable[Any]) -> Any:
    return reduce(operator.mul, it)


def normalize_(t: Tensor, eps: float) -> None:
    mean = t.mean()
    std = t.std()
    t.sub_(mean).div_(std + eps)
