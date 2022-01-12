import numpy as np
from typing import Tuple
from ..prelude import Array


class RunningMeanStd(object):
    """From https://github.com/openai/baselines/blob/master/baselines/common/running_mean_std.py
    """
    def __init__(self, epsilon: float = 1.0e-4, shape: Tuple[int, ...] = ()) -> None:
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x: Array[float]) -> None:
        self.mean, self.var, self.count = _update_mean_var_count_from_moments(
            self.mean,
            self.var,
            self.count,
            np.mean(x, axis=0),
            np.var(x, axis=0),
            x.shape[0]
        )

    def std(self, eps: float = 1.0e-8) -> Array[float]:
        return np.sqrt(self.var + eps)


def _update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count
    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count
    return new_mean, new_var, new_count
