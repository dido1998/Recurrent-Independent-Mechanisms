import numpy as np
from .sample import sample_indices


def test_sample_large():
    arr = sample_indices(100, 90)
    assert np.unique(arr).__len__() == 90


def test_sample_small():
    arr = sample_indices(100, 30)
    assert np.unique(arr).__len__() == 30
