import numpy as np

from nir.ir.utils import _index_tuple


def test_index_tuple():
    assert _index_tuple(1, 0) == 1
    assert _index_tuple([1, 2], 0) == 1
    assert _index_tuple(np.array([1, 2]), 0) == 1
    assert np.all(
        np.equal(_index_tuple(np.array([[1, 2], [3, 4]]), 1), np.array([3, 4]))
    )
