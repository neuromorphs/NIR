import numpy as np
import pytest

import importlib
_HAS_TORCH = importlib.find_loader('torch') is not None

from nir.ir.utils import _index_tuple


def test_index_tuple():
    assert _index_tuple(1, 0) == 1
    assert _index_tuple([1, 2], 0) == 1
    assert _index_tuple(np.array([1, 2]), 0) == 1
    assert np.all(
        np.equal(_index_tuple(np.array([[1, 2], [3, 4]]), 1), np.array([3, 4]))
    )

@pytest.mark.skipif(not _HAS_TORCH, reason="requires torch")
def test_index_tuple_torch():
    import torch
    assert _index_tuple(torch.tensor([1, 2]), 0) == 1
    assert _index_tuple(torch.tensor([[1, 2], [3, 4]]), 1).equal(torch.tensor([3, 4]))