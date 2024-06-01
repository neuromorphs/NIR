import numpy as np
import pytest

from nir.ir.utils import _index_tuple

import importlib

_TORCH_SPEC = importlib.util.find_spec("torch") is not None


def test_index_tuple():
    assert _index_tuple(1, 0) == 1
    assert _index_tuple([1, 2], 0) == 1
    assert _index_tuple(np.array([1, 2]), 0) == 1
    assert np.all(
        np.equal(_index_tuple(np.array([[1, 2], [3, 4]]), 1), np.array([3, 4]))
    )


@pytest.mark.skipif(_TORCH_SPEC is not None, reason="requires torch")
def test_index_tuple_torch():
    torch = _TORCH_SPEC.loader.load_module()
    assert _index_tuple(torch.tensor([1, 2]), 0) == 1
    assert _index_tuple(torch.tensor([[1, 2], [3, 4]]), 1).equal(torch.tensor([3, 4]))
