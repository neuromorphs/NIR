import tempfile

import numpy as np

import nir


def test_simple():
    tmp = tempfile.mktemp()
    ir = nir.NIR(units=[nir.Linear(weights=[1, 2, 3], bias=4)], connectivity=[(0, 0)])
    nir.write(tmp, ir)
    ir2 = nir.read(tmp)
    assert np.array_equal(ir.units[0].weights, ir2.units[0].weights)
    assert np.array_equal(ir.units[0].bias, ir2.units[0].bias)
    assert np.array_equal(ir.connectivity, ir2.connectivity)
