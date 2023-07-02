import nir


def test_simple():
    ir = nir.NIR(units=[nir.Linear(weights=[1, 2, 3], bias=4)], connectivity=[(0, 0)])
    assert ir.units[0].weights == [1, 2, 3]
    assert ir.units[0].bias == 4
    assert ir.connectivity == [(0, 0)]
