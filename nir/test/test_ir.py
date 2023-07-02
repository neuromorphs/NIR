import nir


def test_simple():
    ir = nir.NIR(nodes=[nir.Linear(weights=[1, 2, 3], bias=4)], edges=[(0, 0)])
    assert ir.nodes[0].weights == [1, 2, 3]
    assert ir.nodes[0].bias == 4
    assert ir.edges == [(0, 0)]
