import nir


def test_simple():
    ir = nir.NIR(nodes=[nir.Linear(weights=[1, 2, 3], bias=4)], edges=[(0, 0)])
    assert ir.nodes[0].weights == [1, 2, 3]
    assert ir.nodes[0].bias == 4
    assert ir.edges == [(0, 0)]

def test_simple_with_input_output():
    ir = nir.NIR(
        nodes=[
            nir.Input(shape=[3,]),
            nir.Linear(weights=[1, 2, 3], bias=4),
            nir.Output()],
        edges=[(0, 1), (1,2)])
    assert ir.nodes[0].shape == [3,]
    assert ir.nodes[1].weights == [1, 2, 3]
    assert ir.nodes[1].bias == 4
    assert ir.edges == [(0, 1), (1, 2)]

