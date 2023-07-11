import nir


def test_simple():
    ir = nir.NIRGraph(nodes=[nir.Affine(weight=[1, 2, 3], bias=4)], edges=[(0, 0)])
    assert ir.nodes[0].weight == [1, 2, 3]
    assert ir.nodes[0].bias == 4
    assert ir.edges == [(0, 0)]


def test_nested():
    nested = nir.NIRGraph(
        nodes=[
            nir.I(r=[1, 1]),
            nir.Delay([2, 2]),
        ],
        edges=[(0, 1), (1, 0)]
    )
    ir = nir.NIRGraph(
        nodes=[nir.Affine(weight=[1, 2], bias=4), nested],
        edges=[(0, 1)],
    )
    assert ir.nodes[0].weight == [1, 2]
    assert ir.nodes[1].nodes[0].r == [1, 1]
    assert ir.nodes[1].nodes[1].delay == [2, 2]
    assert ir.nodes[1].edges == [(0, 1), (1, 0)]


def test_simple_with_input_output():
    ir = nir.NIRGraph(
        nodes=[
            nir.Input(
                shape=[
                    3,
                ]
            ),
            nir.Affine(weight=[1, 2, 3], bias=4),
            nir.Output(),
        ],
        edges=[(0, 1), (1, 2)],
    )
    assert ir.nodes[0].shape == [
        3,
    ]
    assert ir.nodes[1].weight == [1, 2, 3]
    assert ir.nodes[1].bias == 4
    assert ir.edges == [(0, 1), (1, 2)]


def test_delay():
    ir = nir.NIRGraph(
        nodes=[
            nir.Input(
                shape=[
                    3,
                ]
            ),
            nir.Delay(delay=[1, 2, 3]),
            nir.Output(),
        ],
        edges=[(0, 1), (1, 2)],
    )
    assert ir.nodes[0].shape == [
        3,
    ]
    assert ir.nodes[1].delay == [1, 2, 3]
    assert ir.edges == [(0, 1), (1, 2)]


def test_threshold():
    ir = nir.NIRGraph(
        nodes=[
            nir.Input(
                shape=[
                    3,
                ]
            ),
            nir.Threshold(threshold=[2.0, 2.5, 2.8]),
            nir.Output(),
        ],
        edges=[(0, 1), (1, 2)],
    )
    assert ir.nodes[0].shape == [
        3,
    ]
    assert ir.nodes[1].threshold == [2.0, 2.5, 2.8]
    assert ir.edges == [(0, 1), (1, 2)]


def test_linear():
    ir = nir.NIRGraph(nodes=[nir.Linear(weight=[1, 2, 3])], edges=[(0, 0)])
    assert ir.nodes[0].weight == [1, 2, 3]
    assert ir.edges == [(0, 0)]
