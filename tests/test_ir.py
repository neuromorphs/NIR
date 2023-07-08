import nir


def test_simple():
    ir = nir.NIR(nodes=[nir.Linear(weights=[1, 2, 3], bias=4)], edges=[(0, 0)])
    assert ir.nodes[0].weights == [1, 2, 3]
    assert ir.nodes[0].bias == 4
    assert ir.edges == [(0, 0)]


def test_simple_with_input_output():
    ir = nir.NIR(
        nodes=[
            nir.Input(
                shape=[
                    3,
                ]
            ),
            nir.Linear(weights=[1, 2, 3], bias=4),
            nir.Output(),
        ],
        edges=[(0, 1), (1, 2)],
    )
    assert ir.nodes[0].shape == [
        3,
    ]
    assert ir.nodes[1].weights == [1, 2, 3]
    assert ir.nodes[1].bias == 4
    assert ir.edges == [(0, 1), (1, 2)]


def test_delay():
    ir = nir.NIR(
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
    ir = nir.NIR(
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
