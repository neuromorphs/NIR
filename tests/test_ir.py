import nir


def test_simple():
    ir = nir.NIRGraph(nodes={0: nir.Affine(weight=[1, 2, 3], bias=4)}, edges=[(0, 0)])
    assert ir.nodes[0].weight == [1, 2, 3]
    assert ir.nodes[0].bias == 4
    assert ir.edges == [(0, 0)]


def test_nested():
    nested = nir.NIRGraph(
        nodes={
            "integrator": nir.I(r=[1, 1]),
            "delay": nir.Delay([2, 2]),
        },
        edges=[("integrator", "delay"), ("delay", "integrator")],
    )
    ir = nir.NIRGraph(
        nodes={"affine": nir.Affine(weight=[1, 2], bias=4), "inner": nested},
        edges=[("affine", "inner")],
    )
    assert ir.nodes["affine"].weight == [1, 2]
    assert ir.nodes["inner"].nodes["integrator"].r == [1, 1]
    assert ir.nodes["inner"].nodes["delay"].delay == [2, 2]
    assert ir.nodes["inner"].edges == [("integrator", "delay"), ("delay", "integrator")]


def test_simple_with_input_output():
    ir = nir.NIRGraph(
        nodes={
            "in": nir.Input(
                shape=[
                    3,
                ]
            ),
            "w": nir.Affine(weight=[1, 2, 3], bias=4),
            "out": nir.Output(),
        },
        edges=[("in", "w"), ("w", "out")],
    )
    assert ir.nodes["in"].shape == [
        3,
    ]
    assert ir.nodes["w"].weight == [1, 2, 3]
    assert ir.nodes["w"].bias == 4
    assert ir.edges == [("in", "w"), ("w", "out")]


def test_delay():
    ir = nir.NIRGraph(
        nodes={
            "in": nir.Input(
                shape=[
                    3,
                ]
            ),
            "d": nir.Delay(delay=[1, 2, 3]),
            "out": nir.Output(),
        },
        edges=[("in", "d"), ("d", "out")],
    )
    assert ir.nodes["in"].shape == [
        3,
    ]
    assert ir.nodes["d"].delay == [1, 2, 3]
    assert ir.edges == [("in", "d"), ("d", "out")]


def test_threshold():
    ir = nir.NIRGraph(
        nodes={
            "in": nir.Input(
                shape=[
                    3,
                ]
            ),
            "thr": nir.Threshold(threshold=[2.0, 2.5, 2.8]),
            "out": nir.Output(),
        },
        edges=[("in", "thr"), ("thr", "out")],
    )
    assert ir.nodes["in"].shape == [
        3,
    ]
    assert ir.nodes["thr"].threshold == [2.0, 2.5, 2.8]
    assert ir.edges == [("in", "thr"), ("thr", "out")]


def test_linear():
    ir = nir.NIRGraph(nodes={0: nir.Linear(weight=[1, 2, 3])}, edges=[(0, 0)])
    assert ir.nodes[0].weight == [1, 2, 3]
    assert ir.edges == [(0, 0)]
