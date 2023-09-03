import numpy as np

import nir
from tests import *


def mock_linear(*shape):
    return nir.Linear(weight=np.random.randn(*shape).T)


def mock_affine(*shape):
    return nir.Affine(weight=np.random.randn(*shape).T, bias=np.random.randn(shape[1]))


def mock_input(*shape):
    return nir.Input(input_shape=np.array(shape))


def mock_integrator(*shape):
    return nir.I(r=np.random.randn(*shape))


def mock_output(*shape):
    return nir.Output(output_shape=np.array(shape))


def mock_delay(*shape):
    return nir.Delay(delay=np.random.randn(*shape))


def test_has_version():
    assert hasattr(nir, "version")
    assert hasattr(nir, "__version__")


def test_simple():
    l = mock_affine(4, 3)
    ir = nir.NIRGraph(nodes={"a": l}, edges=[("a", "a")])
    assert np.allclose(ir.nodes["a"].weight, l.weight)
    assert np.allclose(ir.nodes["a"].bias, l.bias)
    assert ir.edges == [("a", "a")]


def test_nested():
    i = mock_integrator(3)
    d = mock_delay(3)
    a = mock_affine(3, 3)

    nested = nir.NIRGraph(
        nodes={
            "integrator": i,
            "delay": d,
        },
        edges=[("integrator", "delay"), ("delay", "integrator")],
    )
    ir = nir.NIRGraph(
        nodes={"affine": a, "inner": nested},
        edges=[("affine", "inner")],
    )
    assert np.allclose(ir.nodes["affine"].weight, a.weight)
    assert np.allclose(ir.nodes["affine"].bias, a.bias)
    assert np.allclose(ir.nodes["inner"].nodes["integrator"].r, i.r)
    assert np.allclose(ir.nodes["inner"].nodes["delay"].delay, d.delay)
    assert ir.nodes["inner"].edges == [("integrator", "delay"), ("delay", "integrator")]


def test_simple_with_input_output():
    a = mock_affine(3, 3)
    ir = nir.NIRGraph(
        nodes={
            "in": nir.Input(np.array([3])),
            "w": a,
            "out": nir.Output(np.array([3])),
        },
        edges=[("in", "w"), ("w", "out")],
    )
    assert ir.nodes["in"].input_shape == [
        3,
    ]
    assert np.allclose(ir.nodes["w"].weight, a.weight)
    assert np.allclose(ir.nodes["w"].bias, a.bias)
    assert ir.edges == [("in", "w"), ("w", "out")]


def test_delay():
    d = mock_delay(3)
    ir = nir.NIRGraph(
        nodes={
            "in": nir.Input(np.array([3])),
            "d": d,
            "out": nir.Output(np.array([3])),
        },
        edges=[("in", "d"), ("d", "out")],
    )
    assert ir.nodes["in"].input_shape == [
        3,
    ]
    assert np.allclose(ir.nodes["d"].delay, d.delay)
    assert ir.edges == [("in", "d"), ("d", "out")]


def test_cuba_lif():
    a = np.random.randn(10, 10)
    lif = nir.CubaLIF(tau_mem=a, tau_syn=a, r=a, v_leak=a, v_threshold=a)
    assert np.allclose(lif.tau_mem, a)


def test_threshold():
    threshold = np.array([1, 2, 3])
    ir = nir.NIRGraph(
        nodes={
            "in": nir.Input(np.array([3])),
            "thr": nir.Threshold(threshold),
            "out": nir.Output(np.array([3])),
        },
        edges=[("in", "thr"), ("thr", "out")],
    )
    assert ir.nodes["in"].input_shape == [
        3,
    ]
    assert np.allclose(ir.nodes["thr"].threshold, threshold)
    assert ir.edges == [("in", "thr"), ("thr", "out")]


def test_linear():
    l = mock_linear(3, 3)
    ir = nir.NIRGraph(nodes={"a": l}, edges=[("a", "a")])
    assert np.allclose(ir.nodes["a"].weight, l.weight)
    assert ir.edges == [("a", "a")]


def test_flatten():
    ir = nir.NIRGraph(
        nodes={
            "in": nir.Input(input_shape=np.array([4, 5, 2])),
            "flat": nir.Flatten(
                start_dim=0,
                end_dim=0,
                input_shape=np.array([4, 5, 2]),
            ),
            "out": nir.Output(output_shape=np.array([20, 2])),
        },
        edges=[("in", "flat"), ("flat", "out")],
    )
    assert np.allclose(ir.nodes["in"].input_shape, [4, 5, 2])
    assert np.allclose(ir.nodes["out"].input_shape, [20, 2])


def test_from_list_naming():
    ir = nir.NIRGraph.from_list(
        nir.Linear(weight=np.array([[3, 1], [-1, 2], [1, 2]])),
        nir.Linear(weight=np.array([[3, 1], [-1, 4], [1, 2]]).T),
        nir.Affine(
            weight=np.array([[2, 1], [-1, 2], [1, 2]]), bias=np.array([1, 3, 2])
        ),
        nir.Affine(
            weight=np.array([[2, 1], [-1, 4], [1, 2]]).T, bias=np.array([-2, 2])
        ),
        nir.Linear(weight=np.array([[3, 1], [-1, 1], [1, 2]])),
        nir.Linear(weight=np.array([[3, 1], [-1, 3], [1, 2]]).T),
        nir.Affine(
            weight=np.array([[2, 1], [-1, 1], [1, 2]]), bias=np.array([1, 5, 2])
        ),
        nir.Affine(
            weight=np.array([[2, 1], [-1, 3], [1, 2]]).T, bias=np.array([-2, 3])
        ),
    )
    assert "input" in ir.nodes.keys()
    assert "linear" in ir.nodes.keys()
    assert "linear_1" in ir.nodes.keys()
    assert "linear_2" in ir.nodes.keys()
    assert "linear_3" in ir.nodes.keys()
    assert "affine" in ir.nodes.keys()
    assert "affine_1" in ir.nodes.keys()
    assert "affine_2" in ir.nodes.keys()
    assert "affine_3" in ir.nodes.keys()
    assert "output" in ir.nodes.keys()
    assert np.allclose(ir.nodes["input"].input_shape, [3, 2])
    assert np.allclose(ir.nodes["linear"].weight, np.array([[3, 1], [-1, 2], [1, 2]]))
    assert np.allclose(
        ir.nodes["linear_1"].weight, np.array([[3, 1], [-1, 4], [1, 2]]).T
    )
    assert np.allclose(ir.nodes["affine"].weight, np.array([[2, 1], [-1, 2], [1, 2]]))
    assert np.allclose(ir.nodes["affine"].bias, np.array([1, 3, 2]))
    assert np.allclose(
        ir.nodes["affine_1"].weight, np.array([[2, 1], [-1, 4], [1, 2]]).T
    )
    assert np.allclose(ir.nodes["affine_1"].bias, np.array([-2, 2]))
    assert np.allclose(ir.nodes["linear_2"].weight, np.array([[3, 1], [-1, 1], [1, 2]]))
    assert np.allclose(
        ir.nodes["linear_3"].weight, np.array([[3, 1], [-1, 3], [1, 2]]).T
    )
    assert np.allclose(ir.nodes["affine_2"].weight, np.array([[2, 1], [-1, 1], [1, 2]]))
    assert np.allclose(ir.nodes["affine_2"].bias, np.array([1, 5, 2]))
    assert np.allclose(
        ir.nodes["affine_3"].weight, np.array([[2, 1], [-1, 3], [1, 2]]).T
    )
    assert np.allclose(ir.nodes["affine_3"].bias, np.array([-2, 3]))
    assert np.allclose(ir.nodes["output"].input_shape, [2])
    assert ir.edges == [
        ("input", "linear"),
        ("linear", "linear_1"),
        ("linear_1", "affine"),
        ("affine", "affine_1"),
        ("affine_1", "linear_2"),
        ("linear_2", "linear_3"),
        ("linear_3", "affine_2"),
        ("affine_2", "affine_3"),
        ("affine_3", "output"),
    ]


def test_from_list_tuple_or_list():
    nodes = [mock_affine(2, 3), mock_delay(1)]
    assert len(nir.NIRGraph.from_list(*nodes).nodes) == 4
    assert len(nir.NIRGraph.from_list(*nodes).edges) == 3
    assert len(nir.NIRGraph.from_list(tuple(nodes)).nodes) == 4
    assert len(nir.NIRGraph.from_list(tuple(nodes)).nodes) == 4
    assert len(nir.NIRGraph.from_list(nodes[0], nodes[1]).edges) == 3
    assert len(nir.NIRGraph.from_list(nodes[0], nodes[1]).edges) == 3


def test_subgraph_merge():
    """
    ```mermaid
    graph TD;
    A --> B;
    C --> D;
    D --> E;
    B --> E;
    ```
    """
    g1 = nir.NIRGraph.from_list(mock_linear(2, 3), mock_linear(3, 2))
    g2 = nir.NIRGraph.from_list(mock_linear(1, 3), mock_linear(3, 2))
    end = mock_output(2)
    g = nir.NIRGraph(
        nodes={"L": g1, "R": g2, "E": end},
        edges=[("L.output", "E.input"), ("R.output", "E.input")],
    )
    assert np.allclose(g.nodes["L"].nodes["linear"].input_shape, (3, 2))
    assert np.allclose(g.nodes["L"].nodes["linear_1"].input_shape, (2, 3))
    assert np.allclose(g.nodes["R"].nodes["linear"].input_shape, (3, 1))
    assert np.allclose(g.nodes["R"].nodes["linear_1"].input_shape, (2, 3))
    assert np.allclose(g.nodes["E"].input_shape, (2,))
    assert g.edges == [("L.output", "E.input"), ("R.output", "E.input")]
    assert g.nodes["L"].edges == [
        ("input", "linear"),
        ("linear", "linear_1"),
        ("linear_1", "output"),
    ]
    assert g.nodes["R"].edges == [
        ("input", "linear"),
        ("linear", "linear_1"),
        ("linear_1", "output"),
    ]
