import tempfile

import numpy as np

import nir


def assert_equivalence(ir: nir.NIRGraph, ir2: nir.NIRGraph):
    for ik, v in ir.nodes.items():
        if isinstance(ir.nodes[ik], nir.NIRGraph):
            # Handle nested graphs
            assert isinstance(ir2.nodes[ik], nir.NIRGraph)
            assert_equivalence(ir.nodes[ik], ir2.nodes[ik])
        else:
            for k, v in ir.nodes[ik].__dict__.items():
                if isinstance(v, np.ndarray) or isinstance(v, list):
                    assert np.array_equal(v, getattr(ir2.nodes[ik], k))
                else:
                    assert v == getattr(ir2.nodes[ik], k)


def factory_test_graph(ir: nir.NIRGraph):
    tmp = tempfile.mktemp()
    nir.write(tmp, ir)
    ir2 = nir.read(tmp)
    assert_equivalence(ir, ir2)


def test_simple():
    ir = nir.NIRGraph(
        nodes={"a": nir.Affine(weight=[1, 2, 3], bias=4)}, edges=[("a", "a")]
    )
    factory_test_graph(ir)


def test_nested():
    nested = nir.NIRGraph(
        nodes={
            "a": nir.I(r=[1, 1]),
            "b": nir.NIRGraph(
                nodes={
                    "a": nir.Input(np.array([1, 1])),
                    "b": nir.Delay(np.array([1, 1])),
                    "c": nir.Output(),
                },
                edges=[("a", "b"), ("b", "c")],
            ),
            "c": nir.Output(),
        },
        edges=[("a", "b"), ("b", "a")],
    )
    factory_test_graph(nested)


def test_integrator():
    ir = nir.NIRGraph(
        nodes={"a": nir.Affine(weight=[1], bias=0), "b": nir.I(r=2)},
        edges=[("a", "b")],
    )
    factory_test_graph(ir)


def test_integrate_and_fire():
    ir = nir.NIRGraph(
        nodes={"a": nir.Affine(weight=[1], bias=0), "b": nir.IF(r=2, v_threshold=3)},
        edges=[("a", "b")],
    )
    factory_test_graph(ir)


def test_leaky_integrator():
    ir = nir.NIRGraph.from_list(
        nir.Affine(weight=[1], bias=0), nir.LI(tau=1, r=2, v_leak=3)
    )
    factory_test_graph(ir)


def test_linear():
    ir = nir.NIRGraph.from_list(nir.Linear(weight=[1]), nir.LI(tau=1, r=2, v_leak=3))
    factory_test_graph(ir)


def test_leaky_integrator_and_fire():
    ir = nir.NIRGraph.from_list(
        nir.Affine(weight=[1], bias=0),
        nir.LIF(tau=1, r=2, v_leak=3, v_threshold=4),
    )
    factory_test_graph(ir)


def test_current_based_leaky_integrator_and_fire():
    ir = nir.NIRGraph.from_list(
        nir.Linear(weight=[1]),
        nir.CubaLIF(tau_mem=1, tau_syn=1, r=2, v_leak=3, v_threshold=4),
    )
    factory_test_graph(ir)


def test_simple_with_read_write():
    ir = nir.NIRGraph.from_list(
        nir.Input(
            shape=[
                3,
            ]
        ),
        nir.Affine(weight=[1, 2, 3], bias=4),
        nir.Output(),
    )
    factory_test_graph(ir)


def test_delay():
    ir = nir.NIRGraph.from_list(
        nir.Input(
            shape=[
                3,
            ]
        ),
        nir.Delay(delay=[1, 2, 3]),
        nir.Output(),
    )
    factory_test_graph(ir)


def test_threshold():
    ir = nir.NIRGraph.from_list(
        nir.Input(
            shape=[
                3,
            ]
        ),
        nir.Threshold(threshold=[2.0, 2.5, 2.8]),
        nir.Output(),
    )
    factory_test_graph(ir)
