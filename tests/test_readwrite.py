import tempfile

import numpy as np

import nir
from tests import mock_affine


def assert_equivalence(ir: nir.NIRGraph, ir2: nir.NIRGraph):
    for ik, v in ir.nodes.items():
        if isinstance(ir.nodes[ik], nir.NIRGraph):
            # Handle nested graphs
            assert isinstance(ir2.nodes[ik], nir.NIRGraph)
            assert_equivalence(ir.nodes[ik], ir2.nodes[ik])
        else:
            for k, v in ir.nodes[ik].__dict__.items():
                if (
                    isinstance(v, np.ndarray)
                    or isinstance(v, list)
                    or isinstance(v, tuple)
                ):
                    assert np.array_equal(v, getattr(ir2.nodes[ik], k))
                elif isinstance(v, dict):
                    d = getattr(ir2.nodes[ik], k)
                    for a, b in d.items():
                        assert np.array_equal(v[a], b)
                else:
                    assert v == getattr(ir2.nodes[ik], k)
    for i, _ in enumerate(ir.edges):
        assert ir.edges[i][0] == ir2.edges[i][0]
        assert ir.edges[i][1] == ir2.edges[i][1]


def factory_test_graph(ir: nir.NIRGraph):
    tmp = tempfile.mktemp()
    nir.write(tmp, ir)
    ir2 = nir.read(tmp)
    assert_equivalence(ir, ir2)


def test_simple():
    ir = nir.NIRGraph(nodes={"a": mock_affine(2, 2)}, edges=[("a", "a")])
    factory_test_graph(ir)


def test_nested():
    i = np.array([1, 1])
    nested = nir.NIRGraph(
        nodes={
            "a": nir.I(r=np.array([1, 1])),
            "b": nir.NIRGraph(
                nodes={
                    "a": nir.Input(i),
                    "b": nir.Delay(i),
                    "c": nir.Output(np.array([1, 1])),
                },
                edges=[("a", "b"), ("b", "c")],
            ),
            "c": nir.Output(np.array([1, 1])),
        },
        edges=[("a", "b"), ("b", "a")],
    )
    factory_test_graph(nested)


def test_integrator():
    r = np.array([1, 1, 1])
    ir = nir.NIRGraph(
        nodes={"a": mock_affine(2, 2), "b": nir.I(r)},
        edges=[("a", "b")],
    )
    factory_test_graph(ir)


def test_integrate_and_fire():
    r = np.array([1, 1, 1])
    v_threshold = np.array([1, 1, 1])
    ir = nir.NIRGraph(
        nodes={"a": mock_affine(2, 2), "b": nir.IF(r, v_threshold)},
        edges=[("a", "b")],
    )
    factory_test_graph(ir)


def test_leaky_integrator():
    tau = np.array([1, 1, 1])
    r = np.array([1, 1, 1])
    v_leak = np.array([1, 1, 1])

    ir = nir.NIRGraph.from_list(mock_affine(2, 2), nir.LI(tau, r, v_leak))
    factory_test_graph(ir)


def test_linear():
    tau = np.array([1, 1, 1])
    r = np.array([1, 1, 1])
    v_leak = np.array([1, 1, 1])
    ir = nir.NIRGraph.from_list(mock_affine(2, 2), nir.LI(tau, r, v_leak))
    factory_test_graph(ir)


def test_leaky_integrator_and_fire():
    tau = np.array([1, 1, 1])
    r = np.array([1, 1, 1])
    v_leak = np.array([1, 1, 1])
    v_threshold = np.array([3, 3, 3])
    ir = nir.NIRGraph.from_list(
        mock_affine(2, 2),
        nir.LIF(tau, r, v_leak, v_threshold),
    )
    factory_test_graph(ir)


def test_current_based_leaky_integrator_and_fire():
    tau_mem = np.array([1, 1, 1])
    tau_syn = np.array([2, 2, 2])
    r = np.array([1, 1, 1])
    v_leak = np.array([1, 1, 1])
    v_threshold = np.array([3, 3, 3])
    ir = nir.NIRGraph.from_list(
        mock_affine(2, 2),
        nir.CubaLIF(tau_mem, tau_syn, r, v_leak, v_threshold),
    )
    factory_test_graph(ir)


def test_scale():
    ir = nir.NIRGraph.from_list(
        nir.Input(input_type=np.array([3])),
        nir.Scale(scale=np.array([1, 2, 3])),
        nir.Output(output_type=np.array([3])),
    )
    factory_test_graph(ir)


def test_simple_with_read_write():
    ir = nir.NIRGraph.from_list(
        nir.Input(input_type=np.array([3])),
        mock_affine(2, 2),
        nir.Output(output_type=np.array([3])),
    )
    factory_test_graph(ir)


def test_delay():
    delay = np.array([1, 2, 3])
    ir = nir.NIRGraph.from_list(
        nir.Input(np.array([3])),
        nir.Delay(delay),
        nir.Output(np.array([3])),
    )
    factory_test_graph(ir)


def test_threshold():
    threshold = np.array([1, 2, 3])
    ir = nir.NIRGraph.from_list(
        nir.Input(np.array([3])),
        nir.Threshold(threshold),
        nir.Output(np.array([3])),
    )
    factory_test_graph(ir)


def test_flatten():
    ir = nir.NIRGraph.from_list(
        nir.Input(input_type=np.array([2, 3])),
        nir.Flatten(
            start_dim=0,
            end_dim=0,
            input_type={"input": np.array([2, 3])},
        ),
        nir.Output(output_type=np.array([6])),
    )
    factory_test_graph(ir)
