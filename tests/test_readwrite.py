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
                    assert np.array_equal(v, getattr(ir2.nodes[ik], k), equal_nan=True)
                else:
                    assert v == getattr(ir2.nodes[ik], k)


def factory_test_graph(ir: nir.NIRGraph):
    tmp = tempfile.mktemp()
    nir.write(tmp, ir)
    ir2 = nir.read(tmp)
    assert_equivalence(ir, ir2)


def test_simple():
    w = np.array([1, 2, 3])
    b = np.array([4, 4, 4])
    ir = nir.NIRGraph(nodes={"a": nir.Affine(weight=w, bias=b)}, edges=[("a", "a")])
    factory_test_graph(ir)


def test_nested():
    i = np.array([1, 1])
    nested = nir.NIRGraph(
        nodes={
            "a": nir.I(r=[1, 1]),
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
    w = np.array([1, 2, 3])
    b = np.array([4, 4, 4])
    r = np.array([1, 1, 1])
    ir = nir.NIRGraph(
        nodes={"a": nir.Affine(weight=w, bias=b), "b": nir.I(r)},
        edges=[("a", "b")],
    )
    factory_test_graph(ir)


def test_integrate_and_fire():
    w = np.array([1, 2, 3])
    b = np.array([4, 4, 4])
    r = np.array([1, 1, 1])
    v_threshold = np.array([1, 1, 1])
    ir = nir.NIRGraph(
        nodes={"a": nir.Affine(weight=w, bias=b), "b": nir.IF(r, v_threshold)},
        edges=[("a", "b")],
    )
    factory_test_graph(ir)


def test_leaky_integrator():
    w = np.array([1, 2, 3])
    b = np.array([4, 4, 4])
    tau = np.array([1, 1, 1])
    r = np.array([1, 1, 1])
    v_leak = np.array([1, 1, 1])

    ir = nir.NIRGraph.from_list(nir.Affine(weight=w, bias=b), nir.LI(tau, r, v_leak))
    factory_test_graph(ir)


def test_linear():
    w = np.array([1, 2, 3])
    b = np.array([4, 4, 4])
    tau = np.array([1, 1, 1])
    r = np.array([1, 1, 1])
    v_leak = np.array([1, 1, 1])
    ir = nir.NIRGraph.from_list(nir.Affine(w, b), nir.LI(tau, r, v_leak))
    factory_test_graph(ir)


def test_leaky_integrator_and_fire():
    w = np.array([1, 2, 3])
    b = np.array([4, 4, 4])
    tau = np.array([1, 1, 1])
    r = np.array([1, 1, 1])
    v_leak = np.array([1, 1, 1])
    v_threshold = np.array([3, 3, 3])
    ir = nir.NIRGraph.from_list(
        nir.Affine(w, b),
        nir.LIF(tau, r, v_leak, v_threshold),
    )
    factory_test_graph(ir)


def test_current_based_leaky_integrator_and_fire():
    w = np.array([1, 2, 3])
    b = np.array([4, 4, 4])
    tau_mem = np.array([1, 1, 1])
    tau_syn = np.array([2, 2, 2])
    r = np.array([1, 1, 1])
    v_leak = np.array([1, 1, 1])
    v_threshold = np.array([3, 3, 3])
    ir = nir.NIRGraph.from_list(
        nir.Affine(w, b),
        nir.CubaLIF(tau_mem, tau_syn, r, v_leak, v_threshold),
    )
    factory_test_graph(ir)


def test_scale():
    ir = nir.NIRGraph.from_list(
        nir.Input(shape=np.array([3])),
        nir.Scale(scale=np.array([1, 2, 3])),
        nir.Output(shape=np.array([3])),
    )
    factory_test_graph(ir)


def test_simple_with_read_write():
    w = np.array([1, 2, 3])
    b = np.array([4, 4, 4])
    ir = nir.NIRGraph.from_list(
        nir.Input(shape=np.array([3])),
        nir.Affine(w, b),
        nir.Output(shape=np.array([3])),
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
        nir.Input(shape=np.array([2, 3])),
        nir.Flatten(),
        nir.Output(shape=np.array([6])),
    )
    factory_test_graph(ir)


def test_project():
    ir = nir.NIRGraph(
        nodes={
            "in": nir.Input(np.array([4, 5, 2])),
            "project_out": nir.Projection(
                output_indices=np.array([np.nan, 0, 0, np.nan, np.nan])
            ),
            "project_in": nir.Projection(output_indices=np.array([1])),
            "out": nir.Output(np.array([4, 5, 2])),
        },
        edges=[("in", "project_out"), ("project_in", "out")],
    )
    factory_test_graph(ir)
