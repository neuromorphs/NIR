import inspect
import sys
import tempfile

import numpy as np

import nir
from tests import mock_affine, mock_conv, mock_linear

ALL_NODES = []
for name, obj in inspect.getmembers(sys.modules["nir.ir"]):
    if inspect.isclass(obj) and obj.__module__ == "nir.ir":
        ALL_NODES.append(obj)


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
    ir2 = nir.NIRGraph.from_dict(ir.to_dict())
    assert_equivalence(ir, ir2)
    with tempfile.TemporaryFile() as fp:
        nir.write(fp, ir)
        ir2 = nir.read(fp)
    assert_equivalence(ir, ir2)


def factory_test_metadata(ir: nir.NIRGraph):
    def compare_dicts(d1, d2):
        for k, v in d1.items():
            if isinstance(v, np.ndarray):
                assert np.array_equal(v, d2[k])
            elif isinstance(v, bytes):
                assert v.decode("utf8") == d2[k]
            else:
                assert v == d2[k]

    metadata = {"some": "metadata", "with": 2, "data": np.array([1, 2, 3])}
    for node in ir.nodes.values():
        node.metadata = metadata
        compare_dicts(node.metadata, metadata)
    tmp = tempfile.mktemp()
    nir.write(tmp, ir)
    ir2 = nir.read(tmp)
    for node in ir2.nodes.values():
        compare_dicts(node.metadata, metadata)


def test_simple():
    ir = nir.NIRGraph(nodes={"a": mock_affine(2, 2)}, edges=[("a", "a")])
    factory_test_graph(ir)
    factory_test_metadata(ir)


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
    factory_test_metadata(nested)


def test_conv1d():
    ir = nir.NIRGraph.from_list(
        mock_affine(2, 100),
        mock_conv(100, (1, 2, 3)),
        mock_affine(100, 2),
    )
    factory_test_graph(ir)
    factory_test_metadata(ir)


def test_conv1d_2():
    ir = nir.NIRGraph.from_list(
        mock_conv((100, 100), (1, 2, 3, 3)),
    )
    factory_test_graph(ir)
    factory_test_metadata(ir)


def test_integrator():
    r = np.array([1, 1, 1])
    ir = nir.NIRGraph(
        nodes={"a": mock_affine(2, 2), "b": nir.I(r)},
        edges=[("a", "b")],
    )
    factory_test_graph(ir)
    factory_test_metadata(ir)


def test_integrate_and_fire():
    r = np.array([1, 1, 1])
    v_threshold = np.array([1, 1, 1])
    ir = nir.NIRGraph(
        nodes={"a": mock_affine(2, 2), "b": nir.IF(r, v_threshold)},
        edges=[("a", "b")],
    )
    factory_test_graph(ir)
    factory_test_metadata(ir)


def test_leaky_integrator():
    tau = np.array([1, 1, 1])
    r = np.array([1, 1, 1])
    v_leak = np.array([1, 1, 1])

    ir = nir.NIRGraph.from_list(mock_affine(2, 2), nir.LI(tau, r, v_leak))
    factory_test_graph(ir)
    factory_test_metadata(ir)


def test_linear():
    tau = np.array([1, 1, 1])
    r = np.array([1, 1, 1])
    v_leak = np.array([1, 1, 1])
    ir = nir.NIRGraph.from_list(mock_linear(2, 2), nir.LI(tau, r, v_leak))
    factory_test_graph(ir)
    factory_test_metadata(ir)


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
    factory_test_metadata(ir)


def test_current_based_leaky_integrator_and_fire():
    tau_mem = np.array([1, 1, 1])
    tau_syn = np.array([2, 2, 2])
    r = np.array([1, 1, 1])
    v_leak = np.array([1, 1, 1])
    v_threshold = np.array([3, 3, 3])
    w_in = np.array([2, 2, 2])
    ir = nir.NIRGraph.from_list(
        mock_affine(2, 2),
        nir.CubaLIF(tau_mem, tau_syn, r, v_leak, v_threshold, w_in=w_in),
    )
    factory_test_graph(ir)
    factory_test_metadata(ir)


def test_scale():
    ir = nir.NIRGraph.from_list(
        nir.Input(input_type=np.array([3])),
        nir.Scale(scale=np.array([1, 2, 3])),
        nir.Output(output_type=np.array([3])),
    )
    factory_test_graph(ir)
    factory_test_metadata(ir)


def test_simple_with_read_write():
    ir = nir.NIRGraph.from_list(
        nir.Input(input_type=np.array([3])),
        mock_affine(2, 2),
        nir.Output(output_type=np.array([3])),
    )
    factory_test_graph(ir)
    factory_test_metadata(ir)


def test_delay():
    delay = np.array([1, 2, 3])
    ir = nir.NIRGraph.from_list(
        nir.Input(np.array([3])),
        nir.Delay(delay),
        nir.Output(np.array([3])),
    )
    factory_test_graph(ir)
    factory_test_metadata(ir)


def test_threshold():
    threshold = np.array([1, 2, 3])
    ir = nir.NIRGraph.from_list(
        nir.Input(np.array([3])),
        nir.Threshold(threshold),
        nir.Output(np.array([3])),
    )
    factory_test_graph(ir)
    factory_test_metadata(ir)


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
    factory_test_metadata(ir)


def test_sum_pool_2d():
    ir = nir.NIRGraph.from_list(
        [
            nir.Input(input_type=np.array([2, 2, 10, 10])),
            nir.SumPool2d(
                kernel_size=np.array([2, 2]),
                stride=np.array([1, 1]),
                padding=np.ndarray([0, 0]),
            ),
            nir.Output(output_type=np.array([2, 2, 5, 5])),
        ]
    )
    factory_test_graph(ir)


def test_avg_pool_2d():
    ir = nir.NIRGraph.from_list(
        [
            nir.Input(input_type=np.array([2, 2, 10, 10])),
            nir.AvgPool2d(
                kernel_size=np.array([2, 2]),
                stride=np.array([1, 1]),
                padding=np.ndarray([0, 0]),
            ),
            nir.Output(output_type=np.array([2, 2, 5, 5])),
        ]
    )
    factory_test_graph(ir)
    factory_test_metadata(ir)
