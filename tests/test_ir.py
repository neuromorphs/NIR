import numpy as np
import pytest

import nir
from tests import mock_affine, mock_delay, mock_integrator, mock_linear, mock_output


def test_has_version():
    assert hasattr(nir, "version")
    assert hasattr(nir, "__version__")


def test_has_NIRNode():
    assert hasattr(nir, "NIRNode")


def test_eq():
    a = nir.Input(np.array([2, 3]))
    a2 = nir.Input(np.array([2, 3]))
    b = nir.Input(np.array([2, 3]))
    b2 = nir.Input(np.array([2, 2]))
    o = nir.Output(np.array([2, 3]))

    assert a == a
    assert a2 == a2
    assert b == b
    assert b2 == b2
    assert a != a2
    assert a != b
    assert a != b2
    assert a2 != a
    assert a2 != b
    assert a2 != b2
    assert b != a
    assert b != a2
    assert b != b2
    assert b2 != a
    assert b2 != a2
    assert b2 != b
    assert a != o
    assert o != a


def test_simple():
    a = mock_affine(3, 3)
    b = mock_affine(3, 3)
    ir = nir.NIRGraph(
        nodes={
            "a": a,
            "b": b,
        },
        edges=[("a", "b")],
    )
    assert np.allclose(ir.nodes["a"].weight, a.weight)
    assert np.allclose(ir.nodes["a"].bias, a.bias)
    assert np.allclose(ir.nodes["b"].weight, b.weight)
    assert np.allclose(ir.nodes["b"].bias, b.bias)
    assert sorted([("input_a", "a"), ("a", "b"), ("b", "output_b")]) == sorted(ir.edges)


def test_nested():
    i = mock_integrator(3)
    d = mock_delay(3)
    a = mock_affine(3, 3)

    nested = nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type=np.array([3])),
            "integrator": i,
            "delay": d,
            "output": nir.Output(output_type=None),
        },
        edges=[
            ("input", "integrator"),
            ("integrator", "delay"),
            ("delay", "integrator"),
            ("integrator", "output"),
        ],
    )
    ir = nir.NIRGraph(nodes={"affine": a, "inner": nested}, edges=[("affine", "inner")])
    assert np.allclose(ir.nodes["affine"].weight, a.weight)
    assert np.allclose(ir.nodes["affine"].bias, a.bias)
    assert np.allclose(ir.nodes["inner"].nodes["integrator"].r, i.r)
    assert np.allclose(ir.nodes["inner"].nodes["delay"].delay, d.delay)
    assert ir.nodes["inner"].edges == [
        ("input", "integrator"),
        ("integrator", "delay"),
        ("delay", "integrator"),
        ("integrator", "output"),
    ]


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
    assert ir.nodes["in"].input_type == {"input": np.array([3])}
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
    assert ir.nodes["in"].input_type == {"input": np.array([3])}
    assert np.allclose(ir.nodes["d"].delay, d.delay)
    assert ir.edges == [("in", "d"), ("d", "out")]


def test_conv1d():
    w = np.random.randn(2, 1, 3)
    a = nir.Conv1d(
        input_shape=100,
        weight=w,
        stride=2,
        dilation=1,
        groups=1,
        padding=1,
        bias=np.ndarray([1]),
    )
    assert np.allclose(a.weight, w)
    assert np.allclose(a.input_shape, 100)
    assert np.allclose(a.output_type["output"], np.array([2, 50]))


def test_conv2d():
    w = np.random.randn(3, 1, 3, 3)
    a = nir.Conv2d(
        input_shape=(100, 100),
        weight=w,
        padding=(1, 1),
        stride=(1, 2),
        dilation=(1, 1),
        groups=(1, 1),
        bias=np.ndarray([1]),
    )
    assert np.allclose(a.weight, w)
    assert np.allclose(a.input_shape, np.array([100, 100]))
    assert np.allclose(a.output_type["output"], np.array([3, 100, 50]))


def test_conv2d_same():
    # Create a NIR Network
    conv_weights = np.array([[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]])
    li_tau = np.array([0.9, 0.8, 0.7])
    li_r = np.array([1.0, 1.0, 1.0])
    li_v_leak = np.array([0.0, 0.0, 0.0])

    nir_network = nir.NIRGraph.from_list(
        nir.Conv2d(
            input_shape=(3, 3),
            weight=conv_weights,
            stride=1,
            padding="same",
            dilation=1,
            groups=1,
            bias=np.array([0.0] * 9),
        ),
        nir.LI(li_tau, li_r, li_v_leak),
        type_check=False,  # TODO: Add type check
    )
    assert np.allclose(nir_network.nodes["conv2d"].output_type["output"], [1, 3, 3])


def test_cuba_li():
    a = np.random.randn(10, 10)
    li = nir.CubaLI(tau_mem=a, tau_syn=a, r=a, v_leak=a)
    assert np.allclose(li.tau_mem, a)


def test_cuba_lif():
    a = np.random.randn(10, 10)
    lif = nir.CubaLIF(tau_mem=a, tau_syn=a, r=a, v_leak=a, v_reset=a, v_threshold=a)
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
    assert ir.nodes["in"].input_type == {"input": np.array([3])}
    assert np.allclose(ir.nodes["thr"].threshold, threshold)
    assert ir.edges == [("in", "thr"), ("thr", "out")]


def test_linear():
    a = mock_linear(3, 3)
    b = mock_linear(3, 3)
    ir = nir.NIRGraph(nodes={"a": a, "b": b}, edges=[("a", "b")])
    assert np.allclose(ir.nodes["a"].weight, a.weight)
    assert np.allclose(ir.nodes["b"].weight, b.weight)
    assert ("a", "b") in ir.edges


def test_flatten():
    ir = nir.NIRGraph(
        nodes={
            "in": nir.Input(input_type=np.array([4, 5, 2])),
            "flat": nir.Flatten(
                start_dim=0, end_dim=1, input_type={"input": np.array([4, 5, 2])}
            ),
            "out": nir.Output(output_type=np.array([20, 2])),
        },
        edges=[("in", "flat"), ("flat", "out")],
    )
    assert np.allclose(ir.nodes["in"].input_type["input"], np.array([4, 5, 2]))
    assert np.allclose(ir.nodes["out"].input_type["input"], np.array([20, 2]))


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
    assert np.allclose(ir.nodes["input"].input_type["input"], [2])
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
    print(ir.nodes["output"].input_type["input"])
    assert np.allclose(ir.nodes["output"].input_type["input"], [2])
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
    nodes = [mock_affine(2, 3), mock_delay(3)]
    assert len(nir.NIRGraph.from_list(*nodes).nodes) == 4
    assert len(nir.NIRGraph.from_list(*nodes).edges) == 3
    assert len(nir.NIRGraph.from_list(tuple(nodes)).nodes) == 4
    assert len(nir.NIRGraph.from_list(tuple(nodes)).nodes) == 4
    assert len(nir.NIRGraph.from_list(nodes[0], nodes[1]).edges) == 3
    assert len(nir.NIRGraph.from_list(nodes[0], nodes[1]).edges) == 3


def test_graph_from_dict_type_checked():
    nodes = {
        "input": {"type": "Input", "shape": np.array([2])},
        "module": {"type": "Linear", "weight": np.random.random((2, 2))},
        "output": {"type": "Output", "shape": np.array([2])},
    }
    kwargs = {"nodes": nodes, "edges": [("input", "module"), ("module", "output")]}
    nir.NIRGraph.from_dict(kwargs)

    with pytest.raises(AssertionError):
        nir.NIRGraph.from_dict({"type": "Input"})

    with pytest.raises(ValueError):
        nodes = {
            "input": {"type": "Input", "shape": np.array([3])},
            "module": {"type": "Linear", "weight": np.random.random((2, 2))},
            "output": {"type": "Output", "shape": np.array([3])},
        }
        kwargs = {"nodes": nodes, "edges": [("input", "module"), ("module", "output")]}
        nir.NIRGraph.from_dict(kwargs)


@pytest.mark.skip("Not implemented")  # TODO: Fix subgraph nodes for type checking
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
    assert np.allclose(g.nodes["L"].nodes["linear"].input_type["input"], [2])
    assert np.allclose(g.nodes["L"].nodes["linear_1"].input_type["input"], [3])
    assert np.allclose(g.nodes["R"].nodes["linear"].input_type["input"], [1])
    assert np.allclose(g.nodes["R"].nodes["linear_1"].input_type["input"], [3])
    assert np.allclose(g.nodes["E"].input_type["input"], [2])
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


@pytest.mark.skip("Not implemented")  # TODO: Fix subgraph nodes for type checking
def test_inputs_outputs_properties():
    ir = nir.NIRGraph(
        nodes={
            "in1": nir.Input(np.array([4, 5, 2])),
            "in2": nir.Input(np.array([4, 5, 2])),
            "flat": nir.Flatten(
                start_dim=0, end_dim=1, input_type={"input": np.array([4, 5, 2])}
            ),
            "out1": nir.Output(np.array([20, 2])),
            "out2": nir.Output(np.array([20, 2])),
        },
        edges=[("in1", "flat"), ("in2", "flat"), ("flat", "out1"), ("flat", "out2")],
    )
    ir2 = nir.NIRGraph(
        nodes={
            "in": nir.Input(np.array([4, 5, 2])),
            "inner": ir,
            "out": nir.Output(np.array([20, 2])),
        },
        edges=[
            ("in", "inner.in1"),
            ("in", "inner.in2"),
            ("inner.out1", "out"),
            ("inner.out2", "out"),
        ],
    )
    assert np.allclose(ir.nodes["in1"].input_type["input"], [4, 5, 2])
    assert np.allclose(ir.nodes["out1"].input_type["input"], [20, 2])
    assert np.allclose(ir.nodes["in2"].input_type["input"], [4, 5, 2])
    assert np.allclose(ir.nodes["out2"].input_type["input"], [20, 2])
    assert ir.nodes["in1"] == ir.inputs["in1"]
    assert ir.nodes["in2"] == ir.inputs["in2"]
    assert ir.nodes["out1"] == ir.outputs["out1"]
    assert ir.nodes["out2"] == ir.outputs["out2"]
    assert ir.nodes["in1"] not in ir.outputs.values()
    assert ir.nodes["in2"] not in ir.outputs.values()
    assert ir.nodes["out1"] not in ir.inputs.values()
    assert ir.nodes["out2"] not in ir.inputs.values()
    assert ir.nodes["flat"] not in ir.inputs.values()
    assert ir.nodes["flat"] not in ir.outputs.values()
    assert ir.nodes["in1"] in ir2.nodes["inner"].inputs.values()
    assert ir.nodes["in2"] in ir2.nodes["inner"].inputs.values()
    assert ir.nodes["out1"] in ir2.nodes["inner"].outputs.values()
    assert ir.nodes["out2"] in ir2.nodes["inner"].outputs.values()


def test_sumpool_type_inference():
    graph = nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type=np.array([1, 64, 64])),
            "sumpool": nir.SumPool2d(
                kernel_size=np.array([2, 2]),
                stride=np.array([2, 2]),
                padding=np.array([0, 0]),
            ),
            "output": nir.Output(output_type=None),
        },
        edges=[("input", "sumpool"), ("sumpool", "output")],
    )
    assert np.array_equal(graph.output_type["output"], np.array([1, 32, 32]))
    assert np.array_equal(
        graph.nodes["output"].input_type["input"], np.array([1, 32, 32])
    )
    assert np.array_equal(
        graph.nodes["output"].output_type["output"], np.array([1, 32, 32])
    )


def test_avgpool_type_inference():
    graph = nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type=np.array([1, 64, 64])),
            "avgpool": nir.AvgPool2d(
                kernel_size=np.array([2, 2]),
                stride=np.array([2, 2]),
                padding=np.array([0, 0]),
            ),
            "output": nir.Output(output_type=None),
        },
        edges=[("input", "avgpool"), ("avgpool", "output")],
    )
    assert np.array_equal(graph.output_type["output"], np.array([1, 32, 32]))
    assert np.array_equal(
        graph.nodes["output"].input_type["input"], np.array([1, 32, 32])
    )
    assert np.array_equal(
        graph.nodes["output"].output_type["output"], np.array([1, 32, 32])
    )


def test_flatten_type_inference():
    tests = [
        {
            "start_dim": 0,
            "end_dim": 0,
            "input": np.array([1, 64, 64]),
            "output": np.array([1, 64, 64]),
        },
        {
            "start_dim": 0,
            "end_dim": 1,
            "input": np.array([1, 64, 64]),
            "output": np.array([64, 64]),
        },
        {
            "start_dim": 0,
            "end_dim": 2,
            "input": np.array([1, 64, 64]),
            "output": np.array([4096]),
        },
        {
            "start_dim": 1,
            "end_dim": 2,
            "input": np.array([1, 64, 64]),
            "output": np.array([1, 4096]),
        },
    ]

    for test in tests:
        for flatten_input_type in [test["input"], None]:
            graph = nir.NIRGraph(
                nodes={
                    "input": nir.Input(input_type=test["input"]),
                    "flatten": nir.Flatten(
                        start_dim=test["start_dim"],
                        end_dim=test["end_dim"],
                        input_type=flatten_input_type,
                    ),
                    "output": nir.Output(output_type=None),
                },
                edges=[("input", "flatten"), ("flatten", "output")],
            )
            assert np.array_equal(
                graph.nodes["flatten"].input_type["input"], test["input"]
            )
            assert np.array_equal(
                graph.nodes["flatten"].output_type["output"], test["output"]
            )
            assert np.array_equal(
                graph.nodes["output"].input_type["input"], test["output"]
            )
            assert np.array_equal(
                graph.nodes["output"].output_type["output"], test["output"]
            )
            assert np.array_equal(graph.input_type["input"], test["input"])
            assert np.array_equal(graph.output_type["output"], test["output"])


def test_conv2d_type_inference():
    graphs = {
        "undef graph output": nir.NIRGraph(
            nodes={
                "input": nir.Input(input_type=np.array([1, 64, 64])),
                "conv": nir.Conv2d(
                    input_shape=(64, 64),
                    weight=np.zeros((1, 1, 4, 4)),
                    stride=1,
                    padding=0,
                    dilation=1,
                    groups=1,
                    bias=None,
                ),
                "output": nir.Output(output_type=None),
            },
            edges=[("input", "conv"), ("conv", "output")],
            type_check=False,
        ),
        "undef conv.input": nir.NIRGraph(
            nodes={
                "input": nir.Input(input_type=np.array([1, 64, 64])),
                "conv": nir.Conv2d(
                    input_shape=None,
                    weight=np.zeros((1, 1, 4, 4)),
                    stride=1,
                    padding=0,
                    dilation=1,
                    groups=1,
                    bias=None,
                ),
                "output": nir.Output(output_type=np.array([1, 61, 61])),
            },
            edges=[("input", "conv"), ("conv", "output")],
            type_check=False,
        ),
        "undef conv.input and graph output": nir.NIRGraph(
            nodes={
                "input": nir.Input(input_type=np.array([1, 64, 64])),
                "conv": nir.Conv2d(
                    input_shape=None,
                    weight=np.zeros((1, 1, 4, 4)),
                    stride=1,
                    padding=0,
                    dilation=1,
                    groups=1,
                    bias=None,
                ),
                "output": nir.Output(output_type=None),
            },
            edges=[("input", "conv"), ("conv", "output")],
            type_check=False,
        ),
    }
    for name, graph in graphs.items():
        try:
            graph.infer_types()
        except Exception as ex:
            raise AssertionError(f"type inference failed for: {name}: {ex}") from ex
        try:
            graph.check_types()
        except Exception as ex:
            raise AssertionError(f"type check failed for: {name}: {ex}") from ex

        assert np.array_equal(
            graph.nodes["output"].input_type["input"], np.array([1, 61, 61])
        ), name
        assert np.array_equal(
            graph.nodes["output"].output_type["output"], np.array([1, 61, 61])
        ), name
        assert np.array_equal(
            graph.nodes["conv"].output_type["output"], np.array([1, 61, 61])
        ), name
        assert np.array_equal(graph.input_type["input"], np.array([1, 64, 64])), name
        assert np.array_equal(graph.output_type["output"], np.array([1, 61, 61])), name


def test_conv1d_type_inference():
    graphs = {
        "Conv1d undef graph output": nir.NIRGraph(
            nodes={
                "input": nir.Input(input_type=np.array([1, 64])),
                "conv": nir.Conv1d(
                    input_shape=64,
                    weight=np.zeros((1, 1, 4)),
                    stride=1,
                    padding=0,
                    dilation=1,
                    groups=1,
                    bias=None,
                ),
                "output": nir.Output(output_type=None),
            },
            edges=[("input", "conv"), ("conv", "output")],
            type_check=False,
        ),
        "Conv1d undef conv.input and graph output": nir.NIRGraph(
            nodes={
                "input": nir.Input(input_type=np.array([1, 64])),
                "conv": nir.Conv1d(
                    input_shape=None,
                    weight=np.zeros((1, 1, 4)),
                    stride=1,
                    padding=0,
                    dilation=1,
                    groups=1,
                    bias=None,
                ),
                "output": nir.Output(output_type=None),
            },
            edges=[("input", "conv"), ("conv", "output")],
            type_check=False,
        ),
    }
    for name, graph in graphs.items():
        try:
            graph.infer_types()
        except Exception as ex:
            raise AssertionError(f"type inference failed for: {name}: {ex}") from ex
        try:
            graph.check_types()
        except Exception as ex:
            raise AssertionError(f"type check failed for: {name}: {ex}") from ex

        assert np.array_equal(
            graph.nodes["output"].input_type["input"], np.array([1, 61])
        ), name
        assert np.array_equal(
            graph.nodes["output"].output_type["output"], np.array([1, 61])
        ), name
        assert np.array_equal(
            graph.nodes["conv"].output_type["output"], np.array([1, 61])
        ), name
        assert np.array_equal(graph.input_type["input"], np.array([1, 64])), name
        assert np.array_equal(graph.output_type["output"], np.array([1, 61])), name


def test_graph_input_output_type_inference():
    graphs = {
        "inferred Input and Output nodes": nir.NIRGraph(
            nodes={
                "transform": nir.Linear(weight=np.random.random((4, 6))),
            },
            edges=[],
        ),
        "inferred Output node": nir.NIRGraph(
            nodes={
                "graph_input": nir.Input(input_type=np.array([6])),
                "transform": nir.Linear(weight=np.random.random((4, 6))),
            },
            edges=[("graph_input", "transform")],
        ),
        "unset Output(output_type)": nir.NIRGraph(
            nodes={
                "graph_input": nir.Input(input_type=np.array([6])),
                "transform": nir.Linear(weight=np.random.random((4, 6))),
                "graph_output": nir.Output(output_type=None),
            },
            edges=[("graph_input", "transform"), ("transform", "graph_output")],
        ),
    }

    for name, graph in graphs.items():
        # Graph input should be based on Input node input_type
        assert (
            graph.input_type is not None
            and len(graph.input_type) == 1
            and np.array_equal(list(graph.input_type.values())[0], np.array([6]))
        ), f"unexpected graph input type for {name} after type inference"
        # Graph output should be set to the output_type of the Output node
        assert (
            graph.output_type is not None
            and len(graph.output_type) == 1
            and np.array_equal(list(graph.output_type.values())[0], np.array([4]))
        ), f"unexpected graph output type for {name} after type inference"

        # Input nodes should have input and output types set to the same values.
        assert len(graph.inputs) == 1, f"unexpected number of input nodes for {name}"
        input_node = list(graph.inputs.values())[0]
        assert (
            input_node.input_type is not None
            and len(input_node.input_type) == 1
            and "input" in input_node.input_type
            and np.array_equal(input_node.input_type["input"], np.array([6]))
        ), f"unexpected Input node input_type for {name}"
        assert (
            input_node.output_type is not None
            and len(input_node.output_type) == 1
            and "output" in input_node.output_type
            and np.array_equal(input_node.output_type["output"], np.array([6]))
        ), f"unexpected Input node output_type for {name}"

        # Output nodes should have input and output types set to the same values.
        assert len(graph.outputs) == 1, f"unexpected number of output nodes for {name}"
        output_node = list(graph.outputs.values())[0]
        assert (
            output_node.input_type is not None
            and len(output_node.input_type) == 1
            and "input" in output_node.input_type
            and np.array_equal(output_node.input_type["input"], np.array([4]))
        ), f"unexpected Output node input_type for {name}"
        assert (
            output_node.output_type is not None
            and len(output_node.output_type) == 1
            and "output" in output_node.output_type
            and np.array_equal(output_node.output_type["output"], np.array([4]))
        ), f"unexpected Output node output_type for {name}"


def test_type_check_recurrent():
    # Valid recurrent graph
    nir.NIRGraph(
        nodes={
            "a": nir.Input(np.array([2])),
            "b": nir.Linear(np.random.rand(2, 2)),
            "c": nir.IF(
                r=np.random.rand(2), v_threshold=np.random.rand(2), v_reset=np.zeros(2)
            ),
            "d": nir.Output(np.array([2])),
        },
        edges=[("a", "b"), ("b", "c"), ("c", "b"), ("c", "d")],
    )

    # Invalid recurrent graph: c output goes back to b, but shapes do not match
    with pytest.raises(
        ValueError, match="type mismatch: c.output: \\[16\\] -> b.input: \\[1 5 5\\]"
    ):
        nir.NIRGraph(
            nodes={
                "a": nir.Input(input_type=np.array([1, 5, 5])),
                "b": nir.Conv2d(
                    input_shape=None,
                    weight=np.random.rand(1, 1, 2, 2),
                    stride=1,
                    padding=0,
                    dilation=1,
                    groups=1,
                    bias=np.zeros(1),
                ),
                "c": nir.Flatten(input_type=None, start_dim=0, end_dim=2),
                "d": nir.Output(output_type=np.array([16])),
            },
            edges=[("a", "b"), ("b", "c"), ("c", "b"), ("c", "d")],
        )


def test_node():
    try:
        node = nir.ir.NIRNode()
        assert (
            node is None
        ), f"test failed, we should not be able to construct an NIRNode: {node}"
    except AttributeError:
        pass
