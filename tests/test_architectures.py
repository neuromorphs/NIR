import numpy as np
import pytest

import nir
from tests import mock_affine, mock_input, mock_integrator, mock_output

from .test_readwrite import factory_test_graph


def test_sequential():
    a = mock_affine(2, 3)
    b = nir.Delay(np.array([0.5, 0.1, 0.2]))
    c = nir.LIF(
        tau=np.array([10, 20, 30]),
        r=np.array([1, 1, 1]),
        v_leak=np.array([0, 0, 0]),
        v_reset=np.array([0, 0, 0]),
        v_threshold=np.array([1, 2, 3]),
    )
    d = mock_affine(3, 2)

    ir = nir.NIRGraph.from_list(a, b, c, d)
    assert ir.output_type["output"] == [2]
    factory_test_graph(ir)


def test_recurrent_on_first_node():
    """Recurrent node is the first node. Because there is no unambiguous input node,
    Input nodes need to be added manually.

    ```mermaid
    graph TD;
    I --> A;
    A --> I;
    A --> O;
    ```
    """
    affine = mock_affine(2, 2)
    integrator = mock_integrator(2)
    ir = nir.NIRGraph(
        nodes={
            "input": mock_input(2),
            "affine": affine,
            "integrator": integrator,
            "output": mock_output(2),
        },
        edges=[
            ("input", "affine"),
            ("affine", "integrator"),
            ("integrator", "affine"),
            ("integrator", "output"),
        ],
    )
    factory_test_graph(ir)


def test_recurrent_on_second_node():
    """Recurrent node is the second node. There is one node before and after the
    recurrent block, which means the type inference can automatically create the Input
    and Output nodes.

    ```mermaid
    graph TD;
    A1 --> A2;
    A2 --> I;
    I --> A2;
    I --> A3;
    ```
    """
    affine1 = mock_affine(2, 2)
    affine2 = mock_affine(2, 2)
    integrator = mock_integrator(2)
    affine3 = mock_affine(2, 2)
    ir = nir.NIRGraph(
        nodes={
            "affine1": affine1,
            "affine2": affine2,
            "integrator": integrator,
            "affine3": affine3,
        },
        edges=[
            ("affine1", "affine2"),
            ("affine2", "integrator"),
            ("integrator", "affine2"),
            ("integrator", "affine3"),
        ],
    )
    factory_test_graph(ir)


def test_two_independent_branches():
    """
    ```mermaid
    graph TD;
    A --> B;
    B --> C;
    C --> D;
    E --> F;
    F --> G
    ```
    """
    # Branch 1
    a = mock_affine(2, 3)
    b = nir.Delay(np.array([0.5, 0.1, 0.2]))
    c = nir.LIF(
        tau=np.array([10, 20, 30]),
        r=np.array([1, 1, 1]),
        v_leak=np.array([0, 0, 0]),
        v_reset=np.array([0, 0, 0]),
        v_threshold=np.array([1, 2, 3]),
    )
    d = mock_affine(3, 2)

    branch_1 = nir.NIRGraph.from_list(a, b, c, d)

    # Branch 2
    e = mock_affine(2, 2)
    f = nir.LIF(
        tau=np.array([10, 20]),
        r=np.array([1, 1]),
        v_leak=np.array([0, 0]),
        v_reset=np.array([0, 0]),
        v_threshold=np.array([1, 2]),
    )
    g = mock_affine(2, 2)

    branch_2 = nir.NIRGraph.from_list(e, f, g)

    ir = nir.NIRGraph(
        nodes={"branch_1": branch_1, "branch_2": branch_2},
        edges=[],
    )
    factory_test_graph(ir)


def test_two_independent_branches_merging():
    """
    ```mermaid
    graph TD;
    A --> B;
    B --> C;
    C --> D;
    E --> F;
    F --> G;
    G --> H;
    D --> H;
    ```
    """
    # Branch 1
    a = mock_affine(2, 3)
    b = nir.Delay(np.array([0.5, 0.1, 0.2]))
    c = nir.LIF(
        tau=np.array([10, 20, 30]),
        r=np.array([1, 1, 1]),
        v_leak=np.array([0, 0, 0]),
        v_reset=np.array([0, 0, 0]),
        v_threshold=np.array([1, 2, 3]),
    )
    d = mock_affine(3, 3)

    branch_1 = nir.NIRGraph.from_list(a, b, c, d)

    # Branch 2
    e = mock_affine(2, 2)
    f = nir.LIF(
        tau=np.array([10, 20]),
        r=np.array([1, 1]),
        v_leak=np.array([0, 0]),
        v_reset=np.array([0, 0]),
        v_threshold=np.array([1, 2]),
    )
    g = mock_affine(2, 3)

    branch_2 = nir.NIRGraph.from_list(e, f, g)

    # Junction
    # TODO: This should be a node that accepts two input_type
    h = nir.LIF(
        tau=np.array([5, 2, 1]),
        r=np.array([1, 1, 1]),
        v_leak=np.array([0, 0, 0]),
        v_reset=np.array([0, 0, 0]),
        v_threshold=np.array([1, 1, 1]),
    )

    ir = nir.NIRGraph(
        nodes={"branch_1": branch_1, "branch_2": branch_2, "junction": h},
        edges=[("branch_1", "junction"), ("branch_2", "junction")],
    )
    factory_test_graph(ir)


def test_merge_and_split_single_output():
    """
    ```mermaid
    graph TD;
    A --> B;
    B --> C;
    C --> D;
    B --> F;
    F --> G;
    G --> H;
    D --> H;
    ```
    """
    # Part before split
    a = mock_affine(2, 3)
    b = nir.LIF(
        tau=np.array([10, 20, 30]),
        r=np.array([1, 1, 1]),
        v_leak=np.array([0, 0, 0]),
        v_reset=np.array([0, 0, 0]),
        v_threshold=np.array([1, 2, 3]),
    )
    pre_split = nir.NIRGraph.from_list(a, b)

    # Branch 1
    c = mock_affine(3, 2)
    d = nir.LIF(
        tau=np.array([10, 20]),
        r=np.array([1, 1]),
        v_leak=np.array([0, 0]),
        v_reset=np.array([0, 0]),
        v_threshold=np.array([1, 2]),
    )
    branch_1 = nir.NIRGraph.from_list(c, d)

    # Branch 2
    e = mock_affine(3, 2)
    f = nir.LIF(
        tau=np.array([15, 5]),
        r=np.array([1, 1]),
        v_leak=np.array([0, 0]),
        v_reset=np.array([0, 0]),
        v_threshold=np.array([1, 1]),
    )
    branch_2 = nir.NIRGraph.from_list([e, f])

    # Junction
    # TODO: This should be a node that accepts two input_type
    g = nir.Affine(weight=np.array([[2, 0], [1, 3], [4, 1]]), bias=np.array([0, 1]))

    nodes = {
        "pre_split": pre_split,
        "branch_1": branch_1,
        "branch_2": branch_2,
        "junction": g,
    }
    edges = [
        ("pre_split", "branch_1"),
        ("pre_split", "branch_2"),
        ("branch_1", "junction"),
        ("branch_2", "junction"),
    ]
    ir = nir.NIRGraph(nodes=nodes, edges=edges)
    factory_test_graph(ir)


def test_merge_and_split_different_output_type():
    # Part before split
    a = mock_affine(3, 3)
    # TODO: This should be a node with two output_type
    b = nir.LIF(
        tau=np.array([10, 20, 30]),
        r=np.array([1, 1, 1]),
        v_leak=np.array([0, 0, 0]),
        v_reset=np.array([0, 0, 0]),
        v_threshold=np.array([1, 2, 3]),
    )
    pre_split = nir.NIRGraph.from_list([a, b])

    # Branch 1
    c = mock_affine(3, 2)
    d = nir.LIF(
        tau=np.array([10, 20]),
        r=np.array([1, 1]),
        v_leak=np.array([0, 0]),
        v_reset=np.array([0, 0]),
        v_threshold=np.array([1, 2]),
    )
    branch_1 = nir.NIRGraph.from_list([c, d])

    # Branch 2
    e = mock_affine(3, 2)
    f = nir.LIF(
        tau=np.array([15, 5]),
        r=np.array([1, 1]),
        v_leak=np.array([0, 0]),
        v_reset=np.array([0, 0]),
        v_threshold=np.array([1, 1]),
    )
    branch_2 = nir.NIRGraph.from_list([e, f])

    # Junction
    # TODO: This should be a node that accepts two input_type
    g = mock_affine(2, 2)

    nodes = {
        "pre_split": pre_split,
        "branch_1": branch_1,
        "branch_2": branch_2,
        "junction": g,
    }
    edges = [
        ("pre_split", "branch_1"),
        ("pre_split", "branch_2"),
        ("branch_1", "junction"),
        ("branch_2", "junction"),
    ]
    ir = nir.NIRGraph(nodes=nodes, edges=edges)
    factory_test_graph(ir)


def test_residual():
    """
    ```mermaid
    graph TD;
    A --> B;
    B --> C;
    C --> D;
    A --> E;
    D --> E;
    E --> F;
    ```
    """

    # Part before split
    a = mock_affine(2, 2)

    # Residual block
    b = nir.LIF(
        tau=np.array([10, 20]),
        r=np.array([1, 1]),
        v_leak=np.array([0, 0]),
        v_reset=np.array([0, 0]),
        v_threshold=np.array([1, 2]),
    )
    c = mock_affine(2, 2)
    d = nir.LIF(
        tau=np.array([10, 20]),
        r=np.array([1, 1]),
        v_leak=np.array([0, 0]),
        v_reset=np.array([0, 0]),
        v_threshold=np.array([1, 2]),
    )

    # Junction
    # TODO: This should be a node that accepts two input_type
    e = mock_affine(2, 2)
    f = nir.LIF(
        tau=np.array([15, 5]),
        r=np.array([1, 1]),
        v_leak=np.array([0, 0]),
        v_reset=np.array([0, 0]),
        v_threshold=np.array([1, 1]),
    )

    nodes = {
        "a": a,
        "b": b,
        "c": c,
        "d": d,
        "e": e,
        "f": f,
    }
    edges = [
        ("a", "b"),
        ("b", "c"),
        ("c", "d"),
        ("d", "e"),
        ("a", "e"),
        ("e", "f"),
    ]
    ir = nir.NIRGraph(nodes=nodes, edges=edges)
    factory_test_graph(ir)


def test_complex():
    """
    ```mermaid
    graph TD;
    A --> B;
    A --> C;
    C --> D;
    C --> E;
    B --> D;
    D --> F;
    E --> F;
    ```
    """
    a = nir.Affine(weight=np.array([[1], [2], [3]]), bias=np.array([[0, 0, 0]]))
    b = nir.LIF(
        tau=np.array([10, 20, 30]),
        r=np.array([1, 1, 1]),
        v_leak=np.array([0, 0, 0]),
        v_reset=np.array([0, 0, 0]),
        v_threshold=np.array([1, 2, 3]),
    )
    c = nir.LIF(
        tau=np.array([5, 20, 1]),
        r=np.array([1, 1, 1]),
        v_leak=np.array([0, 0, 0]),
        v_reset=np.array([0, 0, 0]),
        v_threshold=np.array([1, 1, 1]),
    )
    # TODO: This should be a node that accepts two input_type
    d = nir.Affine(
        weight=np.array([[1, 3], [2, 3], [1, 4]]).T,
        bias=np.array([0, 0]),
    )
    e = nir.Affine(weight=np.array([[1, 3], [2, 3], [1, 4]]).T, bias=np.array([0, 0]))
    # TODO: This should be a node that accepts two input_type
    f = nir.Affine(weight=np.array([[1, 3], [1, 4]]), bias=np.array([0, 0]))
    nodes = {
        "a": a,
        "b": b,
        "c": c,
        "d": d,
        "e": e,
        "f": f,
    }
    edges = [
        ("a", "b"),
        ("a", "c"),
        ("b", "d"),
        ("c", "d"),
        ("c", "e"),
        ("d", "f"),
        ("e", "f"),
    ]
    ir = nir.NIRGraph(nodes=nodes, edges=edges)
    factory_test_graph(ir)


@pytest.mark.skip("Not implemented")
def test_subgraph_multiple_input_output():
    """
    ```mermaid
    graph TD;
    subgraph G
    B; C;
    end
    A --> B;
    A --> C;
    B --> D;
    C --> D;
    ```
    """
    a = mock_affine(1, 3)
    b = mock_affine(3, 2)
    c = mock_affine(3, 2)
    d = mock_affine(2, 1)

    # Subgraph
    bi = nir.Input(b.input_type)
    ci = nir.Input(b.input_type)
    bo = nir.Output(b.output_type)
    co = nir.Output(c.output_type)
    g = nir.NIRGraph(
        nodes={"b": b, "c": c, "bi": bi, "ci": ci, "bo": bo, "co": co},
        edges=[("bi", "b"), ("b", "bo"), ("ci", "c"), ("c", "co")],
        type_check=False,
    )

    # Supgraph
    nir.NIRGraph(
        nodes={"a": a, "g": g, "d": d},
        edges=[("a", "g.bi"), ("a", "g.ci"), ("g.bo", "d"), ("g.co", "d")],
    )

    # TODO: Add type checking...
