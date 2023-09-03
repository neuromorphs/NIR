import nir
from .test_readwrite import factory_test_graph
from tests import *


def test_sequential():
    a = mock_affine(2, 2)
    b = nir.Delay(np.array([0.5, 0.1, 0.2]))
    c = nir.LIF(
        tau=np.array([10, 20, 30]),
        r=np.array([1, 1, 1]),
        v_leak=np.array([0, 0, 0]),
        v_threshold=np.array([1, 2, 3]),
    )
    d = mock_affine(3, 2)

    ir = nir.NIRGraph.from_list(a, b, c, d)
    factory_test_graph(ir)


def test_two_independent_branches():
    # Branch 1
    a = mock_affine(2, 3)
    b = nir.Delay(np.array([0.5, 0.1, 0.2]))
    c = nir.LIF(
        tau=np.array([10, 20, 30]),
        r=np.array([1, 1, 1]),
        v_leak=np.array([0, 0, 0]),
        v_threshold=np.array([1, 2, 3]),
    )
    d = mock_affine(2, 3)

    branch_1 = nir.NIRGraph.from_list(a, b, c, d)

    # Branch 2
    e = mock_affine(2, 3)
    f = nir.LIF(
        tau=np.array([10, 20]),
        r=np.array([1, 1]),
        v_leak=np.array([0, 0]),
        v_threshold=np.array([1, 2]),
    )
    g = mock_affine(3, 2)

    branch_2 = nir.NIRGraph.from_list(e, f, g)

    ir = nir.NIRGraph(
        nodes={"branch_1": branch_1, "branch_2": branch_2},
        edges=[],
    )
    factory_test_graph(ir)


def test_two_independent_branches_merging():
    # Branch 1
    a = mock_affine(2, 3)
    b = nir.Delay(np.array([0.5, 0.1, 0.2]))
    c = nir.LIF(
        tau=np.array([10, 20, 30]),
        r=np.array([1, 1, 1]),
        v_leak=np.array([0, 0, 0]),
        v_threshold=np.array([1, 2, 3]),
    )
    d = mock_affine(2, 3)

    branch_1 = nir.NIRGraph.from_list(a, b, c, d)

    # Branch 2
    e = mock_affine(2, 3)
    f = nir.LIF(
        tau=np.array([10, 20]),
        r=np.array([1, 1]),
        v_leak=np.array([0, 0]),
        v_threshold=np.array([1, 2]),
    )
    g = mock_affine(3, 2)

    branch_2 = nir.NIRGraph.from_list(e, f, g)

    # Junction
    # TODO: This should be a node that accepts two inputs
    h = nir.LIF(
        tau=np.array([5, 2]),
        r=np.array([1, 1]),
        v_leak=np.array([0, 0]),
        v_threshold=np.array([1, 1]),
    )

    ir = nir.NIRGraph(
        nodes={"branch_1": branch_1, "branch_2": branch_2, "junction": h},
        edges=[("branch_1", "junction"), ("branch_2", "junction")],
    )
    factory_test_graph(ir)


def test_merge_and_split_single_output():
    # Part before split
    a = mock_affine(2, 3)
    b = nir.LIF(
        tau=np.array([10, 20, 30]),
        r=np.array([1, 1, 1]),
        v_leak=np.array([0, 0, 0]),
        v_threshold=np.array([1, 2, 3]),
    )
    pre_split = nir.NIRGraph.from_list(a, b)

    # Branch 1
    c = mock_affine(2, 3)
    d = nir.LIF(
        tau=np.array([10, 20]),
        r=np.array([1, 1]),
        v_leak=np.array([0, 0]),
        v_threshold=np.array([1, 2]),
    )
    branch_1 = nir.NIRGraph.from_list(c, d)

    # Branch 2
    e = mock_affine(2, 3)
    f = nir.LIF(
        tau=np.array([15, 5]),
        r=np.array([1, 1]),
        v_leak=np.array([0, 0]),
        v_threshold=np.array([1, 1]),
    )
    branch_2 = nir.NIRGraph.from_list([e, f])

    # Junction
    # TODO: This should be a node that accepts two inputs
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


def merge_and_split_different_outputs():
    # Part before split
    a = mock_affine(3, 2)
    # TODO: This should be a node with two outputs
    b = nir.LIF(
        tau=np.array([10, 20, 30]),
        r=np.array([1, 1, 1]),
        v_leak=np.array([0, 0, 0]),
        v_threshold=np.array([1, 2, 3]),
    )
    pre_split = nir.NIRGraph.from_list([a, b])

    # Branch 1
    reduce_1 = nir.Project(output_indices=[0])
    c = mock_affine(3, 2)
    d = nir.LIF(
        tau=np.array([10, 20]),
        r=np.array([1, 1]),
        v_leak=np.array([0, 0]),
        v_threshold=np.array([1, 2]),
    )
    branch_1 = nir.NIRGraph.from_list([c, d])
    expand_1 = nir.Project(output_indices=[0, float("nan")])

    # Branch 2
    reduce_2 = nir.Project(output_indices=[1])
    e = mock_affine(3, 2)
    f = nir.LIF(
        tau=np.array([15, 5]),
        r=np.array([1, 1]),
        v_leak=np.array([0, 0]),
        v_threshold=np.array([1, 1]),
    )
    branch_2 = nir.NIRGraph.from_list([e, f])
    expand_2 = nir.Project(output_indices=[float("nan"), 1])

    # Junction
    # TODO: This should be a node that accepts two inputs
    g = mock_affine(3, 2)

    nodes = {
        "pre_split": pre_split,
        "reduce_1": reduce_1,
        "reduce_2": reduce_2,
        "branch_1": branch_1,
        "branch_2": branch_2,
        "expand_1": expand_1,
        "expand_2": expand_2,
        "junction": g,
    }
    edges = [
        ("pre_split", "reduce_1"),
        ("pre_split", "reduce_2"),
        ("reduce_1", "branch_1"),
        ("reduce_2", "branch_2"),
        ("branch_1", "expand_1"),
        ("expand_1", "junction"),
        ("branch_2", "expand_2"),
        ("expand_2", "junction"),
    ]
    ir = nir.NIRGraph(nodes=nodes, edges=edges)
    factory_test_graph(ir)


def test_residual():
    # Part before split
    a = mock_affine(2, 3)

    # Residual block
    b = nir.LIF(
        tau=np.array([10, 20, 30]),
        r=np.array([1, 1, 1]),
        v_leak=np.array([0, 0, 0]),
        v_threshold=np.array([1, 2, 3]),
    )
    c = mock_affine(3, 2)
    d = nir.LIF(
        tau=np.array([10, 20]),
        r=np.array([1, 1]),
        v_leak=np.array([0, 0]),
        v_threshold=np.array([1, 2]),
    )

    # Junction
    # TODO: This should be a node that accepts two inputs
    e = mock_affine(3, 2)
    f = nir.LIF(
        tau=np.array([15, 5]),
        r=np.array([1, 1]),
        v_leak=np.array([0, 0]),
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
    a = nir.Affine(weight=np.array([[1, 2, 3]]), bias=np.array([[0, 0, 0]]))
    b = nir.LIF(
        tau=np.array([10, 20, 30]),
        r=np.array([1, 1, 1]),
        v_leak=np.array([0, 0, 0]),
        v_threshold=np.array([1, 2, 3]),
    )
    c = nir.LIF(
        tau=np.array([5, 20, 1]),
        r=np.array([1, 1, 1]),
        v_leak=np.array([0, 0, 0]),
        v_threshold=np.array([1, 1, 1]),
    )
    # TODO: This should be a node that accepts two inputs
    d = nir.Affine(
        weight=np.array([[[1, 3], [2, 3], [1, 4]], [[2, 3], [1, 2], [1, 4]]]),
        bias=np.array([0, 0]),
    )
    e = nir.Affine(weight=np.array([[1, 3], [2, 3], [1, 4]]), bias=np.array([0, 0]))
    # TODO: This should be a node that accepts two inputs
    f = nir.Affine(
        weight=np.array([[[1, 3], [1, 4]], [[2, 3], [3, 4]]]), bias=np.array([0, 0])
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
        ("a", "c"),
        ("b", "d"),
        ("c", "d"),
        ("c", "e"),
        ("d", "f"),
        ("e", "f"),
    ]
    ir = nir.NIRGraph(nodes=nodes, edges=edges)
    factory_test_graph(ir)
