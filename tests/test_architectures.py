import nir
from .test_readwrite import factory_test_graph

    
def test_sequential():
    a = nir.Affine(weight=[1, 2, 3], bias=[0, 0, 0])
    b = nir.Delay([0.5, 0.1, 0.2])
    c = nir.LIF(tau=[10, 20, 30], r=[1, 1, 1], v_leak=[0, 0, 0], v_threshold=[1, 2, 3])
    d = nir.Affine(weight=[[1, 0], [0, 2], [1, 1]], bias=[1, 1])
 
    ir = nir.NIRGraph.from_list(a, b, c, d)
    factory_test_graph(ir)

def test_two_independent_branches():
   
    # Branch 1
    a = nir.Affine(weight=[1, 2, 3], bias=[0, 0, 0])
    b = nir.Delay([0.5, 0.1, 0.2])
    c = nir.LIF(tau=[10, 20, 30], r=[1, 1, 1], v_leak=[0, 0, 0], v_threshold=[1, 2, 3])
    d = nir.Affine(weight=[[1, 0], [0, 2], [1, 1]], bias=[1, 1])
 
    branch_1 = nir.NIRGraph.from_list(a, b, c, d)
    
    # Branch 2
    e = nir.Affine(weight=[1, 2], bias=[0, 0])
    f = nir.LIF(tau=[10, 20], r=[1, 1], v_leak=[0, 0], v_threshold=[1, 2])
    g = nir.Affine(weight=[[1, 0], [0, 2]], bias=[1, 1])
    
    branch_2 = nir.NIRGraph.from_list(e, f, g)
 
    ir = nir.NIRGraph(
        nodes={"branch_1": branch_1, "branch_2": branch_2},
        edges=[],
    )
    factory_test_graph(ir)

def test_two_independent_branches_merging():
   
    # Branch 1
    a = nir.Affine(weight=[1, 2, 3], bias=[0, 0, 0])
    b = nir.Delay([0.5, 0.1, 0.2])
    c = nir.LIF(tau=[10, 20, 30], r=[1, 1, 1], v_leak=[0, 0, 0], v_threshold=[1, 2, 3])
    d = nir.Affine(weight=[[1, 0], [0, 2], [1, 1]], bias=[1, 1])
 
    branch_1 = nir.NIRGraph.from_list(a, b, c, d)
    
    # Branch 2
    e = nir.Affine(weight=[1, 2], bias=[0, 0])
    f = nir.LIF(tau=[10, 20], r=[1, 1], v_leak=[0, 0], v_threshold=[1, 2])
    g = nir.Affine(weight=[[1, 0], [0, 2]], bias=[1, 1])
    
    branch_2 = nir.NIRGraph.from_list(e, f, g)
 
    # Junction
    # TODO: This should be a node that accepts two inputs
    h = nir.LIF(tau=[5, 2], r=[1, 1], v_leak=[0, 0], v_threshold=[1, 1]) 

    ir = nir.NIRGraph(
        nodes={"branch_1": branch_1, "branch_2": branch_2, "junction": h},
        edges=[("branch_1", "junction"), ("branch_2", "junction")],
    )
    factory_test_graph(ir)

def merge_and_split_single_output():
    # Part before split
    a = nir.Affine(weight=[1, 2, 3], bias=[0, 0, 0])
    b = nir.LIF(tau=[10, 20, 30], r=[1, 1, 1], v_leak=[0, 0, 0], v_threshold=[1, 2, 3])
    pre_split = nir.NIRGraph.from_list(a, b)

    # Branch 1 
    c = nir.Affine(weight=[[1, 0], [0, 2], [1, 1]], bias=[1, 1])
    d = nir.LIF(tau=[10, 20], r=[1, 1], v_leak=[0, 0], v_threshold=[1, 2])
    branch_1 = nir.NIRGraph.from_list(c, d)
    expand_1 = nir.Project(output_indices=[0, float("nan")])

    # Branch 2
    e = nir.Affine(weight=[[2, 4], [1, 0], [0, 1]], bias=[2, 1])
    f = nir.LIF(tau=[15, 5], r=[1, 1], v_leak=[0, 0], v_threshold=[1, 1])
    branch_2 = nir.NIRGraph.from_list([e, f])
    expand_2 = nir.Project(output_indices=[float("nan"), 1])

    # Junction
    # TODO: This should be a node that accepts two inputs
    g = nir.Affine(weight=[[2, 0], [1, 3], [4, 1]], bias=[0, 1])

    nodes={
        "pre_split": pre_split,
        "branch_1": branch_1, 
        "branch_2": branch_2, 
        "expand_1": expand_1,
        "expand_2": expand_2,
        "junction": g,
    }
    edges=[
        ("pre_split", "branch_1"),
        ("pre_split", "branch_2"),
        ("branch_1", "expand_1"), 
        ("expand_1", "junction"), 
        ("branch_2", "expand_2"), 
        ("expand_2", "junction"), 
    ]
    ir = nir.NIRGraph(nodes=nodes, edges=edges)
    factory_test_graph(ir)

def merge_and_split_different_outputs():
    # Part before split
    a = nir.Affine(weight=[1, 2, 3], bias=[0, 0, 0])
    #TODO: This should be a node with two outputs
    b = nir.LIF(tau=[10, 20, 30], r=[1, 1, 1], v_leak=[0, 0, 0], v_threshold=[1, 2, 3])
    pre_split = nir.NIRGraph.from_list([a, b])

    # Branch 1 
    reduce_1 = nir.Project(output_indices=[0])
    c = nir.Affine(weight=[[1, 0], [0, 2], [1, 1]], bias=[1, 1])
    d = nir.LIF(tau=[10, 20], r=[1, 1], v_leak=[0, 0], v_threshold=[1, 2])
    branch_1 = nir.NIRGraph.from_list([c, d])
    expand_1 = nir.Project(output_indices=[0, float("nan")])

    # Branch 2
    reduce_2 = nir.Project(output_indices=[1])
    e = nir.Affine(weight=[[2, 4], [1, 0], [0, 1]], bias=[2, 1])
    f = nir.LIF(tau=[15, 5], r=[1, 1], v_leak=[0, 0], v_threshold=[1, 1])
    branch_2 = nir.NIRGraph.from_list([e, f])
    expand_2 = nir.Project(output_indices=[float("nan"), 1])

    # Junction
    # TODO: This should be a node that accepts two inputs
    g = nir.Affine(weight=[[2, 0], [1, 3], [4, 1]], bias=[0, 1])

    nodes={
        "pre_split": pre_split,
        "reduce_1": reduce_1,
        "reduce_2": reduce_2,
        "branch_1": branch_1, 
        "branch_2": branch_2, 
        "expand_1": expand_1,
        "expand_2": expand_2,
        "junction": g,
    }
    edges=[
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
    a = nir.Affine(weight=[1, 2, 3], bias=[0, 0, 0])

    # Residual block
    b = nir.LIF(tau=[10, 20, 30], r=[1, 1, 1], v_leak=[0, 0, 0], v_threshold=[1, 2, 3])
    c = nir.Affine(weight=[[1, 0], [0, 2], [1, 1]], bias=[1, 1])
    d = nir.LIF(tau=[10, 20], r=[1, 1], v_leak=[0, 0], v_threshold=[1, 2])

    # Junction
    expand_a = nir.Project(output_indices=[0, float("nan")])
    expand_d = nir.Project(output_indices=[float("nan"), 1])
    # TODO: This should be a node that accepts two inputs
    e = nir.Affine(weight=[[2, 4], [1, 0], [0, 1]], bias=[2, 1])
    f = nir.LIF(tau=[15, 5], r=[1, 1], v_leak=[0, 0], v_threshold=[1, 1])
    
    nodes={
        "a": a,
        "b": b,
        "c": c,
        "d": d, 
        "expand_a": expand_a,
        "expand_d": expand_d,
        "e": e,
        "f": f,
    }
    edges=[
        ("a", "b"),
        ("a", "expand_a"),
        ("b", "c"),
        ("d", "expand_d"),
        ("reduce_2", "branch_2"),
        ("branch_1", "expand_1"), 
        ("expand_1", "e"), 
        ("expand_2", "e"), 
        ("e", "f"), 
    ]
    ir = nir.NIRGraph(nodes=nodes, edges=edges)
    factory_test_graph(ir)

def test_complex():
    a = nir.Affine(weight=[1, 2, 3], bias=[0, 0, 0])
    b = nir.LIF(tau=[10, 20, 30], r=[1, 1, 1], v_leak=[0, 0, 0], v_threshold=[1, 2, 3])
    expand_b = nir.Project([0, float("nan")])
    c = nir.LIF(tau=[5, 20, 1], r=[1, 1, 1], v_leak=[0, 0, 0], v_threshold=[1, 1, 1])
    c_first = nir.Project([0])
    c_second = nir.Project([1])
    expand_c = nir.Project([float("nan"), 1])
    # TODO: This should be a node that accepts two inputs
    d = nir.Affine(
        weight=[[[1, 3], [2, 3], [1, 4]], [[2, 3], [1, 2], [1, 4]]],
        bias=[0, 0]
    )
    expand_d = nir.Project([0, float("nan")])
    e = nir.Affine(weight=[[1, 3], [2, 3], [1, 4]], bias=[0, 0])
    expand_e = nir.Project([float("nan"), 1])
    # TODO: This should be a node that accepts two inputs
    f = nir.Affine(
        weight=[[[1, 3], [1, 4]], [[2, 3], [3, 4]]],
        bias=[0, 0]
    )
    nodes={
        "a": a,
        "b": b,
        "expand_b": expand_b,
        "c": c,
        "c_first": c_first,
        "c_second": c_second,
        "expand_c": expand_c,
        "d": d, 
        "expand_d": expand_d,
        "e": e,
        "expand_e": expand_e,
        "f": f,
    }
    edges=[
        ("a", "b"),
        ("a", "c"),
        ("b", "expand_b"),
        ("expand_b", "d"),
        ("c", "c_first"),
        ("c_first", "expand_c"),
        ("expand_c", "d"),
        ("c", "c_second"),
        ("c_second", "e"),
        ("d", "expand_d"),
        ("e", "expand_e"),
        ("expand_d", "f"), 
        ("expand_e", "f"), 
    ]
    ir = nir.NIRGraph(nodes=nodes, edges=edges)
    factory_test_graph(ir)
