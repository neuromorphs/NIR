import numpy as np

import nir


def test_simple():
    w = np.array([1, 2, 3])
    b = np.array([4, 4, 4])
    ir = nir.NIRGraph(nodes={"a": nir.Affine(weight=w, bias=b)}, edges=[("a", "a")])
    assert np.allclose(ir.nodes["a"].weight, w)
    assert np.allclose(ir.nodes["a"].bias, b)
    assert ir.edges == [("a", "a")]


def test_nested():
    r = np.array([1, 1])
    delay = np.array([2, 2])
    w = np.array([1, 2])
    b = np.array([4, 4])
    nested = nir.NIRGraph(
        nodes={
            "integrator": nir.I(r=r),
            "delay": nir.Delay(delay),
        },
        edges=[("integrator", "delay"), ("delay", "integrator")],
    )
    ir = nir.NIRGraph(
        nodes={"affine": nir.Affine(weight=w, bias=b), "inner": nested},
        edges=[("affine", "inner")],
    )
    assert np.allclose(ir.nodes["affine"].weight, w)
    assert np.allclose(ir.nodes["affine"].bias, b)
    assert np.allclose(ir.nodes["inner"].nodes["integrator"].r, r)
    assert np.allclose(ir.nodes["inner"].nodes["delay"].delay, delay)
    assert ir.nodes["inner"].edges == [("integrator", "delay"), ("delay", "integrator")]


def test_simple_with_input_output():
    w = np.array([1, 2, 3])
    b = np.array([4, 4, 4])
    ir = nir.NIRGraph(
        nodes={
            "in": nir.Input(np.array([3])),
            "w": nir.Affine(weight=w, bias=b),
            "out": nir.Output(np.array([3])),
        },
        edges=[("in", "w"), ("w", "out")],
    )
    assert ir.nodes["in"].shape == [
        3,
    ]
    assert np.allclose(ir.nodes["w"].weight, w)
    assert np.allclose(ir.nodes["w"].bias, b)
    assert ir.edges == [("in", "w"), ("w", "out")]


def test_delay():
    delay = np.array([1, 2, 3])
    ir = nir.NIRGraph(
        nodes={
            "in": nir.Input(np.array([3])),
            "d": nir.Delay(delay=delay),
            "out": nir.Output(np.array([3])),
        },
        edges=[("in", "d"), ("d", "out")],
    )
    assert ir.nodes["in"].shape == [
        3,
    ]
    assert np.allclose(ir.nodes["d"].delay, delay)
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
    assert ir.nodes["in"].shape == [
        3,
    ]
    assert np.allclose(ir.nodes["thr"].threshold, threshold)
    assert ir.edges == [("in", "thr"), ("thr", "out")]


def test_linear():
    w = np.array([1, 2, 3])
    ir = nir.NIRGraph(nodes={"a": nir.Linear(weight=w)}, edges=[("a", "a")])
    assert np.allclose(ir.nodes["a"].weight, w)
    assert ir.edges == [("a", "a")]


def test_flatten():
    ir = nir.NIRGraph(
        nodes={
            "in": nir.Input(np.array([4, 5, 2])),
            "flat": nir.Flatten(0),
            "out": nir.Output(np.array([20, 2])),
        },
        edges=[("in", "flat"), ("flat", "out")],
    )
    assert np.allclose(ir.nodes["in"].shape, [4, 5, 2])
    assert np.allclose(ir.nodes["out"].shape, [20, 2])
