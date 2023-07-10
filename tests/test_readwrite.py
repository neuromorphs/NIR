import tempfile

import numpy as np

import nir


def factory_test_graph(ir: nir.NIRGraph):
    tmp = tempfile.mktemp()
    nir.write(tmp, ir)
    ir2 = nir.read(tmp)
    for i in range(len(ir.nodes)):
        for k, v in ir.nodes[i].__dict__.items():
            if isinstance(v, np.ndarray) or isinstance(v, list):
                assert np.array_equal(v, getattr(ir2.nodes[i], k))
            else:
                assert v == getattr(ir2.nodes[i], k)


def test_simple():
    ir = nir.NIRGraph(nodes=[nir.Affine(weight=[1, 2, 3], bias=4)], edges=[(0, 0)])
    factory_test_graph(ir)


def test_integrator():
    ir = nir.NIRGraph(
        nodes=[nir.Affine(weight=[1], bias=0), nir.I(r=2)],
        edges=[(0, 0)],
    )
    factory_test_graph(ir)


def test_integrate_and_fire():
    ir = nir.NIRGraph(
        nodes=[nir.Affine(weight=[1], bias=0), nir.IF(r=2, v_threshold=3)],
        edges=[(0, 0)],
    )
    factory_test_graph(ir)


def test_leaky_integrator():
    ir = nir.NIRGraph(
        nodes=[nir.Affine(weight=[1], bias=0), nir.LI(tau=1, r=2, v_leak=3)],
        edges=[(0, 0)],
    )
    factory_test_graph(ir)


def test_linear():
    ir = nir.NIRGraph(
        nodes=[nir.Linear(weight=[1]), nir.LI(tau=1, r=2, v_leak=3)],
        edges=[(0, 0)],
    )
    factory_test_graph(ir)


def test_leaky_integrator_and_fire():
    ir = nir.NIRGraph(
        nodes=[
            nir.Affine(weight=[1], bias=0),
            nir.LIF(tau=1, r=2, v_leak=3, v_threshold=4),
        ],
        edges=[(0, 0)],
    )
    factory_test_graph(ir)


def test_simple_with_read_write():
    ir = nir.NIRGraph(
        nodes=[
            nir.Input(
                shape=[
                    3,
                ]
            ),
            nir.Affine(weight=[1, 2, 3], bias=4),
            nir.Output(),
        ],
        edges=[(0, 1), (1, 2)],
    )
    factory_test_graph(ir)


def test_delay():
    ir = nir.NIRGraph(
        nodes=[
            nir.Input(
                shape=[
                    3,
                ]
            ),
            nir.Delay(delay=[1, 2, 3]),
            nir.Output(),
        ],
        edges=[(0, 1), (1, 2)],
    )
    factory_test_graph(ir)


def test_threshold():
    ir = nir.NIRGraph(
        nodes=[
            nir.Input(
                shape=[
                    3,
                ]
            ),
            nir.Threshold(threshold=[2.0, 2.5, 2.8]),
            nir.Output(),
        ],
        edges=[(0, 1), (1, 2)],
    )
    factory_test_graph(ir)
