import tempfile

import numpy as np

import nir


def factory_test_graph(ir: nir.NIR):
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
    ir = nir.NIR(nodes=[nir.Linear(weights=[1, 2, 3], bias=4)], edges=[(0, 0)])
    factory_test_graph(ir)


def test_leaky_integrator():
    ir = nir.NIR(
        nodes=[nir.Linear(weights=[1], bias=0), nir.LI(tau=1, r=2, v_leak=3)],
        edges=[(0, 0)],
    )
    factory_test_graph(ir)


def test_leaky_integrator_and_fire():
    ir = nir.NIR(
        nodes=[nir.Linear(weights=[1], bias=0), nir.LIF(tau=1, r=2, v_leak=3, v_th=4)],
        edges=[(0, 0)],
    )
    factory_test_graph(ir)

def test_simple_with_read_write():
    ir = nir.NIR(
        nodes=[nir.Input(shape=[3,]),
               nir.Linear(weights=[1, 2, 3], bias=4),
               nir.Output()],
        edges=[(0, 1), (1,2)]
    )
    factory_test_graph(ir)
