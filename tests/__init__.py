import numpy as np

import nir


def mock_linear(*shape):
    return nir.Linear(weight=np.random.randn(*shape).T)


def mock_affine(*shape):
    return nir.Affine(weight=np.random.randn(*shape).T, bias=np.random.randn(shape[1]))


def mock_input(*shape):
    return nir.Input(input_type=np.array(shape))


def mock_integrator(*shape):
    return nir.I(r=np.random.randn(*shape))


def mock_output(*shape):
    return nir.Output(output_type=np.array(shape))


def mock_delay(*shape):
    return nir.Delay(delay=np.random.randn(*shape))
