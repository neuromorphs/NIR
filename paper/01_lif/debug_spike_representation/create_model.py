import numpy as np

import nir

nir_model = nir.NIRGraph(
    nodes={
        "in": nir.Input(input_type=np.array([1])),  # dummy input, but needed for S2
        "linear1": nir.Linear(
            weight=np.array([[1.0]]),
        ),
        "lif1": nir.LIF(
            tau=np.array([0.01]),
            r=np.array([1.0]),
            v_leak=np.array([1.2]),  # above threshold -> self-spiking
            v_threshold=np.array([1.0]),
        ),
        "linear2": nir.Linear(
            weight=np.array([[1.0]]),
        ),
        "lif2": nir.LIF(
            tau=np.array([0.01]),
            r=np.array([1.0]),
            v_leak=np.array([0.0]),
            v_threshold=np.array([20.0]),
        ),
        "out": nir.Output(output_type=np.array([1])),
    },
    edges=[
        ("in", "linear1"),
        ("linear1", "lif1"),
        ("lif1", "linear2"),
        ("linear2", "lif2"),
        ("lif2", "out"),
    ],
)
nir.write("two_lif_neurons.nir", nir_model)
