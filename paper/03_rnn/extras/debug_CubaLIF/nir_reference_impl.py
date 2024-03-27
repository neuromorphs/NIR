import numpy as np

import nir


class CubaLIFImplementation(object):
    """Numpy forward Euler implmementation of CubaLIF Node."""

    def __init__(self, dt, cuba_node: nir.CubaLIF):
        self.dt = dt  # scalar
        self.node = cuba_node
        self.v = np.zeros_like(self.node.v_threshold)
        self.I = np.zeros_like(self.node.v_threshold)

    def forward(self, x):
        I_last = self.I.copy()
        v_last = self.v.copy()
        self.I = I_last + (self.dt / self.node.tau_syn) * (-I_last + self.node.w_in * x)
        self.v = v_last + (self.dt / self.node.tau_mem) * (
            self.node.v_leak - v_last + self.node.r * I_last
        )
        z = self.v > self.node.v_threshold
        self.v -= z * self.node.v_threshold
        return z, self.v, self.I


def run_cuba_reference_model(ref_model: CubaLIFImplementation, input_data: np.ndarray):
    assert input_data.shape[1:] == ref_model.v.shape

    spikes = np.zeros_like(input_data)
    voltages = np.zeros_like(input_data)
    currents = np.zeros_like(input_data)

    for t, input_this_timestep in enumerate(input_data):
        z, v, curr = ref_model.forward(input_this_timestep)
        spikes[t, :] = z
        voltages[t, :] = v
        currents[t, :] = curr

    return dict(spikes=spikes, voltages=voltages, currents=currents)
