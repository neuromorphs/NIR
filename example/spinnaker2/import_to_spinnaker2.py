import matplotlib.pyplot as plt
import numpy as np
from spinnaker2 import hardware, helpers, s2_nir

import nir

np.random.seed(123)

timesteps = 50

# load NIR model
nir_model = nir.read("nir_model.hdf5")
print(nir_model)
net, inp, outp = s2_nir.from_nir(
    nir_model, outputs=["v", "spikes"], discretization_timestep=1, conn_delay=0
)


# Create some input spikes
input_size = inp[0].size
input_spikes = {}
input_data = np.random.randn(input_size, timesteps)
input_data = (input_data > 1) * 1
print(input_data)

for i in range(input_size):
    input_spikes[i] = input_data[i].nonzero()[0].tolist()

inp[0].params = input_spikes


# Load up hardware + run
hw = hardware.SpiNNaker2Chip(eth_ip="192.168.1.52")  # use ethernet

timesteps += 3
hw.run(net, timesteps)


# get results and plot
spike_times = outp[0].get_spikes()
voltages = outp[0].get_voltages()

fig, axs = plt.subplots(1, 3, sharex=True)

indices, times = helpers.spike_times_dict_to_arrays(input_spikes)
axs[0].plot(times, indices, ".")
axs[0].set_xlim(0, timesteps)
axs[0].set_ylim(-0.5, len(input_spikes.keys()) - 0.5)
axs[0].set_ylabel("neuron")

indices, times = helpers.spike_times_dict_to_arrays(spike_times)
axs[1].plot(times, indices, ".")
axs[1].set_xlim(0, timesteps)
axs[1].set_ylim(-0.5, outp[0].size - 0.5)
axs[1].set_ylabel("neuron")

for i, _ in voltages.items():
    axs[2].plot(_, label=i)
    axs[2].set_xlabel("time step")
    axs[2].set_ylabel("membrane potential")
    axs[2].set_xlim(0, timesteps)
    axs[2].legend()
    axs[2].grid()

plt.show()
