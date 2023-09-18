import matplotlib.pyplot as plt
import numpy as np
from spinnaker2 import brian2_sim, hardware, helpers, s2_nir

import nir

np.random.seed(123)

timesteps = 50

# load NIR model
nir_model = nir.read("lif_norse.nir")
print(nir_model)

# patch nir model
LIF_node = nir_model.nodes["1"]
assert isinstance(LIF_node, nir.LIF)

print(LIF_node)

# scale weights
scale = 102.02
LIF_node.v_threshold =  LIF_node.v_threshold*scale

Affine_node = nir_model.nodes["0"]
assert isinstance(Affine_node, nir.Affine)
Affine_node.weight = Affine_node.weight*scale
Affine_node.bias = Affine_node.bias*scale

net, inp, outp = s2_nir.from_nir(
    nir_model, outputs=["v", "spikes"], discretization_timestep=0.0001, conn_delay=0
)

print(outp[0].params)

# set input data
d0 = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
d = []
for t in d0:
    d.append(t)
    for i in range(9):
        d.append(0)
d = np.array(d, dtype=np.int32)

#d = d[0:900]

print(len(d))
input_spikes = {}
input_spikes[0] = d.nonzero()[0].tolist()
print(input_spikes)

inp[0].params = input_spikes

#hw = hardware.SpiNNaker2Chip(eth_ip="192.168.1.25")  # use ethernet
hw = brian2_sim.Brian2Backend()

timesteps = len(d)+1
hw.run(net, timesteps)


spike_times = outp[0].get_spikes()
voltages = outp[0].get_voltages()

fig, (ax1, ax2, ax3) = plt.subplots(3,1, sharex=True)

indices, times = helpers.spike_times_dict_to_arrays(input_spikes)
ax1.plot(times, indices, "|", ms=20)
ax1.set_ylabel("input spikes")
ax1.set_ylim((-0.5, inp[0].size-0.5))

times = np.arange(timesteps)
ax2.plot(times, voltages[0], label="Neuron 0") 
ax2.axhline(outp[0].params["threshold"][0], ls="--", c="0.5", label="threshold")
ax2.axhline(0, ls="-", c="0.8", zorder=0)
ax2.set_xlim(0,timesteps)
ax2.set_ylabel("voltage")

indices, times = helpers.spike_times_dict_to_arrays(spike_times)
ax3.plot(times, indices, "|", ms=20)
ax3.set_ylabel("output spikes")
ax3.set_xlabel("time step")
ax3.set_ylim((-0.5, outp[0].size-0.5))
fig.suptitle("lif_neuron")
plt.show()


# save data

d = d # input spikes
v = voltages[0][1:]
v = v/scale
spikes = np.zeros(timesteps, dtype=float)
spikes[spike_times[0]] = 1
s = spikes[1:]

ar = np.array((d,v,s))

np.savetxt("lif_spinnaker2.csv", ar.T, delimiter=",")
