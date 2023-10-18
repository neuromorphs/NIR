#!/usr/bin/env python
# coding: utf-8
"""Run Braille inference on SpiNNaker2."""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from spinnaker2 import brian2_sim, hardware, s2_nir
from torch.utils.data import DataLoader

import nir

seed = 42
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.use_deterministic_algorithms(True)

### DEVICE SETTINGS
device = torch.device("cpu")


def input_array_to_spike_list(input_array):
    input_spikes = {}
    for i, row in enumerate(input_array):
        input_spikes[i] = np.where(row == 1)[0].astype(int).tolist()
    return input_spikes


### Load NIR Graph

# load NIR model
nir_model = nir.read("braille_flattened.nir")


def apply_shape_to_one_to_one_nodes(node, new_shape):
    print(new_shape)
    print(np.array(new_shape).dtype)
    node.input_type["input"] = np.array(new_shape)
    node.output_type["output"] = np.array(new_shape)


# apply shapes to CubaLIF nodes.
apply_shape_to_one_to_one_nodes(
    nir_model.nodes["lif1"], nir_model.nodes["fc1"].output_type["output"]
)
apply_shape_to_one_to_one_nodes(
    nir_model.nodes["lif2"], nir_model.nodes["fc2"].output_type["output"]
)

# try if recurrent weights are in right order
# nir_model.nodes["w_rec"].weight =  nir_model.nodes["w_rec"].weight.T

# add recording of lif1
# nir_model.nodes["output_l1"] = nir.Output(np.array([7]))
# nir_model.edges.append(("lif1", "output_l1"))

print("nodes:")
for nodekey, node in nir_model.nodes.items():
    print("\t", nodekey, node.__class__.__name__, node.input_type["input"].dtype)
print("edges:")
for edge in nir_model.edges:
    print("\t", edge)

net, inp, outp = s2_nir.from_nir(
    nir_model,
    outputs=["v", "spikes"],
    discretization_timestep=1.0,
    conn_delay=0,
    scale_weights=False,
)

for pop in net.populations:
    if pop.name == "lif1":
        pop.set_max_atoms_per_core(10)
        # pop.params["i_offset"] = 0.
        print(pop.params)


### TEST DATA
test_data_path = "data/ds_test.pt"
ds_test = torch.load(test_data_path)
letter_written = ["Space", "A", "E", "I", "O", "U", "Y"]

### RUN TESTS
n_samples = 140  # dataset size: 140
do_plot = False
predicted_labels = []
actual_labels = []

for i in range(n_samples):
    single_sample = next(iter(DataLoader(ds_test, batch_size=1, shuffle=True)))
    sample = single_sample[0].numpy()  # shape: (1, 256, 12)

    spike_times = input_array_to_spike_list(sample[0, :, :].T)

    inp[0].params = spike_times

    hw = hardware.SpiNNaker2Chip(eth_ip="192.168.1.25")  # use ethernet
    hw = brian2_sim.Brian2Backend()

    timesteps = sample.shape[1]
    net.reset()  # clear previous spikes and voltages
    hw.run(net, timesteps, quantize_weights=False)

    spike_times = outp[0].get_spikes()
    voltages = outp[0].get_voltages()
    v_scale = outp[0].nir_v_scale
    print(v_scale)

    # print(spike_times)
    # print(outp[0].params)

    n_output_spikes = np.zeros(len(spike_times))
    for i, s in spike_times.items():
        n_output_spikes[i] = len(s)

    print(n_output_spikes)
    predicted_label = int(np.argmax(n_output_spikes))
    actual_label = int(single_sample[1])
    print("Predicted Label:", predicted_label)
    print("Actual Label:   ", actual_label)
    predicted_labels.append(predicted_label)
    actual_labels.append(actual_label)

    if do_plot:
        times = np.arange(timesteps)
        for i, vs in voltages.items():
            plt.plot(times, vs / v_scale, label=str(i))
        plt.xlim(0, timesteps)
        plt.xlabel("time step")
        plt.ylabel("voltage")
        plt.legend(title="Neuron")
        plt.show()

predicted_labels = np.array(predicted_labels)
actual_labels = np.array(actual_labels)
n_correct = np.count_nonzero(predicted_labels == actual_labels)
print("n_correct", n_correct, "out of", n_samples)
