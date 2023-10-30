#!/usr/bin/env python
# coding: utf-8
"""Run Braille inference on SpiNNaker2."""

import os
from itertools import islice

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
    spike_counts_s2 = 0
    for i, row in enumerate(input_array):
        input_spikes[i] = np.where(row == 1)[0].astype(int).tolist()
        spike_counts_s2 += len(input_spikes[i])
    return input_spikes


### Load NIR Graph

# load NIR model
model_name = "retrained_zero"
backend = "S2"  # or "brian2"

if model_name.endswith("zero"):
    reset_method = s2_nir.ResetMethod.ZERO
elif model_name.endswith("subtract"):
    reset_method = s2_nir.ResetMethod.SUBTRACT
else:
    raise Exception("unsupported reset")

nir_model = nir.read(f"braille_{model_name}.nir")

print("nodes:")
for nodekey, node in nir_model.nodes.items():
    print("\t", nodekey, node.__class__.__name__, node.input_type["input"].dtype)
print("edges:")
for edge in nir_model.edges:
    print("\t", edge)

# s2_nir.add_output_to_node("lif1.lif", nir_model, "ouput_lif1")

cfg = s2_nir.ConversionConfig(
    output_record=["v", "spikes"],
    dt=0.0001,
    conn_delay=0,
    scale_weights=True,
    reset=reset_method,
    integrator=s2_nir.IntegratorMethod.FORWARD,
)
net, inp, outp = s2_nir.from_nir(nir_model, cfg)
assert len(inp) == 1  # make sure there is only one input pop

for pop in net.populations:
    if pop.name == "lif1.lif":
        pop.set_max_atoms_per_core(10)

### TEST DATA
test_data_path = "data/ds_test.pt"
ds_test = torch.load(test_data_path)
letter_written = ["Space", "A", "E", "I", "O", "U", "Y"]

### RUN TESTS
n_samples = 140  # dataset size: 140
do_plot = False
predicted_labels = []
actual_labels = []

my_loader = DataLoader(ds_test, batch_size=1, shuffle=False)

for single_sample in islice(my_loader, n_samples):
    sample = single_sample[0].numpy()  # shape: (1, 256, 12)

    spike_times = input_array_to_spike_list(sample[0, :, :].T)

    inp[0].params = spike_times
    # inp[0].params = {}

    timesteps = sample.shape[1]
    net.reset()  # clear previous spikes and voltages

    if backend == "S2":
        hw = hardware.SpiNNaker2Chip(eth_ip="192.168.1.25")  # use ethernet
        hw.run(net, timesteps, debug=False)
    else:
        hw = brian2_sim.Brian2Backend()
        hw.run(net, timesteps, quantize_weights=False)

    spike_times = outp[0].get_spikes()
    voltages = outp[0].get_voltages()
    v_scale = outp[0].nir_v_scale

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
accuracy = np.round(n_correct / n_samples * 100, 2)
np.save(f"s2_accuracy_{model_name}.npy", accuracy)
print(f"Test accuracy: {accuracy}%")
