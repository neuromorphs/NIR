import numpy as np
import tonic
import torch
import torch.nn as nn
from sinabs import layers, set_batch_size
from sinabs.backend.dynapcnn import DynapcnnNetwork
from sinabs.backend.dynapcnn.chip_factory import ChipFactory
from sinabs.nir import from_nir

import nir


def xytp_collate(batch):
    samples = []
    targets = []

    for data, tgt in batch:
        samples.append(data)
        targets.append(tgt)

    return samples, torch.tensor(targets)


nir_graph = nir.read("cnn_sinabs.nir")

# Load sinabs model from nir graph
extracted_sinabs_model = from_nir(nir_graph, batch_size=1)
# print(extracted_sinabs_model)


# Try a forward call on the generated model
sample_data = torch.rand((100, 2, 34, 34))
out = extracted_sinabs_model(sample_data)
print(out.shape)


# Convert model to sequential for speck
sequential_model = nn.Sequential(
    *[
        node.elem
        for node in extracted_sinabs_model.execution_order
        if not isinstance(node.elem, nn.Identity)
    ]
)

# Speck does not support individual parameters per neuron.
# Set threshold to a single value per spiking layer
for mod in sequential_model:
    if isinstance(mod, layers.IAF):
        # Choose threshold to be the same for all neurons
        mod.spike_threshold = torch.nn.Parameter(
            torch.tensor([mod.spike_threshold.flatten()[0]])
        )
        mod.min_v_mem = torch.nn.Parameter(torch.tensor([mod.min_v_mem.flatten()[0]]))

speck_model = DynapcnnNetwork(
    sequential_model, input_shape=(2, 34, 34), discretize=True
)

# Generate dataset
test_dataset = tonic.datasets.NMNIST(".", train=False)

# Update the model to the batch size of the experiment
set_batch_size(speck_model, 1)

# Define device
hw_device = "speck2fmodule:0"
speck_model.to(hw_device, monitor_layers=[0])
layer_in = speck_model.chip_layers_ordering[0]
chip_factory = ChipFactory(hw_device)

data_numbers = {}
activities = {}

# Get the first sample
for data, label in test_dataset:
    if label in data_numbers:
        continue
    data_numbers[label] = data


for label, data in data_numbers.items():
    speck_model.reset_states()
    # Do a forward pass
    with torch.no_grad():
        in_events = chip_factory.xytp_to_events(
            data, layer=layer_in, reset_timestamps=True
        )
        out_events = speck_model(in_events)
        # set dt to high value, don't need time dimension
        out_raster = chip_factory.events_to_raster(
            out_events, dt=0.5, shape=(16, 16, 16)
        )
        activities[label] = out_raster

activities_array = []
for i in range(10):
    activities_array.append(activities[i].numpy())

activities_array = np.stack(activities_array, axis=1)
np.save("speck_activity.npy", activities_array)
