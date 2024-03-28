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


nir_graph = nir.read("scnn_mnist.nir")

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


test_dataset = tonic.datasets.NMNIST(".", train=False)

# Define dataloader
batch_size = 32

data_loader = torch.utils.data.DataLoader(
    test_dataset,
    shuffle=False,
    batch_size=batch_size,
    collate_fn=xytp_collate,
)
# Update the model to the batch size of the experiment
set_batch_size(speck_model, batch_size)

# Testing loop
true_pos = []

# Define device
hw_device = "speck2fmodule:0"
speck_model.to(hw_device, monitor_layers=[-1])
layer_in = speck_model.chip_layers_ordering[0]
chip_factory = ChipFactory(hw_device)


# Iterate through the data
i = 0
for data, label in data_loader:
    batch_size = len(label)
    set_batch_size(speck_model, batch_size)
    with torch.no_grad():
        out_list = []
        for sample_data in data:
            speck_model.reset_states()
            in_events = chip_factory.xytp_to_events(
                sample_data, layer=layer_in, reset_timestamps=True
            )
            out_events = speck_model(in_events)
            # set dt to high value, don't need time dimension
            out_raster = chip_factory.events_to_raster(out_events, dt=1.0)
            out_list.append(out_raster.flatten(-3).sum(0).argmax(0))
        out: torch.Tensor = torch.tensor(out_list)
        true_pos.append(out == label)
        print(
            (
                f"{i: 4d} / {len(test_dataset) // 32}: "
                f"Current batch accuracy: {true_pos[-1].sum()/len(true_pos[-1]):.2%}."
                f" Overall: {torch.cat(true_pos).float().mean():.2%}"
            ),
            end="\r",
        )
        i += 1

true_pos = torch.cat(true_pos)
accuracy = true_pos.sum() / len(true_pos)
np.save("speck_accuracy.npy", accuracy)
print(f"\nFinal test accuracy {accuracy}")
