import tonic
import torch
import torch.nn as nn
from sinabs import from_model, layers, set_batch_size
from sinabs.backend.dynapcnn import DynapcnnNetwork
from sinabs.nir import from_nir, to_nir

import nir

ann = nn.Sequential(
    nn.Conv2d(
        2, 16, kernel_size=(5, 5), stride=(2, 2), padding=(1, 1), bias=False
    ),  # 16, 18, 18
    nn.ReLU(),
    nn.Conv2d(
        16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
    ),  # 8, 18,18
    nn.ReLU(),
    layers.SumPool2d(kernel_size=(2, 2)),  # 8, 17,17
    nn.Conv2d(
        16, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
    ),  # 8, 9, 9
    nn.ReLU(),
    layers.SumPool2d(kernel_size=(2, 2)),
    nn.Flatten(),
    nn.Linear(4 * 4 * 8, 256, bias=False),
    nn.ReLU(),
    nn.Linear(256, 10, bias=False),
    nn.ReLU(),
)

ann.load_state_dict(torch.load("rate_based_gregor.pth"))
input_shape = (2, 34, 34)
# Convert ann to snn
snn = from_model(
    ann, input_shape=(2, 34, 34), batch_size=1, spike_threshold=1.0, min_v_mem=-1.0
).spiking_model

# Convert SNN to NIR graph
nir_graph = to_nir(snn, sample_data=torch.rand((1, 2, 34, 34)))
nir_graph.infer_types()
# Save the graph
nir.write("scnn_mnist.nir", nir_graph)

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


# Test the model against NMNIST dataset

# Create event rasters of 1 ms time bin
to_frame = tonic.transforms.ToFrame(
    sensor_size=tonic.datasets.NMNIST.sensor_size, time_window=1e3
)
test_dataset = tonic.datasets.NMNIST(".", transform=to_frame, train=False)

# Define dataloader
batch_size = 32

data_loader = torch.utils.data.DataLoader(
    test_dataset,
    shuffle=False,
    batch_size=batch_size,
    collate_fn=tonic.collation.PadTensors(),
)
# Update the model to the batch size of the experiment
set_batch_size(speck_model, batch_size)

# Testing loop
true_pos = []

# Define device
device = "mps:0"
speck_model.to(device)

for data, label in data_loader:
    data = data.reshape((-1, *input_shape)).to(device)
    label = label.to(device)
    batch_size = len(label)
    set_batch_size(speck_model, batch_size)
    with torch.no_grad():
        out: torch.Tensor = speck_model(data).reshape((batch_size, -1, 10))
        true_pos.append((out.sum(dim=1).argmax(dim=1)) == label)
        print(f"Current batch accuracy: {true_pos[-1].sum()/len(true_pos[-1])}")

true_pos = torch.cat(true_pos)
print(f"Final test accuracy {true_pos.sum()/len(true_pos)}")
