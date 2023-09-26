import sinabs.layers as sl
import torch
import torch.nn as nn
from sinabs import from_model
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
    sl.SumPool2d(kernel_size=(2, 2)),  # 8, 17,17
    nn.Conv2d(
        16, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
    ),  # 8, 9, 9
    nn.ReLU(),
    sl.SumPool2d(kernel_size=(2, 2)),
    nn.Flatten(),
    nn.Linear(4 * 4 * 8, 256, bias=False),
    nn.ReLU(),
    nn.Linear(256, 10, bias=False),
    nn.ReLU(),
)

ann.load_state_dict(torch.load("rate_based_gregor.pth"))

snn = from_model(ann, input_shape=(2, 34, 34), batch_size=1).spiking_model

print(snn)


nir_graph = to_nir(snn, sample_data=torch.rand((1, 2, 34, 34)))
# Save the graph
nir.write("scnn_mnist.nir", nir_graph)

print(nir_graph)

# Load sinabs model from nir graph

sinabs_model = from_nir(nir_graph, batch_size=1)
print(sinabs_model)
