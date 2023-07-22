import sinabs.layers as sl
import torch
import torch.nn as nn
from sinabs import from_nir, to_nir


batch_size = 4

# Create Sinabs model
orig_model = nn.Sequential(
    torch.nn.Linear(10, 2),
    sl.ExpLeakSqueeze(tau_mem=10.0, batch_size=batch_size),
    sl.LIFSqueeze(tau_mem=10.0, batch_size=batch_size),
    torch.nn.Linear(2, 1),
)

# Convert model to NIR graph with a random input of representative shape
nir_graph = to_nir(orig_model, torch.randn(batch_size, 10))
print(nir_graph)


# Reload sinabs model from NIR
sinabs_model = from_nir(nir_graph, batch_size)
