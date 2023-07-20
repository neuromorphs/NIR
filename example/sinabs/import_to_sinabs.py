import torch
from sinabs import from_nir

import nir

# Create a NIR graph
affine_weights = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
affine_bias = torch.tensor([1.0, 2.0])
li_tau = torch.tensor([0.9, 0.8])
li_r = torch.tensor([1.0, 1.0])
li_v_leak = torch.tensor([0.0, 0.0])
nir_network = nir.NIRGraph.from_list(
    nir.Affine(affine_weights, affine_bias), nir.LI(li_tau, li_r, li_v_leak)
)

# Create Sinabs model from NIR graph.
# You need to define the batch size because Sinabs will use Squeeze
# versions of layers by default.
sinabs_model = from_nir(nir_network, batch_size=10)
print(sinabs_model)
