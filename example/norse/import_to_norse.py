import norse.torch as norse
import torch

import nir

# Create a NIR Network
affine_weights = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
affine_bias = torch.tensor([1.0, 2.0])
li_tau = torch.tensor([0.9, 0.8])
li_r = torch.tensor([1.0, 1.0])
li_v_leak = torch.tensor([0.0, 0.0])
nir_network = nir.NIRGraph.from_list(
    nir.Affine(affine_weights, affine_bias), nir.LI(li_tau, li_r, li_v_leak)
)

# Import to Norse
norse_network = norse.from_nir(nir_network)
