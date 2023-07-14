import norse.torch as norse
import torch

import nir

# Create a network
network = norse.SequentialStateful(norse.LIFCell(), torch.nn.Linear(1, 1))

# Export to nir
nir_model = norse.to_nir(network)

# Save to file
nir.write("nir_model.nir", nir_model)
