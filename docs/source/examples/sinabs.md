# NIR ⇄ Sinabs (SynSense)

NIR integration is supported in Sinabs natively.

`sinabs.to_nir` and `sinabs.from_nir` methods allow you to seemlessly navigate between `nir` and `sinabs`. Once your model is in sinabs, you can use this model to train or directly deploy your models to Speck/DynapCNN. 

## NIR → Sinabs

```python
import torch
import torch.nn as nn
import sinabs.layers as sl
from sinabs import to_nir

batch_size = 4

# Create Sinabs model
orig_model = nn.Sequential(
    nn.Linear(10, 2),
    sl.ExpLeakSqueeze(tau_mem=10.0, batch_size=batch_size),
    sl.LIFSqueeze(tau_mem=10.0, batch_size=batch_size),
    ...
)

# Convert model to NIR graph with a random input of representative shape
sample_data = torch.randn(batch_size, 10)
nir_graph = to_nir(orig_model, sample_data)
```

## Sinabs → NIR
```python
import sinabs.from_nir

nir_model = ...
sinabs_model = sinabs.from_nir(nir_model, batch_size=4)
```