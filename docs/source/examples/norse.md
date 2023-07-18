# NIR ⇄ Norse

NIR integration is supported in Norse natively.

## NIR → Norse

As explained in the [usage]

```python
import torch
import norse.torch as norse

model = norse.SequentialState(
  norse.LIFCell(),
  ...
)

norse.to_nir(model, torch.randn(1, 10))
```

## Norse → NIR
```python
import norse.torch as norse

nir_model = ...
norse.from_nir(nir_model)
```