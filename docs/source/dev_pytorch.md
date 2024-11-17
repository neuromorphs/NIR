# Developing PyTorch extensions

PyTorch is a popular deep learning framework that many of the NIR-supported libraries are built on.
We have built the [`nirtorch` package](https://github.com/neuromorphs/nirtorch) to make it easier to develop PyTorch extensions for the NIR-supported libraries.
`nirtorch` helps you write PyTorch code that (1) exports NIR models from PyTorch and (2) imports NIR models into PyTorch.

## Exporting NIR models from PyTorch
Exporting a NIR model requires two things: exporting the model's nodes and edges.

### Exporting edges
Exporting edges is slightly complicated because PyTorch modules can have multiple inputs and outputs.
And because PyTorch modules are connected via function calls, which only happen at runtime.
Therefore, we need to trace the PyTorch module to get the edges with some sample input.
Luckily, `nirtorch` package helps you do exactly that.
It works behind the scenes, but you can read about it in the [`to_nir.py` file in `nirtorch`](https://github.com/neuromorphs/NIRTorch/blob/main/nirtorch/to_nir.py#L11).

### Exporting nodes
The only thing we really have to do to use `nirtorch` is to export modules.
Since all PyTorch modules inherit from the `torch.nn.Module` class, exporting the nodes is straightforward: we simply need a function that looks at a PyTorch module and returns the corresponding NIR node.
Assume this is done in a function called `export_node`.

```python
import nir
import torch

class MyModule(torch.nn.Module):
    weight: torch.Tensor
    bias: torch.Tensor


def export_node(module: torch.nn.Module) -> Node:
    # Export the module to a NIR node
    if isinstance(module, MyModule):
        return nir.Linear(module.weight, module.bias)
    ...
```
This example converts a custom Linear module to a NIR Linear node.

### Putting it all together
The following code is a snippet taken from the [Norse library](https://github.com/norse/norse) that demonstrates how to export custom PyTorch models to a NIR using the `nirtorch` package.
Note that we only have to declare the `export_node` function for each custom module we want to export.
The edges are traced automatically by the `nirtorch` package.

```python
def _extract_norse_module(module: torch.nn.Module) -> Optional[nir.NIRNode]:
    if isinstance(module, LIFBoxCell):
        return nir.LIF(
            tau=module.p.tau_mem_inv,
            v_th=module.p.v_th,
            v_leak=module.p.v_leak,
            r=torch.ones_like(module.p.v_leak),
        )
    elif isinstance(module, torch.nn.Linear):
        return nir.Linear(module.weight, module.bias)
    elif ...

    return None

def to_nir(
    module: torch.nn.Module, sample_data: torch.Tensor, model_name: str = "norse"
) -> nir.NIRNode:
    return extract_nir_graph(
        module, _extract_norse_module, sample_data, model_name=model_name
    )
```

## Importing NIR models into PyTorch
Importing NIR models into PyTorch with `nirtorch` is also straightforward.
Assuming you have a NIR graph in the Python object `nir_graph` (see [Usage](#usage))