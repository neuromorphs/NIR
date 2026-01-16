# NIRTorch API Documentation

This page lists functions and classes exposed by the `nirtorch` package for converting between PyTorch models and NIR.

## Main Functions

- **Graph extraction**: Convert PyTorch `nn.Module` to NIR graph
- **Graph loading**: Convert NIR graph to PyTorch `nn.Module`

For detailed usage examples and API documentation, see:
- [PyTorch developer guide](dev_pytorch.ipynb)
- [nirtorch source code](https://github.com/neuromorphs/nir/tree/main/nirtorch)

## Key Modules

- **Tracing**: Extract computational graphs from PyTorch models - see [tracing guide](nirtorch/tracing.ipynb)
- **Interpreting**: Load NIR graphs as PyTorch modules - see [interpreting guide](nirtorch/interpreting.ipynb)
- **State management**: Handle stateful operations - see [state guide](nirtorch/state.md)
