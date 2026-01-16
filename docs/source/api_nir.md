# NIR API Documentation

This page lists the main functions and classes exposed by the `nir` package.

## Reading and writing NIR files

The NIR package provides functions for reading and writing neuromorphic models to/from HDF5 files:

- `nir.read(filename)` - Read a NIR graph from an HDF5 file
- `nir.write(filename, graph)` - Write a NIR graph to an HDF5 file

For complete API documentation, see the [nir module source code](https://github.com/neuromorphs/nir/tree/main/nir).

## NIR Nodes

All NIR primitives are defined in the `nir.ir` module. See the [primitives documentation](primitives.md) for details on available node types.

### Core Classes

- `NIRNode` - Base class for all NIR nodes
- `NIRGraph` - Container for NIR graphs with nodes and edges

For the complete list of primitives, see [supported primitives](supported_primitives.md).
