# Using NIR in hardware

NIR is easily portable to any platform and is liberally licensed under the BSD-3 clause, so it can be used in any project, commercial or open-source.
We have a reference implementation in Python, but can export [HDF5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) files that can be read by any language or platform.
**To use NIR, you simply convert the nodes and edges into your platform's primitives**.
That's all.
Let us unpack that statement.

## The NIR format
NIR is an intermediate representation and **only consists of declarations**.
That is, we do not implement any dynamic or describe any runtime behavior---that is up to the individual platforms.
NIR consists of a *hierarchical* structure, with the top-most level being a single [`NIRGraph` object](api_design.md#nir-graphs-and-edges).

```{figure} nir_graph_example.svg
---
height: 200px
name: nir-graph-example
---
An example of a NIR graph with four nodes: Input, Leaky-Integrator, Affine map, and Output.
```

```{note}
See our [API design](api_design.md) for more information about the NIR hierarchical structure.

See the [NIR primitives](primitives.md) for more information about the individual nodes.
```

## Integrating with Python

First step is to load the NIR graph into your platform.
In Python, you can import NIR as a library (installable via [`pip install nir`](https://pypi.org/project/nir/)).
If your graph is stored in a file, you can load it using the `nir.read` function.

```python
import nir
my_graph = nir.read("path_to_my_graph.nir")
```

Once that is done, the graph can be parsed by (1) matching the nodes to your platform's primitives and (2) connecting the nodes together.
Note that the top-level graph may be recursive, so we recommend a recursive function that traverses the graph and evaluates the nodes.
Here's a simple example (without recursion):

```python

import nir

def parse_graph(graph: nir.NIRGraph):
    # Create a dictionary of nodes
    nodes = {}
    for name, node in graph.nodes.items():
        # Match the node to your platform's primitive
        if isinstance(node, nir.Input):
            nodes[name] = MyPlatformInput()
        elif isinstance(node, nir.LI):
            nodes[name] = MyPlatformLeakyIntegrator(node.tau, node.r, node.v_leak)
        elif isinstance(node, nir.Affine):
            nodes[name] = MyPlatformAffine(node.weights, node.bias)
        elif isinstance(node, nir.Output):
            nodes[name] = MyPlatformOutput()
        else:
            raise NotImplementedError(f"Node {node} not supported.")
    
    # Connect the nodes
    for edge in graph.edges:
        # Connect the nodes
        nodes[edge[0]].connect(nodes[edge[1]])

    return nodes

```

```{note}
See the [NIR primitives](primitives.md) for more information about the content of each node.
```

### Integrating with PyTorch

Since several libraries are built on top of PyTorch, we provide default PyTorch mappings in [nirtorch](https://github.com/neuromorphs/nirtorch).
`nirtorch` provides a simple way to write and load NIR graphs, but you still need to let `nirtorch` know how to evaluate the SNN-specific nodes (such as Leaky-Integrator and Spike).

```python
import nir, nirtorch

# Map nodes that are specific to your library
# - nirtorch will map obvious nodes like `Input`, `Output`, `Affine`, `Conv2d` etc.
# - but only if your parsing function do not return a module for that node
def parse_module(node: nir.NIRNode) -> Optional[torch.nn.Module]:
    if isinstance(module, LIFBoxCell):
        return ...
    else:
        return None # Return none to allow nirtorch to map the node

# Load a graph as a PyTorch module (`torch.nn.Module`)
nir_graph = ...
torch_graph = nirtorch.load(nir_graph, parse_module)
```

## Integrating via HDF5 files

If you are not using Python, you can load the NIR graph from the HDF5 file and parse it using your platform's primitives.
The data follows the API structure from before.
Values in the nodes are encoded as [Numpy arrays](https://numpy.org/doc/stable/reference/c-api/array.html).

HDF5 interfaces with numerous languages, including C, C++, Java, and MATLAB.
We refer to the Wikipedia page on [HDF5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) for more information on how to use it in your language.

## What if my platform doesn't support `X`?
NIR contains a number of primitives, like the `LI` (Leaky Integrator) node, or `Spike` (threshold) primitives, that may not be directly supported by your platform.
In this case, you have two options:
* **Approximate the behavior**: You can approximate the behavior of the node using your platform's primitives.
  For example, the `LIF` node can be approximated by a simple leaky integrator.
* **Ignore the node**: If your platform does not support the node, you can simply ignore it.
  This is a valid strategy, as most hardware platforms are naturally constrained. In this case, we advice that you simply raise an exception to the user and inform them that the node is not supported.
  In our [roadmap](roadmap.md), we plan to work on optimization and approximation strategies for these cases.
  If this is interesting for you, we invite you to [about#Contact](get in touch).
