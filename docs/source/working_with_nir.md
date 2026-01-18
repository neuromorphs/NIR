(working_with_nir)=
# Working with NIR graphs

## The NIR format
NIR is an intermediate representation and **only consists of declarations**.
That is, we do not implement any dynamic or describe any runtime behavior---that is up to the individual platforms.
NIR simply describes graphs with (1) nodes and (2) connections between the nodes (edges). 
(1) Nodes describe *what* computation the node performs and the parameters required for the node to compute that.
One example of is the [`Linear` node](https://github.com/neuromorphs/NIR/blob/main/nir/ir/linear.py#L41) which describes a linear mapping $W x$ of some signal $x$, parameterized by weights $W$.
Another example is the [`NIRGraph` node](https://github.com/neuromorphs/NIR/blob/main/nir/ir/graph.py#L21) which is itself a graph. This permits us to have aribtrarily *hierarchical* subgraphs. In principle infinite subgraphs, but we don't recommend that.
The top-most node is typically a single [`NIRGraph` object](api_design.md#nir-graphs-and-edges).
(2) Edges are simply tuples that connect two nodes, e.g. `("A", "B")` in Python. That's it.


```{figure} nir_graph_example.svg
---
height: 200px
label: nir-graph-example
---
An example of a NIR graph with four nodes: Input, Leaky-Integrator, Affine map, and Output.
```

```{note}
See our [API design](api_design.md) for more information about the NIR hierarchical structure.

See the [NIR primitives](primitives.md) for more information about the individual nodes.
```

Graphs can have arbitrary many nodes and arbitrary many connections.
That means that it's entirely possible to create impossible graphs.
You can, in principle, have multiple edges to and from the same nodes.
But it may render your graph useless for many hardware or simulation platforms, so be careful.

Below, we proceed by explaining how you can get NIR graphs, process them, and output them again.

## Getting a NIR graph: Reading NIR
NIR is integrated into numerous frameworks, which provide helper methods to read and write NIR.
One example is the [`export_to_nir` function in snnTorch](https://snntorch.readthedocs.io/en/latest/snntorch.export_nir.html) which translates an snnTorch model to NIR.

```python
import snntorch
my_snntorch_net = torch.nn.Sequential( ... )
nir_graph = snntorch.export_to_nir(my_snntorch_net)
```

Another way is to read NIR graphs from [HDF5 files](https://en.wikipedia.org/wiki/Hierarchical_Data_Format).
This is useful when you want to send a model over email, store it for later, or share it with a colleague.
Note that we provide a reference implementation in Python in the `nir` package, but HDF5 files can be read and written by a host of other languages.

To read a model from a file, use the `nir.read` function with the path to the graph.
```python
import nir
nir_graph = nir.read("my_graph.nir")
```

This gives you a NIR model, which you can process as shown below.

## Traversing NIR graphs

Now that you have a graph in a Python variable (`nir_graph`), we can examine what's inside!
Note that this builds on the reference implementation, but that other implementations may exist.

The topmost node is typically a [`NIRGraph` node](https://github.com/neuromorphs/NIR/blob/main/nir/ir/graph.py#L21) which itself contains nodes and edges.
You can access those by the `.nodes` and `.edges` properties, respectively.

```python
nodes = nir_graph.nodes # A Dictionary of str -> nir.NIRNode
edges = nir_graph.edges # A List tuples (str, str)
```

Edges are pretty boring, because they just list the connections from one node to another.
Nodes are more interesting and contain much more data.
Let's assume we have a graph as in [the figure above](#nir-graph-example) (`Input -> LI -> Affine -> Output`), we can traverse the graph as follows:

```python
for name, node in nodes.items():
    print(name, node)
```

Since each node is required to have a name, we can also use that to look up nodes directly.
Say that the `Affine` node has the (startling) name "affine":

```python
affine_node = nodes["affine"]
print(f"Affine weight shape {affine_node.weight.shape}")
print(f"Affine bias shape {affine_node.bias.shape}")
```

Here, we access the properties of the node, such as the `weight` and `bias` properties of the Affine node.
All elements in NIR nodes are [Numpy](https://numpy.org/) arrays (where applicable), so you can use all the methods from Numpy to manipulate or transform the properties in the nodes.

## Outputting a NIR graph: Writing NIR

Finally, you may want to write a model to a file.
If you have a `NIRGraph` present, you can write it directly using the `nir.write` function.
The `nir.write` function takes two arguments: the file path and the model to write.
```python
import nir
my_nir_graph = ...
nir.write("my_graph.nir", my_model)
```

This produces an [HDF5 file](https://en.wikipedia.org/wiki/Hierarchical_Data_Format), which is a quite universal file format that can be read by multiple platforms and languages, including C, C++, Java, and, of course, Python.