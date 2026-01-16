(api_design)=
# API design

NIR is simple: it consists of a series of objects that *represent* [NIR structures](#primitives).
In other words, they do not implement the functionality of the nodes, but simply represent the necessary parameters required to *eventually* evaluate the node.

We chose Python because the language is straight-forward, known by most, and has excellent [dataclasses](https://docs.python.org/3/library/dataclasses.html) exactly for our purpose.
This permits an incredibly simple structure, where we have encoded all the NIR primitives into a single [`ir.py` file](https://github.com/neuromorphs/NIR/blob/main/nir/ir.py), with a simple structure:

```python
@dataclass
class MyNIRNode(NIRNode):
    some_parameter: np.ndarray
```

In this example, we create a class that inherits from the parent [`NIRNode`](https://github.com/neuromorphs/NIR/blob/main/nir/ir.py#L160) with a single parameter, `some_parameter`.
Instantiating the class is simply `MyNIRNode(np.array([...]))`.

## NIR Graphs and edges
```{figure} nir_graph_example.svg
---
height: 200px
name: nir-graph-example
---
An example of a NIR graph with four nodes: Input, Leaky-Integrator, Affine map, and Output.
```

A collection of nodes is a `NIRGraph`, which is, you guessed it, a `NIRNode`.
But the graph node is special in that it contains a number of named nodes (`.nodes`) and connections between them (`.edges`).
The nodes are named because we need to uniquely distinguish them from each other, so `.nodes` is actually a dictionary (`Dict[str, NIRNode]`).
With our node above, we can define `nodes = {"my_node": MyNIRNode(np.array([...]))}`.

Edges are simply two strings: a beginning and ending node in a tuple (`Tuple[str, str]`).
There are no restrictions on edges; you can connect nodes to themselves---multiple times if you wish.
That would look like this: `edges = [("my_node", "my_node")]`.

In sum, a rudimentary, self-cyclic graph can be described in NIR as follows:

```python
NIRGraph(
    nodes = {"my_node": MyNIRNode(np.array([...]))},
    edges = [("my_node", "my_node")]
)
```

## Input and output types
All nodes are expected to carry two internal variables (`input_type` and `output_type`), that describe the names *and* shape of the input and outputs of the node as a dictionary (`Dict[str, np.ndarray]`).
The variables can be equalled to the function declaration in programming languages, where the names and types of the function arguments are given.
Most nodes have a single input (`"input"`) and output (`"output"`), in which case their `input_type` and `output_type` have single (trivial) entries: `{"input": ...}` and `{"output": ...}`.
Other nodes are more complicated and use explicit names for their arguments (see below).
In most cases the variables are inferred in the [`__post_init__`](https://docs.python.org/3/library/dataclasses.html#post-init-processing) method, but new implementations will have to somehow assign them.

## Subgraphs and types
The decision to include the input and output types were made to disambiguate connectivity between nodes.
Immediately, they allow us to ensure that an edge connecting two nodes are valid; that the type in one end corresponds to the type in the other end.
But the types are necessary in cases where we wish to connect *to* or *from* subgraphs.

Consider a subgraph `G` with two nodes `B` and `C`.
How can we specifically describe connectivity to `B` and not `C`?
By using the input types, we can *subscript* the edge to specify exactly which input we're connecting to: `G.B`.
An edge from `A` to `B` would then look like this: `("A", "G.B")`.
The same process works out of the graph, thanks to the `output_type`: we simply create an outgoing edge from `G.A`.

See [the unit test file `test_architectures.py`](https://github.com/neuromorphs/NIR/blob/main/tests/test_architectures.py) for concrete examples on NIR graphs, input/output types, and subgraphs.