(primitives)=
# Primitives

At its core, NIR is simply a [directed graph](https://en.wikipedia.org/wiki/Directed_graph) (using the [`NIRGraph` primitive](https://github.com/neuromorphs/NIR/blob/main/nir/ir/graph.py)).
The nodes of the graph are computational units, and the edges are the (directed) connections between them.
There are no restrictions on the graph structure, so it can be a simple feedforward network, a recurrent network, a graph with cycles, and even with duplicated connections, if needed.

But, if you plan to execute the graph on restricted neuromorphic hardware, please **verify that the graph is compatible with the hardware**.

## NIR computational primitives

NIR defines 17 fundamental primitives listed in the table below, which backends are free to implement as they want, leading to varying outputs across platforms. While discrepancies could be minimized by constraining implementations or making backends aware of each other's discretization choices, NIR does not do this since it is declarative, specifying only the necessary inputs and outputs. Constraining implementations would cause hardware incompatibilities and making backends aware of each other could create large O(N^2) overhead for N backends. The primitives are already computationally expressive and able to solve complex PDEs.
Units are indicated in brackets, such as [mV] for [millivolt](https://en.wikipedia.org/wiki/Volt).

| Primitive                          | Parameters                                                                 | Computation                                               | Reset                                                                                   |
|------------------------------------|---------------------------------------------------------------------------|----------------------------------------------------------|----------------------------------------------------------------------------------------|
| **Input**                          | Input shape                                                               | -                                                        | -                                                                                      |
| **Output**                         | Output shape                                                              | -                                                        | -                                                                                      |
| **Affine**                         | $W$, $b$                                              | $W \cdot I + b$                                          | -                                                                                      |
| **Convolution**                    | $W$, Stride, Padding, Dilation, Groups, Bias          | $f \star g$                                              | -                                                                                      |
| **Current-based leaky Integrator** | $\tau_\text{syn}$ [ms], $\tau_\text{mem}$ [ms], $R$ [Ω], $v_\text{leak}$ [mV], $w_\text{in}$ | **LI**; **Linear**; **LI**                                       | -                                                                                      |
| **Current-based leaky integrate-and-fire** | $\tau_\text{syn}$ [ms], $\tau_\text{mem}$ [ms], $R$ [Ω], $v_\text{leak}$ [mV], $v_\text{reset}$ [mV], $v_\text{threshold}$ [mV], $w_\text{in}$ | **LI**; **Linear**; **LIF**                              | $\begin{cases} v_\text{reset} & \text{Spike} \\ v_\text{LIF} & \text{else} \end{cases}$ |
| **Delay**                          | $\tau$ [ms]                                                                | $I(t - \tau)$                                            | -                                                                                      |
| **Flatten**                        | Input shape, Start dim., End dim.                                         | -                                                        | -                                                                                      |
| **Integrator**                     | $R$ [Ω]                                                                   | $\dot{v} = R I$                                          | -                                                                                      |
| **Integrate-and-fire**             | $R$ [Ω], $v_\text{reset}$ [mV], $v_\text{threshold}$ [mV]                        | **Integrator**; **Threshold**                            | $\begin{cases} v_\text{reset} & \text{Spike} \\ v & \text{else} \end{cases}$         |
| **Leaky integrator (LI)**          | $\tau$ [ms], $R$ [Ω], $v_\text{leak}$ [mV]                                  | $\tau \dot{v} = (v_\text{leak} - v) + R I$               | -                                                                                      |
| **Linear**                         | $W$                                                      | $W I$                                                    | -                                                                                      |
| **Leaky integrate-fire (LIF)**     | $\tau$ [ms], $R$ [Ω], $v_\text{leak}$ [mV], $v_\text{reset}$ [mV], $v_\text{threshold}$ [mV] | **LI**; **Threshold**                                    | $\begin{cases} v_\text{reset} & \text{Spike} \\ v & \text{else} \end{cases}$         |
| **Scale**                          | $s$                                                                       | $s I$                                                    | -                                                                                      |
| **SumPooling**                     | $p$                                                                       | $\sum_{j} x_j$                                           | -                                                                                      |
| **AvgPooling**                     | $p$                                                                       | **SumPooling**; **Scale**                                | -                                                                                      |
| **Threshold**                          | $\theta_\text{threshold}$ [mV]                                                   | $\delta(I - \theta_\text{threshold})$                               | -                                                                                      |                                                                            |

More details about the equations the computational primitives are implementing are available in the [paper on Neuromorphic Intermediate Representation](https://www.nature.com/articles/s41467-024-52259-9).

## Connectivity

In the graph, each node has a name like "Neuron 1" or, in some cases, simply just an index "1".
Connections between nodes are simply a tuple of the strings desribing the source and target.
As an example, `("A", "B")`, tells us that the output of node `A` is sent to node `B`.

Describing the full connectivity in a graph is as simple as listing all the connections in the graph:
```
[
    ("A", "B"),
    ("B", "C"),
    ("C", "D"),
    ...
]
```

## Input and output nodes
Given a graph, how do we know which nodes should receive inputs? And which nodes should provide outputs?
For that, we define two special nodes: `Input` and `Output`.
Both nodes are "dummies" in the sense that they do not provide any function, apart from marking the beginning and end of the graph.
Note that a single node can be both an input and an output node.

To clarify the dimensionality/input types of the input and output nodes, we require the user to specify the shape *and* name of the input, like so:
```python
import numpy as np
nir.Input(
    input_type = {"input": np.array([28, 28])}
)
nir.Output(
    output_type = {"output": np.array([2])}
)
```

## Metadata

Each node in the graph can have metadata attached to it.
The metadata is a dictionary that can contain any information that may be helpful for the user or backend.
Any dictionary entries can be added, although we recommend restricting the entries to strings, numbers, and arrays.
Here is an example of a metadata dictionary attached to a graph:

```python
import nir

nir.NIRGraph(
    ...,
    metadata = {"some": "metadata", "info": 1}
)
```


```{admonition} Do not rely on the metadata
:class: warning
It's vital to ensure that **no backend should rely on this metadata**.
Metadata entries should contain non-essential meta-information about nodes or graphs, such as the discretization scheme with which the graph was trained, timestamps, etc.
Tidbits that can improve the model or execution, but are not necessary for the execution itself.

If the backend would strictly rely this metadata, it would require everyone else to adhere to this non-enforced standard.
NIR graphs should be self-contained and unambiguous, such that the graph itself (without the metadata) contains all the necessary information to execute the graph.
```

## How to program with primitives

```{admonition} See also
:class: info
See [the usage page](./usage.md) for more information on how to use NIR in practice.
```

Programming with primitives involves creating a graph of nodes whose (directional) connections indicate where signals travel.
For this, we typically import and export one single graph node containing multiple subnodes (which can themselves contain nodes, and so on).

NIR graphs can be loaded from [HDF5 files](https://en.wikipedia.org/wiki/Hierarchical_Data_Format), but we also provide a reference implementation in Python in this repo.

### A Graph Example in Python
To illustrate how a computational graph can be defined using the NIR Python primitives, here is an example of a graph with a single `LIF` neuron with input and output nodes:

```python
import nir

nir.NIRGraph(
    nodes = {
        "input" : nir.Input({"input": np.array([1])}),
        "lif"   : nir.LIF(...),
        "output": nir.Output{"output": np.array([1])}
    },
    edges = [
        ("Input", "LIF"),
        ("LIF"  , "Output"),
    ],
)
```
