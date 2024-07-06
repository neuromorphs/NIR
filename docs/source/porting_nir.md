# Using NIR on your platform

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



## Integrating via HDF5 files


## What if my platform doesn't support `X`?
NIR contains a number of primitives, like the `LI` (Leaky Integrator) node, or `Spike` (threshold) primitives, that may not be directly supported by your platform.
In this case, you have two options:
* **Approximate the behavior**: You can approximate the behavior of the node using your platform's primitives.
  For example, the `LIF` node can be approximated by a simple leaky integrator.
* **Ignore the node**: If your platform does not support the node, you can simply ignore it.
  This is a valid strategy, as most hardware platforms are naturally constrained. In this case, we advice that you simply raise an exception to the user and inform them that the node is not supported.
  In our [roadmap](roadmap.md), we plan to work on optimization and approximation strategies for these cases.
  If this is interesting for you, we invite you to [about#Contact](get in touch).
