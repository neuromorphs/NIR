# NIR primitives

NIR defines a set of primitives that establishes 

On top of popular primitives such as convolutional or fully connected/linear computations, we define additional compuational primitives that are specific to neuromorphic computing and hardware implementations thereof. Computational units that are not specifically neuromorphic take inspiration from the Pytorch ecosystem in terms of naming and parameters (such as Conv2d that uses groups/strides). Example definitons of computational units:

$$
\begin{align}
\text{Affine} &: \mathbb{R}^{m \times n},  \mathbb{R}^n \\
\text{CubaLIF} &:  [ \tau, R, v_{leak}, v_{threshold} ] & \text{Conductance-based LIF}\\
\text{I} &:  [R] \\
\text{LI} &:  [\tau, R, v_{leak}] \\
\text{LI} &:  [\tau, R, v_{leak}] \\
\text{LIF} &:  [ \tau, R, v_{leak}, v_{threshold} ] \\
\text{Linear} &:  \mathbb{R}^{m \times n} \\
\text{Conv1d} &:  \mathbb{R}^{c_{out} \times c_{in} \times x},  \text{Strides}, \text{Groups}, ... \\
\text{Conv2d} &:  \mathbb{R}^{c_{out} \times c_{in} \times y \times x},  \text{Strides}, \text{Groups}, ... \\
\text{SumPool2d} &: \mathbb{R}^{c \times y \times x}, \text{Pooling}, \text{Strides}, \text{Padding} ...\\
\text{Threshold} &:  \begin{cases} 1 & v > v_{threshold} \\ 0 & else \end{cases}
\end{align}
$$

Each primitive is defined by their own dynamical equation, specified in the [API docs](https://nnir.readthedocs.io/en/latest/modindex.html).

## Connectivity 

Each computational unit is a node in a static graph.
Given 3 nodes $A$ which is a LIF node, $B$ which is a Linear node and $C$ which is another LIF node, we can define edges in the graph such as:

$$
    A \rightarrow B \\
    B \rightarrow C
$$

## Format
The intermediate represenation can be stored as hdf5 file, which benefits from compression. 