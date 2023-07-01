# NIR: Neuromorphic Intermediate Representation specification

NIR defines neuron models and connectivity for arbitrary networks that include spiking neurons. Neuron models are defined as dynamical system equations, because time is an essential component of neuromorphic systems. The goal is to provide a common format that different spiking neural network (SNN) frameworks can convert to. That allows a user to train an SNN in framework X and convert it to framework Y. Framework X might offer particularly fast training while framework Y might offer deployment to neuromorphic hardware.

## Computational units
On top of popular primitives such as convolutional or fully connected/linear computations, we define additional compuational primitives that are specific to neuromorphic computing and hardware implementations thereof. Computational units that are not specifically neuromorphic take inspiration from the Pytorch ecosystem in terms of naming and parameters (such as Conv2d that uses groups/strides). Example definitons of computational units:

$$ 
\text{LIF}: [ \tau, \alpha, \beta ] \\
\text{Linear}: \mathbb{R}^{m \times n},  \mathbb{R}^n \\
\text{Conv2d}: \mathbb{R}^{c_{out} \times c_{in} \times y \times x},  \text{Strides}, \text{Groups}, ... \\

$$ 
where LIF is defined as a dynamical equation: $$ \tau \dot{v} = \alpha(v_{leak} - v) + \beta i $$ 


## Connectivity 
Each computational unit is a node in a static graph. Given 3 nodes $A$ which is a LIF node, $B$ which is a Linear node and $C$ which is another LIF node, we can define edges in the graph such as:

$$
A \rightarrow B \\
B \rightarrow C
$$

## Format
The intermediate represenation can be stored as hdf5 file, which benefits from compression. 

## Frameworks that currently support NIR:
* work in progress
* another work in progress