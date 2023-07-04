# NIR: Neuromorphic Intermediate Representation specification

NIR defines neuron models and connectivity for arbitrary networks that include spiking neurons. Neuron models are defined as dynamical system equations, because time is an essential component of neuromorphic systems. The goal is to provide a common format that different spiking neural network (SNN) frameworks can convert to. That allows a user to train an SNN in framework X and convert it to framework Y. Framework X might offer particularly fast training while framework Y might offer deployment to neuromorphic hardware.

## Computational units
On top of popular primitives such as convolutional or fully connected/linear computations, we define additional compuational primitives that are specific to neuromorphic computing and hardware implementations thereof. Computational units that are not specifically neuromorphic take inspiration from the Pytorch ecosystem in terms of naming and parameters (such as Conv2d that uses groups/strides). Example definitons of computational units:


$$\text{LI}: [ \tau, R, v_{leak}]$$

$$\text{LIF}: [ \tau, R, v_{leak}, v_{threshold} ]$$

$$\text{Linear}: \mathbb{R}^{m \times n},  \mathbb{R}^n$$

$$\text{Conv1d}: \mathbb{R}^{c_{out} \times c_{in} \times x},  \text{Strides}, \text{Groups}, ... $$ 

$$\text{Conv2d}: \mathbb{R}^{c_{out} \times c_{in} \times y \times x},  \text{Strides}, \text{Groups}, ... $$ 

where LIF is defined as a dynamical equation:

$$ \tau \dot{v} = v_{leak} - v + R i $$ 


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

## Acknowledgements

Authors (in alphabetical order):
* [Steven Abreu](https://github.com/stevenabreu7)
* [Felix Bauer](https://github.com/bauerfe)
* [Jason Eshraghian](https://github.com/jeshraghian)
* [Matthias Jobst](https://github.com/matjobst)
* [Gregor Lenz](https://github.com/biphasic)
* [Jens Egholm Pedersen](https://github.com/jegp)
* [Sadique Sheik](https://github.com/sheiksadique)

If you use NIR in your work, please cite the [following Zenodo reference](https://zenodo.org/record/8105042)

```
@software{nir2023,
  author       = {Abreu, Steven and
                  Bauer, Felix and
                  Eshraghian, Jason and
                  Jobst, Matthias and
                  Lenz, Gregor and
                  Pedersen, Jens Egholm and
                  Sheik, Sadique},
  title        = {Neuromorphic Intermediate Representation},
  month        = jul,
  year         = 2023,
  publisher    = {Zenodo},
  version      = {0.0.1},
  doi          = {10.5281/zenodo.8105042},
  url          = {https://doi.org/10.5281/zenodo.8105042}
}
```
