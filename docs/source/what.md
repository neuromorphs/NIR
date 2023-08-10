# What is the Neuromorphic Intermediate Representation (NIR)?

NIR is two things: a specification and a community.
The specification defines a common format for physical neural networks, including neuromorphic, [spiking neural networks](https://en.wikipedia.org/wiki/Spiking_neural_network).
The community develops and integrates the NIR specifiation into different frameworks and platforms.

## The NIR specification
NIR is a declarative specification that describes how computational nodes are connected. 
This permits NIR to describe *arbitrary* networks of *any kind*.
NIR does not restrict the type of computation that can be performed or connection that can be made.

Computational nodes are described as continuous ordinary differential equations, because time is an essential component of any physical or neuromorphic system.
is to provide a common format that different spiking neural network (SNN) frameworks can convert to and from.
Ultimately, this allows any user to train a network in framework X and convert it to framework Y.
Framework X might offer particularly fast training while framework Y might offer deployment to neuromorphic hardware.

## The NIR community
This figure shows how NIR works as a middleman between neuromorphic simulators and platforms:

![NIR platforms](nir_platforms.png)
