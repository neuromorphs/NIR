# What is the Neuromorphic Intermediate Representation (NIR)?

NIR defines neuron models and connectivity for arbitrary networks that include spiking neurons. Neuron models are defined as dynamical system equations, because time is an essential component of neuromorphic systems. The goal is to provide a common format that different spiking neural network (SNN) frameworks can convert to. That allows a user to train an SNN in framework X and convert it to framework Y. Framework X might offer particularly fast training while framework Y might offer deployment to neuromorphic hardware.

This figure shows how NIR works as a middleman between neuromorphic simulators and platforms:

![NIR platforms](nir_platforms.png)
