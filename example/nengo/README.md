# Nengo <-> NIR integration

[Nengo](nengo.ai) is a Python package for building, testing, and deploying neural networks.
The examples below shows how to import and export from Nengo to NIR and vice-versa.

Examples:
* Lorentz oscillator: [nir-lorentz.py](https://github.com/neuromorphs/nir/tree/main/example/nengo/nir-lorentz.py)
    * This script creates a Nengo model that simulates the Lorentz oscillator, maps it to NIR, and then back again into Nengo
* Leaky integrate-and-fire (LIF) tests [nir-test.py](https://github.com/neuromorphs/nir/tree/main/example/nengo/nir-test.py)
    * Creates a NIR model for an Affine map and LIF population and map it into Nengo

## Credits
* [Terry Stewart](http://terrystewart.ca/)
* [Jens Pedersen](https://jepedersen.dk)